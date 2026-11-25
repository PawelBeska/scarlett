//! Reference interpreter for RDNA2 shader code: one wave, in wave32 or
//! wave64 mode.
//!
//! This is the oracle for the shader translator. A translated shader can
//! be checked against what this interpreter computes for the same inputs,
//! lane by lane, without a GPU. It favours being obviously right over
//! speed, and refuses (with [`ExecError::Unsupported`]) anything it does
//! not model exactly rather than guessing.
//!
//! Modelled: scalar registers, `vcc`, `exec`, `m0`, `scc`, vector
//! registers per lane, inline constants and literals, the SOPP branches,
//! and the opcodes listed in [`crate::rdna`]'s `sop1`/`sop2`/`sopc`/
//! `vop1`/`vop2`/`vopc`/`vop3` tables.
//!
//! Wave size matters in three places, all handled here: in wave32 only the
//! low halves of `exec` and `vcc` exist (a VOPC writes `vcc_lo`, the high
//! half is left alone), `*_saveexec_b32` is the wave32 form, and lane
//! masks are 32 bits wide. Most console shaders are wave64, so both modes
//! are first-class.

use crate::cp::Memory;
use crate::rdna::{
    self, decode, ds, operand, smem, vintrp, sop1, sop2, sopc, sopp, vop1, vop2, vop3, DecodeError, Fields,
    Inst, Operand,
};

pub const NUM_SGPRS: usize = 106;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveSize {
    W32,
    W64,
}

impl WaveSize {
    pub fn lanes(self) -> usize {
        match self {
            WaveSize::W32 => 32,
            WaveSize::W64 => 64,
        }
    }

    /// Mask of the lanes that exist.
    pub fn mask(self) -> u64 {
        match self {
            WaveSize::W32 => 0xFFFF_FFFF,
            WaveSize::W64 => u64::MAX,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum ExecError {
    Decode(DecodeError),
    /// An instruction or operand this interpreter does not model.
    Unsupported { at: usize, what: String },
    /// Ran past the end of the code without `s_endpgm`.
    PcOutOfRange(usize),
    StepLimit,
    /// A load from unmapped memory.
    MemoryFault { at: usize, addr: u64 },
}

impl From<DecodeError> for ExecError {
    fn from(e: DecodeError) -> Self {
        ExecError::Decode(e)
    }
}

/// State of one wave.
#[derive(Clone, Debug)]
pub struct Wave {
    pub size: WaveSize,
    pub sgpr: [u32; NUM_SGPRS],
    pub vcc: u64,
    pub exec: u64,
    pub m0: u32,
    pub scc: bool,
    /// `vgpr[n][lane]`.
    pub vgpr: Vec<[u32; 64]>,
    /// Dword index of the next instruction.
    pub pc: usize,
    /// LDS of the workgroup (this wave is the whole workgroup), in dwords.
    pub lds: Vec<u32>,
    /// `exp` results in execution order.
    pub exports: Vec<Export>,
    /// Pixel shaders: per attribute, its three vertices' xyzw (bits), the
    /// one primitive this wave shades.
    pub attrs: Vec<[[u32; 4]; 3]>,
    /// GPU address of the code's first dword, for `s_getpc_b64`.
    pub code_base: u64,
}

/// One lane's export: target (0..7 MRT, 9 null), enabled components and
/// the four values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Export {
    pub target: u8,
    pub lane: usize,
    pub en: u8,
    pub values: [u32; 4],
}

impl Wave {
    /// All lanes active, every register zero.
    pub fn new(size: WaveSize, num_vgprs: usize) -> Self {
        Wave {
            size,
            sgpr: [0; NUM_SGPRS],
            vcc: 0,
            exec: size.mask(),
            m0: 0,
            scc: false,
            vgpr: vec![[0; 64]; num_vgprs],
            pc: 0,
            lds: Vec::new(),
            exports: Vec::new(),
            attrs: Vec::new(),
            code_base: 0,
        }
    }

    fn active(&self, lane: usize) -> bool {
        self.exec >> lane & 1 == 1
    }

    pub fn vccz(&self) -> bool {
        self.vcc & self.size.mask() == 0
    }

    pub fn execz(&self) -> bool {
        self.exec & self.size.mask() == 0
    }
}

/// Run from `wave.pc` until `s_endpgm`, at most `max_steps` instructions.
/// Returns the number of instructions executed.
pub fn run(code: &[u32], wave: &mut Wave, max_steps: usize) -> Result<usize, ExecError> {
    run_with_memory(code, wave, &mut NoMemory, max_steps)
}

/// No memory mapped: every access faults.
pub struct NoMemory;

impl Memory for NoMemory {
    fn read(&self, _: u64, _: &mut [u32]) -> bool {
        false
    }
    fn write(&mut self, _: u64, _: &[u32]) -> bool {
        false
    }
}

/// [`run`] with GPU memory for SMEM loads.
pub fn run_with_memory(
    code: &[u32],
    wave: &mut Wave,
    mem: &mut dyn Memory,
    max_steps: usize,
) -> Result<usize, ExecError> {
    let mut total = 0;
    loop {
        let (n, ended) = run_until_barrier(code, wave, mem, max_steps - total)?;
        total += n;
        if ended {
            return Ok(total);
        }
    }
}

/// Run until `s_endpgm` (true) or just past an `s_barrier` (false).
/// Returns the instructions executed.
pub fn run_until_barrier(
    code: &[u32],
    wave: &mut Wave,
    mem: &mut dyn Memory,
    max_steps: usize,
) -> Result<(usize, bool), ExecError> {
    for n in 0..max_steps {
        if wave.pc >= code.len() {
            return Err(ExecError::PcOutOfRange(wave.pc));
        }
        let at = wave.pc;
        let inst = decode(code, at)?;
        if inst.is_endpgm() {
            return Ok((n + 1, true));
        }
        wave.pc = at + inst.len();
        step(&inst, at, wave, mem)?;
        if inst.format == rdna::Format::Sopp && inst.opcode == sopp::S_BARRIER {
            return Ok((n + 1, false));
        }
    }
    Err(ExecError::StepLimit)
}

fn unsupported(at: usize, inst: &Inst, why: &str) -> ExecError {
    ExecError::Unsupported {
        at,
        what: format!("{} ({why})", inst.mnemonic()),
    }
}

struct Ctx<'a> {
    at: usize,
    inst: &'a Inst,
    lit: u32,
}

impl Ctx<'_> {
    fn err(&self, why: &str) -> ExecError {
        unsupported(self.at, self.inst, why)
    }

    /// 32-bit scalar source.
    fn s32(&self, w: &Wave, v: u16) -> Result<u32, ExecError> {
        Ok(match operand(v, self.lit) {
            Operand::Sgpr(n) => w.sgpr[n as usize],
            Operand::VccLo => w.vcc as u32,
            Operand::VccHi => (w.vcc >> 32) as u32,
            Operand::ExecLo => w.exec as u32,
            Operand::ExecHi => (w.exec >> 32) as u32,
            Operand::M0 => w.m0,
            Operand::Null => 0,
            Operand::Int(i) => i as u32,
            Operand::Float(f) => f.to_bits(),
            Operand::InvTwoPi => 0x3E22_F983,
            Operand::Literal(x) => x,
            Operand::Scc => w.scc as u32,
            Operand::Vccz => w.vccz() as u32,
            Operand::Execz => w.execz() as u32,
            other => return Err(self.err(&format!("scalar source {other}"))),
        })
    }

    /// 64-bit scalar source (register pair, vcc, exec or a constant).
    fn s64(&self, w: &Wave, v: u16) -> Result<u64, ExecError> {
        Ok(match operand(v, self.lit) {
            Operand::Sgpr(n) if (n as usize) + 1 < NUM_SGPRS => {
                w.sgpr[n as usize] as u64 | (w.sgpr[n as usize + 1] as u64) << 32
            }
            Operand::VccLo => w.vcc,
            Operand::ExecLo => w.exec,
            Operand::Null => 0,
            Operand::Int(i) => i as i64 as u64,
            // Inline float constants of 64-bit operations are doubles.
            Operand::Float(f) => (f as f64).to_bits(),
            other => return Err(self.err(&format!("64-bit scalar source {other}"))),
        })
    }

    fn set_s32(&self, w: &mut Wave, d: u8, value: u32) -> Result<(), ExecError> {
        match operand(d as u16, 0) {
            Operand::Sgpr(n) => w.sgpr[n as usize] = value,
            Operand::VccLo => w.vcc = (w.vcc & !0xFFFF_FFFF) | value as u64,
            Operand::VccHi => w.vcc = (w.vcc & 0xFFFF_FFFF) | (value as u64) << 32,
            Operand::ExecLo => w.exec = (w.exec & !0xFFFF_FFFF) | value as u64,
            Operand::ExecHi => w.exec = (w.exec & 0xFFFF_FFFF) | (value as u64) << 32,
            Operand::M0 => w.m0 = value,
            Operand::Null => {}
            other => return Err(self.err(&format!("scalar destination {other}"))),
        }
        Ok(())
    }

    fn set_s64(&self, w: &mut Wave, d: u8, value: u64) -> Result<(), ExecError> {
        match operand(d as u16, 0) {
            Operand::Sgpr(n) if (n as usize) + 1 < NUM_SGPRS => {
                w.sgpr[n as usize] = value as u32;
                w.sgpr[n as usize + 1] = (value >> 32) as u32;
            }
            Operand::VccLo => w.vcc = value,
            Operand::ExecLo => w.exec = value,
            Operand::Null => {}
            other => return Err(self.err(&format!("64-bit scalar destination {other}"))),
        }
        Ok(())
    }

    /// 32-bit source for `lane`: a VGPR, or a scalar broadcast.
    fn v32(&self, w: &Wave, v: u16, lane: usize) -> Result<u32, ExecError> {
        match operand(v, self.lit) {
            Operand::Vgpr(n) => w
                .vgpr
                .get(n as usize)
                .map(|r| r[lane])
                .ok_or_else(|| self.err(&format!("v{n} beyond the wave's VGPRs"))),
            _ => self.s32(w, v),
        }
    }

    /// Lane mask held in a scalar source (`vcc`, an SGPR pair, or in
    /// wave32 a single SGPR).
    fn lane_mask(&self, w: &Wave, v: u16) -> Result<u64, ExecError> {
        Ok(match w.size {
            WaveSize::W32 => self.s32(w, v)? as u64,
            WaveSize::W64 => self.s64(w, v)?,
        })
    }

    fn set_lane_mask(&self, w: &mut Wave, d: u8, mask: u64) -> Result<(), ExecError> {
        match w.size {
            WaveSize::W32 => self.set_s32(w, d, mask as u32),
            WaveSize::W64 => self.set_s64(w, d, mask),
        }
    }

    fn vgpr_mut<'w>(&self, w: &'w mut Wave, n: u8) -> Result<&'w mut [u32; 64], ExecError> {
        let at = self.at;
        let inst = self.inst;
        w.vgpr
            .get_mut(n as usize)
            .ok_or_else(|| unsupported(at, inst, &format!("v{n} beyond the wave's VGPRs")))
    }
}

/// Per-lane VALU result for active lanes only.
fn for_active(
    c: &Ctx,
    w: &mut Wave,
    vdst: u8,
    f: impl Fn(&Wave, usize) -> Result<u32, ExecError>,
) -> Result<(), ExecError> {
    let mut out = [0u32; 64];
    let lanes = w.size.lanes();
    for (lane, o) in out.iter_mut().enumerate().take(lanes) {
        if w.active(lane) {
            *o = f(w, lane)?;
        }
    }
    let (exec, d) = (w.exec, c.vgpr_mut(w, vdst)?);
    for (lane, o) in out.iter().enumerate().take(lanes) {
        if exec >> lane & 1 == 1 {
            d[lane] = *o;
        }
    }
    Ok(())
}

/// A VOPC result: bit per active lane, inactive lanes 0.
fn compare_mask(
    w: &Wave,
    f: impl Fn(usize) -> Result<bool, ExecError>,
) -> Result<u64, ExecError> {
    let mut m = 0u64;
    for lane in 0..w.size.lanes() {
        if w.active(lane) && f(lane)? {
            m |= 1 << lane;
        }
    }
    Ok(m)
}

/// A VOPC compare: its kind, condition and whether it is a `v_cmpx`.
/// Opcode layout from LLVM's gfx10_asm_vopc.s / gfx10_asm_vopcx.s: f32
/// conditions 0..15 (`cmpx` +16), f64 32..47 (`cmpx` 48..63), i32 128..135 and u32 192..199 (`cmpx`
/// +16 each).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Compare {
    pub kind: CmpKind,
    pub cond: u16,
    pub cmpx: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CmpKind {
    F32,
    /// Operands are register pairs (VOPC e32 only).
    F64,
    I32,
    U32,
    /// 64-bit integer compares on register pairs (VOPC e32 only).
    I64,
    U64,
}

impl Compare {
    pub fn decode(op: u16) -> Option<Self> {
        let (kind, rel) = match op {
            0..=31 => (CmpKind::F32, op),
            32..=63 => (CmpKind::F64, op - 32),
            128..=151 => (CmpKind::I32, op - 128),
            160..=183 => (CmpKind::I64, op - 160),
            192..=215 => (CmpKind::U32, op - 192),
            224..=247 => (CmpKind::U64, op - 224),
            _ => return None,
        };
        let cmpx = rel & 16 != 0;
        let cond = rel & 15;
        if !matches!(kind, CmpKind::F32 | CmpKind::F64) && cond > 7 {
            return None;
        }
        Some(Compare { kind, cond, cmpx })
    }

    /// Float conditions in opcode order: f, lt, eq, le, gt, lg, ge, o, u,
    /// nge, nlg, ngt, nle, neq, nlt, tru (LLVM's names; "n" forms are the
    /// negation, so they hold for unordered operands). Integer: f, lt, eq,
    /// le, gt, ne, ge, t.
    // The negated comparisons are the point: they are true for NaN.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn eval(&self, a: u32, b: u32) -> bool {
        match self.kind {
            CmpKind::F32 => float_cond(self.cond, f32::from_bits(a) as f64, f32::from_bits(b) as f64),
            CmpKind::F64 => float_cond(self.cond, f64::from_bits(a as u64), f64::from_bits(b as u64)),
            _ => {
                let ord = if self.kind == CmpKind::I32 {
                    (a as i32).cmp(&(b as i32))
                } else {
                    a.cmp(&b)
                };
                int_cond(self.cond, ord)
            }
        }
    }
}

impl Compare {
    /// An f64 / i64 / u64 condition on two register pairs.
    pub fn eval64(&self, a: u64, b: u64) -> bool {
        match self.kind {
            CmpKind::I64 => int_cond(self.cond, (a as i64).cmp(&(b as i64))),
            CmpKind::U64 => int_cond(self.cond, a.cmp(&b)),
            _ => float_cond(self.cond, f64::from_bits(a), f64::from_bits(b)),
        }
    }

    /// Operands are 64-bit register pairs.
    pub fn wide(&self) -> bool {
        matches!(self.kind, CmpKind::F64 | CmpKind::I64 | CmpKind::U64)
    }
}

/// Integer conditions in opcode order: f, lt, eq, le, gt, ne, ge, t (AMD's
/// RDNA2 ISA specification, also for the 64-bit kinds).
fn int_cond(cond: u16, ord: std::cmp::Ordering) -> bool {
    use std::cmp::Ordering::*;
    match cond {
        0 => false,
        1 => ord == Less,
        2 => ord == Equal,
        3 => ord != Greater,
        4 => ord == Greater,
        5 => ord != Equal,
        6 => ord != Less,
        _ => true,
    }
}

impl Compare {
}

/// Float condition `cond` (see `Compare::eval`); f32 operands widen to
/// f64 exactly, NaNs included.
// The negated comparisons are the point: they are true for NaN.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn float_cond(cond: u16, x: f64, y: f64) -> bool {
    let unord = x.is_nan() || y.is_nan();
    match cond {
        0 => false,
        1 => x < y,
        2 => x == y,
        3 => x <= y,
        4 => x > y,
        5 => !unord && x != y,
        6 => x >= y,
        7 => !unord,
        8 => unord,
        9 => !(x >= y),
        10 => !(!unord && x != y),
        11 => !(x > y),
        12 => !(x <= y),
        13 => !(x == y),
        14 => !(x < y),
        _ => true,
    }
}

/// v_cvt_rpi/flr_i32_f32 and v_frexp_exp/mant (see rdna::vop1). None
/// (refused): a NaN or out-of-range result for the conversions (poison in
/// LLVM's patterns), a subnormal input or NaN significand for frexp (FP32
/// denormal mode, NaN bits).
pub fn vop1_f32_special(op: u16, a: u32) -> Option<u32> {
    let x = f32::from_bits(a);
    let to_int = |y: f32| (-2_147_483_648.0..2_147_483_648.0).contains(&y).then_some(y as i32 as u32);
    match op {
        vop1::V_CVT_RPI_I32_F32 => to_int((x + 0.5).floor()),
        vop1::V_CVT_FLR_I32_F32 => to_int(x.floor()),
        _ => {
            if x.is_subnormal() {
                return None;
            }
            if !x.is_normal() {
                // zero, Inf, NaN: exponent 0, significand unchanged
                return if op == vop1::V_FREXP_EXP_I32_F32 { Some(0) } else { (!x.is_nan()).then_some(a) };
            }
            // |x| = m * 2^e with 0.5 <= m < 1
            let e = (a >> 23 & 0xFF) as i32 - 126;
            let m = x as f64 / 2f64.powi(e);
            debug_assert!((0.5..1.0).contains(&m.abs()));
            Some(if op == vop1::V_FREXP_EXP_I32_F32 { e as u32 } else { (m as f32).to_bits() })
        }
    }
}

/// v_ldexp_f32: x * 2^n, exact for a normal result. None (refused): NaN
/// or subnormal x, or a result below the smallest normal (FP32 denormal
/// mode); overflow gives ±Inf.
pub fn ldexp_f32(a: u32, n: i32) -> Option<u32> {
    let x = f32::from_bits(a);
    if x.is_nan() || x.is_subnormal() {
        return None;
    }
    if x == 0.0 || x.is_infinite() {
        return Some(a);
    }
    let r = x as f64 * 2f64.powi(n.clamp(-400, 400));
    if r.abs() >= 2f64.powi(128) {
        return Some(a & 0x8000_0000 | 0x7F80_0000);
    }
    if r.abs() < f32::MIN_POSITIVE as f64 {
        return None;
    }
    Some((r as f32).to_bits())
}

/// f64 -> i32 / u32 (saturating, NaN -> 0: LLVM's `fp_to_*_sat`, Rust's
/// `as`) and f64 -> f32 (round to nearest even). None (refused) for the
/// f32 result: a NaN (its bits), a subnormal input (FP64 denormal mode)
/// or a non-zero value below the smallest normal f32 (FP32 denormal mode).
pub fn cvt_from_f64(op: u16, x: f64) -> Option<u32> {
    Some(match op {
        vop1::V_CVT_I32_F64 => x as i32 as u32,
        vop1::V_CVT_U32_F64 => x as u32,
        vop1::V_CVT_F32_F64 => {
            if x.is_nan() || x.is_subnormal() || (x != 0.0 && x.abs() < f32::MIN_POSITIVE as f64) {
                return None;
            }
            (x as f32).to_bits()
        }
        _ => return None,
    })
}

/// f64 rounding (rdna::vop1::V_TRUNC_F64..V_FLOOR_F64), exact in IEEE
/// arithmetic. None (refused): a NaN (whether the result keeps its bits
/// or is quieted is not settled by the sources) and a subnormal input
/// (ceil/floor of one depend on the shader's FP64 denormal mode).
pub fn round_f64(op: u16, a: u64) -> Option<u64> {
    let x = f64::from_bits(a);
    if x.is_nan() || x.is_subnormal() {
        return None;
    }
    Some(match op {
        vop1::V_TRUNC_F64 => x.trunc(),
        vop1::V_CEIL_F64 => x.ceil(),
        vop1::V_RNDNE_F64 => x.round_ties_even(),
        vop1::V_FLOOR_F64 => x.floor(),
        _ => return None,
    }
    .to_bits())
}

/// f64 add / mul / fma (VOP3, see rdna::vop3::V_ADD_F64) with abs/neg on
/// bit 63, IEEE round-to-nearest-even. Refused: NaN and subnormal inputs
/// or results (the result NaN's bits and the FP64 denormal mode are not
/// settled by the sources).
fn f64_arith(c: &Ctx, w: &mut Wave, op: u16, vdst: u8, src: [u16; 3], abs: u8, neg: u8) -> Result<(), ExecError> {
    let n = if op == vop3::V_FMA_F64 { 3 } else { 2 };
    let mut res = Vec::new();
    for l in 0..w.size.lanes() {
        if !w.active(l) {
            continue;
        }
        let mut x = [0f64; 3];
        for (i, xi) in x.iter_mut().enumerate().take(n) {
            let mut b = src64(c, w, src[i], l)?;
            if abs >> i & 1 != 0 {
                b &= !(1 << 63);
            }
            if neg >> i & 1 != 0 {
                b ^= 1 << 63;
            }
            *xi = f64::from_bits(b);
            if xi.is_nan() || xi.is_subnormal() {
                return Err(c.err("f64 NaN or subnormal operand"));
            }
        }
        let r = match op {
            vop3::V_ADD_F64 => x[0] + x[1],
            vop3::V_MUL_F64 => x[0] * x[1],
            _ => x[0].mul_add(x[1], x[2]),
        };
        if r.is_nan() || r.is_subnormal() {
            return Err(c.err("f64 NaN or subnormal result"));
        }
        res.push((l, r.to_bits()));
    }
    for (l, r) in res {
        c.vgpr_mut(w, vdst)?[l] = r as u32;
        c.vgpr_mut(w, vdst + 1)?[l] = (r >> 32) as u32;
    }
    Ok(())
}

/// A 64-bit VALU source: a VGPR pair, or a 64-bit scalar source (SGPR
/// pair, inline constant as a double); a literal is refused (how a
/// 32-bit literal extends to an f64 operand is not modelled).
fn src64(c: &Ctx, w: &Wave, s: u16, lane: usize) -> Result<u64, ExecError> {
    if s >= 256 {
        return Ok(c.v32(w, s, lane)? as u64 | (c.v32(w, s + 1, lane)? as u64) << 32);
    }
    if s == rdna::SRC_LITERAL {
        return Err(c.err("literal f64 operand"));
    }
    c.s64(w, s)
}

/// SDWA on integer VOP1 / VOP2 (see rdna::sdwa). Refused: float ops,
/// source modifiers, clamp, an SGPR src1, undefined selectors.
fn sdwa_step(inst: &Inst, at: usize, w: &mut Wave) -> Option<Result<(), ExecError>> {
    let (vdst, src1, vop2) = match inst.fields {
        Fields::Vop1 { vdst, src0, .. } if src0 == rdna::SRC_SDWA => (vdst, None, false),
        Fields::Vop2 { vdst, src0, vsrc1 } if src0 == rdna::SRC_SDWA => (vdst, Some(vsrc1), true),
        Fields::Vopc { src0, .. } if src0 == rdna::SRC_SDWA => return Some(Err(unsupported(at, inst, "SDWA on VOPC"))),
        _ => return None,
    };
    Some((|| {
        let c = Ctx { at, inst, lit: 0 };
        let d = rdna::sdwa::parse(*inst.words.get(1).ok_or_else(|| c.err("missing SDWA word"))?);
        if d.clamp || d.mods != [0, 0] || d.src1_sgpr {
            return Err(c.err("SDWA clamp, source modifiers or SGPR src1"));
        }
        let op = inst.opcode;
        let known = if vop2 { vop2_int(op, 0, 0).is_some() } else { op == vop1::V_MOV_B32 || vop1_int(op, 0).is_some() };
        if !known {
            return Err(c.err("SDWA on this opcode"));
        }
        let s0 = if d.src0_sgpr { d.src0 as u16 } else { 256 + d.src0 as u16 };
        let bad = || c.err("SDWA selector");
        let mut out = Vec::new();
        for l in 0..w.size.lanes() {
            if !w.active(l) {
                continue;
            }
            let a = rdna::sdwa::select(c.v32(w, s0, l)?, d.sel[0], d.sext[0]).ok_or_else(bad)?;
            let r = match src1 {
                Some(v1) => {
                    let b = rdna::sdwa::select(c.v32(w, 256 + v1 as u16, l)?, d.sel[1], d.sext[1]).ok_or_else(bad)?;
                    vop2_int(op, a, b).unwrap()
                }
                None if op == vop1::V_MOV_B32 => a,
                None => vop1_int(op, a).unwrap(),
            };
            let old = c.v32(w, 256 + vdst as u16, l)?;
            out.push((l, rdna::sdwa::place(r, old, d.dst_sel, d.dst_unused).ok_or_else(|| c.err("SDWA dst_sel/dst_unused"))?));
        }
        for (l, v) in out {
            c.vgpr_mut(w, vdst)?[l] = v;
        }
        Ok(())
    })())
}

fn step(inst: &Inst, at: usize, w: &mut Wave, mem: &mut dyn Memory) -> Result<(), ExecError> {
    if let Some(r) = sdwa_step(inst, at, w) {
        return r;
    }
    if let Some(l) = rdna::sopk::lower(inst) {
        return step(&l, at, w, mem);
    }
    if inst.format == rdna::Format::Sopk && rdna::sopk::is_waitcnt(inst.opcode) {
        return Ok(());
    }
    if let Some(r) = dpp_step(inst, at, w, mem) {
        return r;
    }
    let c = Ctx {
        at,
        inst,
        lit: inst.literal().unwrap_or(0),
    };
    match (&inst.fields, inst.opcode) {
        (Fields::Sopp { .. }, op) => sopp_step(&c, w, op),
        (&Fields::Sop2 { sdst, ssrc0, ssrc1 }, op) => sop2_step(&c, w, op, sdst, ssrc0, ssrc1),
        (&Fields::Sop1 { sdst, ssrc0 }, op) => sop1_step(&c, w, op, sdst, ssrc0),
        (&Fields::Sopc { ssrc0, ssrc1 }, op) if op == sopc::S_CMP_EQ_U64 || op == sopc::S_CMP_LG_U64 => {
            let (a, b) = (c.s64(w, ssrc0)?, c.s64(w, ssrc1)?);
            w.scc = (a == b) == (op == sopc::S_CMP_EQ_U64);
            Ok(())
        }
        (&Fields::Sopc { ssrc0, ssrc1 }, op) => {
            let (a, b) = (c.s32(w, ssrc0)?, c.s32(w, ssrc1)?);
            w.scc = match op {
                sopc::S_CMP_EQ_I32 | sopc::S_CMP_EQ_U32 => a == b,
                sopc::S_CMP_GT_I32 => (a as i32) > (b as i32),
                sopc::S_CMP_GE_I32 => (a as i32) >= (b as i32),
                sopc::S_CMP_GT_U32 => a > b,
                sopc::S_CMP_GE_U32 => a >= b,
                sopc::S_CMP_LG_I32 | sopc::S_CMP_LG_U32 => a != b,
                sopc::S_CMP_LT_I32 => (a as i32) < (b as i32),
                sopc::S_CMP_LE_I32 => (a as i32) <= (b as i32),
                sopc::S_CMP_LT_U32 => a < b,
                sopc::S_CMP_LE_U32 => a <= b,
                sopc::S_BITCMP0_B32 => a >> (b & 31) & 1 == 0,
                sopc::S_BITCMP1_B32 => a >> (b & 31) & 1 == 1,
                sopc::S_BITCMP0_B64 | sopc::S_BITCMP1_B64 => {
                    let x = c.s64(w, ssrc0)?;
                    (x >> (b & 63) & 1 == 1) == (op == sopc::S_BITCMP1_B64)
                }
                _ => return Err(c.err("opcode")),
            };
            Ok(())
        }
        (&Fields::Vop1 { vdst, src0 }, op) => vop1_step(&c, w, op, vdst, src0),
        (&Fields::Vop2 { vdst, src0, vsrc1 }, op) if rdna::vop2::is_madk(op) => {
            // ASSUMPTION (as v_fma_f32): one rounding.
            let k = f32::from_bits(c.lit);
            for_active(&c, w, vdst, |w, l| {
                let a = f32::from_bits(c.v32(w, src0, l)?);
                let b = f32::from_bits(c.v32(w, 256 + vsrc1 as u16, l)?);
                Ok(if op == rdna::vop2::V_FMAMK_F32 { a.mul_add(k, b) } else { a.mul_add(b, k) }.to_bits())
            })
        }
        (&Fields::Vop2 { vdst, src0, vsrc1 }, op) if carry_op(op, 0, 0, false).is_some() => {
            carry_step(&c, w, op, vdst, src0, 256 + vsrc1 as u16, Some(VCC_LO), VCC_LO as u8)
        }
        (&Fields::Vop2 { vdst, src0, vsrc1 }, op) => {
            vop2_step(&c, w, op, vdst, src0, 256 + vsrc1 as u16, VCC_LO)
        }
        (&Fields::Vopc { src0, vsrc1 }, op) => {
            let cmp = Compare::decode(op).ok_or_else(|| c.err("opcode"))?;
            let b = 256 + vsrc1 as u16;
            let m = if cmp.wide() {
                compare_mask(w, |l| Ok(cmp.eval64(src64(&c, w, src0, l)?, src64(&c, w, b, l)?)))?
            } else {
                compare_mask(w, |l| Ok(cmp.eval(c.v32(w, src0, l)?, c.v32(w, b, l)?)))?
            };
            if cmp.cmpx {
                // GFX10 v_cmpx writes exec only (no vcc operand in LLVM's syntax).
                w.exec = (w.exec & !w.size.mask()) | m;
            } else {
                // wave32 writes vcc_lo only.
                w.vcc = (w.vcc & !w.size.mask()) | m;
            }
            Ok(())
        }
        (Fields::Vop3 { .. }, _) => vop3_step(&c, w),
        (&Fields::Vop3p { vdst, src, neg, neg_hi, op_sel, op_sel_hi, clamp }, op) => {
            if rdna::vop3p::dot(op, 0, 0, 0, false).is_none() {
                return Err(c.err("opcode"));
            }
            if neg != 0 || neg_hi != 0 || op_sel != 0 || op_sel_hi != 7 {
                return Err(c.err("dot with neg / op_sel other than ACO's"));
            }
            for_active(&c, w, vdst, |w, l| {
                let x = |i: usize| c.v32(w, src[i], l);
                rdna::vop3p::dot(op, x(0)?, x(1)?, x(2)?, clamp).ok_or_else(|| c.err("dot result left open by NIR"))
            })
        }
        (Fields::Mimg { .. }, op) => image_step(&c, w, mem, op),
        (&Fields::Mubuf { vdata, vaddr, srsrc, soffset, offset, offen, idxen, .. }, op) => {
            if c.inst.words[0] & rdna::mubuf::LDS != 0 {
                return Err(c.err("LDS load"));
            }
            let glc = c.inst.words[0] & rdna::mubuf::GLC != 0;
            mubuf_step(&c, w, mem, op, glc, vdata, vaddr, srsrc, soffset, offset, offen, idxen)
        }
        (&Fields::Mtbuf { vdata, vaddr, srsrc, soffset, offset, format, offen, idxen }, op) => {
            let m = rdna::mtbuf::as_mubuf(op, format as u32)
                .ok_or_else(|| c.err("typed format with conversion, d16, or channel count"))?;
            // DST_SEL (V# WORD3[11:0]) must be the identity on the channels read.
            let sel = c.s32(w, srsrc as u16 + 3)?;
            let n = rdna::mubuf::access(m).unwrap().0;
            if (0..n).any(|k| sel >> (3 * k) & 7 != 4 + k as u32) {
                return Err(c.err("typed buffer with a non-identity DST_SEL"));
            }
            mubuf_step(&c, w, mem, m, false, vdata, vaddr, srsrc, soffset, offset, offen, idxen)
        }
        (&Fields::Vintrp { vdst, vsrc, attr, chan }, op) => {
            // P0 = vertex 0, P10 = v1 - v0, P20 = v2 - v0 per channel; p1:
            // D = P10 * I + P0, p2: D = P20 * J + D (ISA formulas).
            // ASSUMPTION: the deltas are IEEE single subtractions and the
            // products are fused (one rounding); M0 is not consulted (one
            // primitive per wave).
            let a = w
                .attrs
                .get(attr as usize)
                .ok_or_else(|| c.err("attribute not provided"))?;
            let v = |k: usize| f32::from_bits(a[k][chan as usize]);
            let (p0, p10, p20) = (v(0), v(1) - v(0), v(2) - v(0));
            for_active(&c, w, vdst, |w, l| {
                Ok(match op {
                    vintrp::P1_F32 => p10.mul_add(f32::from_bits(c.v32(w, 256 + vsrc as u16, l)?), p0),
                    vintrp::P2_F32 => p20.mul_add(
                        f32::from_bits(c.v32(w, 256 + vsrc as u16, l)?),
                        f32::from_bits(c.v32(w, 256 + vdst as u16, l)?),
                    ),
                    _ => match vsrc {
                        vintrp::SLOT_P10 => p10,
                        vintrp::SLOT_P20 => p20,
                        vintrp::SLOT_P0 => p0,
                        _ => return Err(c.err("interpolation slot")),
                    },
                }
                .to_bits())
            })
        }
        (&Fields::Exp { target, en, vsrc, compr, .. }, _) => {
            if compr {
                return Err(c.err("compressed (16-bit) export"));
            }
            let known = target <= rdna::exp::MRT_LAST
                || target == rdna::exp::NULL
                || (rdna::exp::POS0..=rdna::exp::POS3).contains(&target)
                || (rdna::exp::PARAM0..=rdna::exp::PARAM_LAST).contains(&target);
            if !known {
                return Err(c.err("export target"));
            }
            for lane in 0..w.size.lanes() {
                if !w.active(lane) {
                    continue;
                }
                let mut values = [0u32; 4];
                for (k, v) in values.iter_mut().enumerate() {
                    if en >> k & 1 != 0 {
                        *v = c.v32(w, 256 + vsrc[k] as u16, lane)?;
                    }
                }
                w.exports.push(Export { target, lane, en, values });
            }
            Ok(())
        }
        (
            &Fields::Ds {
                offset0,
                offset1,
                gds,
                addr,
                data0,
                data1,
                vdst,
                ..
            },
            op,
        ) => {
            if gds {
                return Err(c.err("GDS"));
            }
            if op == ds::SWIZZLE_B32 {
                return ds_swizzle(&c, w, offset0 as u32 | (offset1 as u32) << 8, addr, vdst);
            }
            if op == ds::APPEND || op == ds::CONSUME {
                return ds_append(&c, w, op == ds::CONSUME, offset0 as u32 | (offset1 as u32) << 8, vdst);
            }
            if let Some((base, rtn)) = ds::atomic(op) {
                let off = offset0 as u32 | (offset1 as u32) << 8;
                return ds_atomic(&c, w, base, rtn, off, addr, data0, data1, vdst);
            }
            ds_step(&c, w, op, offset0, offset1, addr, data0, data1, vdst)
        }
        (
            &Fields::Flat {
                vdst,
                addr,
                data,
                saddr,
                offset,
                seg,
                glc,
                ..
            },
            op,
        ) => {
            if seg != rdna::global::SEG_GLOBAL {
                return Err(c.err("FLAT segment other than global"));
            }
            let base = if saddr == rdna::global::SADDR_OFF {
                None
            } else {
                Some(c.s64(w, saddr as u16)?)
            };
            if rdna::global::atomic(op, 0, 0, 0).is_some() {
                return global_atomic(&c, w, mem, op, vdst, addr, data, base, offset, glc);
            }
            if let Some((bytes, signed)) = rdna::global::sub_dword(op) {
                let mut res = Vec::new();
                for lane in 0..w.size.lanes() {
                    if !w.active(lane) {
                        continue;
                    }
                    let a = match base {
                        None => c.v32(w, 256 + addr as u16, lane)? as u64 | (c.v32(w, 256 + addr as u16 + 1, lane)? as u64) << 32,
                        Some(b) => b.wrapping_add(c.v32(w, 256 + addr as u16, lane)? as u64),
                    }
                    .wrapping_add(offset as i64 as u64);
                    if a % bytes as u64 != 0 {
                        return Err(c.err("unaligned sub-dword load"));
                    }
                    let mut d = [0u32];
                    if !mem.read(a & !3, &mut d) {
                        return Err(ExecError::MemoryFault { at: c.at, addr: a });
                    }
                    res.push((lane, rdna::global::extract(d[0], (a & 3) as u32, bytes, signed)));
                }
                for (lane, v) in res {
                    c.vgpr_mut(w, vdst)?[lane] = v;
                }
                return Ok(());
            }
            let (n, store) = rdna::global::access(op).ok_or_else(|| c.err("opcode"))?;
            for lane in 0..w.size.lanes() {
                if !w.active(lane) {
                    continue;
                }
                let v = |r: u8| -> Result<u32, ExecError> {
                    w.vgpr
                        .get(r as usize)
                        .map(|x| x[lane])
                        .ok_or_else(|| c.err("VGPR beyond the wave's VGPRs"))
                };
                let a = match base {
                    None => v(addr)? as u64 | (v(addr + 1)? as u64) << 32,
                    Some(b) => b.wrapping_add(v(addr)? as u64),
                }
                .wrapping_add(offset as i64 as u64);
                if a & 3 != 0 {
                    return Err(c.err("unaligned global access"));
                }
                if store {
                    let mut buf = [0u32; 4];
                    for (k, x) in buf[..n].iter_mut().enumerate() {
                        *x = v(data + k as u8)?;
                    }
                    if !mem.write(a, &buf[..n]) {
                        return Err(ExecError::MemoryFault { at: c.at, addr: a });
                    }
                } else {
                    let mut buf = [0u32; 4];
                    if !mem.read(a, &mut buf[..n]) {
                        return Err(ExecError::MemoryFault { at: c.at, addr: a });
                    }
                    for (k, x) in buf[..n].iter().enumerate() {
                        let r = c.vgpr_mut(w, vdst + k as u8)?;
                        r[lane] = *x;
                    }
                }
            }
            Ok(())
        }
        (
            &Fields::Smem {
                sdata,
                sbase,
                offset,
                soffset,
                ..
            },
            op,
        ) => {
            if let Some(n) = smem::buffer_load_dwords(op) {
                return s_buffer_load(&c, w, mem, n, sdata, sbase, offset, soffset);
            }
            let n = smem::load_dwords(op).ok_or_else(|| c.err("opcode"))?;
            let base = c.s64(w, sbase as u16)?;
            let so = if soffset == smem::SOFFSET_NULL {
                0
            } else {
                c.s32(w, soffset as u16)?
            };
            // Byte address; the low two bits are ignored (dword access).
            let addr = base
                .wrapping_add(offset as i64 as u64)
                .wrapping_add(so as u64)
                & !3;
            let mut buf = [0u32; 16];
            if !mem.read(addr, &mut buf[..n]) {
                return Err(ExecError::MemoryFault { at: c.at, addr });
            }
            for (k, v) in buf[..n].iter().enumerate() {
                c.set_s32(w, sdata + k as u8, *v)?;
            }
            Ok(())
        }
        _ => Err(c.err("format")),
    }
}

const VCC_LO: u16 = 106;

/// 32-bit global atomics, lanes in ascending order. Refused where the
/// result would depend on that order: two active lanes on one address
/// returning the old value (`glc`), or with swap/cmpswap.
#[allow(clippy::too_many_arguments)]
fn global_atomic(
    c: &Ctx,
    w: &mut Wave,
    mem: &mut dyn Memory,
    op: u16,
    vdst: u8,
    addr: u8,
    data: u8,
    base: Option<u64>,
    offset: i16,
    glc: bool,
) -> Result<(), ExecError> {
    let mut seen: Vec<u64> = Vec::new();
    let mut work = Vec::new();
    for lane in 0..w.size.lanes() {
        if !w.active(lane) {
            continue;
        }
        let a = match base {
            None => c.v32(w, 256 + addr as u16, lane)? as u64 | (c.v32(w, 256 + addr as u16 + 1, lane)? as u64) << 32,
            Some(b) => b.wrapping_add(c.v32(w, 256 + addr as u16, lane)? as u64),
        }
        .wrapping_add(offset as i64 as u64);
        if a & 3 != 0 {
            return Err(c.err("unaligned global atomic"));
        }
        if seen.contains(&a) && (glc || !rdna::global::atomic_commutes(op)) {
            return Err(c.err("two lanes' atomics on one address with an order-dependent result"));
        }
        seen.push(a);
        let d = c.v32(w, 256 + data as u16, lane)?;
        let cmp = if op == rdna::global::ATOMIC_CMPSWAP {
            c.v32(w, 256 + data as u16 + 1, lane)?
        } else {
            0
        };
        work.push((lane, a, d, cmp));
    }
    for (lane, a, d, cmp) in work {
        let mut old = [0u32];
        if !mem.read(a, &mut old) {
            return Err(ExecError::MemoryFault { at: c.at, addr: a });
        }
        let new = rdna::global::atomic(op, old[0], d, cmp).expect("atomic opcode");
        if !mem.write(a, &[new]) {
            return Err(ExecError::MemoryFault { at: c.at, addr: a });
        }
        if glc {
            c.vgpr_mut(w, vdst)?[lane] = old[0];
        }
    }
    Ok(())
}

/// MUBUF buffer_load/store_dword*: address = base + soffset + offset +
/// voffset + index * stride, in bounds per [`rdna::BufferDesc::in_bounds`]
/// for each dword (out of bounds: loads read 0, stores are dropped).
/// Refused rather than guessed: LDS loads, ADD_TID, swizzling, a nonzero
/// soffset (whether it takes part in the range check is not confirmed),
/// unaligned addresses, and two lanes storing to the same dword.
#[allow(clippy::too_many_arguments)]
fn mubuf_step(
    c: &Ctx,
    w: &mut Wave,
    mem: &mut dyn Memory,
    op: u16,
    glc: bool,
    vdata: u8,
    vaddr: u8,
    srsrc: u8,
    soffset: u8,
    offset: u16,
    offen: bool,
    idxen: bool,
) -> Result<(), ExecError> {
    let atomic = rdna::global::atomic(op, 0, 0, 0).is_some();
    let sub = rdna::mubuf::sub_dword(op);
    let (n, store) = if atomic {
        (1, true)
    } else if sub.is_some() {
        (1, false)
    } else {
        rdna::mubuf::access(op).ok_or_else(|| c.err("opcode"))?
    };
    let mut dw = [0u32; 4];
    for (k, x) in dw.iter_mut().enumerate() {
        *x = c.s32(w, srsrc as u16 + k as u16)?;
    }
    if rdna::BufferDesc::add_tid(dw[3]) {
        return Err(c.err("ADD_TID_ENABLE"));
    }
    let d = rdna::BufferDesc::parse(dw);
    let so = c.s32(w, soffset as u16)?;
    if so != 0 {
        return Err(c.err("nonzero soffset in a MUBUF access"));
    }
    let mut stored: Vec<u64> = Vec::new();
    let mut atomics: Vec<(usize, u64, u32, u32)> = Vec::new();
    for lane in 0..w.size.lanes() {
        if !w.active(lane) {
            continue;
        }
        let index = if idxen { c.v32(w, 256 + vaddr as u16, lane)? } else { 0 };
        let voff = if offen { c.v32(w, 256 + vaddr as u16 + idxen as u16, lane)? } else { 0 };
        let rec_off = offset as u64 + voff as u64;
        if index as u64 * d.stride as u64 + rec_off + 4 * n as u64 > u32::MAX as u64 {
            return Err(c.err("record offset beyond 32 bits"));
        }
        if let Some((bytes, signed)) = sub {
            let o = rec_off;
            let ok = d.in_bounds(index, o).ok_or_else(|| c.err("swizzled buffer"))?;
            if ok && d.oob_select == 3 && o + bytes as u64 > d.num_records as u64 {
                return Err(c.err("sub-dword load straddles NUM_RECORDS"));
            }
            let addr = d.base + index as u64 * d.stride as u64 + o;
            if !addr.is_multiple_of(bytes as u64) {
                return Err(c.err("unaligned sub-dword load"));
            }
            let mut x = [0u32];
            if ok && !mem.read(addr & !3, &mut x) {
                return Err(ExecError::MemoryFault { at: c.at, addr });
            }
            c.vgpr_mut(w, vdata)?[lane] = if ok { rdna::global::extract(x[0], (addr & 3) as u32, bytes, signed) } else { 0 };
            continue;
        }
        for k in 0..n {
            let o = rec_off + 4 * k as u64;
            let ok = d.in_bounds(index, o).ok_or_else(|| c.err("swizzled buffer"))?;
            if ok && d.oob_select == 3 && o + 4 > d.num_records as u64 {
                return Err(c.err("dword straddles NUM_RECORDS"));
            }
            let addr = d.base + index as u64 * d.stride as u64 + o;
            if addr & 3 != 0 {
                return Err(c.err("unaligned buffer access"));
            }
            if atomic {
                if !ok {
                    if glc {
                        return Err(c.err("out-of-bounds atomic returning a value"));
                    }
                    continue;
                }
                if stored.contains(&addr) && (glc || !rdna::global::atomic_commutes(op)) {
                    return Err(c.err("two lanes' atomics on one address with an order-dependent result"));
                }
                stored.push(addr);
                let d = c.v32(w, 256 + vdata as u16, lane)?;
                let cmp = if op == rdna::global::ATOMIC_CMPSWAP { c.v32(w, 256 + vdata as u16 + 1, lane)? } else { 0 };
                atomics.push((lane, addr, d, cmp));
            } else if store {
                if !ok {
                    continue;
                }
                if stored.contains(&addr) {
                    return Err(c.err("two lanes store to the same dword"));
                }
                stored.push(addr);
                let v = c.v32(w, 256 + vdata as u16 + k as u16, lane)?;
                if !mem.write(addr, &[v]) {
                    return Err(ExecError::MemoryFault { at: c.at, addr });
                }
            } else {
                let mut x = [0u32];
                if ok && !mem.read(addr, &mut x) {
                    return Err(ExecError::MemoryFault { at: c.at, addr });
                }
                c.vgpr_mut(w, vdata + k as u8)?[lane] = if ok { x[0] } else { 0 };
            }
        }
    }
    for (lane, addr, d, cmp) in atomics {
        let mut old = [0u32];
        if !mem.read(addr, &mut old) {
            return Err(ExecError::MemoryFault { at: c.at, addr });
        }
        let new = rdna::global::atomic(op, old[0], d, cmp).expect("atomic opcode");
        if !mem.write(addr, &[new]) {
            return Err(ExecError::MemoryFault { at: c.at, addr });
        }
        if glc {
            c.vgpr_mut(w, vdata)?[lane] = old[0];
        }
    }
    Ok(())
}

/// `image_load` / `image_store` on a 2D linear image (see
/// [`rdna::ImageDesc`]). Stores need dmask covering exactly the format's
/// channels and identity DST_SEL. Refused:
/// other dims, tfe/lwe/a16/d16/r128, tiled (SW_MODE != 0), a base level
/// other than 0, a nonzero DEPTH (a possible custom pitch), formats
/// outside [`rdna::image_format`], `1` selects on integer formats, and
/// out-of-bounds coordinates.
fn image_step(c: &Ctx, w: &mut Wave, mem: &mut dyn Memory, op: u16) -> Result<(), ExecError> {
    let Fields::Mimg {
        vdata,
        vaddr,
        srsrc,
        dmask,
        dim,
        ref nsa_addrs,
        ..
    } = c.inst.fields
    else {
        unreachable!()
    };
    let (w0, w1) = (c.inst.words[0], c.inst.words[1]);
    let store = op == rdna::mimg::IMAGE_STORE;
    if !(op == rdna::mimg::IMAGE_LOAD || store) || dim != rdna::mimg::DIM_2D {
        return Err(c.err("only 2D image_load / image_store"));
    }
    if w0 & (rdna::mimg::TFE | rdna::mimg::LWE | rdna::mimg::R128) != 0
        || w1 & (rdna::mimg::A16 | rdna::mimg::D16) != 0
    {
        return Err(c.err("tfe/lwe/r128/a16/d16"));
    }
    let mut dw = [0u32; 8];
    for (k, x) in dw.iter_mut().enumerate() {
        *x = c.s32(w, srsrc as u16 + k as u16)?;
    }
    let d = rdna::ImageDesc::parse(dw);
    let (bytes, channels, float) =
        rdna::image_format(d.format).ok_or_else(|| c.err("image format"))?;
    // Linear, or a chip-independent tiled mode (rdna::tiling) with one
    // mip level (where tiled mip 0 sits with more levels is not modelled).
    let tiled = d.sw_mode != 0;
    if tiled && (rdna::tiling::pattern(d.sw_mode, bytes.trailing_zeros()).is_none() || d.last_level != 0) {
        return Err(c.err("swizzle mode other than linear/SW_256B_S/SW_4KB_S/SW_64KB_S, or tiled mips"));
    }
    if d.base_level != 0 || d.depth != 0 {
        return Err(c.err("mipmapped or custom-pitch image"));
    }
    if store && d.format == rdna::FORMAT_8_8_8_8_UNORM {
        return Err(c.err("UNORM8 stores (rounding not confirmed)"));
    }
    if store {
        // Only the unambiguous case: every channel written, no swizzle.
        let identity = (0..channels as usize).all(|k| d.dst_sel[k] == 4 + k as u32);
        if dmask as u32 != (1 << channels) - 1 || !identity {
            return Err(c.err("image_store needs dmask = all channels and identity DST_SEL"));
        }
    } else if !float && (d.dst_sel.contains(&1) || (channels < 4 && d.dst_sel.contains(&7))) {
        // Integer "1" (SQ_SEL_1, or a missing alpha) is not confirmed.
        return Err(c.err("constant 1 on an integer format"));
    }
    let pitch = d.linear_pitch(bytes) as u64;
    let addrs: Vec<u8> = std::iter::once(vaddr).chain(nsa_addrs.iter().copied()).collect();
    let (ax, ay) = if nsa_addrs.is_empty() {
        (vaddr, vaddr + 1)
    } else {
        (addrs[0], *addrs.get(1).ok_or_else(|| c.err("NSA address count"))?)
    };
    for lane in 0..w.size.lanes() {
        if !w.active(lane) {
            continue;
        }
        let x = c.v32(w, 256 + ax as u16, lane)?;
        let y = c.v32(w, 256 + ay as u16, lane)?;
        if x >= d.width || y >= d.height {
            return Err(c.err("image coordinates out of bounds"));
        }
        let addr = if tiled {
            d.base + rdna::tiling::offset(d.sw_mode, bytes.trailing_zeros(), d.width, x, y).unwrap()
        } else {
            d.base + (y as u64 * pitch + x as u64) * bytes as u64
        };
        if store {
            let mut texel = [0u32; 4];
            for (k, t) in texel[..channels.min(bytes / 4) as usize].iter_mut().enumerate() {
                *t = c.v32(w, 256 + vdata as u16 + k as u16, lane)?;
            }
            if !mem.write(addr, &texel[..channels.min(bytes / 4) as usize]) {
                return Err(ExecError::MemoryFault { at: c.at, addr });
            }
            continue;
        }
        let mut texel = [0u32; 4];
        if !mem.read(addr, &mut texel[..(bytes / 4) as usize]) {
            return Err(ExecError::MemoryFault { at: c.at, addr });
        }
        if d.format == rdna::FORMAT_8_8_8_8_UNORM {
            let packed = texel[0];
            for (k, t) in texel.iter_mut().enumerate() {
                *t = rdna::unorm8_to_f32(packed >> (8 * k));
            }
        }
        let one = if float { 1.0f32.to_bits() } else { 1 };
        let comp = |sel: u32| -> u32 {
            match sel {
                1 => one,
                4..=7 if sel - 4 < channels => texel[(sel - 4) as usize],
                // Missing channels: (r, 0, 0, 1).
                7 => one,
                _ => 0,
            }
        };
        let mut k = 0u8;
        for (ci, sel) in d.dst_sel.iter().enumerate() {
            if dmask >> ci & 1 != 0 {
                c.vgpr_mut(w, vdata + k)?[lane] = comp(*sel);
                k += 1;
            }
        }
    }
    Ok(())
}

/// ds_append / ds_consume (see rdna::ds::APPEND). Refused: wave64 (the
/// hardware may split it into two halves, which ACO works around and the
/// sources do not pin down), a non-zero M0 (ACO leaves M0 uninitialised on
/// GFX9+ while LLVM puts the base there), unaligned or out-of-range
/// addresses.
fn ds_append(c: &Ctx, w: &mut Wave, consume: bool, off: u32, vdst: u8) -> Result<(), ExecError> {
    if w.size.lanes() != 32 {
        return Err(c.err("ds_append/consume in wave64"));
    }
    if w.m0 != 0 {
        return Err(c.err("ds_append/consume with a non-zero M0"));
    }
    let n = (w.exec & w.size.mask()).count_ones();
    if n == 0 {
        return Ok(());
    }
    if off & 3 != 0 {
        return Err(c.err("unaligned LDS access"));
    }
    let idx = (off / 4) as usize;
    if idx >= w.lds.len() {
        return Err(c.err("LDS access beyond the allocation"));
    }
    let old = w.lds[idx];
    w.lds[idx] = if consume { old.wrapping_sub(n) } else { old.wrapping_add(n) };
    for lane in 0..32 {
        if w.active(lane) {
            c.vgpr_mut(w, vdst)?[lane] = old;
        }
    }
    Ok(())
}

/// ds_swizzle_b32 (see rdna::ds::SWIZZLE_B32). Refused: a source lane that
/// is inactive (what it reads then is not confirmed), unmodelled modes.
fn ds_swizzle(c: &Ctx, w: &mut Wave, offset: u32, addr: u8, vdst: u8) -> Result<(), ExecError> {
    let mut out = Vec::new();
    for l in 0..w.size.lanes() {
        if !w.active(l) {
            continue;
        }
        let src = ds::swizzle_source(offset, l).ok_or_else(|| c.err("ds_swizzle mode"))?;
        if !w.active(src) {
            return Err(c.err("ds_swizzle from an inactive lane"));
        }
        out.push((l, c.v32(w, 256 + addr as u16, src)?));
    }
    for (l, v) in out {
        c.vgpr_mut(w, vdst)?[l] = v;
    }
    Ok(())
}

/// LDS atomics, lanes in ascending order; refused where that order would
/// show: two lanes on one dword returning the old value or with
/// cmpst/wrxchg. Between waves of a workgroup the order of returning
/// atomics is not checked.
#[allow(clippy::too_many_arguments)]
fn ds_atomic(
    c: &Ctx,
    w: &mut Wave,
    base: u16,
    rtn: bool,
    off: u32,
    addr: u8,
    data0: u8,
    data1: u8,
    vdst: u8,
) -> Result<(), ExecError> {
    let mut work: Vec<(usize, usize, u32, u32)> = Vec::new();
    for lane in 0..w.size.lanes() {
        if !w.active(lane) {
            continue;
        }
        let byte = c.v32(w, 256 + addr as u16, lane)?.wrapping_add(off);
        if byte & 3 != 0 {
            return Err(c.err("unaligned LDS access"));
        }
        let idx = (byte / 4) as usize;
        if idx >= w.lds.len() {
            return Err(c.err("LDS access beyond the allocation"));
        }
        if work.iter().any(|t| t.1 == idx) && (rtn || !ds::atomic_commutes(base)) {
            return Err(c.err("two lanes' LDS atomics on one dword with an order-dependent result"));
        }
        let d0 = c.v32(w, 256 + data0 as u16, lane)?;
        let d1 = if base == ds::CMPST_B32 { c.v32(w, 256 + data1 as u16, lane)? } else { 0 };
        work.push((lane, idx, d0, d1));
    }
    for (lane, idx, d0, d1) in work {
        let old = w.lds[idx];
        w.lds[idx] = ds::atomic_apply(base, old, d0, d1);
        if rtn {
            c.vgpr_mut(w, vdst)?[lane] = old;
        }
    }
    Ok(())
}

/// LDS access. Refused rather than guessed: unaligned addresses, addresses
/// beyond the LDS allocation, and two lanes writing the same dword in one
/// instruction.
#[allow(clippy::too_many_arguments)]
fn ds_step(
    c: &Ctx,
    w: &mut Wave,
    op: u16,
    offset0: u8,
    offset1: u8,
    addr: u8,
    data0: u8,
    data1: u8,
    vdst: u8,
) -> Result<(), ExecError> {
    if let Some((bytes, sext, write)) = ds::sub_dword(op) {
        return ds_sub_dword(c, w, offset0 as u32 | (offset1 as u32) << 8, bytes, sext, write, addr, data0, vdst);
    }
    let layout = ds::layout(op, offset0, offset1).ok_or_else(|| c.err("opcode"))?;
    let mut writes: Vec<(usize, u32)> = Vec::new();
    let mut reads: Vec<(usize, Vec<u32>)> = Vec::new();
    for lane in 0..w.size.lanes() {
        if !w.active(lane) {
            continue;
        }
        let a = c.v32(w, 256 + addr as u16, lane)?;
        let mut got = Vec::new();
        for &(o, (sel, sub), align) in &layout.dwords {
            let byte = a.wrapping_add(o);
            if byte % align != 0 {
                return Err(c.err("unaligned LDS access"));
            }
            let idx = (byte / 4) as usize;
            if idx >= w.lds.len() {
                return Err(c.err("LDS access beyond the allocation"));
            }
            if layout.write {
                let r = [data0, data1][sel as usize] + sub;
                let v = c.v32(w, 256 + r as u16, lane)?;
                if writes.iter().any(|(i, _)| *i == idx) {
                    return Err(c.err("two lanes write the same LDS dword"));
                }
                writes.push((idx, v));
            } else {
                got.push(w.lds[idx]);
            }
        }
        if !layout.write {
            reads.push((lane, got));
        }
    }
    for (idx, v) in writes {
        w.lds[idx] = v;
    }
    for (lane, vals) in reads {
        for (k, v) in vals.into_iter().enumerate() {
            c.vgpr_mut(w, vdst + k as u8)?[lane] = v;
        }
    }
    Ok(())
}

/// ds_read_{u,i}{8,16} (zero / sign extended) and ds_write_b{8,16} (the
/// low bytes of data0; the rest of the dword is kept). Refused: a 16-bit
/// access at an odd address, addresses beyond the allocation, and two
/// lanes writing the same byte.
#[allow(clippy::too_many_arguments)]
fn ds_sub_dword(c: &Ctx, w: &mut Wave, off: u32, bytes: u32, sext: bool, write: bool, addr: u8, data0: u8, vdst: u8) -> Result<(), ExecError> {
    let mut writes: Vec<(usize, u32, u32)> = Vec::new();
    let mut reads = Vec::new();
    let mask = if bytes == 1 { 0xFFu32 } else { 0xFFFF };
    for lane in 0..w.size.lanes() {
        if !w.active(lane) {
            continue;
        }
        let byte = c.v32(w, 256 + addr as u16, lane)?.wrapping_add(off);
        if byte % bytes != 0 {
            return Err(c.err("unaligned LDS access"));
        }
        let idx = (byte / 4) as usize;
        if idx >= w.lds.len() {
            return Err(c.err("LDS access beyond the allocation"));
        }
        let sh = (byte & 3) * 8;
        if write {
            let m = mask << sh;
            if writes.iter().any(|&(i, wm, _)| i == idx && wm & m != 0) {
                return Err(c.err("two lanes write the same LDS byte"));
            }
            writes.push((idx, m, (c.v32(w, 256 + data0 as u16, lane)? & mask) << sh));
        } else {
            let v = w.lds[idx] >> sh & mask;
            let v = if sext { ((v << (32 - 8 * bytes)) as i32 >> (32 - 8 * bytes)) as u32 } else { v };
            reads.push((lane, v));
        }
    }
    for (idx, m, v) in writes {
        w.lds[idx] = w.lds[idx] & !m | v;
    }
    for (lane, v) in reads {
        c.vgpr_mut(w, vdst)?[lane] = v;
    }
    Ok(())
}

/// `s_buffer_load_dword*`: raw buffer semantics. Mesa notes that SMEM is
/// bounds-checked against NUM_RECORDS from GFX8 on (GFX6-7 had a bug that
/// skipped the check); out-of-bounds dwords read 0. Modelled per dword:
/// in bounds iff `offset + 4 <= num_records`. Refused rather than guessed:
/// a descriptor with a stride, swizzling or another OOB_SELECT, an offset
/// that wraps past 2^32 within one load, and a dword that straddles a
/// NUM_RECORDS that is not a multiple of 4.
#[allow(clippy::too_many_arguments)]
fn s_buffer_load(
    c: &Ctx,
    w: &mut Wave,
    mem: &mut dyn Memory,
    n: usize,
    sdata: u8,
    sbase: u8,
    offset: i32,
    soffset: u8,
) -> Result<(), ExecError> {
    let mut words = [0u32; 4];
    for (k, x) in words.iter_mut().enumerate() {
        *x = c.s32(w, sbase as u16 + k as u16)?;
    }
    let d = rdna::BufferDesc::parse(words);
    if d.stride != 0 || d.swizzle || d.oob_select != 3 {
        return Err(c.err("s_buffer_load needs a raw descriptor (no stride/swizzle, OOB_SELECT 3)"));
    }
    let so = if soffset == smem::SOFFSET_NULL {
        0
    } else {
        c.s32(w, soffset as u16)?
    };
    let start = (offset as u32).wrapping_add(so) & !3;
    for k in 0..n {
        let Some(o) = start.checked_add(4 * k as u32) else {
            // Whether the hardware wraps the 32-bit offset or treats the
            // dword as out of bounds is not known here; refuse.
            return Err(c.err("buffer offset wraps past 2^32"));
        };
        let o = o as u64;
        let end = o + 4;
        let nr = d.num_records as u64;
        let v = if end <= nr {
            let mut x = [0u32];
            let addr = d.base.wrapping_add(o);
            if !mem.read(addr, &mut x) {
                return Err(ExecError::MemoryFault { at: c.at, addr });
            }
            x[0]
        } else if o < nr {
            return Err(c.err("dword straddles NUM_RECORDS"));
        } else {
            0
        };
        c.set_s32(w, sdata + k as u8, v)?;
    }
    Ok(())
}

fn sopp_step(c: &Ctx, w: &mut Wave, op: u16) -> Result<(), ExecError> {
    let take = match op {
        // One wave per workgroup: a barrier has nobody else to wait for.
        sopp::S_NOP | sopp::S_WAITCNT | sopp::S_BARRIER => return Ok(()),
        sopp::S_BRANCH => true,
        sopp::S_CBRANCH_SCC0 => !w.scc,
        sopp::S_CBRANCH_SCC1 => w.scc,
        sopp::S_CBRANCH_VCCZ => w.vccz(),
        sopp::S_CBRANCH_VCCNZ => !w.vccz(),
        sopp::S_CBRANCH_EXECZ => w.execz(),
        sopp::S_CBRANCH_EXECNZ => !w.execz(),
        _ => return Err(c.err("opcode")),
    };
    if take {
        w.pc = c.inst.branch_target(c.at).ok_or_else(|| c.err("branch target"))?;
    }
    Ok(())
}

/// 64-bit SOP2 bitwise ops: result (SCC = result != 0).
pub fn sop2_b64(op: u16, a: u64, b: u64) -> Option<u64> {
    Some(match op {
        sop2::S_AND_B64 => a & b,
        sop2::S_OR_B64 => a | b,
        sop2::S_XOR_B64 => a ^ b,
        sop2::S_ANDN2_B64 => a & !b,
        sop2::S_ORN2_B64 => a | !b,
        sop2::S_NAND_B64 => !(a & b),
        sop2::S_NOR_B64 => !(a | b),
        sop2::S_XNOR_B64 => !(a ^ b),
        _ => return None,
    })
}

/// 32-bit SOP2 ops without carry: (result, writes SCC as result != 0).
/// Which ops define SCC follows LLVM's SOPInstructions.td; s_bfe takes
/// S1 = offset[4:0] | width[22:16] (Mesa ACO packs `offset | bits << 16`),
/// s_bfm(bits, offset) = ((1 << bits) - 1) << offset. Widths of 32 and
/// more are refused by the caller.
pub fn sop2_b32(op: u16, a: u32, b: u32) -> Option<(u32, bool)> {
    let bfe = |a: u32, b: u32, signed: bool| {
        let (off, width) = (b & 31, (b >> 16) & 0x7F);
        if width == 0 {
            return 0;
        }
        let x = (a >> off) & ((1u32 << width) - 1);
        if signed && x >> (width - 1) & 1 != 0 {
            x | !((1u32 << width) - 1)
        } else {
            x
        }
    };
    Some(match op {
        sop2::S_AND_B32 => (a & b, true),
        sop2::S_OR_B32 => (a | b, true),
        sop2::S_XOR_B32 => (a ^ b, true),
        sop2::S_ANDN2_B32 => (a & !b, true),
        sop2::S_ORN2_B32 => (a | !b, true),
        sop2::S_NAND_B32 => (!(a & b), true),
        sop2::S_NOR_B32 => (!(a | b), true),
        sop2::S_XNOR_B32 => (!(a ^ b), true),
        sop2::S_LSHL_B32 => (a << (b & 31), true),
        sop2::S_LSHR_B32 => (a >> (b & 31), true),
        sop2::S_ASHR_I32 => (((a as i32) >> (b & 31)) as u32, true),
        sop2::S_BFM_B32 => (((1u64 << (a & 31)) as u32).wrapping_sub(1) << (b & 31), false),
        sop2::S_MUL_I32 => (a.wrapping_mul(b), false),
        sop2::S_MUL_HI_U32 => (((a as u64 * b as u64) >> 32) as u32, false),
        sop2::S_MUL_HI_I32 => (((a as i32 as i64 * b as i32 as i64) >> 32) as u32, false),
        sop2::S_BFE_U32 => (bfe(a, b, false), true),
        sop2::S_BFE_I32 => (bfe(a, b, true), true),
        sop2::S_PACK_LL_B32_B16 => (a & 0xFFFF | b << 16, false),
        sop2::S_PACK_LH_B32_B16 => (a & 0xFFFF | b & 0xFFFF_0000, false),
        sop2::S_PACK_HH_B32_B16 => (a >> 16 | b & 0xFFFF_0000, false),
        _ => return None,
    })
}

/// `s_lshl_b64` / `s_lshr_b64` / `s_ashr_i64` by `b & 63`.
pub fn shift64(op: u16, a: u64, b: u32) -> Option<u64> {
    let s = b & 63;
    Some(match op {
        sop2::S_LSHL_B64 => a << s,
        sop2::S_LSHR_B64 => a >> s,
        sop2::S_ASHR_I64 => ((a as i64) >> s) as u64,
        _ => return None,
    })
}

fn sop2_step(c: &Ctx, w: &mut Wave, op: u16, d: u8, s0: u16, s1: u16) -> Result<(), ExecError> {
    if sop2_b64(op, 0, 0).is_some() {
        let r = sop2_b64(op, c.s64(w, s0)?, c.s64(w, s1)?).unwrap();
        w.scc = r != 0;
        return c.set_s64(w, d, r);
    }
    if shift64(op, 0, 0).is_some() {
        let r = shift64(op, c.s64(w, s0)?, c.s32(w, s1)?).unwrap();
        w.scc = r != 0;
        return c.set_s64(w, d, r);
    }
    if op == sop2::S_CSELECT_B64 {
        let v = if w.scc { c.s64(w, s0)? } else { c.s64(w, s1)? };
        return c.set_s64(w, d, v);
    }
    if op == sop2::S_BFM_B64 {
        let (n, off) = (c.s32(w, s0)? & 63, c.s32(w, s1)? & 63);
        return c.set_s64(w, d, (((1u128 << n) - 1) as u64) << off);
    }
    if matches!(op, sop2::S_MIN_I32 | sop2::S_MIN_U32 | sop2::S_MAX_I32 | sop2::S_MAX_U32) {
        let (a, b) = (c.s32(w, s0)?, c.s32(w, s1)?);
        if a == b {
            // which operand counts as selected (SCC) is not stated for a tie
            return Err(c.err("s_min/s_max of equal operands (SCC)"));
        }
        let first = match op {
            sop2::S_MIN_I32 => (a as i32) < (b as i32),
            sop2::S_MIN_U32 => a < b,
            sop2::S_MAX_I32 => (a as i32) > (b as i32),
            _ => a > b,
        };
        w.scc = first;
        return c.set_s32(w, d, if first { a } else { b });
    }
    if op == sop2::S_ABSDIFF_I32 {
        let (a, b) = (c.s32(w, s0)? as i32 as i64, c.s32(w, s1)? as i32 as i64);
        let r = (a - b).abs();
        if r > i32::MAX as i64 {
            return Err(c.err("s_absdiff_i32 beyond i32 (overflow behaviour not stated)"));
        }
        w.scc = r != 0;
        return c.set_s32(w, d, r as u32);
    }
    if sop2_b32(op, 0, 0).is_some() {
        let (a, b) = (c.s32(w, s0)?, c.s32(w, s1)?);
        if matches!(op, sop2::S_BFE_U32 | sop2::S_BFE_I32) && (b >> 16) & 0x7F >= 32 {
            return Err(c.err("bit-field width of 32 or more"));
        }
        let (r, scc) = sop2_b32(op, a, b).unwrap();
        if scc {
            w.scc = r != 0;
        }
        return c.set_s32(w, d, r);
    }
    let (a, b) = (c.s32(w, s0)?, c.s32(w, s1)?);
    let r = match op {
        sop2::S_ADD_U32 => {
            let (r, carry) = a.overflowing_add(b);
            w.scc = carry;
            r
        }
        sop2::S_SUB_U32 => {
            let (r, borrow) = a.overflowing_sub(b);
            w.scc = borrow;
            r
        }
        sop2::S_ADD_I32 => {
            let (r, ovf) = (a as i32).overflowing_add(b as i32);
            w.scc = ovf;
            r as u32
        }
        sop2::S_ADDC_U32 => {
            let wide = a as u64 + b as u64 + w.scc as u64;
            w.scc = wide >> 32 != 0;
            wide as u32
        }
        sop2::S_SUB_I32 => {
            let (r, ovf) = (a as i32).overflowing_sub(b as i32);
            w.scc = ovf;
            r as u32
        }
        sop2::S_SUBB_U32 => {
            let need = b as u64 + w.scc as u64;
            w.scc = (a as u64) < need;
            (a as u64).wrapping_sub(need) as u32
        }
        sop2::S_CSELECT_B32 => {
            if w.scc {
                a
            } else {
                b
            }
        }
        _ => return Err(c.err("opcode")),
    };
    c.set_s32(w, d, r)
}

/// 32-bit SOP1 ops: (result, writes SCC as result != 0), per LLVM's
/// SOPInstructions.td for which ones define SCC.
pub fn sop1_b32(op: u16, a: u32) -> Option<(u32, bool)> {
    Some(match op {
        sop1::S_NOT_B32 => (!a, true),
        sop1::S_BREV_B32 => (a.reverse_bits(), false),
        sop1::S_BCNT0_I32_B32 => (32 - a.count_ones(), true),
        sop1::S_BCNT1_I32_B32 => (a.count_ones(), true),
        // ffbh: -1 for 0
        sop1::S_FLBIT_I32_B32 => (if a == 0 { u32::MAX } else { a.leading_zeros() }, false),
        sop1::S_FF0_I32_B32 => (if a == u32::MAX { u32::MAX } else { a.trailing_ones() }, false),
        sop1::S_FLBIT_I32 => (vop1_int(vop1::V_FFBH_I32, a).unwrap(), false),
        sop1::S_SEXT_I32_I8 => (a as u8 as i8 as i32 as u32, false),
        sop1::S_SEXT_I32_I16 => (a as u16 as i16 as i32 as u32, false),
        _ => return None,
    })
}

fn sop1_step(c: &Ctx, w: &mut Wave, op: u16, d: u8, s0: u16) -> Result<(), ExecError> {
    if sop1_b32(op, 0).is_some() {
        let (r, scc) = sop1_b32(op, c.s32(w, s0)?).unwrap();
        if scc {
            w.scc = r != 0;
        }
        return c.set_s32(w, d, r);
    }
    match op {
        sop1::S_GETPC_B64 => {
            // Address of the next instruction (s_getpc is one dword). PCs
            // from 2^47 up are refused: whether the 48-bit PC is sign- or
            // zero-extended differs between generations in LLVM.
            let pc = w.code_base.wrapping_add(4 * (c.at as u64 + 1));
            if pc >= 1 << 47 || pc < w.code_base {
                return Err(c.err("PC at or above 2^47"));
            }
            c.set_s64(w, d, pc)
        }
        sop1::S_BITSET0_B32 | sop1::S_BITSET1_B32 => {
            let bit = 1u32 << (c.s32(w, s0)? & 31);
            let old = c.s32(w, d as u16)?;
            let r = if op == sop1::S_BITSET1_B32 { old | bit } else { old & !bit };
            c.set_s32(w, d, r)
        }
        sop1::S_BREV_B64 => {
            let r = c.s64(w, s0)?.reverse_bits();
            c.set_s64(w, d, r)
        }
        op if sop1::bit64(op, 0).is_some() => {
            let r = sop1::bit64(op, c.s64(w, s0)?).unwrap();
            if op == sop1::S_BCNT1_I32_B64 || op == sop1::S_BCNT0_I32_B64 {
                w.scc = r != 0;
            }
            c.set_s32(w, d, r)
        }
        sop1::S_BITSET0_B64 | sop1::S_BITSET1_B64 => {
            let bit = 1u64 << (c.s32(w, s0)? & 63);
            let old = c.s64(w, d as u16)?;
            let r = if op == sop1::S_BITSET1_B64 { old | bit } else { old & !bit };
            c.set_s64(w, d, r)
        }
        sop1::S_NOT_B64 => {
            let r = !c.s64(w, s0)?;
            w.scc = r != 0;
            c.set_s64(w, d, r)
        }
        sop1::S_CMOV_B32 => {
            if w.scc {
                let v = c.s32(w, s0)?;
                c.set_s32(w, d, v)?;
            }
            Ok(())
        }
        sop1::S_MOV_B32 => {
            let v = c.s32(w, s0)?;
            c.set_s32(w, d, v)
        }
        sop1::S_BITREPLICATE_B64_B32 => {
            let v = sop1::bitreplicate(c.s32(w, s0)?);
            c.set_s64(w, d, v)
        }
        sop1::S_MOV_B64 => {
            let v = c.s64(w, s0)?;
            c.set_s64(w, d, v)
        }
        sop1::S_FF1_I32_B32 => {
            let v = c.s32(w, s0)?;
            let r = if v == 0 { u32::MAX } else { v.trailing_zeros() };
            c.set_s32(w, d, r)
        }
        sop1::S_ABS_I32 => {
            let r = (c.s32(w, s0)? as i32).wrapping_abs() as u32;
            w.scc = r != 0;
            c.set_s32(w, d, r)
        }
        op if sop1::exec_op(op).is_some() => {
            let (wide, wr, f) = sop1::exec_op(op).unwrap();
            if wide != (w.size == WaveSize::W64) {
                return Err(c.err("saveexec width does not match the wave size"));
            }
            let src = c.lane_mask(w, s0)?;
            let old = w.exec & w.size.mask();
            let new = f(src, old) & w.size.mask();
            c.set_lane_mask(w, d, if wr { new } else { old })?;
            w.exec = (w.exec & !w.size.mask()) | new;
            w.scc = !w.execz();
            Ok(())
        }
        sop1::S_CMOV_B64 => {
            if w.scc {
                let v = c.s64(w, s0)?;
                c.set_s64(w, d, v)?;
            }
            Ok(())
        }
        _ => Err(c.err("opcode")),
    }
}

/// f32 -> f16 bits, round to nearest even or toward zero. None where the
/// result would depend on the shader's FLOAT_MODE or is not confirmed:
/// NaN, an f16 denormal result or input below 2^-14, and overflow (RNE
/// from 65520, RTZ from 65536). Infinities and zeros keep their sign.
pub fn f32_to_f16(bits: u32, rtz: bool) -> Option<u16> {
    let s = (bits >> 16 & 0x8000) as u16;
    let e = bits >> 23 & 0xFF;
    let m = bits & 0x7F_FFFF;
    let mag = f32::from_bits(bits & 0x7FFF_FFFF);
    if e == 0xFF {
        return (m == 0).then_some(s | 0x7C00);
    }
    if bits & 0x7FFF_FFFF == 0 {
        return Some(s);
    }
    if mag < 1.0 / 16384.0 || mag >= if rtz { 65536.0 } else { 65520.0 } {
        return None;
    }
    let mut h = (e - 112) << 10 | m >> 13;
    let rem = m & 0x1FFF;
    if !rtz && (rem > 0x1000 || (rem == 0x1000 && h & 1 == 1)) {
        h += 1;
    }
    Some(s | h as u16)
}

/// f16 bits -> f32 bits; None for NaN and f16 denormals (FLOAT_MODE).
pub fn f16_to_f32(h: u16) -> Option<u32> {
    let s = (h as u32 & 0x8000) << 16;
    let e = (h >> 10 & 0x1F) as u32;
    let m = (h & 0x3FF) as u32;
    Some(match (e, m) {
        (0, 0) => s,
        (0, _) | (31, 1..) => return None,
        (31, 0) => s | 0x7F80_0000,
        _ => s | (e + 112) << 23 | m << 13,
    })
}

fn vop1_step(c: &Ctx, w: &mut Wave, op: u16, vdst: u8, src0: u16) -> Result<(), ExecError> {
    match op {
        vop1::V_CVT_F16_F32 => for_active(c, w, vdst, |w, l| {
            let h = f32_to_f16(c.v32(w, src0, l)?, false)
                .ok_or_else(|| c.err("f16 NaN, denormal or overflow (FLOAT_MODE)"))?;
            Ok(c.v32(w, 256 + vdst as u16, l)? & 0xFFFF_0000 | h as u32)
        }),
        vop1::V_CVT_F32_F16 => for_active(c, w, vdst, |w, l| {
            f16_to_f32(c.v32(w, src0, l)? as u16).ok_or_else(|| c.err("f16 NaN or denormal (FLOAT_MODE)"))
        }),
        vop1::V_NOP => Ok(()),
        vop1::V_MOV_B32 => for_active(c, w, vdst, |w, l| c.v32(w, src0, l)),
        vop1::V_SWAP_B32 => {
            if src0 < 256 {
                return Err(c.err("v_swap_b32 of a non-VGPR"));
            }
            for l in 0..w.size.lanes() {
                if w.active(l) {
                    let (x, y) = (c.v32(w, src0, l)?, c.v32(w, 256 + vdst as u16, l)?);
                    c.vgpr_mut(w, vdst)?[l] = x;
                    c.vgpr_mut(w, (src0 - 256) as u8)?[l] = y;
                }
            }
            Ok(())
        }
        vop1::V_CVT_F32_I32 => for_active(c, w, vdst, |w, l| {
            Ok((c.v32(w, src0, l)? as i32 as f32).to_bits())
        }),
        vop1::V_CVT_F32_U32 => for_active(c, w, vdst, |w, l| {
            Ok((c.v32(w, src0, l)? as f32).to_bits())
        }),
        // Saturating, NaN -> 0: the hardware conversion. LLVM's AMDGPU
        // backend selects these instructions directly for
        // llvm.fpto{u,s}i.sat, which has exactly Rust's `as` semantics.
        vop1::V_CVT_U32_F32 => for_active(c, w, vdst, |w, l| {
            Ok(f32::from_bits(c.v32(w, src0, l)?) as u32)
        }),
        op if matches!(
            op,
            vop1::V_CVT_RPI_I32_F32 | vop1::V_CVT_FLR_I32_F32 | vop1::V_FREXP_EXP_I32_F32 | vop1::V_FREXP_MANT_F32
        ) =>
        {
            for_active(c, w, vdst, |w, l| {
                vop1_f32_special(op, c.v32(w, src0, l)?).ok_or_else(|| c.err("value the sources leave open (NaN, subnormal or out of range)"))
            })
        }
        vop1::V_CVT_I32_F32 => for_active(c, w, vdst, |w, l| {
            Ok(f32::from_bits(c.v32(w, src0, l)?) as i32 as u32)
        }),
        vop1::V_CVT_I32_F64 | vop1::V_CVT_U32_F64 | vop1::V_CVT_F32_F64 => {
            let mut res = Vec::new();
            for l in 0..w.size.lanes() {
                if w.active(l) {
                    let x = f64::from_bits(src64(c, w, src0, l)?);
                    res.push((l, cvt_from_f64(op, x).ok_or_else(|| c.err("f64 -> f32 of a NaN, subnormal or f32-subnormal value"))?));
                }
            }
            for (l, r) in res {
                c.vgpr_mut(w, vdst)?[l] = r;
            }
            Ok(())
        }
        vop1::V_CVT_F64_I32 | vop1::V_CVT_F64_U32 | vop1::V_CVT_F64_F32 => {
            let mut res = Vec::new();
            for l in 0..w.size.lanes() {
                if w.active(l) {
                    let a = c.v32(w, src0, l)?;
                    let x = match op {
                        vop1::V_CVT_F64_I32 => a as i32 as f64,
                        vop1::V_CVT_F64_U32 => a as f64,
                        _ => {
                            let f = f32::from_bits(a);
                            if f.is_nan() || f.is_subnormal() {
                                return Err(c.err("f32 -> f64 of a NaN or subnormal"));
                            }
                            f as f64
                        }
                    };
                    res.push((l, x.to_bits()));
                }
            }
            for (l, r) in res {
                c.vgpr_mut(w, vdst)?[l] = r as u32;
                c.vgpr_mut(w, vdst + 1)?[l] = (r >> 32) as u32;
            }
            Ok(())
        }
        vop1::V_TRUNC_F64..=vop1::V_FLOOR_F64 => {
            let mut res = Vec::new();
            for l in 0..w.size.lanes() {
                if w.active(l) {
                    let r = round_f64(op, src64(c, w, src0, l)?).ok_or_else(|| c.err("f64 rounding of a NaN or subnormal"))?;
                    res.push((l, r));
                }
            }
            for (l, r) in res {
                c.vgpr_mut(w, vdst)?[l] = r as u32;
                c.vgpr_mut(w, vdst + 1)?[l] = (r >> 32) as u32;
            }
            Ok(())
        }
        op if vop1_float(op, 0).is_some() => for_active(c, w, vdst, |w, l| {
            Ok(vop1_float(op, c.v32(w, src0, l)?).unwrap())
        }),
        op if vop1_int(op, 0).is_some() => for_active(c, w, vdst, |w, l| {
            Ok(vop1_int(op, c.v32(w, src0, l)?).unwrap())
        }),
        vop1::V_READFIRSTLANE_B32 => {
            // The first active lane, or lane 0 when none is.
            let lane = (0..w.size.lanes()).find(|&l| w.active(l)).unwrap_or(0);
            let v = c.v32(w, src0, lane)?;
            c.set_s32(w, vdst, v)
        }
        _ => Err(c.err("opcode")),
    }
}

/// v_min_f32 / v_max_f32 as LLVM's `minnum`/`maxnum`, which its AMDGPU
/// backend selects these instructions for (IEEE mode): a quiet NaN
/// yields the other operand, -0.0 < +0.0. ASSUMPTION: IEEE mode (the
/// compute default). None (refused) for signaling NaNs and two NaNs,
/// whose result the reference leaves open.
pub fn min_max_f32(max: bool, a: u32, b: u32) -> Option<u32> {
    let (x, y) = (f32::from_bits(a), f32::from_bits(b));
    let snan = |v: u32| f32::from_bits(v).is_nan() && v & 0x0040_0000 == 0;
    if snan(a) || snan(b) || (x.is_nan() && y.is_nan()) {
        return None;
    }
    Some(if x.is_nan() {
        b
    } else if y.is_nan() {
        a
    } else if x == y {
        // equal values, or +0 vs -0: min keeps the minus sign, max drops it
        if max { a & b } else { a | b }
    } else if (x < y) != max {
        a
    } else {
        b
    })
}

/// v_med3_f32 as LLVM's `fmed3AMDGCN`: the max of the two operands other
/// than the (bitwise) max of all three. None for any NaN (LLVM handles
/// those separately).
pub fn med3_f32(a: u32, b: u32, c: u32) -> Option<u32> {
    if [a, b, c].iter().any(|&v| f32::from_bits(v).is_nan()) {
        return None;
    }
    let max = |x, y| min_max_f32(true, x, y).unwrap();
    let m3 = max(max(a, b), c);
    Some(if m3 == a {
        max(b, c)
    } else if m3 == b {
        max(a, c)
    } else {
        max(a, b)
    })
}

/// v_min_f64 / v_max_f64 with the rules of `min_max_f32` (quiet NaN →
/// the other operand, -0 < +0; a signalling NaN or two NaNs refused).
pub fn min_max_f64(max: bool, a: u64, b: u64) -> Option<u64> {
    let (x, y) = (f64::from_bits(a), f64::from_bits(b));
    let snan = |v: u64| f64::from_bits(v).is_nan() && v & 0x0008_0000_0000_0000 == 0;
    if snan(a) || snan(b) || (x.is_nan() && y.is_nan()) {
        return None;
    }
    Some(if x.is_nan() {
        b
    } else if y.is_nan() {
        a
    } else if x == y {
        if max { a & b } else { a | b }
    } else if (x < y) != max {
        a
    } else {
        b
    })
}

/// Float VOP2 opcodes that accept DPP source modifiers.
fn vop2_takes_float_mods(op: u16) -> bool {
    matches!(
        op,
        vop2::V_ADD_F32 | vop2::V_SUB_F32 | vop2::V_SUBREV_F32 | vop2::V_MUL_F32
    )
}

/// A VOP1/VOP2 with a DPP16 src0: permute src0 across lanes, apply source
/// modifiers, and write only the lanes DPP enables (row/bank masks; an
/// invalid source lane reads 0 with bound_ctrl, else leaves the lane
/// alone). Done by running the plain instruction on a scratch VGPR under
/// a narrowed exec. None when the instruction has no DPP16 operand.
fn dpp_step(inst: &Inst, at: usize, w: &mut Wave, mem: &mut dyn Memory) -> Option<Result<(), ExecError>> {
    let (src0, is_vop2) = match inst.fields {
        Fields::Vop1 { src0, .. } => (src0, false),
        Fields::Vop2 { src0, .. } => (src0, true),
        Fields::Vopc { src0, .. } if src0 == rdna::SRC_DPP16 => {
            return Some(Err(unsupported(at, inst, "DPP on VOPC")));
        }
        _ => return None,
    };
    if src0 != rdna::SRC_DPP16 {
        return None;
    }
    Some((|| {
        let d = rdna::Dpp16::parse(*inst.words.get(1).ok_or_else(|| unsupported(at, inst, "missing DPP word"))?);
        let rule = d.rule().ok_or_else(|| unsupported(at, inst, "DPP control"))?;
        if d.fi {
            return Err(unsupported(at, inst, "DPP fetch-inactive"));
        }
        let mods = d.src0_neg || d.src0_abs || d.src1_neg || d.src1_abs;
        if mods && !(is_vop2 && vop2_takes_float_mods(inst.opcode)) {
            return Err(unsupported(at, inst, "DPP source modifiers on this opcode"));
        }
        let src = w
            .vgpr
            .get(d.src0 as usize)
            .ok_or_else(|| unsupported(at, inst, "VGPR beyond the wave's VGPRs"))?
            .to_owned();
        let mut vals = [0u32; 64];
        let mut enable = 0u64;
        #[allow(clippy::needless_range_loop)] // `lane` also indexes masks and exec
        for lane in 0..w.size.lanes() {
            let row_on = d.row_mask >> (lane / 16) & 1 != 0;
            let bank_on = d.bank_mask >> ((lane % 16) / 4) & 1 != 0;
            if !(row_on && bank_on) {
                continue;
            }
            let v = match rule.source(lane) {
                Some(sl) if sl < w.size.lanes() && w.active(sl) => Some(src[sl]),
                _ if d.bound_ctrl => Some(0),
                _ => None,
            };
            if let Some(mut v) = v {
                if d.src0_abs {
                    v &= 0x7FFF_FFFF;
                }
                if d.src0_neg {
                    v ^= 0x8000_0000;
                }
                vals[lane] = v;
                enable |= 1 << lane;
            }
        }
        // Run the plain instruction with src0 = a scratch VGPR.
        let tmp = w.vgpr.len();
        w.vgpr.push(vals);
        let mut plain = inst.clone();
        let mut src1_scratch = None;
        match &mut plain.fields {
            Fields::Vop1 { src0, .. } => *src0 = 256 + tmp as u16,
            Fields::Vop2 { src0, vsrc1, .. } => {
                *src0 = 256 + tmp as u16;
                if d.src1_neg || d.src1_abs {
                    let mut v1 = w.vgpr[*vsrc1 as usize];
                    for x in v1.iter_mut() {
                        if d.src1_abs {
                            *x &= 0x7FFF_FFFF;
                        }
                        if d.src1_neg {
                            *x ^= 0x8000_0000;
                        }
                    }
                    w.vgpr.push(v1);
                    *vsrc1 = (tmp + 1) as u8;
                    src1_scratch = Some(());
                }
            }
            _ => unreachable!(),
        }
        plain.words.truncate(1);
        let saved = w.exec;
        w.exec &= enable;
        let r = step(&plain, at, w, mem);
        w.exec = saved;
        if src1_scratch.is_some() {
            w.vgpr.pop();
        }
        w.vgpr.pop();
        r
    })())
}

/// Integer VOP2 semantics (RDNA2 ISA; shift amounts use S0[4:0], and the
/// `*rev` forms shift S1 by S0).
pub fn vop2_int(op: u16, a: u32, b: u32) -> Option<u32> {
    Some(match op {
        vop2::V_MUL_U32_U24 => (a & 0xFF_FFFF).wrapping_mul(b & 0xFF_FFFF),
        vop2::V_MUL_HI_U32_U24 => (((a as u64 & 0xFF_FFFF) * (b as u64 & 0xFF_FFFF)) >> 32) as u32,
        vop2::V_MUL_I32_I24 | vop2::V_MUL_HI_I32_I24 => {
            let s = |x: u32| ((x << 8) as i32 >> 8) as i64;
            let p = s(a) * s(b);
            if op == vop2::V_MUL_I32_I24 { p as u32 } else { (p >> 32) as u32 }
        }
        vop2::V_MIN_I32 => (a as i32).min(b as i32) as u32,
        vop2::V_MAX_I32 => (a as i32).max(b as i32) as u32,
        vop2::V_MIN_U32 => a.min(b),
        vop2::V_MAX_U32 => a.max(b),
        vop2::V_LSHRREV_B32 => b >> (a & 31),
        vop2::V_ASHRREV_I32 => ((b as i32) >> (a & 31)) as u32,
        vop2::V_LSHLREV_B32 => b << (a & 31),
        vop2::V_AND_B32 => a & b,
        vop2::V_OR_B32 => a | b,
        vop2::V_XOR_B32 => a ^ b,
        vop2::V_XNOR_B32 => !(a ^ b),
        vop2::V_ADD_NC_U32 => a.wrapping_add(b),
        vop2::V_SUB_NC_U32 => a.wrapping_sub(b),
        vop2::V_SUBREV_NC_U32 => b.wrapping_sub(a),
        _ => return None,
    })
}

/// Exact float VOP1 ops: rounding (round-to-nearest-even for rndne) and
/// byte-to-float conversion. ASSUMPTION: a NaN input comes out as the
/// same quiet NaN (what Rust's operations produce); to be confirmed.
pub fn vop1_float(op: u16, a: u32) -> Option<u32> {
    let x = f32::from_bits(a);
    Some(match op {
        vop1::V_FLOOR_F32 => x.floor().to_bits(),
        vop1::V_CEIL_F32 => x.ceil().to_bits(),
        vop1::V_TRUNC_F32 => x.trunc().to_bits(),
        vop1::V_RNDNE_F32 => x.round_ties_even().to_bits(),
        vop1::V_CVT_F32_UBYTE0..=vop1::V_CVT_F32_UBYTE3 => {
            let n = (op - vop1::V_CVT_F32_UBYTE0) as u32;
            (((a >> (8 * n)) & 0xFF) as f32).to_bits()
        }
        _ => return None,
    })
}

/// Integer VOP1 semantics. `ffbl`/`ffbh` return all ones for 0.
pub fn vop1_int(op: u16, a: u32) -> Option<u32> {
    Some(match op {
        vop1::V_NOT_B32 => !a,
        vop1::V_BFREV_B32 => a.reverse_bits(),
        vop1::V_FFBH_U32 => if a == 0 { u32::MAX } else { a.leading_zeros() },
        vop1::V_FFBL_B32 => if a == 0 { u32::MAX } else { a.trailing_zeros() },
        vop1::V_FFBH_I32 => {
            let x = if (a as i32) < 0 { !a } else { a };
            if x == 0 { u32::MAX } else { x.leading_zeros() }
        }
        _ => return None,
    })
}

/// Carry arithmetic: (result, carry/borrow out) of `op` (a VOP2 co_ci
/// opcode, its VOP3 form, or a VOP3b co opcode) on `a`, `b`, carry in.
/// Subtractions borrow when the subtrahend (plus carry in) exceeds the
/// minuend; subrev swaps the operands.
pub fn carry_op(op: u16, a: u32, b: u32, cin: bool) -> Option<(u32, bool)> {
    let e = |o: u16| vop3::FROM_VOP2 + o;
    let (kind, cin) = match op {
        vop3::V_ADD_CO_U32 => (0, false),
        vop3::V_SUB_CO_U32 => (1, false),
        vop3::V_SUBREV_CO_U32 => (2, false),
        vop2::V_ADD_CO_CI_U32 => (0, cin),
        vop2::V_SUB_CO_CI_U32 => (1, cin),
        vop2::V_SUBREV_CO_CI_U32 => (2, cin),
        o if o == e(vop2::V_ADD_CO_CI_U32) => (0, cin),
        o if o == e(vop2::V_SUB_CO_CI_U32) => (1, cin),
        o if o == e(vop2::V_SUBREV_CO_CI_U32) => (2, cin),
        _ => return None,
    };
    let (a, b) = if kind == 2 { (b, a) } else { (a, b) };
    let c = cin as u64;
    Some(if kind == 0 {
        let s = a as u64 + b as u64 + c;
        (s as u32, s >> 32 != 0)
    } else {
        (a.wrapping_sub(b).wrapping_sub(cin as u32), b as u64 + c > a as u64)
    })
}

/// Carry ops: per active lane, the result to `vdst` and the carry bit to
/// the mask `sdst` (inactive lanes' bits 0, as for VOPC). The carry in is
/// read from `cin` before anything is written.
#[allow(clippy::too_many_arguments)]
fn carry_step(c: &Ctx, w: &mut Wave, op: u16, vdst: u8, s0: u16, s1: u16, cin: Option<u16>, sdst: u8) -> Result<(), ExecError> {
    let m = match cin {
        Some(src) => c.lane_mask(w, src)?,
        None => 0,
    };
    let mut res = Vec::new();
    let mut out = 0u64;
    for l in 0..w.size.lanes() {
        if !w.active(l) {
            continue;
        }
        let (r, co) = carry_op(op, c.v32(w, s0, l)?, c.v32(w, s1, l)?, m >> l & 1 != 0).expect("carry op");
        res.push((l, r));
        out |= (co as u64) << l;
    }
    for (l, r) in res {
        c.vgpr_mut(w, vdst)?[l] = r;
    }
    c.set_lane_mask(w, sdst, out)
}

/// 16-bit integer VOP3 (rdna::vop3::V_*_U16 etc.): the low-half result.
pub fn vop3_16(op: u16, a: u32, b: u32) -> Option<u16> {
    let (a, b) = (a as u16, b as u16);
    let (sa, sb) = (a as i16, b as i16);
    Some(match op {
        vop3::V_ADD_NC_U16 => a.wrapping_add(b),
        vop3::V_SUB_NC_U16 => a.wrapping_sub(b),
        vop3::V_MUL_LO_U16 => a.wrapping_mul(b),
        vop3::V_LSHLREV_B16 => b << (a & 15),
        vop3::V_LSHRREV_B16 => b >> (a & 15),
        vop3::V_ASHRREV_I16 => (sb >> (a & 15)) as u16,
        vop3::V_MAX_U16 => a.max(b),
        vop3::V_MAX_I16 => sa.max(sb) as u16,
        vop3::V_MIN_U16 => a.min(b),
        vop3::V_MIN_I16 => sa.min(sb) as u16,
        _ => return None,
    })
}

/// Integer three-operand VOP3 semantics.
pub fn vop3_int(op: u16, a: u32, b: u32, c: u32) -> Option<u32> {
    Some(match op {
        vop3::V_MUL_LO_U32 => a.wrapping_mul(b),
        vop3::V_LSHL_ADD_U32 => (a << (b & 31)).wrapping_add(c),
        vop3::V_ADD3_U32 => a.wrapping_add(b).wrapping_add(c),
        vop3::V_AND_OR_B32 => (a & b) | c,
        vop3::V_OR3_B32 => a | b | c,
        vop3::V_XOR3_B32 => a ^ b ^ c,
        // AMD RDNA2 ISA specification descriptions
        vop3::V_ADD_NC_I32 => a.wrapping_add(b),
        vop3::V_SUB_NC_I32 => a.wrapping_sub(b),
        vop3::V_MAD_U32_U16 => (a & 0xFFFF).wrapping_mul(b & 0xFFFF).wrapping_add(c),
        vop3::V_MAD_I32_I16 => ((a as u16 as i16 as i32).wrapping_mul(b as u16 as i16 as i32) as u32).wrapping_add(c),
        vop3::V_SAD_U8 | vop3::V_SAD_HI_U8 => {
            let sum: u32 = (0..4).map(|k| ((a >> (8 * k)) as u8).abs_diff((b >> (8 * k)) as u8) as u32).sum();
            (if op == vop3::V_SAD_HI_U8 { sum << 16 } else { sum }).wrapping_add(c)
        }
        vop3::V_SAD_U16 => {
            let sum: u32 = (0..2).map(|k| ((a >> (16 * k)) as u16).abs_diff((b >> (16 * k)) as u16) as u32).sum();
            sum.wrapping_add(c)
        }
        vop3::V_SAD_U32 => a.abs_diff(b).wrapping_add(c),
        vop3::V_LERP_U8 => (0..4).fold(0, |r, k| {
            let byte = |x: u32| x >> (8 * k) & 0xFF;
            r | (byte(a) + byte(b) + (byte(c) & 1)) >> 1 << (8 * k)
        }),
        vop3::V_ADD_LSHL_U32 => a.wrapping_add(b) << (c & 31),
        vop3::V_BFM_B32 => (((1u64 << (a & 31)) as u32).wrapping_sub(1)) << (b & 31),
        vop3::V_BFE_U32 => (a >> (b & 31)) & ((1u32 << (c & 31)) - 1),
        vop3::V_BFI_B32 => (a & b) | (!a & c),
        vop3::V_MED3_U32 => a.min(b).max(a.max(b).min(c)),
        vop3::V_MUL_HI_U32 => ((a as u64 * b as u64) >> 32) as u32,
        vop3::V_MUL_LO_I32 => a.wrapping_mul(b),
        vop3::V_MUL_HI_I32 => ((a as i32 as i64 * b as i32 as i64) >> 32) as u32,
        vop3::V_MAD_U32_U24 => (a & 0xFF_FFFF).wrapping_mul(b & 0xFF_FFFF).wrapping_add(c),
        vop3::V_MAD_I32_I24 => {
            let s = |x: u32| ((x << 8) as i32 >> 8) as u32;
            s(a).wrapping_mul(s(b)).wrapping_add(c)
        }
        vop3::V_BCNT_U32_B32 => a.count_ones().wrapping_add(b),
        vop3::V_ALIGNBIT_B32 => (((a as u64) << 32 | b as u64) >> (c & 31)) as u32,
        vop3::V_BFE_I32 => {
            let (off, width) = (b & 31, c & 31);
            if width == 0 {
                0
            } else if off + width < 32 {
                ((a << (32 - off - width)) as i32 >> (32 - width)) as u32
            } else {
                (a as i32 >> off) as u32
            }
        }
        vop3::V_MIN3_I32 => (a as i32).min(b as i32).min(c as i32) as u32,
        vop3::V_MIN3_U32 => a.min(b).min(c),
        vop3::V_MAX3_I32 => (a as i32).max(b as i32).max(c as i32) as u32,
        vop3::V_MAX3_U32 => a.max(b).max(c),
        vop3::V_MED3_I32 => {
            let (a, b, c) = (a as i32, b as i32, c as i32);
            a.min(b).max(a.max(b).min(c)) as u32
        }
        vop3::V_XAD_U32 => (a ^ b).wrapping_add(c),
        vop3::V_LSHL_OR_B32 => (a << (b & 31)) | c,
        vop3::V_PERM_B32 => {
            let src = (a as u64) << 32 | b as u64;
            (0..4)
                .map(|i| {
                    let s = c >> (8 * i) & 0xFF;
                    let byte = match s {
                        0..=7 => (src >> (8 * s) & 0xFF) as u32,
                        8..=11 => ((src >> (((s - 8) * 2 + 1) * 8 + 7)) & 1) as u32 * 0xFF,
                        12 => 0,
                        _ => 0xFF,
                    };
                    byte << (8 * i)
                })
                .sum()
        }
        vop3::V_CVT_PK_U16_U32 => a.min(0xFFFF) | b.min(0xFFFF) << 16,
        vop3::V_CVT_PK_I16_I32 => {
            let s = |x: u32| (x as i32).clamp(-32768, 32767) as u32 & 0xFFFF;
            s(a) | s(b) << 16
        }
        vop3::V_ALIGNBYTE_B32 => (((a as u64) << 32 | b as u64) >> (8 * (c & 3))) as u32,
        vop3::V_MSAD_U8 => (0..4).fold(c, |acc, i| {
            let (r, s) = (b >> (8 * i) & 0xFF, a >> (8 * i) & 0xFF);
            if r != 0 { acc.wrapping_add(r.abs_diff(s)) } else { acc }
        }),
        _ => return None,
    })
}

/// Float source modifiers of VOP3: bit `i` of `abs`/`neg` for source `i`.
fn fmod(bits: u32, i: usize, abs: u8, neg: u8) -> f32 {
    let mut x = f32::from_bits(bits);
    if abs >> i & 1 == 1 {
        x = x.abs();
    }
    if neg >> i & 1 == 1 {
        x = -x;
    }
    x
}

fn vop2_step(
    c: &Ctx,
    w: &mut Wave,
    op: u16,
    vdst: u8,
    s0: u16,
    s1: u16,
    mask_src: u16,
) -> Result<(), ExecError> {
    let f = |a: u32, b: u32, g: fn(f32, f32) -> f32| g(f32::from_bits(a), f32::from_bits(b)).to_bits();
    match op {
        vop2::V_CVT_PKRTZ_F16_F32 => for_active(c, w, vdst, |w, l| {
            let h = |v: u32| f32_to_f16(v, true).ok_or_else(|| c.err("f16 NaN, denormal or overflow"));
            Ok(h(c.v32(w, s0, l)?)? as u32 | (h(c.v32(w, s1, l)?)? as u32) << 16)
        }),
        vop2::V_CNDMASK_B32 => {
            let m = c.lane_mask(w, mask_src)?;
            for_active(c, w, vdst, |w, l| {
                if m >> l & 1 == 1 {
                    c.v32(w, s1, l)
                } else {
                    c.v32(w, s0, l)
                }
            })
        }
        vop2::V_ADD_F32 => for_active(c, w, vdst, |w, l| {
            Ok(f(c.v32(w, s0, l)?, c.v32(w, s1, l)?, |a, b| a + b))
        }),
        vop2::V_SUB_F32 => for_active(c, w, vdst, |w, l| {
            Ok(f(c.v32(w, s0, l)?, c.v32(w, s1, l)?, |a, b| a - b))
        }),
        vop2::V_SUBREV_F32 => for_active(c, w, vdst, |w, l| {
            Ok(f(c.v32(w, s0, l)?, c.v32(w, s1, l)?, |a, b| b - a))
        }),
        vop2::V_MUL_F32 => for_active(c, w, vdst, |w, l| {
            Ok(f(c.v32(w, s0, l)?, c.v32(w, s1, l)?, |a, b| a * b))
        }),
        vop2::V_MIN_F32 | vop2::V_MAX_F32 => for_active(c, w, vdst, |w, l| {
            min_max_f32(op == vop2::V_MAX_F32, c.v32(w, s0, l)?, c.v32(w, s1, l)?)
                .ok_or_else(|| c.err("signaling NaN or two NaNs"))
        }),
        vop2::V_FMAC_F32 => for_active(c, w, vdst, |w, l| {
            let x = f32::from_bits(c.v32(w, s0, l)?);
            let y = f32::from_bits(c.v32(w, s1, l)?);
            let d = f32::from_bits(c.v32(w, 256 + vdst as u16, l)?);
            Ok(x.mul_add(y, d).to_bits())
        }),
        op if vop2_int(op, 0, 0).is_some() => for_active(c, w, vdst, |w, l| {
            Ok(vop2_int(op, c.v32(w, s0, l)?, c.v32(w, s1, l)?).unwrap())
        }),
        _ => Err(c.err("opcode")),
    }
}

fn vop3_step(c: &Ctx, w: &mut Wave) -> Result<(), ExecError> {
    let &Fields::Vop3 {
        vdst,
        src,
        abs,
        neg,
        clamp,
        omod,
        op_sel,
    } = &c.inst.fields
    else {
        unreachable!()
    };
    let op = c.inst.opcode;
    if op == vop3::V_MAD_U64_U32 {
        if clamp || omod != 0 || neg != 0 {
            return Err(c.err("clamp/omod/neg on v_mad_u64_u32"));
        }
        let sdst = (c.inst.words[0] >> 8 & 0x7F) as u8;
        let mut res = Vec::new();
        let mut out = 0u64;
        for l in 0..w.size.lanes() {
            if !w.active(l) {
                continue;
            }
            let full = c.v32(w, src[0], l)? as u128 * c.v32(w, src[1], l)? as u128 + src64(c, w, src[2], l)? as u128;
            res.push((l, full as u64));
            out |= ((full >> 64) as u64) << l;
        }
        for (l, r) in res {
            c.vgpr_mut(w, vdst)?[l] = r as u32;
            c.vgpr_mut(w, vdst + 1)?[l] = (r >> 32) as u32;
        }
        return c.set_lane_mask(w, sdst, out);
    }
    if carry_op(op, 0, 0, false).is_some() {
        // VOP3b: dword0 [14:8] is the carry-out SGPR, not abs/op_sel.
        if clamp || omod != 0 || neg != 0 {
            return Err(c.err("clamp/omod/neg on a carry op"));
        }
        let sdst = (c.inst.words[0] >> 8 & 0x7F) as u8;
        let cin = (op >= vop3::FROM_VOP2).then_some(src[2]).filter(|_| op < vop3::FROM_VOP1);
        return carry_step(c, w, op, vdst, src[0], src[1], cin, sdst);
    }
    if clamp || omod != 0 || op_sel != 0 {
        return Err(c.err("clamp/omod/op_sel"));
    }
    if let Some(cmp) = Compare::decode(op) {
        // VOPC in VOP3 form: the mask goes to the SGPR(s) named by vdst.
        if cmp.cmpx {
            return Err(c.err("v_cmpx e64"));
        }
        if cmp.wide() {
            return Err(c.err("64-bit compare e64"));
        }
        if (abs != 0 || neg != 0) && cmp.kind != CmpKind::F32 {
            return Err(c.err("source modifiers on an integer compare"));
        }
        let m = compare_mask(w, |l| {
            let x = |i: usize| -> Result<u32, ExecError> {
                Ok(fmod(c.v32(w, src[i], l)?, i, abs, neg).to_bits())
            };
            let (a, b) = if cmp.kind == CmpKind::F32 {
                (x(0)?, x(1)?)
            } else {
                (c.v32(w, src[0], l)?, c.v32(w, src[1], l)?)
            };
            Ok(cmp.eval(a, b))
        })?;
        return c.set_lane_mask(w, vdst, m);
    }
    if matches!(op, vop3::V_MIN_F64 | vop3::V_MAX_F64) {
        let mut res = Vec::new();
        for l in 0..w.size.lanes() {
            if !w.active(l) {
                continue;
            }
            let mut x = [0u64; 2];
            for (i, xi) in x.iter_mut().enumerate() {
                let mut b = src64(c, w, src[i], l)?;
                if abs >> i & 1 != 0 {
                    b &= !(1 << 63);
                }
                if neg >> i & 1 != 0 {
                    b ^= 1 << 63;
                }
                if f64::from_bits(b).is_subnormal() {
                    return Err(c.err("f64 subnormal operand"));
                }
                *xi = b;
            }
            let r = min_max_f64(op == vop3::V_MAX_F64, x[0], x[1]).ok_or_else(|| c.err("v_min/max_f64 of a signalling NaN or two NaNs"))?;
            res.push((l, r));
        }
        for (l, r) in res {
            c.vgpr_mut(w, vdst)?[l] = r as u32;
            c.vgpr_mut(w, vdst + 1)?[l] = (r >> 32) as u32;
        }
        return Ok(());
    }
    if matches!(op, vop3::V_ADD_F64 | vop3::V_MUL_F64 | vop3::V_FMA_F64) {
        return f64_arith(c, w, op, vdst, src, abs, neg);
    }
    let float_op = matches!(
        op,
        vop3::V_FMA_F32 | 0x103 | 0x104 | 0x105 | 0x108 | 0x10F | 0x110 | 0x12B // add/sub/subrev/mul/min/max/fmac e64
            | vop3::V_MIN3_F32 | vop3::V_MAX3_F32 | vop3::V_MED3_F32
    );
    if (abs != 0 || neg != 0) && !float_op {
        return Err(c.err("source modifiers on an integer operation"));
    }
    let lanes_below = |l: usize| if l >= 64 { u64::MAX } else { (1u64 << l) - 1 };
    match op {
        vop3::V_FMA_F32 => for_active(c, w, vdst, |w, l| {
            let x = |i| Ok::<f32, ExecError>(fmod(c.v32(w, src[i], l)?, i, abs, neg));
            Ok(x(0)?.mul_add(x(1)?, x(2)?).to_bits())
        }),
        vop3::V_MIN3_F32 | vop3::V_MAX3_F32 | vop3::V_MED3_F32 => for_active(c, w, vdst, |w, l| {
            let x = |i| Ok::<u32, ExecError>(fmod(c.v32(w, src[i], l)?, i, abs, neg).to_bits());
            let (a, b, d) = (x(0)?, x(1)?, x(2)?);
            let r = match op {
                vop3::V_MIN3_F32 => min_max_f32(false, a, b).and_then(|t| min_max_f32(false, t, d)),
                vop3::V_MAX3_F32 => min_max_f32(true, a, b).and_then(|t| min_max_f32(true, t, d)),
                _ => med3_f32(a, b, d),
            };
            r.ok_or_else(|| c.err("min3/max3/med3 case left open (NaN rules)"))
        }),
        vop3::V_LDEXP_F32 => for_active(c, w, vdst, |w, l| {
            ldexp_f32(c.v32(w, src[0], l)?, c.v32(w, src[1], l)? as i32).ok_or_else(|| c.err("v_ldexp_f32 of a NaN or subnormal, or a subnormal result"))
        }),
        vop3::V_READLANE_B32 | vop3::V_WRITELANE_B32 => {
            if abs != 0 || neg != 0 {
                return Err(c.err("source modifiers on a lane access"));
            }
            // A lane select at or beyond the wave size is refused (whether
            // the hardware masks it is not confirmed).
            let lane = c.s32(w, src[1])? as usize;
            if lane >= w.size.lanes() {
                return Err(c.err("lane select beyond the wave"));
            }
            if op == vop3::V_READLANE_B32 {
                let v = c.v32(w, src[0], lane)?;
                c.set_s32(w, vdst, v)
            } else {
                let v = c.s32(w, src[0])?;
                c.vgpr_mut(w, vdst)?[lane] = v;
                Ok(())
            }
        }
        vop3::V_LSHLREV_B64 | vop3::V_LSHRREV_B64 | vop3::V_ASHRREV_I64 => {
            if abs != 0 || neg != 0 {
                return Err(c.err("source modifiers on an integer operation"));
            }
            let sop = match op {
                vop3::V_LSHLREV_B64 => sop2::S_LSHL_B64,
                vop3::V_LSHRREV_B64 => sop2::S_LSHR_B64,
                _ => sop2::S_ASHR_I64,
            };
            let mut res = Vec::new();
            for l in 0..w.size.lanes() {
                if !w.active(l) {
                    continue;
                }
                let x = if src[1] >= 256 {
                    c.v32(w, src[1], l)? as u64 | (c.v32(w, src[1] + 1, l)? as u64) << 32
                } else {
                    c.s64(w, src[1])?
                };
                res.push((l, shift64(sop, x, c.v32(w, src[0], l)?).unwrap()));
            }
            for (l, r) in res {
                c.vgpr_mut(w, vdst)?[l] = r as u32;
                c.vgpr_mut(w, vdst + 1)?[l] = (r >> 32) as u32;
            }
            Ok(())
        }
        op if vop3_16(op, 0, 0).is_some() => {
            if abs != 0 || neg != 0 {
                return Err(c.err("source modifiers on an integer operation"));
            }
            for_active(c, w, vdst, |w, l| {
                let r = vop3_16(op, c.v32(w, src[0], l)?, c.v32(w, src[1], l)?).unwrap();
                Ok(c.v32(w, 256 + vdst as u16, l)? & 0xFFFF_0000 | r as u32)
            })
        }
        op if vop3_int(op, 0, 0, 0).is_some() => for_active(c, w, vdst, |w, l| {
            let x = |i: usize| c.v32(w, src[i], l);
            Ok(vop3_int(op, x(0)?, x(1)?, x(2)?).unwrap())
        }),
        vop3::V_MBCNT_LO_U32_B32 => for_active(c, w, vdst, |w, l| {
            let m = lanes_below(l) as u32;
            Ok((c.v32(w, src[0], l)? & m).count_ones() + c.v32(w, src[1], l)?)
        }),
        vop3::V_MBCNT_HI_U32_B32 => for_active(c, w, vdst, |w, l| {
            let m = (lanes_below(l) >> 32) as u32;
            Ok((c.v32(w, src[0], l)? & m).count_ones() + c.v32(w, src[1], l)?)
        }),
        op if (vop3::FROM_VOP1..vop3::FROM_VOP1 + 0x80).contains(&op) => {
            if abs != 0 || neg != 0 {
                return Err(c.err("source modifiers"));
            }
            vop1_step(c, w, op - vop3::FROM_VOP1, vdst, src[0])
        }
        op if (vop3::FROM_VOP2..vop3::FROM_VOP1).contains(&op) => {
            let base = op - vop3::FROM_VOP2;
            if base == vop2::V_CNDMASK_B32 {
                return vop2_step(c, w, base, vdst, src[0], src[1], src[2]);
            }
            if base == vop2::V_MIN_F32 || base == vop2::V_MAX_F32 || base == vop2::V_FMAC_F32 {
                return for_active(c, w, vdst, |w, l| {
                    let x = |i: usize| Ok::<f32, ExecError>(fmod(c.v32(w, src[i], l)?, i, abs, neg));
                    if base == vop2::V_FMAC_F32 {
                        let d = f32::from_bits(c.v32(w, 256 + vdst as u16, l)?);
                        return Ok(x(0)?.mul_add(x(1)?, d).to_bits());
                    }
                    min_max_f32(base == vop2::V_MAX_F32, x(0)?.to_bits(), x(1)?.to_bits())
                        .ok_or_else(|| c.err("signaling NaN or two NaNs"))
                });
            }
            if vop2_int(base, 0, 0).is_some() {
                if abs != 0 || neg != 0 {
                    return Err(c.err("source modifiers on an integer operation"));
                }
                return for_active(c, w, vdst, |w, l| {
                    Ok(vop2_int(base, c.v32(w, src[0], l)?, c.v32(w, src[1], l)?).unwrap())
                });
            }
            let g: fn(f32, f32) -> f32 = match base {
                vop2::V_ADD_F32 => |a, b| a + b,
                vop2::V_SUB_F32 => |a, b| a - b,
                vop2::V_MUL_F32 => |a, b| a * b,
                vop2::V_SUBREV_F32 => |a, b| b - a,
                _ => return Err(c.err("opcode")),
            };
            for_active(c, w, vdst, |w, l| {
                let x = |i| Ok::<f32, ExecError>(fmod(c.v32(w, src[i], l)?, i, abs, neg));
                Ok(g(x(0)?, x(1)?).to_bits())
            })
        }
        _ => Err(c.err("opcode")),
    }
}

/// Disassembly line of the instruction at `pc`, for error messages.
pub fn describe(code: &[u32], pc: usize) -> String {
    match rdna::decode(code, pc) {
        Ok(i) => format!("{pc:#x}: {i}"),
        Err(e) => format!("{pc:#x}: <{e:?}>"),
    }
}

/// Run a compute dispatch of `groups` work-groups set up per
/// [`crate::translate::ComputeIo`]: each work-group is `ceil(threads /
/// wave)` waves sharing `lds_words` of fresh LDS; wave `w` holds local
/// indices `w * wave ..`, the last one possibly partial (its extra lanes
/// start inactive). User SGPRs from `user`, then the enabled work-group
/// ids, then work-group info ({first_wavefront, .., waves in group});
/// v0..v2 the work-item ids. Waves run in turn up to each `s_barrier`,
/// which releases once every wave that has not ended reached it. Memory
/// is shared.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_compute(
    code: &[u32],
    io: &crate::translate::ComputeIo,
    user: &[u32],
    groups: [u32; 3],
    size: WaveSize,
    num_vgprs: usize,
    lds_words: usize,
    mem: &mut dyn Memory,
    max_steps: usize,
) -> Result<(), ExecError> {
    dispatch_compute_at(code, 0, io, user, groups, size, num_vgprs, lds_words, mem, max_steps)
}

/// [`dispatch_compute`] with the code at GPU address `code_base`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_compute_at(
    code: &[u32],
    code_base: u64,
    io: &crate::translate::ComputeIo,
    user: &[u32],
    groups: [u32; 3],
    size: WaveSize,
    num_vgprs: usize,
    lds_words: usize,
    mem: &mut dyn Memory,
    max_steps: usize,
) -> Result<(), ExecError> {
    let threads = io.threads() as usize;
    let lanes = size.lanes();
    if threads == 0 || num_vgprs < 3 {
        return Err(ExecError::Unsupported {
            at: 0,
            what: "empty work-group or fewer than 3 VGPRs".into(),
        });
    }
    let n_waves = threads.div_ceil(lanes);
    for gz in 0..groups[2] {
        for gy in 0..groups[1] {
            for gx in 0..groups[0] {
                let mut waves: Vec<Wave> = (0..n_waves)
                    .map(|wi| {
                        let mut w = Wave::new(size, num_vgprs);
                        w.code_base = code_base;
                        let here = (threads - wi * lanes).min(lanes);
                        w.exec = if here >= 64 { u64::MAX } else { (1u64 << here) - 1 };
                        let mut next = 0;
                        for i in 0..io.user_sgprs as usize {
                            w.sgpr[next] = user.get(i).copied().unwrap_or(0);
                            next += 1;
                        }
                        for (d, id) in [gx, gy, gz].into_iter().enumerate() {
                            if io.tgid[d] {
                                w.sgpr[next] = id;
                                next += 1;
                            }
                        }
                        if io.tg_size {
                            w.sgpr[next] = crate::translate::ComputeIo::workgroup_info(wi as u32, n_waves as u32);
                        }
                        let [lx, ly, _] = io.workgroup;
                        for lane in 0..here {
                            let li = (wi * lanes + lane) as u32;
                            let id = [li % lx, li / lx % ly, li / (lx * ly)];
                            let n = io.tidig_comp_cnt.min(2) as usize + 1;
                            for (reg, v) in w.vgpr.iter_mut().zip(id).take(n) {
                                reg[lane] = v;
                            }
                        }
                        w
                    })
                    .collect();
                let mut lds = vec![0u32; lds_words];
                let mut ended = vec![false; n_waves];
                let mut steps = 0;
                while ended.iter().any(|e| !e) {
                    for (w, done) in waves.iter_mut().zip(ended.iter_mut()) {
                        if *done {
                            continue;
                        }
                        w.lds = std::mem::take(&mut lds);
                        let r = run_until_barrier(code, w, mem, max_steps.saturating_sub(steps));
                        lds = std::mem::take(&mut w.lds);
                        let (n, end) = r?;
                        steps += n;
                        *done = end;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Run a pixel-shader wave: lane `i` is the pixel with `frag_coord[i]`
/// (x, y, z, 1/w as Vulkan's FragCoord); lanes past the slice start
/// inactive. Inputs follow SPI_PS_INPUT_ADDR/ENA (see
/// [`rdna::ps_input`]); only the POS_*_FLOAT inputs are modelled, other
/// enabled inputs are refused. User SGPRs from s0.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_pixels(
    code: &[u32],
    ena: u32,
    addr: u32,
    user: &[u32],
    frag_coord: &[[f32; 4]],
    size: WaveSize,
    num_vgprs: usize,
    max_steps: usize,
) -> Result<Wave, ExecError> {
    dispatch_pixels_interp(code, ena, addr, user, frag_coord, &[], &[], &[], size, num_vgprs, max_steps)
}

/// [`dispatch_pixels`] with PERSP_CENTER barycentrics (`bary[lane]` =
/// (i, j)), FRONT_FACE (`front[lane]`, true = front face; missing lanes
/// are front) and the primitive's attributes for `v_interp_*`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_pixels_interp(
    code: &[u32],
    ena: u32,
    addr: u32,
    user: &[u32],
    frag_coord: &[[f32; 4]],
    bary: &[[f32; 2]],
    front: &[bool],
    attrs: &[[[f32; 4]; 3]],
    size: WaveSize,
    num_vgprs: usize,
    max_steps: usize,
) -> Result<Wave, ExecError> {
    let pos_bits = 0xF << rdna::ps_input::POS_X_FLOAT
        | 1 << rdna::ps_input::PERSP_CENTER
        | 1 << rdna::ps_input::FRONT_FACE;
    if ena & !pos_bits != 0 || ena & !addr != 0 || frag_coord.len() > size.lanes() {
        return Err(ExecError::Unsupported {
            at: 0,
            what: "pixel inputs other than POS_*_FLOAT/PERSP_CENTER/FRONT_FACE, ENA outside ADDR, or too many pixels"
                .into(),
        });
    }
    let mut w = Wave::new(size, num_vgprs);
    let n = frag_coord.len();
    w.exec = if n >= 64 { u64::MAX } else { (1u64 << n) - 1 };
    w.sgpr[..user.len()].copy_from_slice(user);
    w.attrs = attrs
        .iter()
        .map(|a| a.map(|vtx| vtx.map(f32::to_bits)))
        .collect();
    if ena >> rdna::ps_input::PERSP_CENTER & 1 != 0 {
        let r = rdna::ps_input::vgpr(addr, rdna::ps_input::PERSP_CENTER).unwrap() as usize;
        for (lane, ij) in bary.iter().enumerate().take(n) {
            for (k, x) in ij.iter().enumerate() {
                w.vgpr
                    .get_mut(r + k)
                    .ok_or(ExecError::Unsupported { at: 0, what: "input VGPR beyond num_vgprs".into() })?[lane] =
                    x.to_bits();
            }
        }
    }
    for (k, bit) in (rdna::ps_input::POS_X_FLOAT..=rdna::ps_input::POS_W_FLOAT).enumerate() {
        if ena >> bit & 1 == 0 {
            continue;
        }
        let r = rdna::ps_input::vgpr(addr, bit).unwrap() as usize;
        for (lane, fc) in frag_coord.iter().enumerate() {
            w.vgpr
                .get_mut(r)
                .ok_or(ExecError::Unsupported { at: 0, what: "input VGPR beyond num_vgprs".into() })?[lane] =
                fc[k].to_bits();
        }
    }
    if ena >> rdna::ps_input::FRONT_FACE & 1 != 0 {
        let r = rdna::ps_input::vgpr(addr, rdna::ps_input::FRONT_FACE).unwrap() as usize;
        let reg = w
            .vgpr
            .get_mut(r)
            .ok_or(ExecError::Unsupported { at: 0, what: "input VGPR beyond num_vgprs".into() })?;
        for (lane, x) in reg.iter_mut().enumerate().take(n) {
            *x = if front.get(lane).copied().unwrap_or(true) { 1.0f32 } else { -1.0 }.to_bits();
        }
    }
    run(code, &mut w, max_steps)?;
    Ok(w)
}

/// One vertex for [`dispatch_vertices`]: zero-based vertex id, primitive
/// id and zero-based instance id.
#[derive(Clone, Copy, Debug)]
pub struct VertexIn {
    pub vertex_id: u32,
    pub prim_id: u32,
    pub instance_id: u32,
}

/// Run a legacy vertex-shader wave: lane `i` is `verts[i]`, inputs per
/// [`rdna::vs_input`] up to `comp_cnt`, user SGPRs from s0. Exports are
/// collected in the wave.
pub fn dispatch_vertices(
    code: &[u32],
    comp_cnt: u32,
    user: &[u32],
    verts: &[VertexIn],
    size: WaveSize,
    num_vgprs: usize,
    max_steps: usize,
) -> Result<Wave, ExecError> {
    if verts.len() > size.lanes() || num_vgprs < 4 {
        return Err(ExecError::Unsupported { at: 0, what: "too many vertices or fewer than 4 VGPRs".into() });
    }
    let mut w = Wave::new(size, num_vgprs);
    let n = verts.len();
    w.exec = if n >= 64 { u64::MAX } else { (1u64 << n) - 1 };
    w.sgpr[..user.len()].copy_from_slice(user);
    for (lane, v) in verts.iter().enumerate() {
        let vals = [v.vertex_id, 0, v.prim_id, v.instance_id];
        for (r, x) in vals.iter().enumerate().take(comp_cnt as usize + 1) {
            w.vgpr[r][lane] = *x;
        }
    }
    run(code, &mut w, max_steps)?;
    Ok(w)
}
