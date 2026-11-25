//! RDNA2 → SPIR-V translation.
//!
//! # Execution model
//!
//! One shader invocation per lane, one subgroup per wave (the subgroup
//! size must equal the wave size; wave64 on 32-wide hardware needs the
//! emulation stage that comes later).
//!
//! RDNA2 control flow is scalar: branches test `scc`, `vccz` or `execz`,
//! which are the same for the whole wave. Divergence is expressed with the
//! `exec` mask, not with branches. The translation keeps it that way:
//!
//! * every scalar register (SGPRs, `vcc`, `exec`, `m0`, `scc`) is a
//!   `Private` variable that holds the same value in every invocation;
//! * every VGPR is a `Private` variable of its lane;
//! * a VALU write stores `active ? new : old`, where `active` is the lane's
//!   bit of `exec`, so inactive lanes keep their values exactly as on the
//!   hardware;
//! * a VOPC result is `OpGroupNonUniformBallot(active && cond)`, giving the
//!   whole wave's mask in every invocation;
//! * the shader's basic blocks become arms of one dispatcher loop over a
//!   uniform block index, so SPIR-V control flow is structured and uniform
//!   for any RDNA2 control-flow graph, loops and gotos included.
//!
//! The dispatcher is simple rather than fast; structurizing common shapes
//! into real `if`/`loop` constructs is an optimisation for later, checked
//! against this version.
//!
//! # I/O model
//!
//! [`Io::RegisterDump`] (tests and the differential check against
//! [`crate::interp`]): descriptor set 0, binding 0 is a storage buffer of
//! u32 words, read at the start and written back at `s_endpgm`:
//!
//! ```text
//! [0 .. num_sgprs)                      s0..
//! [num_sgprs + SPECIAL_*]               exec_lo, exec_hi, vcc_lo, vcc_hi, m0, scc
//! [vgpr_base + lane * num_vgprs + n]    VGPR n of `lane`
//! ```
//!
//! with `vgpr_base = num_sgprs + SPECIALS`, and the lane index is
//! `GlobalInvocationId.x` (one workgroup of `lanes` invocations). The real
//! pipeline contract replaces this I/O model later; the register lowering
//! stays the same.

use crate::rdna::{
    decode_shader, ds, global, mimg, mubuf, operand, smem, sop1, sop2, sopc, sopp, vop1, vop2, vop3, DecodeError, Fields,
    Format, Inst, Operand,
};
use crate::spirv::{
    addressing, builtin, capability, decoration, execution_mode, execution_model,
    glsl, memory_model, op, scope, semantics, storage, Builder, Id,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Io {
    RegisterDump,
    /// The hardware compute contract (see [`ComputeIo`]); results leave
    /// only through memory stores.
    Compute(ComputeIo),
    /// A pixel shader (see [`PixelIo`]).
    Pixel(PixelIo),
    /// A legacy (non-NGG) vertex shader (see [`VertexIo`]).
    Vertex(VertexIo),
}

/// Vertex-shader contract, legacy VS (see `rdna::vs_input`): v0 =
/// VertexIndex - BaseVertex, v3 = InstanceIndex - BaseInstance (both
/// zero-based like the hardware's), v2 (primitive id) reads 0 because a
/// Vulkan vertex shader has none, user SGPRs from push constants;
/// `exp pos0` → Position, `exp param N` → vec4 output at Location N.
/// NGG (merged ES/GS) vertex shaders have another layout and are not
/// covered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexIo {
    pub user_sgprs: u32,
    /// SPI_SHADER_PGM_RSRC1_VS.VGPR_COMP_CNT.
    pub vgpr_comp_cnt: u32,
}

/// Pixel-shader contract, first stage: SPI_PS_INPUT_ENA/ADDR with only the
/// POS_*_FLOAT inputs (from FragCoord; Mesa maps frag_pos[0..3] to
/// FragCoord x, y, z and 1/w, which is FragCoord.w in Vulkan), user SGPRs
/// from push constants, `exp mrt0..7` to vec4 outputs at Location 0..7.
/// A pixel that never exports while active is discarded (OpKill), as the
/// hardware writes nothing for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelIo {
    pub ena: u32,
    pub addr: u32,
    pub user_sgprs: u32,
}

/// Initial wave state of a GFX10 compute dispatch, as COMPUTE_PGM_RSRC2 and
/// COMPUTE_NUM_THREAD_* describe it. Order from LLVM's AMDGPUUsage
/// "Initial Kernel Execution State": the user SGPRs from s0, then the
/// enabled work-group ids X, Y, Z, then the work-group info SGPR; VGPRs v0,
/// v1, v2 hold the work-item id x, y, z (unpacked on GFX10; Mesa's
/// `local_invocation_ids_packed` is GFX11+).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComputeIo {
    /// RSRC2.USER_SGPR: user data dwords, read from push constants
    /// (a block of u32[16] at offset 0).
    pub user_sgprs: u32,
    /// RSRC2.TGID_{X,Y,Z}_EN.
    pub tgid: [bool; 3],
    /// RSRC2.TG_SIZE_EN: the work-group info SGPR.
    pub tg_size: bool,
    /// RSRC2.TIDIG_COMP_CNT: 0 = x only, 1 = x, y, 2 = x, y, z.
    pub tidig_comp_cnt: u32,
    /// COMPUTE_NUM_THREAD_{X,Y,Z}. One wave per work-group for now: the
    /// product must not exceed the wave size; lanes past it start inactive.
    pub workgroup: [u32; 3],
}

impl ComputeIo {
    /// From the COMPUTE_PGM_RSRC2 register value (field layout from Mesa's
    /// `gfx10.json`) and the work-group size.
    pub fn from_rsrc2(rsrc2: u32, workgroup: [u32; 3]) -> Self {
        let bit = |n: u32| rsrc2 >> n & 1 != 0;
        ComputeIo {
            user_sgprs: (rsrc2 >> 1) & 0x1F,
            tgid: [bit(7), bit(8), bit(9)],
            tg_size: bit(10),
            tidig_comp_cnt: (rsrc2 >> 11) & 3,
            workgroup,
        }
    }

    pub fn threads(&self) -> u32 {
        self.workgroup.iter().product()
    }

    /// Work-group info SGPR for a work-group of one wave:
    /// {first_wavefront, 14'b0, ordered_append_term[10:0],
    /// threadgroup_size_in_wavefronts[5:0]}.
    pub const WORKGROUP_INFO_ONE_WAVE: u32 = 1 << 31 | 1;

    /// Work-group info SGPR of wave `wave` in a group of `waves`.
    pub fn workgroup_info(wave: u32, waves: u32) -> u32 {
        ((wave == 0) as u32) << 31 | (waves & 0x3F)
    }
}

/// Push-constant block of [`Io::Compute`]: 16 user SGPRs.
pub const MAX_USER_SGPRS: u32 = 16;

#[derive(Clone, Debug)]
pub struct Options {
    pub io: Io,
    pub num_sgprs: u32,
    pub num_vgprs: u32,
    /// Wave size: 32 or 64. Also the workgroup size.
    pub lanes: u32,
    /// GPU memory visible to SMEM loads, if any.
    pub memory: Option<Window>,
    /// How cross-lane operations (VOPC masks, readfirstlane) are built.
    pub cross_lane: CrossLane,
    /// LDS allocation in dwords (0: the shader may not use LDS). It
    /// becomes a Workgroup array with one extra sink dword per lane, where
    /// inactive and out-of-range lanes store, so stores need no per-lane
    /// branch and no two lanes ever share a sink.
    pub lds_words: u32,
    /// GPU address of the code's first dword, for `s_getpc_b64`.
    pub code_base: u64,
}

/// Cross-lane strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CrossLane {
    /// Subgroup operations; needs subgroup size == wave size.
    #[default]
    Subgroup,
    /// Workgroup memory and barriers: exact on any subgroup size (wave64
    /// on 32-wide hardware such as Apple GPUs or NVIDIA), slower. Each
    /// lane writes its value to a shared array, a barrier publishes it,
    /// every lane reads what it needs, and a second barrier keeps the next
    /// write from racing those reads.
    Workgroup,
}

/// A window of GPU virtual memory bound as descriptor set 0, binding 1:
/// a storage buffer of `words + 1` u32 words, the first `words` holding
/// `[base, base + 4*words)`. The extra last word is a sink: stores from
/// inactive lanes and stores outside the window go there, which keeps
/// control flow uniform (no per-lane branch around a store).
///
/// A load outside the window reads 0 and a store outside it is dropped. That is a choice of this stage, not
/// hardware behaviour (an unmapped address faults on the console); it
/// keeps the SPIR-V free of out-of-bounds accesses, which are undefined.
#[derive(Clone, Copy, Debug)]
pub struct Window {
    pub base: u64,
    pub words: u32,
}

/// Offsets of the special registers after the SGPRs in the dump.
pub const SPECIAL_EXEC_LO: u32 = 0;
pub const SPECIAL_EXEC_HI: u32 = 1;
pub const SPECIAL_VCC_LO: u32 = 2;
pub const SPECIAL_VCC_HI: u32 = 3;
pub const SPECIAL_M0: u32 = 4;
pub const SPECIAL_SCC: u32 = 5;
pub const SPECIALS: u32 = 6;

#[derive(Debug, PartialEq)]
pub enum TranslateError {
    Decode(DecodeError),
    Unsupported { at: usize, what: String },
    BadOptions(&'static str),
}

impl From<DecodeError> for TranslateError {
    fn from(e: DecodeError) -> Self {
        TranslateError::Decode(e)
    }
}

type Res<T> = Result<T, TranslateError>;

struct Types {
    u32t: Id,
    i32t: Id,
    f32t: Id,
    boolt: Id,
    uvec4: Id,
}

/// A 64-bit scalar value as two u32 ids.
#[derive(Clone, Copy)]
struct Pair {
    lo: Id,
    hi: Id,
}

struct Tr {
    code_base: u64,
    b: Builder,
    t: Types,
    wave64: bool,
    sgpr: Vec<Id>,
    vgpr: Vec<Id>,
    exec_lo: Id,
    exec_hi: Id,
    vcc_lo: Id,
    vcc_hi: Id,
    m0: Id,
    scc: Id,
    /// Per-invocation lane index and derived values, computed once.
    lane: Id,
    lane_is_hi: Id,
    lane_shift: Id,
    subgroup_scope: Id,
    /// (buffer variable, pointer-to-u32 type, window) for SMEM.
    mem: Option<(Id, Id, Window)>,
    /// Workgroup emulation: (shared u32[64] variable, pointer type).
    xlane: Option<(Id, Id)>,
    /// LDS: (Workgroup u32[words + 64] variable, pointer type, words).
    lds: Option<(Id, Id, u32)>,
    /// Multi-wave work-group: this wave's "waiting at s_barrier" flag.
    waiting: Option<Id>,
    /// While translating a DPP16 instruction: the permuted src0 value, the
    /// modified src1 (operand encoding, value) and the lane write enable.
    dpp_src0: Option<Id>,
    dpp_src1: Option<(u16, Id)>,
    dpp_enable: Option<Id>,
    /// UNORM8 → float table (Private u32[256], pointer type), when the
    /// shader reads images.
    unorm_lut: Option<(Id, Id)>,
    /// The GLSL.std.450 import, created on first use.
    glsl: Option<Id>,
    /// Pixel shader: (MRT output variables by target, "exported" flag).
    pixel: Option<([Option<Id>; 8], Id)>,
    /// Vertex shader: export target → output variable.
    vertex_outs: Option<Vec<(u8, Id)>>,
    /// Pixel shader attributes: per attribute, its Input vec4[3]
    /// (PerVertexKHR) variable, and the pointer type to one vec4.
    attr_vars: Vec<(u8, Id)>,
    p_v4_in: Id,
    // where we are, for errors
    at: usize,
    text: String,
}

const VCC_LO_ENC: u16 = 106;

impl Tr {
    fn err(&self, why: &str) -> TranslateError {
        TranslateError::Unsupported {
            at: self.at,
            what: format!("{} ({why})", self.text),
        }
    }

    fn c(&mut self, v: u32) -> Id {
        self.b.constant_u32(v)
    }

    fn u(&mut self, opcode: u16, ops: &[Id]) -> Id {
        let t = self.t.u32t;
        self.b.op(t, opcode, ops)
    }

    fn bl(&mut self, opcode: u16, ops: &[Id]) -> Id {
        let t = self.t.boolt;
        self.b.op(t, opcode, ops)
    }

    fn load_u(&mut self, var: Id) -> Id {
        let t = self.t.u32t;
        self.b.load(t, var)
    }

    fn select_u(&mut self, cond: Id, a: Id, b: Id) -> Id {
        self.u(op::SELECT, &[cond, a, b])
    }

    fn select_b(&mut self, cond: Id, a: Id, b: Id) -> Id {
        self.bl(op::SELECT, &[cond, a, b])
    }

    fn bool_to_u(&mut self, x: Id) -> Id {
        let (one, zero) = (self.c(1), self.c(0));
        self.select_u(x, one, zero)
    }

    fn nonzero(&mut self, x: Id) -> Id {
        let z = self.c(0);
        self.bl(op::I_NOT_EQUAL, &[x, z])
    }

    /// Variable for a 32-bit scalar register operand, if it is one.
    fn sreg(&self, o: Operand) -> Res<Option<Id>> {
        Ok(Some(match o {
            Operand::Sgpr(n) => *self
                .sgpr
                .get(n as usize)
                .ok_or_else(|| self.err("SGPR beyond num_sgprs"))?,
            Operand::VccLo => self.vcc_lo,
            Operand::VccHi => self.vcc_hi,
            Operand::ExecLo => self.exec_lo,
            Operand::ExecHi => self.exec_hi,
            Operand::M0 => self.m0,
            _ => return Ok(None),
        }))
    }

    /// A 32-bit scalar source (also used for VALU sources that are not VGPRs).
    fn s32(&mut self, v: u16, lit: u32) -> Res<Id> {
        let o = operand(v, lit);
        if let Some(var) = self.sreg(o)? {
            return Ok(self.load_u(var));
        }
        Ok(match o {
            Operand::Int(i) => self.c(i as u32),
            Operand::Float(f) => self.c(f.to_bits()),
            Operand::InvTwoPi => self.c(0x3E22_F983),
            Operand::Literal(x) => self.c(x),
            Operand::Null => self.c(0),
            Operand::Scc => {
                let s = self.b.load(self.t.boolt, self.scc);
                self.bool_to_u(s)
            }
            Operand::Vccz => {
                let z = self.mask_is_zero(self.vcc_lo, self.vcc_hi);
                self.bool_to_u(z)
            }
            Operand::Execz => {
                let z = self.mask_is_zero(self.exec_lo, self.exec_hi);
                self.bool_to_u(z)
            }
            other => return Err(self.err(&format!("scalar source {other}"))),
        })
    }

    fn set_s32(&mut self, d: u8, value: Id) -> Res<()> {
        let o = operand(d as u16, 0);
        if o == Operand::Null {
            return Ok(());
        }
        let var = self
            .sreg(o)?
            .ok_or_else(|| self.err(&format!("scalar destination {o}")))?;
        self.b.store(var, value);
        Ok(())
    }

    /// Register pair for a 64-bit scalar operand.
    fn pair_vars(&self, o: Operand) -> Res<Option<(Id, Id)>> {
        Ok(Some(match o {
            Operand::Sgpr(n) => {
                let lo = self.sgpr.get(n as usize);
                let hi = self.sgpr.get(n as usize + 1);
                match (lo, hi) {
                    (Some(&lo), Some(&hi)) => (lo, hi),
                    _ => return Err(self.err("SGPR pair beyond num_sgprs")),
                }
            }
            Operand::VccLo => (self.vcc_lo, self.vcc_hi),
            Operand::ExecLo => (self.exec_lo, self.exec_hi),
            _ => return Ok(None),
        }))
    }

    fn s64(&mut self, v: u16, lit: u32) -> Res<Pair> {
        let o = operand(v, lit);
        if let Some((lo, hi)) = self.pair_vars(o)? {
            return Ok(Pair {
                lo: self.load_u(lo),
                hi: self.load_u(hi),
            });
        }
        let bits: u64 = match o {
            Operand::Int(i) => i as i64 as u64,
            Operand::Float(f) => (f as f64).to_bits(),
            Operand::Null => 0,
            other => return Err(self.err(&format!("64-bit scalar source {other}"))),
        };
        Ok(Pair {
            lo: self.c(bits as u32),
            hi: self.c((bits >> 32) as u32),
        })
    }

    fn set_s64(&mut self, d: u8, v: Pair) -> Res<()> {
        let o = operand(d as u16, 0);
        if o == Operand::Null {
            return Ok(());
        }
        let (lo, hi) = self
            .pair_vars(o)?
            .ok_or_else(|| self.err(&format!("64-bit scalar destination {o}")))?;
        self.b.store(lo, v.lo);
        self.b.store(hi, v.hi);
        Ok(())
    }

    /// Lane mask from a scalar source: one SGPR in wave32, a pair in wave64.
    fn mask(&mut self, v: u16, lit: u32) -> Res<Pair> {
        if self.wave64 {
            self.s64(v, lit)
        } else {
            let lo = self.s32(v, lit)?;
            Ok(Pair { lo, hi: self.c(0) })
        }
    }

    fn set_mask(&mut self, d: u8, v: Pair) -> Res<()> {
        if self.wave64 {
            self.set_s64(d, v)
        } else {
            self.set_s32(d, v.lo)
        }
    }

    fn mask_is_zero(&mut self, lo_var: Id, hi_var: Id) -> Id {
        let lo = self.load_u(lo_var);
        let m = if self.wave64 {
            let hi = self.load_u(hi_var);
            self.u(op::BITWISE_OR, &[lo, hi])
        } else {
            lo
        };
        let z = self.c(0);
        self.bl(op::I_EQUAL, &[m, z])
    }

    /// This lane's bit of a mask.
    fn lane_bit(&mut self, m: Pair) -> Id {
        let word = self.select_u(self.lane_is_hi, m.hi, m.lo);
        let shifted = self.u(op::SHIFT_RIGHT_LOGICAL, &[word, self.lane_shift]);
        let one = self.c(1);
        let bit = self.u(op::BITWISE_AND, &[shifted, one]);
        self.nonzero(bit)
    }

    fn active(&mut self) -> Id {
        let m = Pair {
            lo: self.load_u(self.exec_lo),
            hi: self.load_u(self.exec_hi),
        };
        self.lane_bit(m)
    }

    fn barrier(&mut self) {
        let wg = self.c(scope::WORKGROUP);
        let sem = self.c(semantics::ACQUIRE_RELEASE | semantics::WORKGROUP_MEMORY);
        self.b.op_void(op::CONTROL_BARRIER, &[wg, wg, sem]);
    }

    /// Order this wave's workgroup-memory accesses: a subgroup-scope
    /// barrier when the work-group has several waves (control flow is then
    /// only uniform per wave), a workgroup barrier otherwise.
    fn wave_barrier(&mut self) {
        if self.waiting.is_some() {
            let sg = self.c(scope::SUBGROUP);
            let wg = self.c(scope::WORKGROUP);
            let sem = self.c(semantics::ACQUIRE_RELEASE | semantics::WORKGROUP_MEMORY);
            self.b.op_void(op::CONTROL_BARRIER, &[sg, wg, sem]);
        } else {
            self.barrier();
        }
    }

    /// Publish `value` for this lane in the shared array (Workgroup mode).
    fn publish(&mut self, value: Id) -> (Id, Id) {
        let (arr, ptr) = self.xlane.expect("workgroup mode");
        let p = self.b.op(ptr, op::ACCESS_CHAIN, &[arr, self.lane]);
        self.b.store(p, value);
        self.barrier();
        (arr, ptr)
    }

    fn ballot(&mut self, pred: Id) -> Pair {
        if self.xlane.is_some() {
            let bit = self.bool_to_u(pred);
            let (arr, ptr) = self.publish(bit);
            let lanes = if self.wave64 { 64 } else { 32 };
            let mut words = [self.c(0), self.c(0)];
            for k in 0..lanes {
                let kc = self.c(k);
                let p = self.b.op(ptr, op::ACCESS_CHAIN, &[arr, kc]);
                let v = self.load_u(p);
                let sh = self.c(k % 32);
                let b = self.u(op::SHIFT_LEFT_LOGICAL, &[v, sh]);
                let w = &mut words[(k / 32) as usize];
                *w = self.u(op::BITWISE_OR, &[*w, b]);
            }
            self.barrier();
            return Pair {
                lo: words[0],
                hi: words[1],
            };
        }
        let t = self.t.uvec4;
        let b = self
            .b
            .op(t, op::GROUP_NON_UNIFORM_BALLOT, &[self.subgroup_scope, pred]);
        let lo = self.u(op::COMPOSITE_EXTRACT, &[b, 0]);
        let hi = if self.wave64 {
            self.u(op::COMPOSITE_EXTRACT, &[b, 1])
        } else {
            self.c(0)
        };
        Pair { lo, hi }
    }

    /// 32-bit VALU source for this lane.
    fn vsrc(&mut self, v: u16, lit: u32) -> Res<Id> {
        if v == crate::rdna::SRC_DPP16 {
            return self.dpp_src0.ok_or_else(|| self.err("DPP operand outside a DPP instruction"));
        }
        if let Some((enc, val)) = self.dpp_src1 {
            if enc == v {
                return Ok(val);
            }
        }
        match operand(v, lit) {
            Operand::Vgpr(n) => {
                let var = *self
                    .vgpr
                    .get(n as usize)
                    .ok_or_else(|| self.err("VGPR beyond num_vgprs"))?;
                Ok(self.load_u(var))
            }
            _ => self.s32(v, lit),
        }
    }

    /// Predicated VGPR write: inactive lanes keep their value.
    fn set_vgpr(&mut self, n: u8, value: Id) -> Res<()> {
        let var = *self
            .vgpr
            .get(n as usize)
            .ok_or_else(|| self.err("VGPR beyond num_vgprs"))?;
        let old = self.load_u(var);
        let mut act = self.active();
        if let Some(e) = self.dpp_enable {
            act = self.bl(op::LOGICAL_AND, &[act, e]);
        }
        let v = self.select_u(act, value, old);
        self.b.store(var, v);
        Ok(())
    }

    /// This lane's bit of exec at lane index `l` (a value < wave size).
    fn exec_bit_at(&mut self, l: Id) -> Id {
        let (k31, k32, one) = (self.c(31), self.c(32), self.c(1));
        let hi = self.bl(op::U_GREATER_THAN_EQUAL, &[l, k32]);
        let (lo_w, hi_w) = (self.load_u(self.exec_lo), self.load_u(self.exec_hi));
        let word = self.select_u(hi, hi_w, lo_w);
        let sh = self.u(op::BITWISE_AND, &[l, k31]);
        let x = self.u(op::SHIFT_RIGHT_LOGICAL, &[word, sh]);
        let b = self.u(op::BITWISE_AND, &[x, one]);
        self.nonzero(b)
    }

    /// Value of `v` (this lane's id) in lane `idx` of the wave.
    fn lane_value(&mut self, v: Id, idx: Id) -> Id {
        if self.xlane.is_some() {
            let (arr, ptr) = self.publish(v);
            let p = self.b.op(ptr, op::ACCESS_CHAIN, &[arr, idx]);
            let r = self.load_u(p);
            self.barrier();
            r
        } else {
            let u = self.t.u32t;
            self.b
                .op(u, op::GROUP_NON_UNIFORM_SHUFFLE, &[self.subgroup_scope, v, idx])
        }
    }

    /// Translate a VOP1/VOP2 whose src0 is DPP16 (see `interp::dpp_step`
    /// for the semantics both follow).
    fn dpp(&mut self, inst: &Inst) -> Res<()> {
        let d = crate::rdna::Dpp16::parse(*inst.words.get(1).ok_or_else(|| self.err("missing DPP word"))?);
        let rule = d.rule().ok_or_else(|| self.err("DPP control"))?;
        if d.fi {
            return Err(self.err("DPP fetch-inactive"));
        }
        let float_mods = matches!(inst.fields, Fields::Vop2 { .. })
            && matches!(
                inst.opcode,
                vop2::V_ADD_F32 | vop2::V_SUB_F32 | vop2::V_SUBREV_F32 | vop2::V_MUL_F32
            );
        if (d.src0_neg || d.src0_abs || d.src1_neg || d.src1_abs) && !float_mods {
            return Err(self.err("DPP source modifiers on this opcode"));
        }
        let lane = self.lane;
        let c = |tr: &mut Self, v: u32| tr.c(v);
        let (k15, k16) = (c(self, 15), c(self, 16));
        let r = self.u(op::BITWISE_AND, &[lane, k15]);
        let not15 = c(self, !15);
        let row = self.u(op::BITWISE_AND, &[lane, not15]);
        let t = self.b.constant_bool(true);
        let (src, in_row) = match rule {
            crate::rdna::DppRule::QuadPerm(sel) => {
                let (k3, not3, two) = (c(self, 3), c(self, !3), c(self, 2));
                let q = self.u(op::BITWISE_AND, &[lane, k3]);
                let sh = self.u(op::I_MUL, &[q, two]);
                let selc = c(self, sel as u32);
                let x = self.u(op::SHIFT_RIGHT_LOGICAL, &[selc, sh]);
                let x = self.u(op::BITWISE_AND, &[x, k3]);
                let base = self.u(op::BITWISE_AND, &[lane, not3]);
                (self.u(op::BITWISE_OR, &[base, x]), t)
            }
            crate::rdna::DppRule::RowShl(n) => {
                let nc = c(self, n as u32);
                let rn = self.u(op::I_ADD, &[r, nc]);
                let ok = self.bl(op::U_LESS_THAN, &[rn, k16]);
                (self.u(op::I_ADD, &[lane, nc]), ok)
            }
            crate::rdna::DppRule::RowShr(n) => {
                let nc = c(self, n as u32);
                let ok = self.bl(op::U_GREATER_THAN_EQUAL, &[r, nc]);
                (self.u(op::I_SUB, &[lane, nc]), ok)
            }
            crate::rdna::DppRule::RowRor(n) => {
                let k = c(self, 16 - n as u32);
                let x = self.u(op::I_ADD, &[r, k]);
                let x = self.u(op::BITWISE_AND, &[x, k15]);
                (self.u(op::BITWISE_OR, &[row, x]), t)
            }
            crate::rdna::DppRule::RowMirror => {
                let x = self.u(op::I_SUB, &[k15, r]);
                (self.u(op::BITWISE_OR, &[row, x]), t)
            }
            crate::rdna::DppRule::RowHalfMirror => {
                let (k7, not7) = (c(self, 7), c(self, !7));
                let h = self.u(op::BITWISE_AND, &[lane, k7]);
                let x = self.u(op::I_SUB, &[k7, h]);
                let base = self.u(op::BITWISE_AND, &[lane, not7]);
                (self.u(op::BITWISE_OR, &[base, x]), t)
            }
            crate::rdna::DppRule::RowShare(n) => {
                let nc = c(self, n as u32);
                (self.u(op::BITWISE_OR, &[row, nc]), t)
            }
            crate::rdna::DppRule::RowXmask(n) => {
                let nc = c(self, n as u32);
                let x = self.u(op::BITWISE_XOR, &[r, nc]);
                (self.u(op::BITWISE_OR, &[row, x]), t)
            }
        };
        // Out-of-row sources read lane itself (never used) so the shuffle
        // index stays in range.
        let safe = self.select_u(in_row, src, lane);
        let v = self.vsrc(256 + d.src0 as u16, 0)?;
        let moved = self.lane_value(v, safe);
        let src_active = self.exec_bit_at(safe);
        let valid = self.bl(op::LOGICAL_AND, &[in_row, src_active]);
        let zero = self.c(0);
        let mut val = self.select_u(valid, moved, zero);
        let sign_ops = |tr: &mut Self, x: Id, abs: bool, neg: bool| {
            let mut x = x;
            if abs {
                let m = tr.c(0x7FFF_FFFF);
                x = tr.u(op::BITWISE_AND, &[x, m]);
            }
            if neg {
                let m = tr.c(0x8000_0000);
                x = tr.u(op::BITWISE_XOR, &[x, m]);
            }
            x
        };
        val = sign_ops(self, val, d.src0_abs, d.src0_neg);
        // Lane enable: row and bank masks, and a valid source or bound_ctrl.
        let (four, k3) = (self.c(4), self.c(3));
        let row_idx = self.u(op::SHIFT_RIGHT_LOGICAL, &[lane, four]);
        let two = self.c(2);
        let bank_idx = self.u(op::SHIFT_RIGHT_LOGICAL, &[r, two]);
        let bank_idx = self.u(op::BITWISE_AND, &[bank_idx, k3]);
        let (rm, bm, one) = (self.c(d.row_mask as u32), self.c(d.bank_mask as u32), self.c(1));
        let rbit = self.u(op::SHIFT_RIGHT_LOGICAL, &[rm, row_idx]);
        let rbit = self.u(op::BITWISE_AND, &[rbit, one]);
        let bbit = self.u(op::SHIFT_RIGHT_LOGICAL, &[bm, bank_idx]);
        let bbit = self.u(op::BITWISE_AND, &[bbit, one]);
        let rb = self.u(op::BITWISE_AND, &[rbit, bbit]);
        let masks = self.nonzero(rb);
        let source_ok = if d.bound_ctrl { t } else { valid };
        let enable = self.bl(op::LOGICAL_AND, &[masks, source_ok]);
        self.dpp_src0 = Some(val);
        self.dpp_enable = Some(enable);
        if let Fields::Vop2 { vsrc1, .. } = inst.fields {
            if d.src1_abs || d.src1_neg {
                let enc = 256 + vsrc1 as u16;
                let x = self.vsrc(enc, 0)?;
                let x = sign_ops(self, x, d.src1_abs, d.src1_neg);
                self.dpp_src1 = Some((enc, x));
            }
        }
        let mut plain = inst.clone();
        plain.words.truncate(1);
        let r = self.inst(&plain);
        self.dpp_src0 = None;
        self.dpp_src1 = None;
        self.dpp_enable = None;
        r
    }

    fn as_f32(&mut self, x: Id) -> Id {
        let f = self.t.f32t;
        self.b.op(f, op::BITCAST, &[x])
    }

    fn as_i32(&mut self, x: Id) -> Id {
        let i = self.t.i32t;
        self.b.op(i, op::BITCAST, &[x])
    }

    fn bits_of(&mut self, x: Id) -> Id {
        self.u(op::BITCAST, &[x])
    }

    fn fbin(&mut self, (opcode, swap): (u16, bool), a: Id, b: Id, neg: u8) -> Id {
        let (mut fa, mut fb) = (self.as_f32(a), self.as_f32(b));
        let f = self.t.f32t;
        if neg & 1 != 0 {
            fa = self.b.op(f, op::F_NEGATE, &[fa]);
        }
        if neg & 2 != 0 {
            fb = self.b.op(f, op::F_NEGATE, &[fb]);
        }
        let r = if swap {
            self.b.op(f, opcode, &[fb, fa])
        } else {
            self.b.op(f, opcode, &[fa, fb])
        };
        self.bits_of(r)
    }

    /// SPIR-V op for a float VOP2 opcode, and whether operands swap
    /// (after source modifiers) as for `v_subrev`.
    fn vop2_float(op2: u16) -> Option<(u16, bool)> {
        Some(match op2 {
            vop2::V_ADD_F32 => (op::F_ADD, false),
            vop2::V_SUB_F32 => (op::F_SUB, false),
            vop2::V_SUBREV_F32 => (op::F_SUB, true),
            vop2::V_MUL_F32 => (op::F_MUL, false),
            _ => return None,
        })
    }

    fn glsl_set(&mut self) -> Id {
        match self.glsl {
            Some(id) => id,
            None => {
                let id = self.b.ext_inst_import(glsl::NAME);
                self.glsl = Some(id);
                id
            }
        }
    }

    /// v_min/max_f32 with `interp::min_max_f32`'s rules (quiet NaN → the
    /// other operand, -0 < +0); cases the interpreter refuses get some
    /// value.
    fn min_max(&mut self, max: bool, a: Id, b: Id) -> Id {
        let (fa, fb) = (self.as_f32(a), self.as_f32(b));
        let a_num = self.bl(op::F_ORD_EQUAL, &[fa, fa]);
        let b_num = self.bl(op::F_ORD_EQUAL, &[fb, fb]);
        let eq = self.bl(op::F_ORD_EQUAL, &[fa, fb]);
        let lt = self.bl(op::F_ORD_LESS_THAN, &[fa, fb]);
        let pick = if max { self.select_u(lt, b, a) } else { self.select_u(lt, a, b) };
        let tie = if max {
            self.u(op::BITWISE_AND, &[a, b])
        } else {
            self.u(op::BITWISE_OR, &[a, b])
        };
        let r0 = self.select_u(eq, tie, pick);
        let r1 = self.select_u(b_num, r0, a);
        self.select_u(a_num, r1, b)
    }

    fn fma(&mut self, a: Id, b: Id, c: Id) -> Id {
        let set = self.glsl_set();
        let f = self.t.f32t;
        let (x, y, z) = (self.as_f32(a), self.as_f32(b), self.as_f32(c));
        let r = self.b.op(f, op::EXT_INST, &[set, glsl::FMA, x, y, z]);
        self.bits_of(r)
    }

    /// The 64-bit float type (declares the Float64 capability on first
    /// use, so shaders without f64 do not need the device feature).
    fn f64t(&mut self) -> Id {
        self.b.capability(crate::spirv::capability::FLOAT64);
        self.b.type_float(64)
    }

    fn uvec2t(&mut self) -> Id {
        self.b.type_vector(self.t.u32t, 2)
    }

    /// A register pair as a double (OpBitcast of uvec2: component 0 is
    /// the low word).
    fn as_f64(&mut self, p: Pair) -> Id {
        let v2 = self.uvec2t();
        let v = self.b.op(v2, op::COMPOSITE_CONSTRUCT, &[p.lo, p.hi]);
        let d = self.f64t();
        self.b.op(d, op::BITCAST, &[v])
    }

    fn f64_bits(&mut self, x: Id) -> Pair {
        let v2 = self.uvec2t();
        let v = self.b.op(v2, op::BITCAST, &[x]);
        Pair { lo: self.u(op::COMPOSITE_EXTRACT, &[v, 0]), hi: self.u(op::COMPOSITE_EXTRACT, &[v, 1]) }
    }

    /// A 64-bit VALU source (see `interp::src64`): VGPR pair or 64-bit
    /// scalar source; a literal is refused.
    fn src64(&mut self, v: u16, lit: u32) -> Res<Pair> {
        if v >= 256 {
            return Ok(Pair { lo: self.vsrc(v, lit)?, hi: self.vsrc(v + 1, lit)? });
        }
        if v == crate::rdna::SRC_LITERAL {
            return Err(self.err("literal f64 operand"));
        }
        self.s64(v, lit)
    }

    fn and_b(&mut self, a: Id, b: Id) -> Id {
        self.bl(op::LOGICAL_AND, &[a, b])
    }

    fn not_b(&mut self, a: Id) -> Id {
        self.bl(op::LOGICAL_NOT, &[a])
    }

    /// a < b as unsigned 64-bit pairs.
    fn ult64(&mut self, a: Pair, b: Pair) -> Id {
        let hl = self.bl(op::U_LESS_THAN, &[a.hi, b.hi]);
        let he = self.bl(op::I_EQUAL, &[a.hi, b.hi]);
        let ll = self.bl(op::U_LESS_THAN, &[a.lo, b.lo]);
        let x = self.and_b(he, ll);
        self.bl(op::LOGICAL_OR, &[hl, x])
    }

    /// Is the double `p` a NaN (exponent all ones, mantissa non-zero)?
    fn nan64(&mut self, p: Pair) -> Id {
        let m7 = self.c(0x7FFF_FFFF);
        let m = self.u(op::BITWISE_AND, &[p.hi, m7]);
        let inf = self.c(0x7FF0_0000);
        let gt = self.bl(op::U_GREATER_THAN, &[m, inf]);
        let ex = self.bl(op::I_EQUAL, &[m, inf]);
        let lnz = self.nonzero(p.lo);
        let x = self.and_b(ex, lnz);
        self.bl(op::LOGICAL_OR, &[gt, x])
    }

    /// Unsigned key ordering non-NaN doubles as numbers, with -0 < +0.
    fn key64(&mut self, p: Pair) -> Pair {
        let sb = self.c(0x8000_0000);
        let s = self.u(op::BITWISE_AND, &[p.hi, sb]);
        let neg = self.nonzero(s);
        let nh = self.u(op::NOT, &[p.hi]);
        let nl = self.u(op::NOT, &[p.lo]);
        let fh = self.u(op::BITWISE_XOR, &[p.hi, sb]);
        Pair { hi: self.select_u(neg, nh, fh), lo: self.select_u(neg, nl, p.lo) }
    }

    /// f64 VOPC condition `cond` on bit patterns (no Float64 needed):
    /// NaN from the exponent and mantissa, equality with -0 == +0, and
    /// order through the usual sign-magnitude to unsigned key.
    fn compare64(&mut self, cond: u16, a: Pair, b: Pair) -> Id {
        let (na, nb) = (self.nan64(a), self.nan64(b));
        let unord = self.bl(op::LOGICAL_OR, &[na, nb]);
        let ord = self.not_b(unord);
        let m7 = self.c(0x7FFF_FFFF);
        let hh = self.u(op::BITWISE_OR, &[a.hi, b.hi]);
        let hh = self.u(op::BITWISE_AND, &[hh, m7]);
        let ll = self.u(op::BITWISE_OR, &[a.lo, b.lo]);
        let all = self.u(op::BITWISE_OR, &[hh, ll]);
        let z = self.c(0);
        let both_zero = self.bl(op::I_EQUAL, &[all, z]);
        let he = self.bl(op::I_EQUAL, &[a.hi, b.hi]);
        let le = self.bl(op::I_EQUAL, &[a.lo, b.lo]);
        let same = self.and_b(he, le);
        let eqb = self.bl(op::LOGICAL_OR, &[same, both_zero]);
        let eq = self.and_b(ord, eqb);
        let (ka, kb) = (self.key64(a), self.key64(b));
        let lt_key = self.ult64(ka, kb);
        let ne = self.not_b(eq);
        let oe = self.and_b(ord, ne);
        let lt = self.and_b(oe, lt_key);
        let nlk = self.not_b(lt_key);
        let gt = self.and_b(oe, nlk);
        let le_ = self.bl(op::LOGICAL_OR, &[lt, eq]);
        let ge = self.bl(op::LOGICAL_OR, &[gt, eq]);
        let (t, f) = (self.b.constant_bool(true), self.b.constant_bool(false));
        match cond {
            0 => f,
            1 => lt,
            2 => eq,
            3 => le_,
            4 => gt,
            5 => oe,
            6 => ge,
            7 => ord,
            8 => unord,
            9 => self.not_b(ge),
            10 => self.not_b(oe),
            11 => self.not_b(gt),
            12 => self.not_b(le_),
            13 => ne,
            14 => self.not_b(lt),
            _ => t,
        }
    }

    /// f64 trunc/ceil/rndne/floor on the bit pattern (see
    /// `interp::round_f64`): |x| < 1 gives ±0 or ±1; exponents with no
    /// fraction bits (and Inf/NaN) pass through; otherwise the fraction
    /// bits are cleared and, when rounding away from zero, one unit of
    /// the last integer bit is added (a carry into the exponent is the
    /// IEEE result).
    fn round_f64(&mut self, opc: u16, x: Pair) -> Pair {
        let k = |t: &mut Self, v: u32| t.c(v);
        let sb = k(self, 0x8000_0000);
        let sign = self.u(op::BITWISE_AND, &[x.hi, sb]);
        let neg = self.nonzero(sign);
        let pos = self.not_b(neg);
        let k20 = k(self, 20);
        let e = self.u(op::SHIFT_RIGHT_LOGICAL, &[x.hi, k20]);
        let m11 = k(self, 0x7FF);
        let e = self.u(op::BITWISE_AND, &[e, m11]);
        let m7 = k(self, 0x7FFF_FFFF);
        let mh = self.u(op::BITWISE_AND, &[x.hi, m7]);
        let mag = self.u(op::BITWISE_OR, &[mh, x.lo]);
        let nonzero = self.nonzero(mag);
        let k1023 = k(self, 1023);
        let small = self.bl(op::U_LESS_THAN, &[e, k1023]);
        let k1075 = k(self, 1075);
        let big = self.bl(op::U_GREATER_THAN_EQUAL, &[e, k1075]);
        // |x| < 1
        let one = k(self, 0x3FF0_0000);
        let sone = self.u(op::BITWISE_OR, &[sign, one]);
        let small_hi = match opc {
            vop1::V_TRUNC_F64 => sign,
            vop1::V_FLOOR_F64 => {
                let c = self.and_b(neg, nonzero);
                self.select_u(c, sone, sign)
            }
            vop1::V_CEIL_F64 => {
                let c = self.and_b(pos, nonzero);
                self.select_u(c, sone, sign)
            }
            _ => {
                // above one half: exponent 1022 with a non-zero mantissa
                let k1022 = k(self, 1022);
                let e22 = self.bl(op::I_EQUAL, &[e, k1022]);
                let mm = k(self, 0x000F_FFFF);
                let fh = self.u(op::BITWISE_AND, &[x.hi, mm]);
                let fm = self.u(op::BITWISE_OR, &[fh, x.lo]);
                let fnz = self.nonzero(fm);
                let c = self.and_b(e22, fnz);
                self.select_u(c, sone, sign)
            }
        };
        // 1 <= |x| < 2^52: f = 1075 - e fraction bits (1..=52)
        let mid = {
            let ns = self.not_b(small);
            let nb = self.not_b(big);
            self.and_b(ns, nb)
        };
        let fr = self.u(op::I_SUB, &[k1075, e]);
        let k1 = k(self, 1);
        let f = self.select_u(mid, fr, k1);
        let bit = |t: &mut Self, n: Id| -> Pair {
            // 1 << n as a pair, n in 0..64
            let k32 = t.c(32);
            let ge = t.bl(op::U_GREATER_THAN_EQUAL, &[n, k32]);
            let k31 = t.c(31);
            let sh = t.u(op::BITWISE_AND, &[n, k31]);
            let one = t.c(1);
            let v = t.u(op::SHIFT_LEFT_LOGICAL, &[one, sh]);
            let zero = t.c(0);
            Pair { lo: t.select_u(ge, zero, v), hi: t.select_u(ge, v, zero) }
        };
        let ulp = bit(self, f);
        let all1 = k(self, u32::MAX);
        let zero = k(self, 0);
        // mask = ulp - 1
        let lo_m = self.u(op::I_SUB, &[ulp.lo, k1]);
        let lz = self.bl(op::I_EQUAL, &[ulp.lo, zero]);
        let mlo = self.select_u(lz, all1, lo_m);
        let hi_m = self.u(op::I_SUB, &[ulp.hi, k1]);
        let mhi = self.select_u(lz, hi_m, zero);
        let nm_lo = self.u(op::NOT, &[mlo]);
        let nm_hi = self.u(op::NOT, &[mhi]);
        let t = Pair { lo: self.u(op::BITWISE_AND, &[x.lo, nm_lo]), hi: self.u(op::BITWISE_AND, &[x.hi, nm_hi]) };
        let frac = Pair { lo: self.u(op::BITWISE_AND, &[x.lo, mlo]), hi: self.u(op::BITWISE_AND, &[x.hi, mhi]) };
        let fo = self.u(op::BITWISE_OR, &[frac.lo, frac.hi]);
        let frnz = self.nonzero(fo);
        let up = match opc {
            vop1::V_TRUNC_F64 => self.b.constant_bool(false),
            vop1::V_FLOOR_F64 => self.and_b(neg, frnz),
            vop1::V_CEIL_F64 => self.and_b(pos, frnz),
            _ => {
                let h = self.u(op::I_SUB, &[f, k1]);
                let half = bit(self, h);
                let gt = self.ult64(half, frac);
                let he = self.bl(op::I_EQUAL, &[half.hi, frac.hi]);
                let le = self.bl(op::I_EQUAL, &[half.lo, frac.lo]);
                let eq = self.and_b(he, le);
                let ol = self.u(op::BITWISE_AND, &[t.lo, ulp.lo]);
                let oh = self.u(op::BITWISE_AND, &[t.hi, ulp.hi]);
                let oo = self.u(op::BITWISE_OR, &[ol, oh]);
                let odd = self.nonzero(oo);
                let tie = self.and_b(eq, odd);
                self.bl(op::LOGICAL_OR, &[gt, tie])
            }
        };
        let r = self.add64(t, ulp);
        let r = Pair { lo: self.select_u(up, r.lo, t.lo), hi: self.select_u(up, r.hi, t.hi) };
        let lo = self.select_u(big, x.lo, r.lo);
        let hi = self.select_u(big, x.hi, r.hi);
        Pair { lo: self.select_u(small, zero, lo), hi: self.select_u(small, small_hi, hi) }
    }

    /// A VOPC condition on two 32-bit values (see `interp::Compare`).
    fn compare_with(&mut self, cmp: crate::interp::Compare, a: Id, b: Id) -> Id {
        use crate::interp::CmpKind;
        let (t, f) = (self.b.constant_bool(true), self.b.constant_bool(false));
        if cmp.kind == CmpKind::F32 {
            let (x, y) = (self.as_f32(a), self.as_f32(b));
            let ordc = |tr: &mut Self, o: u16| tr.bl(o, &[x, y]);
            let not = |tr: &mut Self, v: Id| tr.bl(op::LOGICAL_NOT, &[v]);
            match cmp.cond {
                0 => f,
                1 => ordc(self, op::F_ORD_LESS_THAN),
                2 => ordc(self, op::F_ORD_EQUAL),
                3 => ordc(self, op::F_ORD_LESS_THAN_EQUAL),
                4 => ordc(self, op::F_ORD_GREATER_THAN),
                5 => ordc(self, op::F_ORD_NOT_EQUAL),
                6 => ordc(self, op::F_ORD_GREATER_THAN_EQUAL),
                7 | 8 => {
                    // ordered: neither is NaN (x == x is false only for NaN)
                    let xa = self.bl(op::F_ORD_EQUAL, &[x, x]);
                    let ya = self.bl(op::F_ORD_EQUAL, &[y, y]);
                    let o = self.bl(op::LOGICAL_AND, &[xa, ya]);
                    if cmp.cond == 7 { o } else { not(self, o) }
                }
                9 => {
                    let v = ordc(self, op::F_ORD_GREATER_THAN_EQUAL);
                    not(self, v)
                }
                10 => {
                    let v = ordc(self, op::F_ORD_NOT_EQUAL);
                    not(self, v)
                }
                11 => {
                    let v = ordc(self, op::F_ORD_GREATER_THAN);
                    not(self, v)
                }
                12 => {
                    let v = ordc(self, op::F_ORD_LESS_THAN_EQUAL);
                    not(self, v)
                }
                13 => ordc(self, op::F_UNORD_NOT_EQUAL),
                14 => {
                    let v = ordc(self, op::F_ORD_LESS_THAN);
                    not(self, v)
                }
                _ => t,
            }
        } else {
            let signed = cmp.kind == CmpKind::I32;
            let (x, y) = if signed { (self.as_i32(a), self.as_i32(b)) } else { (a, b) };
            let pick = |s_op: u16, u_op: u16| if signed { s_op } else { u_op };
            let o = match cmp.cond {
                0 => return f,
                1 => pick(op::S_LESS_THAN, op::U_LESS_THAN),
                2 => op::I_EQUAL,
                3 => pick(op::S_LESS_THAN_EQUAL, op::U_LESS_THAN_EQUAL),
                4 => pick(op::S_GREATER_THAN, op::U_GREATER_THAN),
                5 => op::I_NOT_EQUAL,
                6 => pick(op::S_GREATER_THAN_EQUAL, op::U_GREATER_THAN_EQUAL),
                _ => return t,
            };
            self.bl(o, &[x, y])
        }
    }

    /// (1 << n) - 1 for n in 0..=32 (all ones for n >= 32).
    fn low_mask(&mut self, n: Id) -> Id {
        let (one, k32, all) = (self.c(1), self.c(32), self.c(u32::MAX));
        let small = self.bl(op::U_LESS_THAN, &[n, k32]);
        let zero = self.c(0);
        let safe = self.select_u(small, n, zero);
        let bit = self.u(op::SHIFT_LEFT_LOGICAL, &[one, safe]);
        let m = self.u(op::I_SUB, &[bit, one]);
        self.select_u(small, m, all)
    }

    fn and31(&mut self, x: Id) -> Id {
        let k = self.c(31);
        self.u(op::BITWISE_AND, &[x, k])
    }

    fn pick(&mut self, cmp: u16, a: Id, b: Id, signed: bool, want_less: bool) -> Id {
        let (x, y) = if signed {
            (self.as_i32(a), self.as_i32(b))
        } else {
            (a, b)
        };
        let lt = self.bl(cmp, &[x, y]);
        if want_less {
            self.select_u(lt, a, b)
        } else {
            self.select_u(lt, b, a)
        }
    }

    /// Integer VOP2 lowering; mirrors `interp::vop2_int`.
    fn vop2_int(&mut self, opc: u16, a: Id, b: Id) -> Option<Id> {
        Some(match opc {
            vop2::V_MUL_U32_U24 => {
                let m = self.c(0xFF_FFFF);
                let x = self.u(op::BITWISE_AND, &[a, m]);
                let y = self.u(op::BITWISE_AND, &[b, m]);
                self.u(op::I_MUL, &[x, y])
            }
            vop2::V_MIN_I32 => self.pick(op::S_LESS_THAN, a, b, true, true),
            vop2::V_MAX_I32 => self.pick(op::S_LESS_THAN, a, b, true, false),
            vop2::V_MIN_U32 => self.pick(op::U_LESS_THAN, a, b, false, true),
            vop2::V_MAX_U32 => self.pick(op::U_LESS_THAN, a, b, false, false),
            vop2::V_LSHRREV_B32 => {
                let s = self.and31(a);
                self.u(op::SHIFT_RIGHT_LOGICAL, &[b, s])
            }
            vop2::V_ASHRREV_I32 => {
                let s = self.and31(a);
                let i = self.as_i32(b);
                let t = self.t.i32t;
                let r = self.b.op(t, op::SHIFT_RIGHT_ARITHMETIC, &[i, s]);
                self.bits_of(r)
            }
            vop2::V_LSHLREV_B32 => {
                let s = self.and31(a);
                self.u(op::SHIFT_LEFT_LOGICAL, &[b, s])
            }
            vop2::V_AND_B32 => self.u(op::BITWISE_AND, &[a, b]),
            vop2::V_OR_B32 => self.u(op::BITWISE_OR, &[a, b]),
            vop2::V_XOR_B32 => self.u(op::BITWISE_XOR, &[a, b]),
            vop2::V_ADD_NC_U32 => self.u(op::I_ADD, &[a, b]),
            vop2::V_SUB_NC_U32 => self.u(op::I_SUB, &[a, b]),
            vop2::V_SUBREV_NC_U32 => self.u(op::I_SUB, &[b, a]),
            _ => return None,
        })
    }

    /// Integer VOP1 lowering; mirrors `interp::vop1_int`.
    fn vop1_int(&mut self, opc: u16, a: Id) -> Option<Id> {
        Some(match opc {
            vop1::V_NOT_B32 => self.u(op::NOT, &[a]),
            vop1::V_BFREV_B32 => {
                // swap halves, bytes, nibbles, pairs, bits
                let mut x = a;
                for (sh, mask) in [
                    (16u32, 0x0000_FFFFu32),
                    (8, 0x00FF_00FF),
                    (4, 0x0F0F_0F0F),
                    (2, 0x3333_3333),
                    (1, 0x5555_5555),
                ] {
                    let (s, m) = (self.c(sh), self.c(mask));
                    let lo = self.u(op::BITWISE_AND, &[x, m]);
                    let lo = self.u(op::SHIFT_LEFT_LOGICAL, &[lo, s]);
                    let hi = self.u(op::SHIFT_RIGHT_LOGICAL, &[x, s]);
                    let hi = self.u(op::BITWISE_AND, &[hi, m]);
                    x = self.u(op::BITWISE_OR, &[lo, hi]);
                }
                x
            }
            vop1::V_FFBH_I32 => {
                // ffbh_u32 of the value, complemented when negative
                let zero = self.c(0);
                let (ai, zi) = (self.as_i32(a), self.as_i32(zero));
                let neg = self.bl(op::S_LESS_THAN, &[ai, zi]);
                let na = self.u(op::NOT, &[a]);
                let x = self.select_u(neg, na, a);
                return self.vop1_int(vop1::V_FFBH_U32, x);
            }
            vop1::V_FFBL_B32 | vop1::V_FFBH_U32 => {
                let n = if opc == vop1::V_FFBL_B32 {
                    self.ctz(a)
                } else {
                    // clz = 32 - popcount(smear(x))
                    let mut x = a;
                    for sh in [1u32, 2, 4, 8, 16] {
                        let s = self.c(sh);
                        let y = self.u(op::SHIFT_RIGHT_LOGICAL, &[x, s]);
                        x = self.u(op::BITWISE_OR, &[x, y]);
                    }
                    let k32 = self.c(32);
                    let p = self.u(op::BIT_COUNT, &[x]);
                    self.u(op::I_SUB, &[k32, p])
                };
                let k32 = self.c(32);
                let none = self.bl(op::I_EQUAL, &[n, k32]);
                let all = self.c(u32::MAX);
                self.select_u(none, all, n)
            }
            _ => return None,
        })
    }

    /// Integer three-operand VOP3 lowering; mirrors `interp::vop3_int`.
    fn vop3_int(&mut self, opc: u16, a: Id, b: Id, c3: Id) -> Option<Id> {
        Some(match opc {
            vop3::V_MUL_LO_U32 => self.u(op::I_MUL, &[a, b]),
            vop3::V_LSHL_ADD_U32 => {
                let s = self.and31(b);
                let x = self.u(op::SHIFT_LEFT_LOGICAL, &[a, s]);
                self.u(op::I_ADD, &[x, c3])
            }
            vop3::V_ADD3_U32 => {
                let x = self.u(op::I_ADD, &[a, b]);
                self.u(op::I_ADD, &[x, c3])
            }
            vop3::V_AND_OR_B32 => {
                let x = self.u(op::BITWISE_AND, &[a, b]);
                self.u(op::BITWISE_OR, &[x, c3])
            }
            vop3::V_OR3_B32 => {
                let x = self.u(op::BITWISE_OR, &[a, b]);
                self.u(op::BITWISE_OR, &[x, c3])
            }
            vop3::V_XOR3_B32 => {
                let x = self.u(op::BITWISE_XOR, &[a, b]);
                self.u(op::BITWISE_XOR, &[x, c3])
            }
            vop3::V_ADD_LSHL_U32 => {
                let x = self.u(op::I_ADD, &[a, b]);
                let sh = self.and31(c3);
                self.u(op::SHIFT_LEFT_LOGICAL, &[x, sh])
            }
            vop3::V_BFM_B32 => {
                let n = self.and31(a);
                let m = self.low_mask(n);
                let sh = self.and31(b);
                self.u(op::SHIFT_LEFT_LOGICAL, &[m, sh])
            }
            vop3::V_BFE_U32 => {
                let one = self.c(1);
                let off = self.and31(b);
                let width = self.and31(c3);
                let sh = self.u(op::SHIFT_RIGHT_LOGICAL, &[a, off]);
                let bit = self.u(op::SHIFT_LEFT_LOGICAL, &[one, width]);
                let m = self.u(op::I_SUB, &[bit, one]);
                self.u(op::BITWISE_AND, &[sh, m])
            }
            vop3::V_BFI_B32 => {
                let x = self.u(op::BITWISE_AND, &[a, b]);
                let na = self.u(op::NOT, &[a]);
                let y = self.u(op::BITWISE_AND, &[na, c3]);
                self.u(op::BITWISE_OR, &[x, y])
            }
            vop3::V_MED3_U32 | vop3::V_MED3_I32 => {
                let (cmp, s) = if opc == vop3::V_MED3_I32 { (op::S_LESS_THAN, true) } else { (op::U_LESS_THAN, false) };
                let mn = self.pick(cmp, a, b, s, true);
                let mx = self.pick(cmp, a, b, s, false);
                let m2 = self.pick(cmp, mx, c3, s, true);
                self.pick(cmp, mn, m2, s, false)
            }
            vop3::V_MIN3_I32 | vop3::V_MIN3_U32 | vop3::V_MAX3_I32 | vop3::V_MAX3_U32 => {
                let s = matches!(opc, vop3::V_MIN3_I32 | vop3::V_MAX3_I32);
                let cmp = if s { op::S_LESS_THAN } else { op::U_LESS_THAN };
                let less = matches!(opc, vop3::V_MIN3_I32 | vop3::V_MIN3_U32);
                let x = self.pick(cmp, a, b, s, less);
                self.pick(cmp, x, c3, s, less)
            }
            vop3::V_MUL_LO_I32 => self.u(op::I_MUL, &[a, b]),
            vop3::V_MUL_HI_U32 | vop3::V_MUL_HI_I32 => {
                // 16-bit limbs: every partial product fits in 32 bits.
                let (k16, m16) = (self.c(16), self.c(0xFFFF));
                let al = self.u(op::BITWISE_AND, &[a, m16]);
                let ah = self.u(op::SHIFT_RIGHT_LOGICAL, &[a, k16]);
                let bl = self.u(op::BITWISE_AND, &[b, m16]);
                let bh = self.u(op::SHIFT_RIGHT_LOGICAL, &[b, k16]);
                let ll = self.u(op::I_MUL, &[al, bl]);
                let lh = self.u(op::I_MUL, &[al, bh]);
                let hl = self.u(op::I_MUL, &[ah, bl]);
                let hh = self.u(op::I_MUL, &[ah, bh]);
                let ll_hi = self.u(op::SHIFT_RIGHT_LOGICAL, &[ll, k16]);
                let lh_lo = self.u(op::BITWISE_AND, &[lh, m16]);
                let hl_lo = self.u(op::BITWISE_AND, &[hl, m16]);
                let mid = self.u(op::I_ADD, &[ll_hi, lh_lo]);
                let mid = self.u(op::I_ADD, &[mid, hl_lo]);
                let lh_hi = self.u(op::SHIFT_RIGHT_LOGICAL, &[lh, k16]);
                let hl_hi = self.u(op::SHIFT_RIGHT_LOGICAL, &[hl, k16]);
                let mid_hi = self.u(op::SHIFT_RIGHT_LOGICAL, &[mid, k16]);
                let mut hi = self.u(op::I_ADD, &[hh, lh_hi]);
                hi = self.u(op::I_ADD, &[hi, hl_hi]);
                hi = self.u(op::I_ADD, &[hi, mid_hi]);
                if opc == vop3::V_MUL_HI_I32 {
                    // signed hi = unsigned hi - (a < 0 ? b : 0) - (b < 0 ? a : 0)
                    let zero = self.c(0);
                    let (ai, bi) = (self.as_i32(a), self.as_i32(b));
                    let zi = self.as_i32(zero);
                    let an = self.bl(op::S_LESS_THAN, &[ai, zi]);
                    let bn = self.bl(op::S_LESS_THAN, &[bi, zi]);
                    let ta = self.select_u(an, b, zero);
                    let tb = self.select_u(bn, a, zero);
                    hi = self.u(op::I_SUB, &[hi, ta]);
                    hi = self.u(op::I_SUB, &[hi, tb]);
                }
                hi
            }
            vop3::V_MAD_U32_U24 | vop3::V_MAD_I32_I24 => {
                let (x, y) = if opc == vop3::V_MAD_U32_U24 {
                    let m = self.c(0xFF_FFFF);
                    (self.u(op::BITWISE_AND, &[a, m]), self.u(op::BITWISE_AND, &[b, m]))
                } else {
                    let k8 = self.c(8);
                    let mut s = |v: Id| {
                        let t = self.u(op::SHIFT_LEFT_LOGICAL, &[v, k8]);
                        self.u(op::SHIFT_RIGHT_ARITHMETIC, &[t, k8])
                    };
                    (s(a), s(b))
                };
                let p = self.u(op::I_MUL, &[x, y]);
                self.u(op::I_ADD, &[p, c3])
            }
            vop3::V_BCNT_U32_B32 => {
                let n = self.u(op::BIT_COUNT, &[a]);
                self.u(op::I_ADD, &[n, b])
            }
            vop3::V_ALIGNBIT_B32 => {
                // fshr: s == 0 -> b, else b >> s | a << (32 - s)
                let s = self.and31(c3);
                let k32 = self.c(32);
                let inv = self.u(op::I_SUB, &[k32, s]);
                let inv = self.and31(inv);
                let lo = self.u(op::SHIFT_RIGHT_LOGICAL, &[b, s]);
                let hi = self.u(op::SHIFT_LEFT_LOGICAL, &[a, inv]);
                let r = self.u(op::BITWISE_OR, &[lo, hi]);
                let zero = self.c(0);
                let z = self.bl(op::I_EQUAL, &[s, zero]);
                self.select_u(z, b, r)
            }
            vop3::V_BFE_I32 => {
                let off = self.and31(b);
                let width = self.and31(c3);
                let (zero, k32) = (self.c(0), self.c(32));
                let ow = self.u(op::I_ADD, &[off, width]);
                let l = self.u(op::I_SUB, &[k32, ow]);
                let l = self.and31(l);
                let r = self.u(op::I_SUB, &[k32, width]);
                let r = self.and31(r);
                let t = self.u(op::SHIFT_LEFT_LOGICAL, &[a, l]);
                let field = self.u(op::SHIFT_RIGHT_ARITHMETIC, &[t, r]);
                let whole = self.u(op::SHIFT_RIGHT_ARITHMETIC, &[a, off]);
                let fits = self.bl(op::U_LESS_THAN, &[ow, k32]);
                let v = self.select_u(fits, field, whole);
                let w0 = self.bl(op::I_EQUAL, &[width, zero]);
                self.select_u(w0, zero, v)
            }
            vop3::V_XAD_U32 => {
                let x = self.u(op::BITWISE_XOR, &[a, b]);
                self.u(op::I_ADD, &[x, c3])
            }
            vop3::V_LSHL_OR_B32 => {
                let s = self.and31(b);
                let x = self.u(op::SHIFT_LEFT_LOGICAL, &[a, s]);
                self.u(op::BITWISE_OR, &[x, c3])
            }
            vop3::V_CVT_PK_U16_U32 | vop3::V_CVT_PK_I16_I32 => {
                let signed = opc == vop3::V_CVT_PK_I16_I32;
                let (cmp, lo, hi) = if signed {
                    (op::S_LESS_THAN, self.c(-32768i32 as u32), self.c(32767))
                } else {
                    (op::U_LESS_THAN, self.c(0), self.c(0xFFFF))
                };
                let m = self.c(0xFFFF);
                let k16 = self.c(16);
                let sat = |tr: &mut Self, x: Id| {
                    let y = tr.pick(cmp, x, hi, signed, true); // min(x, hi)
                    let y = if signed { tr.pick(cmp, y, lo, true, false) } else { y }; // max(y, lo)
                    tr.u(op::BITWISE_AND, &[y, m])
                };
                let l = sat(self, a);
                let h = sat(self, b);
                let h = self.u(op::SHIFT_LEFT_LOGICAL, &[h, k16]);
                self.u(op::BITWISE_OR, &[l, h])
            }
            vop3::V_ALIGNBYTE_B32 => {
                let k3 = self.c(3);
                let k8 = self.c(8);
                let s = self.u(op::BITWISE_AND, &[c3, k3]);
                let s = self.u(op::I_MUL, &[s, k8]);
                let k32 = self.c(32);
                let inv = self.u(op::I_SUB, &[k32, s]);
                let inv = self.and31(inv);
                let lo = self.u(op::SHIFT_RIGHT_LOGICAL, &[b, s]);
                let hi = self.u(op::SHIFT_LEFT_LOGICAL, &[a, inv]);
                let r = self.u(op::BITWISE_OR, &[lo, hi]);
                let zero = self.c(0);
                let z = self.bl(op::I_EQUAL, &[s, zero]);
                self.select_u(z, b, r)
            }
            vop3::V_MSAD_U8 => {
                let kff = self.c(0xFF);
                let zero = self.c(0);
                let mut acc = c3;
                for i in 0..4u32 {
                    let sh = self.c(8 * i);
                    let r = self.u(op::SHIFT_RIGHT_LOGICAL, &[b, sh]);
                    let r = self.u(op::BITWISE_AND, &[r, kff]);
                    let s = self.u(op::SHIFT_RIGHT_LOGICAL, &[a, sh]);
                    let s = self.u(op::BITWISE_AND, &[s, kff]);
                    let mx = self.pick(op::U_LESS_THAN, r, s, false, false);
                    let mn = self.pick(op::U_LESS_THAN, r, s, false, true);
                    let d = self.u(op::I_SUB, &[mx, mn]);
                    let nz = self.bl(op::I_NOT_EQUAL, &[r, zero]);
                    let d = self.select_u(nz, d, zero);
                    acc = self.u(op::I_ADD, &[acc, d]);
                }
                acc
            }
            vop3::V_PERM_B32 => {
                // Per byte: select by the selector byte (interp::vop3_int).
                let (k8, kff, k3, k4, k12) = (self.c(8), self.c(0xFF), self.c(3), self.c(4), self.c(12));
                let mut out = self.c(0);
                for i in 0..4u32 {
                    let sh = self.c(8 * i);
                    let s = self.u(op::SHIFT_RIGHT_LOGICAL, &[c3, sh]);
                    let s = self.u(op::BITWISE_AND, &[s, kff]);
                    let s3 = self.u(op::BITWISE_AND, &[s, k3]);
                    let bs = self.u(op::I_MUL, &[s3, k8]);
                    let from_b = self.u(op::SHIFT_RIGHT_LOGICAL, &[b, bs]);
                    let from_b = self.u(op::BITWISE_AND, &[from_b, kff]);
                    let from_a = self.u(op::SHIFT_RIGHT_LOGICAL, &[a, bs]);
                    let from_a = self.u(op::BITWISE_AND, &[from_a, kff]);
                    // 8..11: sign of byte 2 (sel - 8) + 1 of S0:S1, replicated
                    let k = self.u(op::I_SUB, &[s, k8]);
                    let one = self.c(1);
                    let k1 = self.u(op::BITWISE_AND, &[k, one]);
                    let k16 = self.c(16);
                    let k15 = self.c(15);
                    let bit = self.u(op::I_MUL, &[k1, k16]);
                    let bit = self.u(op::I_ADD, &[bit, k15]);
                    let two = self.c(2);
                    let hi_word = self.bl(op::U_GREATER_THAN_EQUAL, &[k, two]);
                    let word = self.select_u(hi_word, a, b);
                    let sb = self.u(op::SHIFT_RIGHT_LOGICAL, &[word, bit]);
                    let sb = self.u(op::BITWISE_AND, &[sb, one]);
                    let sign = self.u(op::I_MUL, &[sb, kff]);
                    let lt4 = self.bl(op::U_LESS_THAN, &[s, k4]);
                    let lt8 = self.bl(op::U_LESS_THAN, &[s, k8]);
                    let lt12 = self.bl(op::U_LESS_THAN, &[s, k12]);
                    let is12 = self.bl(op::I_EQUAL, &[s, k12]);
                    let zero = self.c(0);
                    let v = self.select_u(is12, zero, kff);
                    let v = self.select_u(lt12, sign, v);
                    let v = self.select_u(lt8, from_a, v);
                    let v = self.select_u(lt4, from_b, v);
                    let v = self.u(op::SHIFT_LEFT_LOGICAL, &[v, sh]);
                    out = self.u(op::BITWISE_OR, &[out, v]);
                }
                out
            }
            _ => return None,
        })
    }

    /// Carry arithmetic; mirrors `interp::carry_op` / `carry_step`.
    #[allow(clippy::too_many_arguments)]
    fn carry(&mut self, opc: u16, vdst: u8, s0: u16, s1: u16, cin: Option<u16>, sdst: u8, lit: u32) -> Res<()> {
        let e = |o: u16| vop3::FROM_VOP2 + o;
        let kind = match opc {
            vop3::V_ADD_CO_U32 | vop2::V_ADD_CO_CI_U32 => 0,
            vop3::V_SUB_CO_U32 | vop2::V_SUB_CO_CI_U32 => 1,
            o if o == e(vop2::V_ADD_CO_CI_U32) => 0,
            o if o == e(vop2::V_SUB_CO_CI_U32) => 1,
            _ => 2,
        };
        let x = self.vsrc(s0, lit)?;
        let y = self.vsrc(s1, lit)?;
        let (a, b) = if kind == 2 { (y, x) } else { (x, y) };
        let zero = self.c(0);
        let ci = match cin {
            Some(src) => {
                let m = self.mask(src, lit)?;
                let bit = self.lane_bit(m);
                self.bool_to_u(bit)
            }
            None => zero,
        };
        let (r, co) = if kind == 0 {
            let s1v = self.u(op::I_ADD, &[a, b]);
            let c1 = self.bl(op::U_LESS_THAN, &[s1v, a]);
            let s = self.u(op::I_ADD, &[s1v, ci]);
            let c2 = self.bl(op::U_LESS_THAN, &[s, s1v]);
            (s, self.bl(op::LOGICAL_OR, &[c1, c2]))
        } else {
            let d1 = self.u(op::I_SUB, &[a, b]);
            let b1 = self.bl(op::U_LESS_THAN, &[a, b]);
            let d = self.u(op::I_SUB, &[d1, ci]);
            let b2 = self.bl(op::U_LESS_THAN, &[d1, ci]);
            (d, self.bl(op::LOGICAL_OR, &[b1, b2]))
        };
        let act = self.active();
        let p = self.bl(op::LOGICAL_AND, &[act, co]);
        let m = self.ballot(p);
        self.set_vgpr(vdst, r)?;
        self.set_mask(sdst, m)
    }

    /// SDWA source selector at translation time (rdna::sdwa::select).
    fn sdwa_select(&mut self, x: Id, sel: u32, sext: bool) -> Res<Id> {
        let (sh, w) = crate::rdna::sdwa::field(sel).ok_or_else(|| self.err("SDWA selector"))?;
        if w == 32 {
            return Ok(x);
        }
        let up = self.c(32 - w - sh);
        let t = self.u(op::SHIFT_LEFT_LOGICAL, &[x, up]);
        let down = self.c(32 - w);
        Ok(self.u(if sext { op::SHIFT_RIGHT_ARITHMETIC } else { op::SHIFT_RIGHT_LOGICAL }, &[t, down]))
    }

    /// SDWA on integer VOP1 / VOP2; mirrors `interp::sdwa_step`.
    fn sdwa(&mut self, inst: &Inst) -> Res<()> {
        let (vdst, src1) = match inst.fields {
            Fields::Vop1 { vdst, .. } => (vdst, None),
            Fields::Vop2 { vdst, vsrc1, .. } => (vdst, Some(vsrc1)),
            _ => return Err(self.err("SDWA on VOPC")),
        };
        let d = crate::rdna::sdwa::parse(*inst.words.get(1).ok_or_else(|| self.err("missing SDWA word"))?);
        if d.clamp || d.mods != [0, 0] || d.src1_sgpr {
            return Err(self.err("SDWA clamp, source modifiers or SGPR src1"));
        }
        let opc = inst.opcode;
        let s0 = if d.src0_sgpr { d.src0 as u16 } else { 256 + d.src0 as u16 };
        let x = self.vsrc(s0, 0)?;
        let a = self.sdwa_select(x, d.sel[0], d.sext[0])?;
        let r = match src1 {
            Some(v1) => {
                let y = self.vsrc(256 + v1 as u16, 0)?;
                let b = self.sdwa_select(y, d.sel[1], d.sext[1])?;
                self.vop2_int(opc, a, b).ok_or_else(|| self.err("SDWA on this opcode"))?
            }
            None if opc == vop1::V_MOV_B32 => a,
            None => self.vop1_int(opc, a).ok_or_else(|| self.err("SDWA on this opcode"))?,
        };
        let (sh, w) = crate::rdna::sdwa::field(d.dst_sel).ok_or_else(|| self.err("SDWA dst_sel"))?;
        let out = if w == 32 {
            r
        } else {
            let m = self.c(((1u32 << w) - 1) << sh);
            let shc = self.c(sh);
            let f = self.u(op::SHIFT_LEFT_LOGICAL, &[r, shc]);
            let f = self.u(op::BITWISE_AND, &[f, m]);
            match d.dst_unused {
                0 => f,
                1 if sh == 0 => {
                    let k = self.c(32 - w);
                    let t = self.u(op::SHIFT_LEFT_LOGICAL, &[f, k]);
                    self.u(op::SHIFT_RIGHT_ARITHMETIC, &[t, k])
                }
                2 => {
                    let old = self.vsrc(256 + vdst as u16, 0)?;
                    let nm = self.c(!(((1u32 << w) - 1) << sh));
                    let keep = self.u(op::BITWISE_AND, &[old, nm]);
                    self.u(op::BITWISE_OR, &[keep, f])
                }
                _ => return Err(self.err("SDWA dst_sel/dst_unused")),
            }
        };
        self.set_vgpr(vdst, out)
    }

    /// VOP3P dot products; mirrors rdna::vop3p::dot wherever it is defined.
    fn dot(&mut self, opc: u16, a: Id, b: Id, c: Id, sat: bool) -> Id {
        use crate::rdna::vop3p::*;
        let (w, signed) = match opc {
            V_DOT2_I32_I16 => (16u32, true),
            V_DOT2_U32_U16 => (16, false),
            V_DOT4_I32_I8 => (8, true),
            _ => (8, false),
        };
        let field = |tr: &mut Self, x: Id, k: u32| {
            let up = tr.c(32 - w - k * w);
            let down = tr.c(32 - w);
            let t = tr.u(op::SHIFT_LEFT_LOGICAL, &[x, up]);
            tr.u(if signed { op::SHIFT_RIGHT_ARITHMETIC } else { op::SHIFT_RIGHT_LOGICAL }, &[t, down])
        };
        let mut dp = self.c(0);
        for k in 0..32 / w {
            let x = field(self, a, k);
            let y = field(self, b, k);
            let p = self.u(op::I_MUL, &[x, y]);
            dp = self.u(op::I_ADD, &[dp, p]);
        }
        let t = self.u(op::I_ADD, &[dp, c]);
        if !sat {
            return t;
        }
        if signed {
            // signed overflow of dp + c: operands of one sign, result of the other
            let (di, ci, ti) = (self.as_i32(dp), self.as_i32(c), self.as_i32(t));
            let zero = self.c(0);
            let zi = self.as_i32(zero);
            let dn = self.bl(op::S_LESS_THAN, &[di, zi]);
            let cn = self.bl(op::S_LESS_THAN, &[ci, zi]);
            let tn = self.bl(op::S_LESS_THAN, &[ti, zi]);
            // boolean (in)equality via select: x != y == (x ? !y : y)
            let ncn = self.bl(op::LOGICAL_NOT, &[cn]);
            let diff = self.select_b(dn, ncn, cn);
            let same = self.bl(op::LOGICAL_NOT, &[diff]);
            let ntn = self.bl(op::LOGICAL_NOT, &[tn]);
            let flip = self.select_b(dn, ntn, tn);
            let ovf = self.bl(op::LOGICAL_AND, &[same, flip]);
            let (mx, mn) = (self.c(i32::MAX as u32), self.c(i32::MIN as u32));
            let lim = self.select_u(dn, mn, mx);
            self.select_u(ovf, lim, t)
        } else {
            let carry = self.bl(op::U_LESS_THAN, &[t, dp]);
            let mx = self.c(u32::MAX);
            self.select_u(carry, mx, t)
        }
    }

    fn cndmask(&mut self, s0: Id, s1: Id, m: Pair) -> Id {
        let bit = self.lane_bit(m);
        self.select_u(bit, s1, s0)
    }

    /// Count trailing zeros of a 32-bit value (32 for 0), as
    /// popcount((x & -x) - 1).
    fn ctz(&mut self, x: Id) -> Id {
        let zero = self.c(0);
        let one = self.c(1);
        let neg = self.u(op::I_SUB, &[zero, x]);
        let low = self.u(op::BITWISE_AND, &[x, neg]);
        let below = self.u(op::I_SUB, &[low, one]);
        self.u(op::BIT_COUNT, &[below])
    }

    /// Mask of the lanes below this one, split in halves.
    fn lanes_below(&mut self) -> Pair {
        let one = self.c(1);
        let all = self.c(u32::MAX);
        let zero = self.c(0);
        let bit = self.u(op::SHIFT_LEFT_LOGICAL, &[one, self.lane_shift]);
        let below = self.u(op::I_SUB, &[bit, one]);
        Pair {
            lo: self.select_u(self.lane_is_hi, all, below),
            hi: self.select_u(self.lane_is_hi, below, zero),
        }
    }

    fn inst(&mut self, inst: &Inst) -> Res<()> {
        if let Some(l) = crate::rdna::sopk::lower(inst) {
            return self.inst(&l);
        }
        if inst.format == Format::Sopk && crate::rdna::sopk::is_waitcnt(inst.opcode) {
            return Ok(());
        }
        let is_dpp = match inst.fields {
            Fields::Vop1 { src0, .. } | Fields::Vop2 { src0, .. } | Fields::Vopc { src0, .. } => {
                src0 == crate::rdna::SRC_DPP16
            }
            _ => false,
        };
        let sdwa = matches!(inst.fields,
            Fields::Vop1 { src0, .. } | Fields::Vop2 { src0, .. } | Fields::Vopc { src0, .. }
                if src0 == crate::rdna::SRC_SDWA);
        if sdwa {
            return self.sdwa(inst);
        }
        if is_dpp && self.dpp_src0.is_none() {
            if matches!(inst.fields, Fields::Vopc { .. }) {
                return Err(self.err("DPP on VOPC"));
            }
            return self.dpp(inst);
        }
        let lit = inst.literal().unwrap_or(0);
        match (&inst.fields, inst.format, inst.opcode) {
            (_, Format::Sopp, sopp::S_NOP | sopp::S_WAITCNT) => Ok(()),
            (_, Format::Sopp, sopp::S_BARRIER) => {
                match self.waiting {
                    // The dispatcher releases waiting waves together.
                    Some(w) => {
                        let t = self.b.constant_bool(true);
                        self.b.store(w, t);
                    }
                    None => self.barrier(),
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
                _,
                opc,
            ) => {
                if gds {
                    return Err(self.err("GDS"));
                }
                if opc == ds::SWIZZLE_B32 {
                    return self.ds_swizzle(offset0 as u32 | (offset1 as u32) << 8, addr, vdst);
                }
                if opc == ds::APPEND || opc == ds::CONSUME {
                    return self.ds_append(opc == ds::CONSUME, offset0 as u32 | (offset1 as u32) << 8, vdst);
                }
                if let Some((base, rtn)) = ds::atomic(opc) {
                    let off = offset0 as u32 | (offset1 as u32) << 8;
                    return self.ds_atomic(base, rtn, off, addr, data0, data1, vdst);
                }
                self.ds(opc, offset0, offset1, addr, data0, data1, vdst)
            }
            (&Fields::Sop1 { sdst, ssrc0 }, _, opc) => self.sop1(opc, sdst, ssrc0, lit),
            (&Fields::Sop2 { sdst, ssrc0, ssrc1 }, _, opc) => self.sop2(opc, sdst, ssrc0, ssrc1, lit),
            (&Fields::Sopc { ssrc0, ssrc1 }, _, opc)
                if opc == sopc::S_CMP_EQ_U64 || opc == sopc::S_CMP_LG_U64 =>
            {
                let a = self.s64(ssrc0, lit)?;
                let b = self.s64(ssrc1, lit)?;
                let x = self.u(op::BITWISE_XOR, &[a.lo, b.lo]);
                let y = self.u(op::BITWISE_XOR, &[a.hi, b.hi]);
                let d = self.u(op::BITWISE_OR, &[x, y]);
                let ne = self.nonzero(d);
                let r = if opc == sopc::S_CMP_LG_U64 { ne } else { self.bl(op::LOGICAL_NOT, &[ne]) };
                self.b.store(self.scc, r);
                Ok(())
            }
            (&Fields::Sopc { ssrc0, ssrc1 }, _, opc) => {
                let a = self.s32(ssrc0, lit)?;
                let b = self.s32(ssrc1, lit)?;
                let (spv, signed) = match opc {
                    sopc::S_CMP_EQ_I32 | sopc::S_CMP_EQ_U32 => (op::I_EQUAL, false),
                    sopc::S_CMP_GT_I32 => (op::S_GREATER_THAN, true),
                    sopc::S_CMP_GE_I32 => (op::S_GREATER_THAN_EQUAL, true),
                    sopc::S_CMP_GT_U32 => (op::U_GREATER_THAN, false),
                    sopc::S_CMP_GE_U32 => (op::U_GREATER_THAN_EQUAL, false),
                    sopc::S_CMP_LG_I32 | sopc::S_CMP_LG_U32 => (op::I_NOT_EQUAL, false),
                    sopc::S_CMP_LT_I32 => (op::S_LESS_THAN, true),
                    sopc::S_CMP_LE_I32 => (op::S_LESS_THAN_EQUAL, true),
                    sopc::S_CMP_LT_U32 => (op::U_LESS_THAN, false),
                    sopc::S_CMP_LE_U32 => (op::U_LESS_THAN_EQUAL, false),
                    sopc::S_BITCMP0_B32 | sopc::S_BITCMP1_B32 => {
                        let s = self.and31(b);
                        let x = self.u(op::SHIFT_RIGHT_LOGICAL, &[a, s]);
                        let one = self.c(1);
                        let bit = self.u(op::BITWISE_AND, &[x, one]);
                        let want = self.c((opc == sopc::S_BITCMP1_B32) as u32);
                        let r = self.bl(op::I_EQUAL, &[bit, want]);
                        self.b.store(self.scc, r);
                        return Ok(());
                    }
                    _ => return Err(self.err("opcode")),
                };
                let (a, b) = if signed {
                    (self.as_i32(a), self.as_i32(b))
                } else {
                    (a, b)
                };
                let r = self.bl(spv, &[a, b]);
                self.b.store(self.scc, r);
                Ok(())
            }
            (&Fields::Vop1 { vdst, src0 }, _, opc) => self.vop1(opc, vdst, src0, lit),
            (&Fields::Vop2 { vdst, src0, vsrc1 }, _, opc) if crate::rdna::vop2::is_madk(opc) => {
                let a = self.vsrc(src0, lit)?;
                let b = self.vsrc(256 + vsrc1 as u16, lit)?;
                let k = self.c(lit);
                let r = if opc == vop2::V_FMAMK_F32 { self.fma(a, k, b) } else { self.fma(a, b, k) };
                self.set_vgpr(vdst, r)
            }
            (&Fields::Vop2 { vdst, src0, vsrc1 }, _, opc) if crate::interp::carry_op(opc, 0, 0, false).is_some() => {
                self.carry(opc, vdst, src0, 256 + vsrc1 as u16, Some(VCC_LO_ENC), VCC_LO_ENC as u8, lit)
            }
            (&Fields::Vop2 { vdst, src0, vsrc1 }, _, opc) => {
                let a = self.vsrc(src0, lit)?;
                let b = self.vsrc(256 + vsrc1 as u16, lit)?;
                let r = if opc == vop2::V_CVT_PKRTZ_F16_F32 {
                    let lo = self.f32_to_f16(a, true);
                    let hi = self.f32_to_f16(b, true);
                    let k16 = self.c(16);
                    let hi = self.u(op::SHIFT_LEFT_LOGICAL, &[hi, k16]);
                    self.u(op::BITWISE_OR, &[lo, hi])
                } else if opc == vop2::V_CNDMASK_B32 {
                    let m = self.mask(VCC_LO_ENC, 0)?;
                    self.cndmask(a, b, m)
                } else if opc == vop2::V_MIN_F32 || opc == vop2::V_MAX_F32 {
                    self.min_max(opc == vop2::V_MAX_F32, a, b)
                } else if opc == vop2::V_FMAC_F32 {
                    let d = self.vsrc(256 + vdst as u16, 0)?;
                    self.fma(a, b, d)
                } else if let Some(r) = self.vop2_int(opc, a, b) {
                    r
                } else {
                    let spv = Self::vop2_float(opc).ok_or_else(|| self.err("opcode"))?;
                    self.fbin(spv, a, b, 0)
                };
                self.set_vgpr(vdst, r)
            }
            (&Fields::Vopc { src0, vsrc1 }, _, opc) => {
                let cmp = crate::interp::Compare::decode(opc).ok_or_else(|| self.err("opcode"))?;
                let cond = if cmp.kind == crate::interp::CmpKind::F64 {
                    let a = self.src64(src0, lit)?;
                    let b = self.src64(256 + vsrc1 as u16, lit)?;
                    self.compare64(cmp.cond, a, b)
                } else {
                    let a = self.vsrc(src0, lit)?;
                    let b = self.vsrc(256 + vsrc1 as u16, lit)?;
                    self.compare_with(cmp, a, b)
                };
                let act = self.active();
                let p = self.bl(op::LOGICAL_AND, &[act, cond]);
                let m = self.ballot(p);
                // v_cmpx (GFX10) writes exec only; v_cmp writes vcc.
                let (lo, hi) = if cmp.cmpx {
                    (self.exec_lo, self.exec_hi)
                } else {
                    (self.vcc_lo, self.vcc_hi)
                };
                self.b.store(lo, m.lo);
                if self.wave64 {
                    self.b.store(hi, m.hi);
                }
                Ok(())
            }
            (Fields::Vop3 { .. }, _, _) => self.vop3(inst, lit),
            (&Fields::Vop3p { vdst, src, neg, neg_hi, op_sel, op_sel_hi, clamp }, _, opc) => {
                if crate::rdna::vop3p::dot(opc, 0, 0, 0, false).is_none() {
                    return Err(self.err("opcode"));
                }
                if neg != 0 || neg_hi != 0 || op_sel != 0 || op_sel_hi != 7 {
                    return Err(self.err("dot with neg / op_sel other than ACO's"));
                }
                let a = self.vsrc(src[0], lit)?;
                let b = self.vsrc(src[1], lit)?;
                let c3 = self.vsrc(src[2], lit)?;
                let r = self.dot(opc, a, b, c3, clamp);
                self.set_vgpr(vdst, r)
            }
            (
                &Fields::Smem {
                    sdata,
                    sbase,
                    offset,
                    soffset,
                    ..
                },
                _,
                opc,
            ) => self.smem(opc, sdata, sbase, offset, soffset),
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
                _,
                opc,
            ) => self.global(opc, vdst, addr, data, saddr, offset, seg, glc),
            (Fields::Mimg { .. }, _, _) => self.image_load(inst),
            (
                &Fields::Mubuf {
                    vdata,
                    vaddr,
                    srsrc,
                    soffset,
                    offset,
                    offen,
                    idxen,
                    glc,
                },
                _,
                opc,
            ) => {
                if inst.words[0] & mubuf::LDS != 0 {
                    return Err(self.err("MUBUF load to LDS"));
                }
                self.mubuf(opc, vdata, vaddr, srsrc, soffset, offset, offen, idxen, glc)
            }
            (
                &Fields::Mtbuf {
                    vdata,
                    vaddr,
                    srsrc,
                    soffset,
                    offset,
                    format,
                    offen,
                    idxen,
                },
                _,
                opc,
            ) => {
                // As interp: the equivalent MUBUF dword access (DST_SEL is
                // checked there, at run time).
                let m = crate::rdna::mtbuf::as_mubuf(opc, format as u32)
                    .ok_or_else(|| self.err("typed format with conversion, d16, or channel count"))?;
                self.mubuf(m, vdata, vaddr, srsrc, soffset, offset, offen, idxen, false)
            }
            (&Fields::Vintrp { vdst, vsrc, attr, chan }, _, opc) => {
                // Same formulas as interp (P0 = v0, P10 = v1 - v0,
                // P20 = v2 - v0; fused multiply-adds).
                let var = self
                    .attr_vars
                    .iter()
                    .find(|(a, _)| *a == attr)
                    .map(|(_, v)| *v)
                    .ok_or_else(|| self.err("attribute without an input"))?;
                let f = self.t.f32t;
                let v4 = self.b.type_vector(f, 4);
                let p_v4 = self.p_v4_in;
                let mut vtx = [0; 3];
                for (k, x) in vtx.iter_mut().enumerate() {
                    let kc = self.c(k as u32);
                    let p = self.b.op(p_v4, op::ACCESS_CHAIN, &[var, kc]);
                    let v = self.b.load(v4, p);
                    *x = self.b.op(f, op::COMPOSITE_EXTRACT, &[v, chan as u32]);
                }
                let p0 = vtx[0];
                let p10 = self.b.op(f, op::F_SUB, &[vtx[1], vtx[0]]);
                let p20 = self.b.op(f, op::F_SUB, &[vtx[2], vtx[0]]);
                let as_bits = |tr: &mut Self, x: Id| tr.bits_of(x);
                let r = match opc {
                    crate::rdna::vintrp::P1_F32 => {
                        let i = self.vsrc(256 + vsrc as u16, 0)?;
                        let (a, b) = (as_bits(self, p10), as_bits(self, p0));
                        self.fma(a, i, b)
                    }
                    crate::rdna::vintrp::P2_F32 => {
                        let j = self.vsrc(256 + vsrc as u16, 0)?;
                        let d = self.vsrc(256 + vdst as u16, 0)?;
                        let a = as_bits(self, p20);
                        self.fma(a, j, d)
                    }
                    _ => {
                        let x = match vsrc {
                            crate::rdna::vintrp::SLOT_P10 => p10,
                            crate::rdna::vintrp::SLOT_P20 => p20,
                            crate::rdna::vintrp::SLOT_P0 => p0,
                            _ => return Err(self.err("interpolation slot")),
                        };
                        as_bits(self, x)
                    }
                };
                self.set_vgpr(vdst, r)
            }
            (&Fields::Exp { target, en, vsrc, compr, .. }, _, _) => {
                if compr {
                    return Err(self.err("compressed (16-bit) export"));
                }
                if target == crate::rdna::exp::NULL {
                    return Ok(());
                }
                let (out, exported) = match (self.pixel, &self.vertex_outs) {
                    (Some((outs, exported)), _) => (
                        outs.get(target as usize)
                            .copied()
                            .flatten()
                            .ok_or_else(|| self.err("export target other than MRT or null"))?,
                        Some(exported),
                    ),
                    (None, Some(vo)) => (
                        vo.iter()
                            .find(|(t, _)| *t == target)
                            .map(|(_, v)| *v)
                            .ok_or_else(|| self.err("export target other than pos0 or a parameter"))?,
                        None,
                    ),
                    _ => return Err(self.err("exp outside a pixel or vertex shader")),
                };
                if en != 0xF {
                    return Err(self.err("partial color export (component write mask)"));
                }
                let act = self.active();
                let v4 = self.b.type_vector(self.t.f32t, 4);
                let f = self.t.f32t;
                let old = self.b.load(v4, out);
                let mut comps = Vec::new();
                for (k, &r) in vsrc.iter().enumerate() {
                    let v = self.vsrc(256 + r as u16, 0)?;
                    let nv = self.as_f32(v);
                    let ov = self.b.op(f, op::COMPOSITE_EXTRACT, &[old, k as u32]);
                    // inactive lanes keep what they had (SPIR-V 1.3 OpSelect
                    // needs a scalar result for a scalar condition)
                    comps.push(self.b.op(f, op::SELECT, &[act, nv, ov]));
                }
                let newv = self.b.op(v4, op::COMPOSITE_CONSTRUCT, &comps);
                self.b.store(out, newv);
                if let Some(exported) = exported {
                    let boolt = self.t.boolt;
                    let was = self.b.load(boolt, exported);
                    let now = self.bl(op::LOGICAL_OR, &[was, act]);
                    self.b.store(exported, now);
                }
                Ok(())
            }
            _ => Err(self.err("not translated yet")),
        }
    }

    fn smem(&mut self, opc: u16, sdata: u8, sbase: u8, offset: i32, soffset: u8) -> Res<()> {
        let (buf, p_u32, win) = self.mem.ok_or_else(|| self.err("no memory window bound"))?;
        if let Some(n) = smem::buffer_load_dwords(opc) {
            return self.s_buffer_load(n, sdata, sbase, offset, soffset, buf, p_u32, win);
        }
        let n = smem::load_dwords(opc).ok_or_else(|| self.err("opcode"))?;
        let base = self.s64(sbase as u16, 0)?;
        let so = if soffset == smem::SOFFSET_NULL {
            self.c(0)
        } else {
            self.s32(soffset as u16, 0)?
        };
        // 64-bit address = base + sext(offset) + soffset, low bits cleared.
        let off_lo = self.c(offset as u32);
        let off_hi = self.c(if offset < 0 { u32::MAX } else { 0 });
        let a = self.add64(base, Pair { lo: off_lo, hi: off_hi });
        let zero = self.c(0);
        let a = self.add64(a, Pair { lo: so, hi: zero });
        let mask = self.c(!3);
        let lo = self.u(op::BITWISE_AND, &[a.lo, mask]);
        let addr = Pair { lo, hi: a.hi };
        for k in 0..n as u32 {
            let kc = self.c(4 * k);
            let zero = self.c(0);
            let at = self.add64(addr, Pair { lo: kc, hi: zero });
            let v = self.load_window(buf, p_u32, win, at);
            self.set_s32(sdata + k as u8, v)?;
        }
        Ok(())
    }

    /// ds_swizzle_b32: the source lane from this lane's id (as
    /// rdna::ds::swizzle_source), read with a shuffle; no LDS access.
    fn ds_swizzle(&mut self, offset: u32, addr: u8, vdst: u8) -> Res<()> {
        if ds::swizzle_source(offset, 0).is_none() {
            return Err(self.err("ds_swizzle mode"));
        }
        let l = self.lane;
        let idx = if offset & 0xFF00 == 0x8000 {
            let (k3, nk3) = (self.c(3), self.c(!3));
            let q = self.u(op::BITWISE_AND, &[l, k3]);
            let mut sel = self.c(offset & 3);
            for i in 1..4u32 {
                let ic = self.c(i);
                let is = self.bl(op::I_EQUAL, &[q, ic]);
                let v = self.c(offset >> (2 * i) & 3);
                sel = self.select_u(is, v, sel);
            }
            let base = self.u(op::BITWISE_AND, &[l, nk3]);
            self.u(op::BITWISE_OR, &[base, sel])
        } else {
            let (and, or, xor) = (offset & 31, offset >> 5 & 31, offset >> 10 & 31);
            let (ka, ko, kx, n31) = (self.c(and), self.c(or), self.c(xor), self.c(!31));
            let t = self.u(op::BITWISE_AND, &[l, ka]);
            let t = self.u(op::BITWISE_OR, &[t, ko]);
            let t = self.u(op::BITWISE_XOR, &[t, kx]);
            let base = self.u(op::BITWISE_AND, &[l, n31]);
            self.u(op::BITWISE_OR, &[base, t])
        };
        let v = self.vsrc(256 + addr as u16, 0)?;
        let r = self.lane_value(v, idx);
        self.set_vgpr(vdst, r)
    }

    /// ds_append / ds_consume (wave32): the first active lane adds or
    /// subtracts the active-lane count at M0 + offset with one atomic, and
    /// its pre-op value goes to every lane. An out-of-range address hits
    /// the leader's sink word and returns 0.
    fn ds_append(&mut self, consume: bool, off: u32, vdst: u8) -> Res<()> {
        if self.wave64 {
            return Err(self.err("ds_append/consume in wave64"));
        }
        let (arr, ptr, words) = self.lds.ok_or_else(|| self.err("no LDS allocated"))?;
        let exec = self.load_u(self.exec_lo);
        let n = self.u(op::BIT_COUNT, &[exec]);
        let leader = self.ctz(exec);
        let is_leader = self.bl(op::I_EQUAL, &[self.lane, leader]);
        let m0 = self.load_u(self.m0);
        let oc = self.c(off);
        let byte = self.u(op::I_ADD, &[m0, oc]);
        let two = self.c(2);
        let idx = self.u(op::SHIFT_RIGHT_LOGICAL, &[byte, two]);
        let wc = self.c(words);
        let inr = self.bl(op::U_LESS_THAN, &[idx, wc]);
        let ok = self.bl(op::LOGICAL_AND, &[inr, is_leader]);
        let sink = self.u(op::I_ADD, &[wc, self.lane]);
        let slot = self.select_u(ok, idx, sink);
        let p = self.b.op(ptr, op::ACCESS_CHAIN, &[arr, slot]);
        let (sc, relaxed) = (self.c(scope::WORKGROUP), self.c(0));
        let aop = if consume { op::ATOMIC_I_SUB } else { op::ATOMIC_I_ADD };
        let r = self.u(aop, &[p, sc, relaxed, n]);
        let zero = self.c(0);
        self.wave_barrier();
        let r = self.select_u(ok, r, zero);
        let k31 = self.c(31);
        let src = self.u(op::BITWISE_AND, &[leader, k31]);
        let all = self.lane_value(r, src);
        self.set_vgpr(vdst, all)
    }

    /// LDS atomic as `OpAtomic*` (Workgroup scope, relaxed) on the LDS
    /// array; inactive lanes and out-of-range addresses hit the lane's
    /// sink word and return 0. A barrier follows, as for other DS ops.
    #[allow(clippy::too_many_arguments)]
    fn ds_atomic(&mut self, base: u16, rtn: bool, off: u32, addr: u8, data0: u8, data1: u8, vdst: u8) -> Res<()> {
        let (arr, ptr, words) = self.lds.ok_or_else(|| self.err("no LDS allocated"))?;
        let a = self.vsrc(256 + addr as u16, 0)?;
        let act = self.active();
        let wc = self.c(words);
        let sink = self.u(op::I_ADD, &[wc, self.lane]);
        let oc = self.c(off);
        let byte = self.u(op::I_ADD, &[a, oc]);
        let two = self.c(2);
        let idx = self.u(op::SHIFT_RIGHT_LOGICAL, &[byte, two]);
        let inr = self.bl(op::U_LESS_THAN, &[idx, wc]);
        let ok = self.bl(op::LOGICAL_AND, &[inr, act]);
        let slot = self.select_u(ok, idx, sink);
        let p = self.b.op(ptr, op::ACCESS_CHAIN, &[arr, slot]);
        let (sc, relaxed) = (self.c(scope::WORKGROUP), self.c(0));
        let d0 = self.vsrc(256 + data0 as u16, 0)?;
        let aop = match base {
            ds::ADD_U32 => op::ATOMIC_I_ADD,
            ds::SUB_U32 => op::ATOMIC_I_SUB,
            ds::MIN_I32 => op::ATOMIC_S_MIN,
            ds::MAX_I32 => op::ATOMIC_S_MAX,
            ds::MIN_U32 => op::ATOMIC_U_MIN,
            ds::MAX_U32 => op::ATOMIC_U_MAX,
            ds::AND_B32 => op::ATOMIC_AND,
            ds::OR_B32 => op::ATOMIC_OR,
            ds::XOR_B32 => op::ATOMIC_XOR,
            ds::WRXCHG_RTN_B32 => op::ATOMIC_EXCHANGE,
            _ => op::ATOMIC_COMPARE_EXCHANGE,
        };
        let r = if aop == op::ATOMIC_COMPARE_EXCHANGE {
            // ds_cmpst: data0 = comparand, data1 = new value.
            let d1 = self.vsrc(256 + data1 as u16, 0)?;
            self.u(aop, &[p, sc, relaxed, relaxed, d1, d0])
        } else {
            self.u(aop, &[p, sc, relaxed, d0])
        };
        self.wave_barrier();
        if rtn {
            let zero = self.c(0);
            let r = self.select_u(ok, r, zero);
            self.set_vgpr(vdst, r)?;
        }
        Ok(())
    }

    /// LDS access; a workgroup barrier follows every DS instruction, so
    /// other lanes' later accesses see it (within one wave the hardware
    /// orders LDS operations; in SPIR-V that needs a barrier).
    #[allow(clippy::too_many_arguments)]
    fn ds(&mut self, opc: u16, offset0: u8, offset1: u8, addr: u8, data0: u8, data1: u8, vdst: u8) -> Res<()> {
        let (arr, ptr, words) = self.lds.ok_or_else(|| self.err("no LDS allocated"))?;
        let off16 = offset0 as u32 | (offset1 as u32) << 8;
        let (offs, write): (Vec<u32>, bool) = match opc {
            ds::WRITE_B32 => (vec![off16], true),
            ds::WRITE_B64 => (vec![off16, off16 + 4], true),
            ds::READ_B32 => (vec![off16], false),
            ds::READ_B64 => (vec![off16, off16 + 4], false),
            ds::READ2_B32 => (vec![offset0 as u32 * 4, offset1 as u32 * 4], false),
            ds::READ2ST64_B32 => (vec![offset0 as u32 * 256, offset1 as u32 * 256], false),
            ds::WRITE2_B32 => (vec![offset0 as u32 * 4, offset1 as u32 * 4], true),
            ds::WRITE2ST64_B32 => (vec![offset0 as u32 * 256, offset1 as u32 * 256], true),
            _ => return Err(self.err("opcode")),
        };
        let two_regs = matches!(opc, ds::WRITE2_B32 | ds::WRITE2ST64_B32);
        let a = self.vsrc(256 + addr as u16, 0)?;
        let act = self.active();
        let wc = self.c(words);
        let sink = self.u(op::I_ADD, &[wc, self.lane]);
        let zero = self.c(0);
        let mut loaded = Vec::new();
        for (k, o) in offs.iter().enumerate() {
            let oc = self.c(*o);
            let byte = self.u(op::I_ADD, &[a, oc]);
            let two = self.c(2);
            let idx = self.u(op::SHIFT_RIGHT_LOGICAL, &[byte, two]);
            let inr = self.bl(op::U_LESS_THAN, &[idx, wc]);
            let ok = self.bl(op::LOGICAL_AND, &[inr, act]);
            let slot = self.select_u(ok, idx, sink);
            let p = self.b.op(ptr, op::ACCESS_CHAIN, &[arr, slot]);
            if write {
                let r = if two_regs { [data0, data1][k] } else { data0 + k as u8 };
                let v = self.vsrc(256 + r as u16, 0)?;
                self.b.store(p, v);
            } else {
                let v = self.load_u(p);
                loaded.push(self.select_u(ok, v, zero));
            }
        }
        self.wave_barrier();
        for (k, v) in loaded.into_iter().enumerate() {
            self.set_vgpr(vdst + k as u8, v)?;
        }
        Ok(())
    }

    /// `image_load` / `image_store` on a 2D linear image through the
    /// memory window, with
    /// the descriptor read at run time (see `interp::image_step` for the
    /// rules). What the interpreter refuses at run time (unsupported
    /// format, tiling, mip level, custom pitch, out-of-bounds coordinates)
    /// reads 0 here.
    fn image_load(&mut self, inst: &Inst) -> Res<()> {
        let Fields::Mimg {
            vdata,
            vaddr,
            srsrc,
            dmask,
            dim,
            ref nsa_addrs,
            ..
        } = inst.fields
        else {
            unreachable!()
        };
        let (w0, w1) = (inst.words[0], inst.words[1]);
        let store = inst.opcode == mimg::IMAGE_STORE;
        if !(inst.opcode == mimg::IMAGE_LOAD || store) || dim != mimg::DIM_2D {
            return Err(self.err("only 2D image_load / image_store"));
        }
        if w0 & (mimg::TFE | mimg::LWE | mimg::R128) != 0 || w1 & (mimg::A16 | mimg::D16) != 0 {
            return Err(self.err("tfe/lwe/r128/a16/d16"));
        }
        let (buf, p_u32, win) = self.mem.ok_or_else(|| self.err("no memory window bound"))?;
        let (ax, ay) = match nsa_addrs.first() {
            None => (vaddr, vaddr + 1),
            Some(&y) => (vaddr, y),
        };
        let mut d = [0; 8];
        for (k, x) in d.iter_mut().enumerate() {
            *x = self.s32(srsrc as u16 + k as u16, 0)?;
        }
        let (zero, one) = (self.c(0), self.c(1));
        let bits = |tr: &mut Self, w: Id, sh: u32, mask: u32| {
            let (s, m) = (tr.c(sh), tr.c(mask));
            let x = tr.u(op::SHIFT_RIGHT_LOGICAL, &[w, s]);
            tr.u(op::BITWISE_AND, &[x, m])
        };
        let eq = |tr: &mut Self, a: Id, v: u32| {
            let c = tr.c(v);
            tr.bl(op::I_EQUAL, &[a, c])
        };
        // base = WORD0 << 8 | WORD1[7:0] << 40
        let k24 = self.c(24);
        let k8 = self.c(8);
        let lo = self.u(op::SHIFT_LEFT_LOGICAL, &[d[0], k8]);
        let top = self.u(op::SHIFT_RIGHT_LOGICAL, &[d[0], k24]);
        let b8 = bits(self, d[1], 0, 0xFF);
        let b8 = self.u(op::SHIFT_LEFT_LOGICAL, &[b8, k8]);
        let hi = self.u(op::BITWISE_OR, &[top, b8]);
        let base = Pair { lo, hi };
        let fmt = bits(self, d[1], 20, 0x1FF);
        let wlo = bits(self, d[1], 30, 3);
        let whi = bits(self, d[2], 0, 0xFFF);
        let two = self.c(2);
        let whi = self.u(op::SHIFT_LEFT_LOGICAL, &[whi, two]);
        let wm1 = self.u(op::BITWISE_OR, &[wlo, whi]);
        let width = self.u(op::I_ADD, &[wm1, one]);
        let hm1 = bits(self, d[2], 14, 0x3FFF);
        let height = self.u(op::I_ADD, &[hm1, one]);
        let base_level = bits(self, d[3], 12, 0xF);
        let sw_mode = bits(self, d[3], 20, 0x1F);
        let depth = bits(self, d[4], 0, 0x1FFF);
        // Format: bytes, channels, float.
        let is4u = {
            let a = eq(self, fmt, 20);
            let b = eq(self, fmt, 21);
            self.bl(op::LOGICAL_OR, &[a, b])
        };
        let is4f = eq(self, fmt, 22);
        let is8 = eq(self, fmt, 62);
        let is16 = eq(self, fmt, 77);
        let isu8 = eq(self, fmt, crate::rdna::FORMAT_8_8_8_8_UNORM);
        let is4f = self.bl(op::LOGICAL_OR, &[is4f, isu8]);
        let is4 = self.bl(op::LOGICAL_OR, &[is4u, is4f]);
        let known0 = self.bl(op::LOGICAL_OR, &[is4, is8]);
        let known = self.bl(op::LOGICAL_OR, &[known0, is16]);
        let float = self.bl(op::LOGICAL_OR, &[is4f, is16]);
        let (c4, c8, c16) = (self.c(4), self.c(8), self.c(16));
        let bytes0 = self.select_u(is8, c8, c4);
        let bytes = self.select_u(is16, c16, bytes0);
        let (ch1, ch2, ch4) = (self.c(1), self.c(2), self.c(4));
        let chans0 = self.select_u(is8, ch2, ch1);
        let four_ch = self.bl(op::LOGICAL_OR, &[is16, isu8]);
        let chans = self.select_u(four_ch, ch4, chans0);
        // pitch = align(width, 256 / bytes)
        let k256 = self.c(256);
        let a = self.u(op::U_DIV, &[k256, bytes]);
        let am1 = self.u(op::I_SUB, &[a, one]);
        let wa = self.u(op::I_ADD, &[width, am1]);
        let nam1 = self.u(op::NOT, &[am1]);
        let pitch = self.u(op::BITWISE_AND, &[wa, nam1]);
        let x = self.vsrc(256 + ax as u16, 0)?;
        let y = self.vsrc(256 + ay as u16, 0)?;
        let xin = self.bl(op::U_LESS_THAN, &[x, width]);
        let yin = self.bl(op::U_LESS_THAN, &[y, height]);
        let lin = eq(self, sw_mode, 0);
        let lvl0 = eq(self, base_level, 0);
        let dep0 = eq(self, depth, 0);
        // texel address = base + (y * pitch + x) * bytes when linear, else
        // the rdna::tiling offset for the (mode, element size) at hand
        // (offsets fit 32 bits for any image the window can hold).
        let row = self.u(op::I_MUL, &[y, pitch]);
        let idx = self.u(op::I_ADD, &[row, x]);
        let mut off = self.u(op::I_MUL, &[idx, bytes]);
        let last0 = {
            let l = bits(self, d[3], 16, 0xF);
            eq(self, l, 0)
        };
        let mut layout_ok = lin;
        for sw in [1u32, 5, 9] {
            for e in 2u32..=4 {
                let p = crate::rdna::tiling::pattern(sw, e).unwrap();
                let (bw, bh) = crate::rdna::tiling::block_dims(p);
                let m1 = eq(self, sw_mode, sw);
                let ec = self.c(1 << e);
                let m2 = self.bl(op::I_EQUAL, &[bytes, ec]);
                let hit = self.bl(op::LOGICAL_AND, &[m1, m2]);
                let mut inblk = zero;
                for (i, &(mx, my)) in p.iter().enumerate() {
                    if mx == 0 && my == 0 {
                        continue;
                    }
                    let (cx, cy) = (self.c(mx as u32), self.c(my as u32));
                    let a = self.u(op::BITWISE_AND, &[x, cx]);
                    let b = self.u(op::BITWISE_AND, &[y, cy]);
                    let v = self.u(op::BITWISE_XOR, &[a, b]);
                    let n = self.u(op::BIT_COUNT, &[v]);
                    let bit = self.u(op::BITWISE_AND, &[n, one]);
                    let sh = self.c(i as u32);
                    let bit = self.u(op::SHIFT_LEFT_LOGICAL, &[bit, sh]);
                    inblk = self.u(op::BITWISE_OR, &[inblk, bit]);
                }
                let (kbw, kbh, kbits) = (self.c(bw), self.c(bh), self.c(p.len() as u32));
                let wround = self.c((1 << bw) - 1);
                let wr = self.u(op::I_ADD, &[width, wround]);
                let pb = self.u(op::SHIFT_RIGHT_LOGICAL, &[wr, kbw]);
                let yb = self.u(op::SHIFT_RIGHT_LOGICAL, &[y, kbh]);
                let xb = self.u(op::SHIFT_RIGHT_LOGICAL, &[x, kbw]);
                let blk = self.u(op::I_MUL, &[yb, pb]);
                let blk = self.u(op::I_ADD, &[blk, xb]);
                let blk = self.u(op::SHIFT_LEFT_LOGICAL, &[blk, kbits]);
                let t = self.u(op::BITWISE_OR, &[blk, inblk]);
                off = self.select_u(hit, t, off);
                let tok = self.bl(op::LOGICAL_AND, &[hit, last0]);
                layout_ok = self.bl(op::LOGICAL_OR, &[layout_ok, tok]);
            }
        }
        let mut ok = known;
        for c in [xin, yin, layout_ok, lvl0, dep0] {
            ok = self.bl(op::LOGICAL_AND, &[ok, c]);
        }
        let addr = self.add64(base, Pair { lo: off, hi: zero });
        if store {
            // Accepted only like the interpreter: dmask = all channels,
            // identity DST_SEL; otherwise the store is dropped.
            let nu8 = self.bl(op::LOGICAL_NOT, &[isu8]);
            let mut fine = self.bl(op::LOGICAL_AND, &[ok, nu8]);
            let m = self.c(dmask as u32);
            let full = self.low_mask(chans);
            let dm = self.bl(op::I_EQUAL, &[m, full]);
            fine = self.bl(op::LOGICAL_AND, &[fine, dm]);
            for k in 0..4u32 {
                let sel = bits(self, d[3], 3 * k, 7);
                let want = eq(self, sel, 4 + k);
                let kc = self.c(k);
                let used = self.bl(op::U_LESS_THAN, &[kc, chans]);
                let nu = self.bl(op::LOGICAL_NOT, &[used]);
                let good = self.bl(op::LOGICAL_OR, &[nu, want]);
                fine = self.bl(op::LOGICAL_AND, &[fine, good]);
            }
            let active = self.active();
            let fine = self.bl(op::LOGICAL_AND, &[fine, active]);
            for k in 0..(dmask as u32).count_ones().min(4) {
                let kc = self.c(k);
                let used = self.bl(op::U_LESS_THAN, &[kc, chans]);
                let en = self.bl(op::LOGICAL_AND, &[fine, used]);
                let off = self.c(4 * k);
                let at = self.add64(addr, Pair { lo: off, hi: zero });
                let v = self.vsrc(256 + vdata as u16 + k as u16, 0)?;
                self.store_window(buf, p_u32, win, at, v, en);
            }
            return Ok(());
        }
        let mut texel = Vec::new();
        for k in 0..4u32 {
            let kc = self.c(4 * k);
            let at = self.add64(addr, Pair { lo: kc, hi: zero });
            let v = self.load_window(buf, p_u32, win, at);
            texel.push(v);
        }
        // UNORM8: four bytes of the first dword through the exact table.
        let (lut, p_lut) = self.unorm_lut.expect("table built when images are used");
        let packed = texel[0];
        for (k, t) in texel.iter_mut().enumerate() {
            let sh = self.c(8 * k as u32);
            let ff = self.c(0xFF);
            let byte = self.u(op::SHIFT_RIGHT_LOGICAL, &[packed, sh]);
            let byte = self.u(op::BITWISE_AND, &[byte, ff]);
            let p = self.b.op(p_lut, op::ACCESS_CHAIN, &[lut, byte]);
            let f = self.load_u(p);
            *t = self.select_u(isu8, f, *t);
        }
        let fone = self.c(1.0f32.to_bits());
        let one_v = self.select_u(float, fone, one);
        let mut k = 0u8;
        for ci in 0..4u32 {
            if dmask >> ci & 1 == 0 {
                continue;
            }
            let sel = bits(self, d[3], 3 * ci, 7);
            // sel: 0 -> 0, 1 -> one, 4..7 -> channel (missing: 0, 0, 0, one)
            let mut v = zero;
            let is1 = eq(self, sel, 1);
            v = self.select_u(is1, one_v, v);
            for ch in 0..4u32 {
                let is = eq(self, sel, 4 + ch);
                let chc = self.c(ch);
                let present = self.bl(op::U_LESS_THAN, &[chc, chans]);
                let missing = if ch == 3 { one_v } else { zero };
                let val = self.select_u(present, texel[ch as usize], missing);
                v = self.select_u(is, val, v);
            }
            let v = self.select_u(ok, v, zero);
            self.set_vgpr(vdata + k, v)?;
            k += 1;
        }
        Ok(())
    }

    /// `buffer_load/store_dword*` with the V# read at run time and the
    /// GFX10 OOB_SELECT range check (see `rdna::BufferDesc::in_bounds`);
    /// out of bounds loads read 0 and stores go to the sink. What the
    /// interpreter refuses (swizzle, ADD_TID, nonzero soffset, record
    /// offsets past 32 bits, unaligned or colliding accesses) is not
    /// distinguished here.
    #[allow(clippy::too_many_arguments)]
    fn mubuf(
        &mut self,
        opc: u16,
        vdata: u8,
        vaddr: u8,
        srsrc: u8,
        soffset: u8,
        offset: u16,
        offen: bool,
        idxen: bool,
        glc: bool,
    ) -> Res<()> {
        let atomic = atomic_op(opc);
        let sub = mubuf::sub_dword(opc);
        let (n, store) = match (atomic, sub) {
            (Some(_), _) => (1, true),
            (_, Some(_)) => (1, false),
            _ => mubuf::access(opc).ok_or_else(|| self.err("opcode"))?,
        };
        let (buf, p_u32, win) = self.mem.ok_or_else(|| self.err("no memory window bound"))?;
        let mut d = [0; 4];
        for (k, x) in d.iter_mut().enumerate() {
            *x = self.s32(srsrc as u16 + k as u16, 0)?;
        }
        let bits = |tr: &mut Self, w: Id, sh: u32, mask: u32| {
            let (s, m) = (tr.c(sh), tr.c(mask));
            let x = tr.u(op::SHIFT_RIGHT_LOGICAL, &[w, s]);
            tr.u(op::BITWISE_AND, &[x, m])
        };
        let base = Pair {
            lo: d[0],
            hi: bits(self, d[1], 0, 0xFFFF),
        };
        let stride = bits(self, d[1], 16, 0x3FFF);
        let nr = d[2];
        let oob = bits(self, d[3], 28, 3);
        let so = self.s32(soffset as u16, 0)?;
        let zero = self.c(0);
        let index = if idxen { self.vsrc(256 + vaddr as u16, 0)? } else { zero };
        let voff = if offen {
            self.vsrc(256 + vaddr as u16 + idxen as u16, 0)?
        } else {
            zero
        };
        let ic = self.c(offset as u32);
        let rec_off = self.u(op::I_ADD, &[ic, voff]);
        let idx_ok = self.bl(op::U_LESS_THAN, &[index, nr]);
        let nr_nz = self.bl(op::I_NOT_EQUAL, &[nr, zero]);
        let row = self.u(op::I_MUL, &[index, stride]);
        let active = self.active();
        let sel = |tr: &mut Self, v: u32| {
            let c = tr.c(v);
            tr.bl(op::I_EQUAL, &[oob, c])
        };
        let (is0, is1, is2) = (sel(self, 0), sel(self, 1), sel(self, 2));
        for k in 0..n as u32 {
            let kc = self.c(4 * k);
            let o = self.u(op::I_ADD, &[rec_off, kc]);
            let o_stride = self.bl(op::U_LESS_THAN, &[o, stride]);
            let o_nr = self.bl(op::U_LESS_THAN, &[o, nr]);
            let ok0 = self.bl(op::LOGICAL_AND, &[idx_ok, o_stride]);
            let mut ok = o_nr;
            ok = self.select_b(is2, nr_nz, ok);
            ok = self.select_b(is1, idx_ok, ok);
            ok = self.select_b(is0, ok0, ok);
            let rel = self.u(op::I_ADD, &[row, o]);
            let rel = self.u(op::I_ADD, &[rel, so]);
            let at = self.add64(base, Pair { lo: rel, hi: zero });
            if let Some((bytes, signed)) = sub {
                // The containing dword, shifted and extended (see global).
                let (k3, nk3, k8) = (self.c(3), self.c(!3), self.c(8));
                let lo = self.u(op::BITWISE_AND, &[at.lo, nk3]);
                let dw = self.load_window(buf, p_u32, win, Pair { lo, hi: at.hi });
                let byte = self.u(op::BITWISE_AND, &[at.lo, k3]);
                let sh = self.u(op::I_MUL, &[byte, k8]);
                let x = self.u(op::SHIFT_RIGHT_LOGICAL, &[dw, sh]);
                let up = self.c(32 - 8 * bytes);
                let t = self.u(op::SHIFT_LEFT_LOGICAL, &[x, up]);
                let r = self.u(if signed { op::SHIFT_RIGHT_ARITHMETIC } else { op::SHIFT_RIGHT_LOGICAL }, &[t, up]);
                let r = self.select_u(ok, r, zero);
                self.set_vgpr(vdata, r)?;
                continue;
            }
            if let Some(aop) = atomic {
                let (widx, inwin) = self.window_index(win, at);
                let en = self.bl(op::LOGICAL_AND, &[ok, active]);
                let en = self.bl(op::LOGICAL_AND, &[en, inwin]);
                let sink = self.c(win.words);
                let slot = self.select_u(en, widx, sink);
                let p = self.b.op(p_u32, op::ACCESS_CHAIN, &[buf, zero, slot]);
                let (sc, relaxed) = (self.c(1), self.c(0));
                let d0 = self.vsrc(256 + vdata as u16, 0)?;
                let r = if aop == op::ATOMIC_COMPARE_EXCHANGE {
                    let cmp = self.vsrc(256 + vdata as u16 + 1, 0)?;
                    self.u(aop, &[p, sc, relaxed, relaxed, d0, cmp])
                } else {
                    self.u(aop, &[p, sc, relaxed, d0])
                };
                if glc {
                    let r = self.select_u(en, r, zero);
                    self.set_vgpr(vdata, r)?;
                }
            } else if store {
                let en = self.bl(op::LOGICAL_AND, &[ok, active]);
                let v = self.vsrc(256 + vdata as u16 + k as u16, 0)?;
                self.store_window(buf, p_u32, win, at, v, en);
            } else {
                let v = self.load_window(buf, p_u32, win, at);
                let v = self.select_u(ok, v, zero);
                self.set_vgpr(vdata + k as u8, v)?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn global(
        &mut self,
        opc: u16,
        vdst: u8,
        addr: u8,
        data: u8,
        saddr: u8,
        offset: i16,
        seg: u8,
        glc: bool,
    ) -> Res<()> {
        if seg != global::SEG_GLOBAL {
            return Err(self.err("FLAT segment other than global"));
        }
        let atomic = atomic_op(opc);
        let sub = global::sub_dword(opc);
        let (n, store) = match (atomic, sub) {
            (Some(_), _) | (_, Some(_)) => (1, false),
            _ => global::access(opc).ok_or_else(|| self.err("opcode"))?,
        };
        let (buf, p_u32, win) = self.mem.ok_or_else(|| self.err("no memory window bound"))?;
        let zero = self.c(0);
        let base = if saddr == global::SADDR_OFF {
            Pair {
                lo: self.vsrc(256 + addr as u16, 0)?,
                hi: self.vsrc(256 + addr as u16 + 1, 0)?,
            }
        } else {
            let b = self.s64(saddr as u16, 0)?;
            let lo = self.vsrc(256 + addr as u16, 0)?;
            self.add64(b, Pair { lo, hi: zero })
        };
        let off_lo = self.c(offset as i32 as u32);
        let off_hi = self.c(if offset < 0 { u32::MAX } else { 0 });
        let a = self.add64(base, Pair { lo: off_lo, hi: off_hi });
        let active = self.active();
        if let Some((bytes, signed)) = sub {
            // The containing dword, then shift and zero/sign-extend (the
            // interpreter refuses misaligned addresses).
            let k3 = self.c(3);
            let nk3 = self.c(!3);
            let lo = self.u(op::BITWISE_AND, &[a.lo, nk3]);
            let dw = self.load_window(buf, p_u32, win, Pair { lo, hi: a.hi });
            let byte = self.u(op::BITWISE_AND, &[a.lo, k3]);
            let k8 = self.c(8);
            let sh = self.u(op::I_MUL, &[byte, k8]);
            let x = self.u(op::SHIFT_RIGHT_LOGICAL, &[dw, sh]);
            let up = self.c(32 - 8 * bytes);
            let t = self.u(op::SHIFT_LEFT_LOGICAL, &[x, up]);
            let r = self.u(if signed { op::SHIFT_RIGHT_ARITHMETIC } else { op::SHIFT_RIGHT_LOGICAL }, &[t, up]);
            return self.set_vgpr(vdst, r);
        }
        if let Some(aop) = atomic {
            // Inactive lanes and addresses outside the window operate on
            // the sink word; the returned value is then 0.
            let (idx, ok) = self.window_index(win, a);
            let ok = self.bl(op::LOGICAL_AND, &[ok, active]);
            let sink = self.c(win.words);
            let slot = self.select_u(ok, idx, sink);
            let p = self.b.op(p_u32, op::ACCESS_CHAIN, &[buf, zero, slot]);
            let (scope, relaxed) = (self.c(1), self.c(0)); // Device, relaxed
            let d0 = self.vsrc(256 + data as u16, 0)?;
            let r = if aop == op::ATOMIC_COMPARE_EXCHANGE {
                let cmp = self.vsrc(256 + data as u16 + 1, 0)?;
                self.u(aop, &[p, scope, relaxed, relaxed, d0, cmp])
            } else {
                self.u(aop, &[p, scope, relaxed, d0])
            };
            if glc {
                let r = self.select_u(ok, r, zero);
                self.set_vgpr(vdst, r)?;
            }
            return Ok(());
        }
        for k in 0..n as u32 {
            let kc = self.c(4 * k);
            let at = self.add64(a, Pair { lo: kc, hi: zero });
            if store {
                let v = self.vsrc(256 + data as u16 + k as u16, 0)?;
                self.store_window(buf, p_u32, win, at, v, active);
            } else {
                let v = self.load_window(buf, p_u32, win, at);
                self.set_vgpr(vdst + k as u8, v)?;
            }
        }
        Ok(())
    }

    /// Store `v` at byte address `a` if `enable` and inside the window;
    /// otherwise into the sink word after the window.
    fn store_window(&mut self, buf: Id, p_u32: Id, win: Window, a: Pair, v: Id, enable: Id) {
        let (idx, ok) = self.window_index(win, a);
        let ok = self.bl(op::LOGICAL_AND, &[ok, enable]);
        let sink = self.c(win.words);
        let slot = self.select_u(ok, idx, sink);
        let zero = self.c(0);
        let p = self.b.op(p_u32, op::ACCESS_CHAIN, &[buf, zero, slot]);
        self.b.store(p, v);
    }

    /// Word index of byte address `a` in the window, and whether it is
    /// inside.
    fn window_index(&mut self, win: Window, a: Pair) -> (Id, Id) {
        let zero = self.c(0);
        let wb = Pair {
            lo: self.c(win.base as u32),
            hi: self.c((win.base >> 32) as u32),
        };
        let d = self.sub64(a, wb);
        let two = self.c(2);
        let idx = self.u(op::SHIFT_RIGHT_LOGICAL, &[d.lo, two]);
        let words = self.c(win.words);
        let hi_ok = self.bl(op::I_EQUAL, &[d.hi, zero]);
        let lt = self.bl(op::U_LESS_THAN, &[idx, words]);
        let ok = self.bl(op::LOGICAL_AND, &[hi_ok, lt]);
        (idx, ok)
    }

    /// One dword of the memory window at byte address `a` (dword aligned),
    /// 0 outside the window.
    fn load_window(&mut self, buf: Id, p_u32: Id, win: Window, a: Pair) -> Id {
        let zero = self.c(0);
        let (idx, ok) = self.window_index(win, a);
        let safe = self.select_u(ok, idx, zero);
        let p = self.b.op(p_u32, op::ACCESS_CHAIN, &[buf, zero, safe]);
        let v = self.load_u(p);
        self.select_u(ok, v, zero)
    }

    /// `s_buffer_load_dword*` with raw-buffer semantics: dword `k` at
    /// `offset + soffset + 4k` (dword aligned) reads 0 unless it lies
    /// within NUM_RECORDS bytes. Strided, swizzled and non-raw descriptors
    /// are not distinguished here (the interpreter refuses them).
    #[allow(clippy::too_many_arguments)]
    fn s_buffer_load(
        &mut self,
        n: usize,
        sdata: u8,
        sbase: u8,
        offset: i32,
        soffset: u8,
        buf: Id,
        p_u32: Id,
        win: Window,
    ) -> Res<()> {
        let mut d = [0; 4];
        for (k, x) in d.iter_mut().enumerate() {
            *x = self.s32(sbase as u16 + k as u16, 0)?;
        }
        let k16 = self.c(0xFFFF);
        let base = Pair {
            lo: d[0],
            hi: self.u(op::BITWISE_AND, &[d[1], k16]),
        };
        let so = if soffset == smem::SOFFSET_NULL {
            self.c(0)
        } else {
            self.s32(soffset as u16, 0)?
        };
        let off = self.c(offset as u32);
        let start0 = self.u(op::I_ADD, &[off, so]);
        let mask = self.c(!3);
        let start = self.u(op::BITWISE_AND, &[start0, mask]);
        let (zero, four) = (self.c(0), self.c(4));
        let nr = d[2];
        let big = self.bl(op::U_GREATER_THAN_EQUAL, &[nr, four]);
        let limit = self.u(op::I_SUB, &[nr, four]);
        for k in 0..n as u32 {
            let kc = self.c(4 * k);
            let o = self.u(op::I_ADD, &[start, kc]);
            // o + 4 <= nr without overflow: nr >= 4 && o <= nr - 4; and o
            // must not have wrapped past 2^32.
            let le = self.bl(op::U_LESS_THAN_EQUAL, &[o, limit]);
            let no_wrap = self.bl(op::U_GREATER_THAN_EQUAL, &[o, start]);
            let in0 = self.bl(op::LOGICAL_AND, &[big, le]);
            let inb = self.bl(op::LOGICAL_AND, &[in0, no_wrap]);
            let at = self.add64(base, Pair { lo: o, hi: zero });
            let v = self.load_window(buf, p_u32, win, at);
            let r = self.select_u(inb, v, zero);
            self.set_s32(sdata + k as u8, r)?;
        }
        Ok(())
    }

    fn add64(&mut self, a: Pair, b: Pair) -> Pair {
        let lo = self.u(op::I_ADD, &[a.lo, b.lo]);
        let carry_b = self.bl(op::U_LESS_THAN, &[lo, a.lo]);
        let carry = self.bool_to_u(carry_b);
        let hi0 = self.u(op::I_ADD, &[a.hi, b.hi]);
        let hi = self.u(op::I_ADD, &[hi0, carry]);
        Pair { lo, hi }
    }

    fn sub64(&mut self, a: Pair, b: Pair) -> Pair {
        let lo = self.u(op::I_SUB, &[a.lo, b.lo]);
        let borrow_b = self.bl(op::U_LESS_THAN, &[a.lo, b.lo]);
        let borrow = self.bool_to_u(borrow_b);
        let hi0 = self.u(op::I_SUB, &[a.hi, b.hi]);
        let hi = self.u(op::I_SUB, &[hi0, borrow]);
        Pair { lo, hi }
    }

    /// 64-bit shift of a register pair by `b & 63`; mirrors
    /// `interp::shift64`. Every SPIR-V shift amount stays below 32.
    fn shift64(&mut self, opc: u16, a: Pair, b: Id) -> Pair {
        let (k63, k32, zero) = (self.c(63), self.c(32), self.c(0));
        let s = self.u(op::BITWISE_AND, &[b, k63]);
        let big = self.bl(op::U_GREATER_THAN_EQUAL, &[s, k32]);
        let is0 = self.bl(op::I_EQUAL, &[s, zero]);
        let lo_s = self.and31(s); // s mod 32
        let inv = self.u(op::I_SUB, &[k32, lo_s]);
        let inv = self.and31(inv); // 32 - s (only used when s mod 32 != 0)
        let small0 = self.bl(op::I_EQUAL, &[lo_s, zero]);
        match opc {
            sop2::S_LSHL_B64 => {
                let lo = self.u(op::SHIFT_LEFT_LOGICAL, &[a.lo, lo_s]);
                let hi = self.u(op::SHIFT_LEFT_LOGICAL, &[a.hi, lo_s]);
                let spill = self.u(op::SHIFT_RIGHT_LOGICAL, &[a.lo, inv]);
                let spill = self.select_u(small0, zero, spill);
                let hi = self.u(op::BITWISE_OR, &[hi, spill]);
                let big_hi = self.u(op::SHIFT_LEFT_LOGICAL, &[a.lo, lo_s]);
                let rlo = self.select_u(big, zero, lo);
                let rhi = self.select_u(big, big_hi, hi);
                Pair {
                    lo: self.select_u(is0, a.lo, rlo),
                    hi: self.select_u(is0, a.hi, rhi),
                }
            }
            _ => {
                let arith = opc == sop2::S_ASHR_I64;
                let sr = if arith { op::SHIFT_RIGHT_ARITHMETIC } else { op::SHIFT_RIGHT_LOGICAL };
                let lo = self.u(op::SHIFT_RIGHT_LOGICAL, &[a.lo, lo_s]);
                let spill = self.u(op::SHIFT_LEFT_LOGICAL, &[a.hi, inv]);
                let spill = self.select_u(small0, zero, spill);
                let lo = self.u(op::BITWISE_OR, &[lo, spill]);
                let hi = self.u(sr, &[a.hi, lo_s]);
                let big_lo = self.u(sr, &[a.hi, lo_s]);
                let fill = if arith {
                    let k31 = self.c(31);
                    self.u(op::SHIFT_RIGHT_ARITHMETIC, &[a.hi, k31])
                } else {
                    zero
                };
                let rlo = self.select_u(big, big_lo, lo);
                let rhi = self.select_u(big, fill, hi);
                Pair {
                    lo: self.select_u(is0, a.lo, rlo),
                    hi: self.select_u(is0, a.hi, rhi),
                }
            }
        }
    }

    fn sop1(&mut self, opc: u16, d: u8, s0: u16, lit: u32) -> Res<()> {
        match opc {
            sop1::S_GETPC_B64 => {
                // Known at translation time; see interp for the 2^47 limit.
                let pc = self.code_base.wrapping_add(4 * (self.at as u64 + 1));
                if pc >= 1 << 47 || pc < self.code_base {
                    return Err(self.err("PC at or above 2^47"));
                }
                let v = Pair {
                    lo: self.c(pc as u32),
                    hi: self.c((pc >> 32) as u32),
                };
                self.set_s64(d, v)
            }
            sop1::S_BITSET0_B32 | sop1::S_BITSET1_B32 => {
                let n = self.s32(s0, lit)?;
                let n = self.and31(n);
                let one = self.c(1);
                let bit = self.u(op::SHIFT_LEFT_LOGICAL, &[one, n]);
                let old = self.s32(d as u16, 0)?;
                let r = if opc == sop1::S_BITSET1_B32 {
                    self.u(op::BITWISE_OR, &[old, bit])
                } else {
                    let nb = self.u(op::NOT, &[bit]);
                    self.u(op::BITWISE_AND, &[old, nb])
                };
                self.set_s32(d, r)
            }
            sop1::S_MOV_B32 => {
                let v = self.s32(s0, lit)?;
                self.set_s32(d, v)
            }
            sop1::S_BITREPLICATE_B64_B32 => {
                // Spread each 16-bit half to the even bits, then double.
                let x = self.s32(s0, lit)?;
                let lo16 = self.c(0xFFFF);
                let l = self.u(op::BITWISE_AND, &[x, lo16]);
                let sixteen = self.c(16);
                let h = self.u(op::SHIFT_RIGHT_LOGICAL, &[x, sixteen]);
                let mut half = |h: Id| {
                    let mut v = h;
                    for (sh, m) in [(8, 0x00FF_00FF), (4, 0x0F0F_0F0F), (2, 0x3333_3333), (1, 0x5555_5555)] {
                        let k = self.c(sh);
                        let t = self.u(op::SHIFT_LEFT_LOGICAL, &[v, k]);
                        let o = self.u(op::BITWISE_OR, &[v, t]);
                        let mk = self.c(m);
                        v = self.u(op::BITWISE_AND, &[o, mk]);
                    }
                    let one = self.c(1);
                    let t = self.u(op::SHIFT_LEFT_LOGICAL, &[v, one]);
                    self.u(op::BITWISE_OR, &[v, t])
                };
                let lo = half(l);
                let hi = half(h);
                self.set_s64(d, Pair { lo, hi })
            }
            sop1::S_MOV_B64 => {
                let v = self.s64(s0, lit)?;
                self.set_s64(d, v)
            }
            sop1::S_FF1_I32_B32 => {
                let v = self.s32(s0, lit)?;
                let t = self.ctz(v);
                let k32 = self.c(32);
                let none = self.bl(op::I_EQUAL, &[t, k32]);
                let all = self.c(u32::MAX);
                let r = self.select_u(none, all, t);
                self.set_s32(d, r)
            }
            sop1::S_ABS_I32 => {
                let v = self.s32(s0, lit)?;
                let i = self.as_i32(v);
                let zero = self.b.constant_u32(0);
                let zi = self.as_i32(zero);
                let neg = self.bl(op::S_LESS_THAN, &[i, zi]);
                let n = self.u(op::I_SUB, &[zero, v]);
                let r = self.select_u(neg, n, v);
                let s = self.nonzero(r);
                self.b.store(self.scc, s);
                self.set_s32(d, r)
            }
            sop1::S_AND_SAVEEXEC_B64
            | sop1::S_AND_SAVEEXEC_B32
            | sop1::S_OR_SAVEEXEC_B64
            | sop1::S_OR_SAVEEXEC_B32
            | sop1::S_ANDN2_SAVEEXEC_B64
            | sop1::S_ANDN2_SAVEEXEC_B32 => {
                let wide = matches!(
                    opc,
                    sop1::S_AND_SAVEEXEC_B64 | sop1::S_OR_SAVEEXEC_B64 | sop1::S_ANDN2_SAVEEXEC_B64
                );
                if wide != self.wave64 {
                    return Err(self.err("saveexec width does not match the wave size"));
                }
                let src = self.mask(s0, lit)?;
                let old = Pair {
                    lo: self.load_u(self.exec_lo),
                    hi: self.load_u(self.exec_hi),
                };
                self.set_mask(d, old)?;
                let comb = |tr: &mut Self, a: Id, e: Id| match opc {
                    sop1::S_OR_SAVEEXEC_B64 | sop1::S_OR_SAVEEXEC_B32 => tr.u(op::BITWISE_OR, &[a, e]),
                    sop1::S_ANDN2_SAVEEXEC_B64 | sop1::S_ANDN2_SAVEEXEC_B32 => {
                        let ne = tr.u(op::NOT, &[e]);
                        tr.u(op::BITWISE_AND, &[a, ne])
                    }
                    _ => tr.u(op::BITWISE_AND, &[a, e]),
                };
                let lo = comb(self, src.lo, old.lo);
                self.b.store(self.exec_lo, lo);
                let any = if self.wave64 {
                    let hi = comb(self, src.hi, old.hi);
                    self.b.store(self.exec_hi, hi);
                    self.u(op::BITWISE_OR, &[lo, hi])
                } else {
                    lo
                };
                let s = self.nonzero(any);
                self.b.store(self.scc, s);
                Ok(())
            }
            sop1::S_NOT_B64 => {
                let a = self.s64(s0, lit)?;
                let lo = self.u(op::NOT, &[a.lo]);
                let hi = self.u(op::NOT, &[a.hi]);
                let any = self.u(op::BITWISE_OR, &[lo, hi]);
                let s = self.nonzero(any);
                self.b.store(self.scc, s);
                self.set_s64(d, Pair { lo, hi })
            }
            sop1::S_CMOV_B32 => {
                let v = self.s32(s0, lit)?;
                let old = self.s32(d as u16, 0)?;
                let sc = self.b.load(self.t.boolt, self.scc);
                let r = self.select_u(sc, v, old);
                self.set_s32(d, r)
            }
            o if crate::interp::sop1_b32(o, 0).is_some() => {
                let a = self.s32(s0, lit)?;
                let (r, scc) = match o {
                    sop1::S_NOT_B32 => (self.u(op::NOT, &[a]), true),
                    sop1::S_BREV_B32 => (self.vop1_int(vop1::V_BFREV_B32, a).unwrap(), false),
                    sop1::S_BCNT1_I32_B32 => (self.u(op::BIT_COUNT, &[a]), true),
                    sop1::S_BCNT0_I32_B32 => {
                        let n = self.u(op::BIT_COUNT, &[a]);
                        let k32 = self.c(32);
                        (self.u(op::I_SUB, &[k32, n]), true)
                    }
                    sop1::S_FLBIT_I32_B32 => (self.vop1_int(vop1::V_FFBH_U32, a).unwrap(), false),
                    sop1::S_FLBIT_I32 => (self.vop1_int(vop1::V_FFBH_I32, a).unwrap(), false),
                    _ => {
                        // sext from 8 or 16 bits: shl then ashr
                        let n = self.c(if o == sop1::S_SEXT_I32_I8 { 24 } else { 16 });
                        let x = self.u(op::SHIFT_LEFT_LOGICAL, &[a, n]);
                        let xi = self.as_i32(x);
                        let t = self.t.i32t;
                        let y = self.b.op(t, op::SHIFT_RIGHT_ARITHMETIC, &[xi, n]);
                        (self.bits_of(y), false)
                    }
                };
                if scc {
                    let s = self.nonzero(r);
                    self.b.store(self.scc, s);
                }
                self.set_s32(d, r)
            }
            _ => Err(self.err("opcode")),
        }
    }

    fn sop2(&mut self, opc: u16, d: u8, s0: u16, s1: u16, lit: u32) -> Res<()> {
        if crate::interp::sop2_b64(opc, 0, 0).is_some() {
            let a = self.s64(s0, lit)?;
            let b = self.s64(s1, lit)?;
            let f = |tr: &mut Self, x: Id, y: Id| match opc {
                sop2::S_AND_B64 => tr.u(op::BITWISE_AND, &[x, y]),
                sop2::S_OR_B64 => tr.u(op::BITWISE_OR, &[x, y]),
                sop2::S_XOR_B64 => tr.u(op::BITWISE_XOR, &[x, y]),
                _ => {
                    let ny = tr.u(op::NOT, &[y]);
                    tr.u(op::BITWISE_AND, &[x, ny])
                }
            };
            let lo = f(self, a.lo, b.lo);
            let hi = f(self, a.hi, b.hi);
            let any = self.u(op::BITWISE_OR, &[lo, hi]);
            let s = self.nonzero(any);
            self.b.store(self.scc, s);
            return self.set_s64(d, Pair { lo, hi });
        }
        if crate::interp::shift64(opc, 0, 0).is_some() {
            let a = self.s64(s0, lit)?;
            let b = self.s32(s1, lit)?;
            let r = self.shift64(opc, a, b);
            let any = self.u(op::BITWISE_OR, &[r.lo, r.hi]);
            let s = self.nonzero(any);
            self.b.store(self.scc, s);
            return self.set_s64(d, r);
        }
        if opc == sop2::S_MUL_HI_U32 || opc == sop2::S_MUL_HI_I32 {
            let a = self.s32(s0, lit)?;
            let b = self.s32(s1, lit)?;
            let v = if opc == sop2::S_MUL_HI_U32 { vop3::V_MUL_HI_U32 } else { vop3::V_MUL_HI_I32 };
            let r = self.vop3_int(v, a, b, a).unwrap();
            return self.set_s32(d, r);
        }
        if opc == sop2::S_CSELECT_B64 {
            let a = self.s64(s0, lit)?;
            let b = self.s64(s1, lit)?;
            let sc = self.b.load(self.t.boolt, self.scc);
            let lo = self.select_u(sc, a.lo, b.lo);
            let hi = self.select_u(sc, a.hi, b.hi);
            return self.set_s64(d, Pair { lo, hi });
        }
        let a = self.s32(s0, lit)?;
        let b = self.s32(s1, lit)?;
        if crate::interp::sop2_b32(opc, 0, 0).is_some() && opc != sop2::S_AND_B32 {
            let not = |tr: &mut Self, x: Id| tr.u(op::NOT, &[x]);
            let (r, scc) = match opc {
                sop2::S_OR_B32 => (self.u(op::BITWISE_OR, &[a, b]), true),
                sop2::S_XOR_B32 => (self.u(op::BITWISE_XOR, &[a, b]), true),
                sop2::S_ANDN2_B32 => {
                    let nb = not(self, b);
                    (self.u(op::BITWISE_AND, &[a, nb]), true)
                }
                sop2::S_ORN2_B32 => {
                    let nb = not(self, b);
                    (self.u(op::BITWISE_OR, &[a, nb]), true)
                }
                sop2::S_NAND_B32 => {
                    let x = self.u(op::BITWISE_AND, &[a, b]);
                    (not(self, x), true)
                }
                sop2::S_NOR_B32 => {
                    let x = self.u(op::BITWISE_OR, &[a, b]);
                    (not(self, x), true)
                }
                sop2::S_XNOR_B32 => {
                    let x = self.u(op::BITWISE_XOR, &[a, b]);
                    (not(self, x), true)
                }
                sop2::S_LSHL_B32 => (self.vop2_int(vop2::V_LSHLREV_B32, b, a).unwrap(), true),
                sop2::S_LSHR_B32 => (self.vop2_int(vop2::V_LSHRREV_B32, b, a).unwrap(), true),
                sop2::S_ASHR_I32 => (self.vop2_int(vop2::V_ASHRREV_I32, b, a).unwrap(), true),
                sop2::S_MUL_I32 => (self.u(op::I_MUL, &[a, b]), false),
                sop2::S_BFM_B32 => {
                    let n = self.and31(a);
                    let m = self.low_mask(n);
                    let sh = self.and31(b);
                    (self.u(op::SHIFT_LEFT_LOGICAL, &[m, sh]), false)
                }
                _ => {
                    // s_bfe_u32 / s_bfe_i32: S1 = offset[4:0] | width[22:16]
                    let (k16, k7f) = (self.c(16), self.c(0x7F));
                    let off = self.and31(b);
                    let wd = self.u(op::SHIFT_RIGHT_LOGICAL, &[b, k16]);
                    let wd = self.u(op::BITWISE_AND, &[wd, k7f]);
                    let x = self.u(op::SHIFT_RIGHT_LOGICAL, &[a, off]);
                    let m = self.low_mask(wd);
                    let x = self.u(op::BITWISE_AND, &[x, m]);
                    let r = if opc == sop2::S_BFE_I32 {
                        // sign-extend from bit width - 1 (width in 1..31)
                        let k32 = self.c(32);
                        let zero = self.c(0);
                        let wz = self.bl(op::I_EQUAL, &[wd, zero]);
                        let sh = self.u(op::I_SUB, &[k32, wd]);
                        let sh = self.and31(sh);
                        let y = self.u(op::SHIFT_LEFT_LOGICAL, &[x, sh]);
                        let yi = self.as_i32(y);
                        let t = self.t.i32t;
                        let z = self.b.op(t, op::SHIFT_RIGHT_ARITHMETIC, &[yi, sh]);
                        let z = self.bits_of(z);
                        self.select_u(wz, zero, z)
                    } else {
                        x
                    };
                    (r, true)
                }
            };
            if scc {
                let s = self.nonzero(r);
                self.b.store(self.scc, s);
            }
            return self.set_s32(d, r);
        }
        let (r, scc) = match opc {
            sop2::S_ADD_U32 => {
                let r = self.u(op::I_ADD, &[a, b]);
                let c = self.bl(op::U_LESS_THAN, &[r, a]);
                (r, Some(c))
            }
            sop2::S_SUB_U32 => {
                let r = self.u(op::I_SUB, &[a, b]);
                let c = self.bl(op::U_LESS_THAN, &[a, b]);
                (r, Some(c))
            }
            sop2::S_ADD_I32 => {
                // overflow <=> a and b have the same sign and r differs
                let r = self.u(op::I_ADD, &[a, b]);
                let x = self.u(op::BITWISE_XOR, &[a, r]);
                let y = self.u(op::BITWISE_XOR, &[b, r]);
                let both = self.u(op::BITWISE_AND, &[x, y]);
                let k31 = self.c(31);
                let sign = self.u(op::SHIFT_RIGHT_LOGICAL, &[both, k31]);
                let c = self.nonzero(sign);
                (r, Some(c))
            }
            sop2::S_SUB_I32 => {
                // overflow <=> a and b have different signs and r's sign differs from a's
                let r = self.u(op::I_SUB, &[a, b]);
                let x = self.u(op::BITWISE_XOR, &[a, b]);
                let y = self.u(op::BITWISE_XOR, &[a, r]);
                let both = self.u(op::BITWISE_AND, &[x, y]);
                let k31 = self.c(31);
                let sign = self.u(op::SHIFT_RIGHT_LOGICAL, &[both, k31]);
                let c = self.nonzero(sign);
                (r, Some(c))
            }
            sop2::S_SUBB_U32 => {
                let sc = self.b.load(self.t.boolt, self.scc);
                let bi = self.bool_to_u(sc);
                let ab = self.u(op::I_SUB, &[a, b]);
                let r = self.u(op::I_SUB, &[ab, bi]);
                // borrow <=> a < b, or a == b with a borrow in
                let lt = self.bl(op::U_LESS_THAN, &[a, b]);
                let eq = self.bl(op::I_EQUAL, &[a, b]);
                let full = self.bl(op::LOGICAL_AND, &[eq, sc]);
                let c = self.bl(op::LOGICAL_OR, &[lt, full]);
                (r, Some(c))
            }
            sop2::S_ADDC_U32 => {
                let sc = self.b.load(self.t.boolt, self.scc);
                let ci = self.bool_to_u(sc);
                let ab = self.u(op::I_ADD, &[a, b]);
                let r = self.u(op::I_ADD, &[ab, ci]);
                // carry <=> r < a, or r == a with a carry in (b + c wrapped
                // a full turn)
                let lt = self.bl(op::U_LESS_THAN, &[r, a]);
                let eq = self.bl(op::I_EQUAL, &[r, a]);
                let full = self.bl(op::LOGICAL_AND, &[eq, sc]);
                let c = self.bl(op::LOGICAL_OR, &[lt, full]);
                (r, Some(c))
            }
            sop2::S_CSELECT_B32 => {
                let sc = self.b.load(self.t.boolt, self.scc);
                (self.select_u(sc, a, b), None)
            }
            sop2::S_AND_B32 => {
                let r = self.u(op::BITWISE_AND, &[a, b]);
                let c = self.nonzero(r);
                (r, Some(c))
            }
            _ => return Err(self.err("opcode")),
        };
        if let Some(c) = scc {
            self.b.store(self.scc, c);
        }
        self.set_s32(d, r)
    }

    /// f32 bits -> f16 bits as `interp::f32_to_f16` for every input it
    /// accepts (normal range, zero, infinity).
    fn f32_to_f16(&mut self, x: Id, rtz: bool) -> Id {
        let c = |tr: &mut Self, v: u32| tr.c(v);
        let (k16, k23, k13) = (c(self, 16), c(self, 23), c(self, 13));
        let (m8000, mff, mm, mrem) = (c(self, 0x8000), c(self, 0xFF), c(self, 0x7F_FFFF), c(self, 0x1FFF));
        let t = self.u(op::SHIFT_RIGHT_LOGICAL, &[x, k16]);
        let s = self.u(op::BITWISE_AND, &[t, m8000]);
        let t = self.u(op::SHIFT_RIGHT_LOGICAL, &[x, k23]);
        let e = self.u(op::BITWISE_AND, &[t, mff]);
        let m = self.u(op::BITWISE_AND, &[x, mm]);
        let k112 = c(self, 112);
        let eb = self.u(op::I_SUB, &[e, k112]);
        let k10 = c(self, 10);
        let eh = self.u(op::SHIFT_LEFT_LOGICAL, &[eb, k10]);
        let mh = self.u(op::SHIFT_RIGHT_LOGICAL, &[m, k13]);
        let mut h = self.u(op::BITWISE_OR, &[eh, mh]);
        if !rtz {
            let rem = self.u(op::BITWISE_AND, &[m, mrem]);
            let half = c(self, 0x1000);
            let gt = self.bl(op::U_GREATER_THAN, &[rem, half]);
            let eqh = self.bl(op::I_EQUAL, &[rem, half]);
            let one = c(self, 1);
            let odd = self.u(op::BITWISE_AND, &[h, one]);
            let oddb = self.bl(op::I_EQUAL, &[odd, one]);
            let tie = self.bl(op::LOGICAL_AND, &[eqh, oddb]);
            let up = self.bl(op::LOGICAL_OR, &[gt, tie]);
            let inc = self.bool_to_u(up);
            h = self.u(op::I_ADD, &[h, inc]);
        }
        let normal = self.u(op::BITWISE_OR, &[s, h]);
        let k7c = c(self, 0x7C00);
        let inf = self.u(op::BITWISE_OR, &[s, k7c]);
        let is_inf = self.bl(op::I_EQUAL, &[e, mff]);
        let zero = c(self, 0);
        let mag_mask = c(self, 0x7FFF_FFFF);
        let mag = self.u(op::BITWISE_AND, &[x, mag_mask]);
        let is_zero = self.bl(op::I_EQUAL, &[mag, zero]);
        let r = self.select_u(is_inf, inf, normal);
        self.select_u(is_zero, s, r)
    }

    fn vop1(&mut self, opc: u16, vdst: u8, src0: u16, lit: u32) -> Res<()> {
        match opc {
            vop1::V_CVT_F16_F32 => {
                // GFX10 keeps D[31:16] (see rdna::vop1::V_CVT_F16_F32).
                let a = self.vsrc(src0, lit)?;
                let h = self.f32_to_f16(a, false);
                let old = self.vsrc(256 + vdst as u16, 0)?;
                let km = self.c(0xFFFF_0000);
                let hi = self.u(op::BITWISE_AND, &[old, km]);
                let r = self.u(op::BITWISE_OR, &[hi, h]);
                self.set_vgpr(vdst, r)
            }
            vop1::V_CVT_F32_F16 => {
                // As interp::f16_to_f32 for normals, zeros and infinities.
                let a = self.vsrc(src0, lit)?;
                let c = |tr: &mut Self, v: u32| tr.c(v);
                let (k16, k10, k13, k23) = (c(self, 16), c(self, 10), c(self, 13), c(self, 23));
                let (m8000, m1f, m3ff) = (c(self, 0x8000), c(self, 0x1F), c(self, 0x3FF));
                let sv = self.u(op::BITWISE_AND, &[a, m8000]);
                let s = self.u(op::SHIFT_LEFT_LOGICAL, &[sv, k16]);
                let t = self.u(op::SHIFT_RIGHT_LOGICAL, &[a, k10]);
                let e = self.u(op::BITWISE_AND, &[t, m1f]);
                let m = self.u(op::BITWISE_AND, &[a, m3ff]);
                let k112 = c(self, 112);
                let ef = self.u(op::I_ADD, &[e, k112]);
                let ef = self.u(op::SHIFT_LEFT_LOGICAL, &[ef, k23]);
                let mf = self.u(op::SHIFT_LEFT_LOGICAL, &[m, k13]);
                let n = self.u(op::BITWISE_OR, &[ef, mf]);
                let normal = self.u(op::BITWISE_OR, &[s, n]);
                let kinf = c(self, 0x7F80_0000);
                let inf = self.u(op::BITWISE_OR, &[s, kinf]);
                let zero = c(self, 0);
                let is_zero = self.bl(op::I_EQUAL, &[e, zero]);
                let is_inf = self.bl(op::I_EQUAL, &[e, m1f]);
                let r = self.select_u(is_inf, inf, normal);
                let r = self.select_u(is_zero, s, r);
                self.set_vgpr(vdst, r)
            }
            vop1::V_NOP => Ok(()),
            vop1::V_MOV_B32 => {
                let a = self.vsrc(src0, lit)?;
                self.set_vgpr(vdst, a)
            }
            vop1::V_CVT_F32_I32 => {
                let a = self.vsrc(src0, lit)?;
                let i = self.as_i32(a);
                let f = self.t.f32t;
                let x = self.b.op(f, op::CONVERT_S_TO_F, &[i]);
                let r = self.bits_of(x);
                self.set_vgpr(vdst, r)
            }
            vop1::V_CVT_I32_F64 | vop1::V_CVT_U32_F64 => {
                // saturate as the hardware does: clamp on the bit pattern
                // first, since SPIR-V leaves out-of-range conversions undefined
                let p = self.src64(src0, lit)?;
                let signed = opc == vop1::V_CVT_I32_F64;
                let (lo_d, hi_d, lo_v, hi_v) = if signed {
                    (-2_147_483_648.0f64, 2_147_483_648.0f64, i32::MIN as u32, i32::MAX as u32)
                } else {
                    (0.0, 4_294_967_296.0, 0, u32::MAX)
                };
                let pair = |t: &mut Self, d: f64| Pair { lo: t.c(d.to_bits() as u32), hi: t.c((d.to_bits() >> 32) as u32) };
                let (lo_p, hi_p) = (pair(self, lo_d), pair(self, hi_d));
                let above = self.compare64(6, p, hi_p);
                let in_low = self.compare64(4, p, lo_p);
                let x = self.as_f64(p);
                let conv = if signed {
                    let i = self.t.i32t;
                    let v = self.b.op(i, op::CONVERT_F_TO_S, &[x]);
                    self.bits_of(v)
                } else {
                    self.u(op::CONVERT_F_TO_U, &[x])
                };
                let (lo_vc, hi_vc, zero) = (self.c(lo_v), self.c(hi_v), self.c(0));
                let nan = self.nan64(p);
                let low = self.select_u(nan, zero, lo_vc);
                let r0 = self.select_u(in_low, conv, low);
                let r = self.select_u(above, hi_vc, r0);
                self.set_vgpr(vdst, r)
            }
            vop1::V_CVT_F32_F64 => {
                let p = self.src64(src0, lit)?;
                let x = self.as_f64(p);
                let f = self.t.f32t;
                let v = self.b.op(f, op::F_CONVERT, &[x]);
                let r = self.bits_of(v);
                self.set_vgpr(vdst, r)
            }
            vop1::V_CVT_F64_I32 | vop1::V_CVT_F64_U32 | vop1::V_CVT_F64_F32 => {
                let a = self.vsrc(src0, lit)?;
                let d = self.f64t();
                let x = match opc {
                    vop1::V_CVT_F64_I32 => {
                        let i = self.as_i32(a);
                        self.b.op(d, op::CONVERT_S_TO_F, &[i])
                    }
                    vop1::V_CVT_F64_U32 => self.b.op(d, op::CONVERT_U_TO_F, &[a]),
                    _ => {
                        let fa = self.as_f32(a);
                        self.b.op(d, op::F_CONVERT, &[fa])
                    }
                };
                let r = self.f64_bits(x);
                self.set_vgpr(vdst, r.lo)?;
                self.set_vgpr(vdst + 1, r.hi)
            }
            vop1::V_TRUNC_F64..=vop1::V_FLOOR_F64 => {
                let x = self.src64(src0, lit)?;
                let r = self.round_f64(opc, x);
                self.set_vgpr(vdst, r.lo)?;
                self.set_vgpr(vdst + 1, r.hi)
            }
            o if crate::interp::vop1_float(o, 0).is_some() => {
                let a = self.vsrc(src0, lit)?;
                let f = self.t.f32t;
                let r = if (vop1::V_CVT_F32_UBYTE0..=vop1::V_CVT_F32_UBYTE3).contains(&o) {
                    let sh = self.c(8 * (o - vop1::V_CVT_F32_UBYTE0) as u32);
                    let ff = self.c(0xFF);
                    let x = self.u(op::SHIFT_RIGHT_LOGICAL, &[a, sh]);
                    let x = self.u(op::BITWISE_AND, &[x, ff]);
                    let v = self.b.op(f, op::CONVERT_U_TO_F, &[x]);
                    self.bits_of(v)
                } else {
                    let which = match o {
                        vop1::V_FLOOR_F32 => glsl::FLOOR,
                        vop1::V_CEIL_F32 => glsl::CEIL,
                        vop1::V_TRUNC_F32 => glsl::TRUNC,
                        _ => glsl::ROUND_EVEN,
                    };
                    let set = self.glsl_set();
                    let x = self.as_f32(a);
                    let v = self.b.op(f, op::EXT_INST, &[set, which, x]);
                    self.bits_of(v)
                };
                self.set_vgpr(vdst, r)
            }
            vop1::V_CVT_F32_U32 => {
                let a = self.vsrc(src0, lit)?;
                let f = self.t.f32t;
                let x = self.b.op(f, op::CONVERT_U_TO_F, &[a]);
                let r = self.bits_of(x);
                self.set_vgpr(vdst, r)
            }
            vop1::V_CVT_RPI_I32_F32 | vop1::V_CVT_FLR_I32_F32 => {
                // in range only (the interpreter refuses the rest)
                let a = self.vsrc(src0, lit)?;
                let f = self.t.f32t;
                let mut x = self.as_f32(a);
                if opc == vop1::V_CVT_RPI_I32_F32 {
                    let half = self.b.constant_f32(0.5);
                    x = self.b.op(f, op::F_ADD, &[x, half]);
                }
                let set = self.glsl_set();
                let y = self.b.op(f, op::EXT_INST, &[set, glsl::FLOOR, x]);
                let i = self.t.i32t;
                let v = self.b.op(i, op::CONVERT_F_TO_S, &[y]);
                let r = self.bits_of(v);
                self.set_vgpr(vdst, r)
            }
            vop1::V_FREXP_EXP_I32_F32 | vop1::V_FREXP_MANT_F32 => {
                // on the bit pattern: a normal value keeps sign and
                // mantissa with exponent -1 and reports e - 126; zero,
                // Inf and NaN give exponent 0 and themselves
                let a = self.vsrc(src0, lit)?;
                let (k23, kff) = (self.c(23), self.c(0xFF));
                let e = self.u(op::SHIFT_RIGHT_LOGICAL, &[a, k23]);
                let e = self.u(op::BITWISE_AND, &[e, kff]);
                let (zero, k255) = (self.c(0), self.c(255));
                let nz = self.bl(op::I_NOT_EQUAL, &[e, zero]);
                let ni = self.bl(op::I_NOT_EQUAL, &[e, k255]);
                let normal = self.and_b(nz, ni);
                let r = if opc == vop1::V_FREXP_EXP_I32_F32 {
                    let k126 = self.c(126);
                    let x = self.u(op::I_SUB, &[e, k126]);
                    self.select_u(normal, x, zero)
                } else {
                    let keep = self.c(0x807F_FFFF);
                    let x = self.u(op::BITWISE_AND, &[a, keep]);
                    let half = self.c(126 << 23);
                    let x = self.u(op::BITWISE_OR, &[x, half]);
                    self.select_u(normal, x, a)
                };
                self.set_vgpr(vdst, r)
            }
            vop1::V_CVT_U32_F32 | vop1::V_CVT_I32_F32 => {
                // The hardware saturates and maps NaN to 0; SPIR-V leaves
                // out-of-range conversions undefined, so clamp first.
                let a = self.vsrc(src0, lit)?;
                let x = self.as_f32(a);
                let signed = opc == vop1::V_CVT_I32_F32;
                let (lo_f, hi_f, lo_v, hi_v) = if signed {
                    (-2_147_483_648.0f32, 2_147_483_648.0f32, i32::MIN as u32, i32::MAX as u32)
                } else {
                    (0.0, 4_294_967_296.0, 0, u32::MAX)
                };
                let lo_c = self.b.constant_f32(lo_f);
                let hi_c = self.b.constant_f32(hi_f);
                let above = self.bl(op::F_ORD_GREATER_THAN_EQUAL, &[x, hi_c]);
                // Ordered: false for NaN, so NaN takes the "not in range" path.
                let in_low = self.bl(op::F_ORD_GREATER_THAN, &[x, lo_c]);
                let conv = if signed {
                    let i = self.t.i32t;
                    let v = self.b.op(i, op::CONVERT_F_TO_S, &[x]);
                    self.bits_of(v)
                } else {
                    self.u(op::CONVERT_F_TO_U, &[x])
                };
                let (lo_vc, hi_vc, zero) = (self.c(lo_v), self.c(hi_v), self.c(0));
                // x <= lo: lo value, except NaN -> 0.
                let is_num = self.bl(op::F_ORD_EQUAL, &[x, x]);
                let low = self.select_u(is_num, lo_vc, zero);
                let r0 = self.select_u(in_low, conv, low);
                let r = self.select_u(above, hi_vc, r0);
                self.set_vgpr(vdst, r)
            }
            o if matches!(
                o,
                vop1::V_NOT_B32 | vop1::V_BFREV_B32 | vop1::V_FFBH_U32 | vop1::V_FFBH_I32 | vop1::V_FFBL_B32
            ) =>
            {
                let a = self.vsrc(src0, lit)?;
                let r = self.vop1_int(o, a).unwrap();
                self.set_vgpr(vdst, r)
            }
            vop1::V_READFIRSTLANE_B32 => {
                // First active lane, or lane 0 when exec is empty.
                let lo = self.load_u(self.exec_lo);
                let t_lo = self.ctz(lo);
                let idx = if self.wave64 {
                    let hi = self.load_u(self.exec_hi);
                    let t_hi = self.ctz(hi);
                    let k32 = self.c(32);
                    let hi_idx = self.u(op::I_ADD, &[t_hi, k32]);
                    let lo_empty = self.bl(op::I_EQUAL, &[t_lo, k32]);
                    let t = self.select_u(lo_empty, hi_idx, t_lo);
                    let k64 = self.c(64);
                    let none = self.bl(op::I_EQUAL, &[t, k64]);
                    let zero = self.c(0);
                    self.select_u(none, zero, t)
                } else {
                    let k32 = self.c(32);
                    let none = self.bl(op::I_EQUAL, &[t_lo, k32]);
                    let zero = self.c(0);
                    self.select_u(none, zero, t_lo)
                };
                let v = self.vsrc(src0, lit)?;
                let r = if self.xlane.is_some() {
                    let (arr, ptr) = self.publish(v);
                    let p = self.b.op(ptr, op::ACCESS_CHAIN, &[arr, idx]);
                    let r = self.load_u(p);
                    self.barrier();
                    r
                } else {
                    let u = self.t.u32t;
                    self.b
                        .op(u, op::GROUP_NON_UNIFORM_SHUFFLE, &[self.subgroup_scope, v, idx])
                };
                self.set_s32(vdst, r)
            }
            _ => Err(self.err("opcode")),
        }
    }

    fn vop3(&mut self, inst: &Inst, lit: u32) -> Res<()> {
        let &Fields::Vop3 {
            vdst,
            src,
            abs,
            neg,
            clamp,
            omod,
            op_sel,
        } = &inst.fields
        else {
            unreachable!()
        };
        let opc = inst.opcode;
        if opc == vop3::V_MAD_U64_U32 {
            if clamp || omod != 0 || neg != 0 {
                return Err(self.err("clamp/omod/neg on v_mad_u64_u32"));
            }
            let sdst = (inst.words[0] >> 8 & 0x7F) as u8;
            let a = self.vsrc(src[0], lit)?;
            let b = self.vsrc(src[1], lit)?;
            let addend = self.src64(src[2], lit)?;
            let lo = self.u(op::I_MUL, &[a, b]);
            let hi = self.vop3_int(vop3::V_MUL_HI_U32, a, b, a).expect("mul_hi");
            let s_lo = self.u(op::I_ADD, &[lo, addend.lo]);
            let c0 = self.bl(op::U_LESS_THAN, &[s_lo, lo]);
            let h1 = self.u(op::I_ADD, &[hi, addend.hi]);
            let c1 = self.bl(op::U_LESS_THAN, &[h1, hi]);
            let ci = self.bool_to_u(c0);
            let s_hi = self.u(op::I_ADD, &[h1, ci]);
            let c2 = self.bl(op::U_LESS_THAN, &[s_hi, h1]);
            let co = self.bl(op::LOGICAL_OR, &[c1, c2]);
            let act = self.active();
            let p = self.bl(op::LOGICAL_AND, &[act, co]);
            let m = self.ballot(p);
            self.set_vgpr(vdst, s_lo)?;
            self.set_vgpr(vdst + 1, s_hi)?;
            return self.set_mask(sdst, m);
        }
        let opc = inst.opcode;
        if crate::interp::carry_op(opc, 0, 0, false).is_some() {
            // VOP3b: dword0 [14:8] is the carry-out SGPR.
            if clamp || omod != 0 || neg != 0 {
                return Err(self.err("clamp/omod/neg on a carry op"));
            }
            let sdst = (inst.words[0] >> 8 & 0x7F) as u8;
            let cin = (vop3::FROM_VOP2..vop3::FROM_VOP1).contains(&opc).then_some(src[2]);
            return self.carry(opc, vdst, src[0], src[1], cin, sdst, lit);
        }
        if let Some(cmp) = crate::interp::Compare::decode(opc) {
            if clamp || omod != 0 || op_sel != 0 {
                return Err(self.err("clamp/omod/op_sel"));
            }
            if cmp.cmpx {
                return Err(self.err("v_cmpx e64"));
            }
            if cmp.kind == crate::interp::CmpKind::F64 {
                return Err(self.err("f64 compare e64"));
            }
            if (abs != 0 || neg != 0) && cmp.kind != crate::interp::CmpKind::F32 {
                return Err(self.err("source modifiers on an integer compare"));
            }
            let mut ab = [self.vsrc(src[0], lit)?, self.vsrc(src[1], lit)?];
            for (i, x) in ab.iter_mut().enumerate() {
                if abs >> i & 1 != 0 {
                    let m = self.c(0x7FFF_FFFF);
                    *x = self.u(op::BITWISE_AND, &[*x, m]);
                }
                if neg >> i & 1 != 0 {
                    let m = self.c(0x8000_0000);
                    *x = self.u(op::BITWISE_XOR, &[*x, m]);
                }
            }
            let cond = self.compare_with(cmp, ab[0], ab[1]);
            let act = self.active();
            let p = self.bl(op::LOGICAL_AND, &[act, cond]);
            let m = self.ballot(p);
            return self.set_mask(vdst, m);
        }
        if matches!(opc, vop3::V_MIN_F64 | vop3::V_MAX_F64) {
            if clamp || omod != 0 || op_sel != 0 {
                return Err(self.err("clamp/omod/op_sel"));
            }
            let mut x = Vec::new();
            for (i, &s) in src.iter().enumerate().take(2) {
                let mut p = self.src64(s, lit)?;
                if abs >> i & 1 != 0 {
                    let m = self.c(0x7FFF_FFFF);
                    p.hi = self.u(op::BITWISE_AND, &[p.hi, m]);
                }
                if neg >> i & 1 != 0 {
                    let m = self.c(0x8000_0000);
                    p.hi = self.u(op::BITWISE_XOR, &[p.hi, m]);
                }
                x.push(p);
            }
            let (a, b) = (x[0], x[1]);
            // order by the sign-magnitude key (-0 < +0); a quiet NaN gives
            // the other operand
            let (ka, kb) = (self.key64(a), self.key64(b));
            let pick_a = if opc == vop3::V_MAX_F64 { self.ult64(kb, ka) } else { self.ult64(ka, kb) };
            let (na, nb) = (self.nan64(a), self.nan64(b));
            let nnb = self.not_b(nb);
            let pa = self.and_b(pick_a, nnb);
            let pa = self.bl(op::LOGICAL_OR, &[pa, nb]);
            let nna = self.not_b(na);
            let pa = self.and_b(pa, nna);
            let lo = self.select_u(pa, a.lo, b.lo);
            let hi = self.select_u(pa, a.hi, b.hi);
            self.set_vgpr(vdst, lo)?;
            return self.set_vgpr(vdst + 1, hi);
        }
        if matches!(opc, vop3::V_ADD_F64 | vop3::V_MUL_F64 | vop3::V_FMA_F64) {
            if clamp || omod != 0 || op_sel != 0 {
                return Err(self.err("clamp/omod/op_sel"));
            }
            let n = if opc == vop3::V_FMA_F64 { 3 } else { 2 };
            let mut x = Vec::new();
            for (i, &s) in src.iter().enumerate().take(n) {
                let mut p = self.src64(s, lit)?;
                if abs >> i & 1 != 0 {
                    let m = self.c(0x7FFF_FFFF);
                    p.hi = self.u(op::BITWISE_AND, &[p.hi, m]);
                }
                if neg >> i & 1 != 0 {
                    let m = self.c(0x8000_0000);
                    p.hi = self.u(op::BITWISE_XOR, &[p.hi, m]);
                }
                x.push(self.as_f64(p));
            }
            let d = self.f64t();
            let r = match opc {
                vop3::V_ADD_F64 => self.b.op(d, op::F_ADD, &[x[0], x[1]]),
                vop3::V_MUL_F64 => self.b.op(d, op::F_MUL, &[x[0], x[1]]),
                _ => {
                    let set = self.glsl_set();
                    self.b.op(d, op::EXT_INST, &[set, glsl::FMA, x[0], x[1], x[2]])
                }
            };
            let r = self.f64_bits(r);
            self.set_vgpr(vdst, r.lo)?;
            return self.set_vgpr(vdst + 1, r.hi);
        }
        let float3 = opc == vop3::V_FMA_F32
            || opc == vop3::FROM_VOP2 + vop2::V_MIN_F32
            || opc == vop3::FROM_VOP2 + vop2::V_MAX_F32
            || opc == vop3::FROM_VOP2 + vop2::V_FMAC_F32
            || matches!(opc, vop3::V_MIN3_F32 | vop3::V_MAX3_F32 | vop3::V_MED3_F32);
        if float3 {
            if clamp || omod != 0 || op_sel != 0 {
                return Err(self.err("clamp/omod/op_sel"));
            }
            let mut x = [self.vsrc(src[0], lit)?, self.vsrc(src[1], lit)?, self.vsrc(src[2], lit)?];
            for (i, v) in x.iter_mut().enumerate() {
                if abs >> i & 1 != 0 {
                    let m = self.c(0x7FFF_FFFF);
                    *v = self.u(op::BITWISE_AND, &[*v, m]);
                }
                if neg >> i & 1 != 0 {
                    let m = self.c(0x8000_0000);
                    *v = self.u(op::BITWISE_XOR, &[*v, m]);
                }
            }
            let r = match opc {
                vop3::V_FMA_F32 => self.fma(x[0], x[1], x[2]),
                o if o == vop3::FROM_VOP2 + vop2::V_FMAC_F32 => {
                    let d = self.vsrc(256 + vdst as u16, 0)?;
                    self.fma(x[0], x[1], d)
                }
                vop3::V_MIN3_F32 | vop3::V_MAX3_F32 => {
                    let max = opc == vop3::V_MAX3_F32;
                    let t = self.min_max(max, x[0], x[1]);
                    self.min_max(max, t, x[2])
                }
                vop3::V_MED3_F32 => {
                    // fmed3AMDGCN: drop the operand equal to the max of all three
                    let ab = self.min_max(true, x[0], x[1]);
                    let m3 = self.min_max(true, ab, x[2]);
                    let bc = self.min_max(true, x[1], x[2]);
                    let ac = self.min_max(true, x[0], x[2]);
                    let is_a = self.bl(op::I_EQUAL, &[m3, x[0]]);
                    let is_b = self.bl(op::I_EQUAL, &[m3, x[1]]);
                    let r = self.select_u(is_b, ac, ab);
                    self.select_u(is_a, bc, r)
                }
                o => self.min_max(o == vop3::FROM_VOP2 + vop2::V_MAX_F32, x[0], x[1]),
            };
            return self.set_vgpr(vdst, r);
        }
        if clamp || omod != 0 || op_sel != 0 || abs != 0 {
            return Err(self.err("abs/clamp/omod/op_sel"));
        }
        match opc {
            vop3::V_BFE_U32
            | vop3::V_BFI_B32
            | vop3::V_MED3_U32
            | vop3::V_MUL_LO_U32
            | vop3::V_LSHL_ADD_U32
            | vop3::V_ADD3_U32
            | vop3::V_AND_OR_B32
            | vop3::V_OR3_B32
            | vop3::V_XOR3_B32
            | vop3::V_BFM_B32
            | vop3::V_ADD_LSHL_U32
            | vop3::V_MUL_HI_U32
            | vop3::V_MUL_LO_I32
            | vop3::V_MUL_HI_I32
            | vop3::V_MAD_I32_I24
            | vop3::V_MAD_U32_U24
            | vop3::V_BCNT_U32_B32
            | vop3::V_ALIGNBIT_B32
            | vop3::V_BFE_I32
            | vop3::V_MIN3_I32
            | vop3::V_MIN3_U32
            | vop3::V_MAX3_I32
            | vop3::V_MAX3_U32
            | vop3::V_MED3_I32
            | vop3::V_XAD_U32
            | vop3::V_PERM_B32
            | vop3::V_ALIGNBYTE_B32
            | vop3::V_CVT_PK_U16_U32
            | vop3::V_CVT_PK_I16_I32
            | vop3::V_MSAD_U8
            | vop3::V_LSHL_OR_B32 => {
                if neg != 0 {
                    return Err(self.err("neg on an integer operation"));
                }
                let a = self.vsrc(src[0], lit)?;
                let b = self.vsrc(src[1], lit)?;
                let c3 = self.vsrc(src[2], lit)?;
                let r = self.vop3_int(opc, a, b, c3).unwrap();
                self.set_vgpr(vdst, r)
            }
            vop3::V_ADD_NC_U16
            | vop3::V_SUB_NC_U16
            | vop3::V_MUL_LO_U16
            | vop3::V_LSHLREV_B16
            | vop3::V_LSHRREV_B16
            | vop3::V_ASHRREV_I16
            | vop3::V_MAX_U16
            | vop3::V_MAX_I16
            | vop3::V_MIN_U16
            | vop3::V_MIN_I16 => {
                if neg != 0 {
                    return Err(self.err("neg on an integer operation"));
                }
                // Mirrors interp::vop3_16 on the low halves (sign-extended
                // for the i16 ops), then D[31:16] kept.
                let x = self.vsrc(src[0], lit)?;
                let y = self.vsrc(src[1], lit)?;
                let k16 = self.c(16);
                let lo = |tr: &mut Self, v: Id, signed: bool| {
                    let t = tr.u(op::SHIFT_LEFT_LOGICAL, &[v, k16]);
                    tr.u(if signed { op::SHIFT_RIGHT_ARITHMETIC } else { op::SHIFT_RIGHT_LOGICAL }, &[t, k16])
                };
                let signed = matches!(opc, vop3::V_ASHRREV_I16 | vop3::V_MAX_I16 | vop3::V_MIN_I16);
                let (a, b) = (lo(self, x, signed), lo(self, y, signed));
                let k15 = self.c(15);
                let r = match opc {
                    vop3::V_ADD_NC_U16 => self.u(op::I_ADD, &[a, b]),
                    vop3::V_SUB_NC_U16 => self.u(op::I_SUB, &[a, b]),
                    vop3::V_MUL_LO_U16 => self.u(op::I_MUL, &[a, b]),
                    vop3::V_LSHLREV_B16 | vop3::V_LSHRREV_B16 | vop3::V_ASHRREV_I16 => {
                        let s = self.u(op::BITWISE_AND, &[a, k15]);
                        let sop = match opc {
                            vop3::V_LSHLREV_B16 => op::SHIFT_LEFT_LOGICAL,
                            vop3::V_LSHRREV_B16 => op::SHIFT_RIGHT_LOGICAL,
                            _ => op::SHIFT_RIGHT_ARITHMETIC,
                        };
                        self.u(sop, &[b, s])
                    }
                    vop3::V_MAX_U16 => self.pick(op::U_LESS_THAN, a, b, false, false),
                    vop3::V_MIN_U16 => self.pick(op::U_LESS_THAN, a, b, false, true),
                    vop3::V_MAX_I16 => self.pick(op::S_LESS_THAN, a, b, true, false),
                    _ => self.pick(op::S_LESS_THAN, a, b, true, true),
                };
                let m = self.c(0xFFFF);
                let r = self.u(op::BITWISE_AND, &[r, m]);
                let old = self.vsrc(256 + vdst as u16, 0)?;
                let km = self.c(0xFFFF_0000);
                let hi = self.u(op::BITWISE_AND, &[old, km]);
                let out = self.u(op::BITWISE_OR, &[hi, r]);
                self.set_vgpr(vdst, out)
            }
            vop3::V_LDEXP_F32 => {
                if neg != 0 {
                    return Err(self.err("neg on v_ldexp_f32"));
                }
                // add n to the exponent field of a normal value; overflow
                // gives ±Inf, zero/Inf/NaN pass through
                let a = self.vsrc(src[0], lit)?;
                let n = self.vsrc(src[1], lit)?;
                let (k23, kff) = (self.c(23), self.c(0xFF));
                let e = self.u(op::SHIFT_RIGHT_LOGICAL, &[a, k23]);
                let e = self.u(op::BITWISE_AND, &[e, kff]);
                let (zero, k255) = (self.c(0), self.c(255));
                let ez = self.bl(op::I_EQUAL, &[e, zero]);
                let ei = self.bl(op::I_EQUAL, &[e, k255]);
                let special = self.bl(op::LOGICAL_OR, &[ez, ei]);
                let (lo, hi) = (self.c((-512i32) as u32), self.c(512));
                let (ni, loi, hii) = (self.as_i32(n), self.as_i32(lo), self.as_i32(hi));
                let below = self.bl(op::S_LESS_THAN, &[ni, loi]);
                let above = self.bl(op::S_GREATER_THAN, &[ni, hii]);
                let nc = self.select_u(below, lo, n);
                let nc = self.select_u(above, hi, nc);
                let e2 = self.u(op::I_ADD, &[e, nc]);
                let (e2i, k255i, zi) = (self.as_i32(e2), self.as_i32(k255), self.as_i32(zero));
                let over = self.bl(op::S_GREATER_THAN_EQUAL, &[e2i, k255i]);
                let under = self.bl(op::S_LESS_THAN_EQUAL, &[e2i, zi]);
                let sb = self.c(0x8000_0000);
                let sign = self.u(op::BITWISE_AND, &[a, sb]);
                let infc = self.c(0x7F80_0000);
                let inf = self.u(op::BITWISE_OR, &[sign, infc]);
                let keep = self.c(0x807F_FFFF);
                let m = self.u(op::BITWISE_AND, &[a, keep]);
                let e2s = self.u(op::SHIFT_LEFT_LOGICAL, &[e2, k23]);
                let norm = self.u(op::BITWISE_OR, &[m, e2s]);
                let r = self.select_u(under, sign, norm);
                let r = self.select_u(over, inf, r);
                let r = self.select_u(special, a, r);
                self.set_vgpr(vdst, r)
            }
            vop3::V_READLANE_B32 | vop3::V_WRITELANE_B32 => {
                if neg != 0 {
                    return Err(self.err("neg on a lane access"));
                }
                // Lane select masked to the wave (the interpreter refuses
                // larger ones), so the shuffle index stays in range.
                let l = self.s32(src[1], lit)?;
                let m = self.c(if self.wave64 { 63 } else { 31 });
                let l = self.u(op::BITWISE_AND, &[l, m]);
                if opc == vop3::V_READLANE_B32 {
                    let v = self.vsrc(src[0], lit)?;
                    let r = self.lane_value(v, l);
                    self.set_s32(vdst, r)
                } else {
                    // Not predicated by exec.
                    let v = self.s32(src[0], lit)?;
                    let var = *self.vgpr.get(vdst as usize).ok_or_else(|| self.err("VGPR beyond num_vgprs"))?;
                    let old = self.load_u(var);
                    let me = self.bl(op::I_EQUAL, &[self.lane, l]);
                    let r = self.select_u(me, v, old);
                    self.b.store(var, r);
                    Ok(())
                }
            }
            vop3::V_LSHLREV_B64 | vop3::V_LSHRREV_B64 | vop3::V_ASHRREV_I64 => {
                if neg != 0 {
                    return Err(self.err("neg on an integer operation"));
                }
                let s = self.vsrc(src[0], lit)?;
                let x = if src[1] >= 256 {
                    Pair {
                        lo: self.vsrc(src[1], lit)?,
                        hi: self.vsrc(src[1] + 1, lit)?,
                    }
                } else {
                    self.s64(src[1], lit)?
                };
                let sop = match opc {
                    vop3::V_LSHLREV_B64 => sop2::S_LSHL_B64,
                    vop3::V_LSHRREV_B64 => sop2::S_LSHR_B64,
                    _ => sop2::S_ASHR_I64,
                };
                let r = self.shift64(sop, x, s);
                self.set_vgpr(vdst, r.lo)?;
                self.set_vgpr(vdst + 1, r.hi)
            }
            vop3::V_MBCNT_LO_U32_B32 | vop3::V_MBCNT_HI_U32_B32 => {
                if neg != 0 {
                    return Err(self.err("neg on an integer operation"));
                }
                let a = self.vsrc(src[0], lit)?;
                let b = self.vsrc(src[1], lit)?;
                let below = self.lanes_below();
                let m = if opc == vop3::V_MBCNT_LO_U32_B32 {
                    below.lo
                } else {
                    below.hi
                };
                let x = self.u(op::BITWISE_AND, &[a, m]);
                let n = self.u(op::BIT_COUNT, &[x]);
                let r = self.u(op::I_ADD, &[n, b]);
                self.set_vgpr(vdst, r)
            }
            o if o == vop3::FROM_VOP2 + vop2::V_CNDMASK_B32 => {
                if neg != 0 {
                    return Err(self.err("neg on v_cndmask"));
                }
                let a = self.vsrc(src[0], lit)?;
                let b = self.vsrc(src[1], lit)?;
                let m = self.mask(src[2], lit)?;
                let r = self.cndmask(a, b, m);
                self.set_vgpr(vdst, r)
            }
            o if (vop3::FROM_VOP1..vop3::FROM_VOP1 + 0x80).contains(&o) => {
                if neg != 0 {
                    return Err(self.err("source modifiers"));
                }
                if o - vop3::FROM_VOP1 == vop1::V_READFIRSTLANE_B32 {
                    return Err(self.err("readfirstlane e64"));
                }
                self.vop1(o - vop3::FROM_VOP1, vdst, src[0], lit)
            }
            o => {
                let base = o.checked_sub(vop3::FROM_VOP2).filter(|x| *x < 0x80);
                if let Some(base) = base {
                    let a = self.vsrc(src[0], lit)?;
                    let b = self.vsrc(src[1], lit)?;
                    if neg == 0 {
                        if let Some(r) = self.vop2_int(base, a, b) {
                            return self.set_vgpr(vdst, r);
                        }
                    }
                }
                let spv = base
                    .and_then(Self::vop2_float)
                    .ok_or_else(|| self.err("opcode"))?;
                let a = self.vsrc(src[0], lit)?;
                let b = self.vsrc(src[1], lit)?;
                let r = self.fbin(spv, a, b, neg);
                self.set_vgpr(vdst, r)
            }
        }
    }
}

/// Basic blocks: (start dword, instructions), in address order.
fn blocks(insts: &[(usize, Inst)]) -> Vec<(usize, Vec<(usize, Inst)>)> {
    let mut starts = crate::rdna::block_starts(insts);
    for (at, i) in insts {
        // A block also ends after s_barrier, where a wave of a multi-wave
        // work-group stops until the others arrive.
        if i.is_endpgm() || (i.format == Format::Sopp && i.opcode == sopp::S_BARRIER) {
            starts.push(at + i.len());
        }
    }
    starts.sort_unstable();
    starts.dedup();
    let mut out: Vec<(usize, Vec<(usize, Inst)>)> = Vec::new();
    for (at, i) in insts {
        if out.is_empty() || starts.contains(at) {
            out.push((*at, Vec::new()));
        }
        out.last_mut().unwrap().1.push((*at, i.clone()));
    }
    out
}

/// Translate one shader to a GLCompute SPIR-V module.
/// SPIR-V atomic for a GLOBAL / MUBUF atomic opcode (same numbers on GFX10).
fn atomic_op(opc: u16) -> Option<u16> {
    Some(match opc {
        global::ATOMIC_SWAP => op::ATOMIC_EXCHANGE,
        global::ATOMIC_CMPSWAP => op::ATOMIC_COMPARE_EXCHANGE,
        global::ATOMIC_ADD => op::ATOMIC_I_ADD,
        global::ATOMIC_SUB => op::ATOMIC_I_SUB,
        global::ATOMIC_SMIN => op::ATOMIC_S_MIN,
        global::ATOMIC_UMIN => op::ATOMIC_U_MIN,
        global::ATOMIC_SMAX => op::ATOMIC_S_MAX,
        global::ATOMIC_UMAX => op::ATOMIC_U_MAX,
        global::ATOMIC_AND => op::ATOMIC_AND,
        global::ATOMIC_OR => op::ATOMIC_OR,
        global::ATOMIC_XOR => op::ATOMIC_XOR,
        _ => return None,
    })
}

pub fn translate(code: &[u32], opts: &Options) -> Res<Vec<u32>> {
    let wave64 = match opts.lanes {
        32 => false,
        64 => true,
        _ => return Err(TranslateError::BadOptions("lanes must be 32 or 64")),
    };
    if let Io::Vertex(_) = opts.io {
        if opts.cross_lane != CrossLane::Subgroup || opts.num_vgprs < 4 {
            return Err(TranslateError::BadOptions("vertex shaders need CrossLane::Subgroup and 4 VGPRs"));
        }
    }
    if let Io::Pixel(p) = opts.io {
        let pos_bits = 0xF << crate::rdna::ps_input::POS_X_FLOAT
            | 1 << crate::rdna::ps_input::PERSP_CENTER
            | 1 << crate::rdna::ps_input::FRONT_FACE;
        if p.ena & !pos_bits != 0 || p.ena & !p.addr != 0 {
            return Err(TranslateError::BadOptions(
                "pixel inputs other than POS_*_FLOAT/PERSP_CENTER/FRONT_FACE, or ENA outside ADDR",
            ));
        }
        if opts.cross_lane != CrossLane::Subgroup {
            return Err(TranslateError::BadOptions("pixel shaders need CrossLane::Subgroup"));
        }
    }
    if let Io::Compute(c) = opts.io {
        if c.threads() == 0 {
            return Err(TranslateError::BadOptions("empty work-group"));
        }
        if c.threads() > opts.lanes && opts.cross_lane != CrossLane::Subgroup {
            return Err(TranslateError::BadOptions(
                "multi-wave work-groups need CrossLane::Subgroup (one subgroup per wave)",
            ));
        }
        let needed = c.user_sgprs + c.tgid.iter().filter(|&&x| x).count() as u32 + c.tg_size as u32;
        if c.user_sgprs > MAX_USER_SGPRS || needed > opts.num_sgprs || opts.num_vgprs < 3 {
            return Err(TranslateError::BadOptions("not enough SGPRs/VGPRs for the compute inputs"));
        }
    }
    let insts = decode_shader(code)?;
    let blocks = blocks(&insts);
    let index_of = |at: usize| blocks.iter().position(|(s, _)| *s == at);
    let end_index = blocks.len() as u32;

    let mut b = Builder::new();
    b.capability(capability::SHADER);
    if opts.cross_lane == CrossLane::Subgroup {
        b.capability(capability::GROUP_NON_UNIFORM);
        b.capability(capability::GROUP_NON_UNIFORM_BALLOT);
        b.capability(capability::GROUP_NON_UNIFORM_SHUFFLE);
    }
    b.memory_model(addressing::LOGICAL, memory_model::GLSL450);

    let void = b.type_void();
    let u32t = b.type_u32();
    let i32t = b.type_int(32, true);
    let f32t = b.type_f32();
    let boolt = b.type_bool();
    let uvec3 = b.type_vector(u32t, 3);
    let uvec4 = b.type_vector(u32t, 4);
    let p_u32_priv = b.type_pointer(storage::PRIVATE, u32t);
    let p_bool_priv = b.type_pointer(storage::PRIVATE, boolt);
    let p_u32_sb = b.type_pointer(storage::STORAGE_BUFFER, u32t);
    let p_uvec3_in = b.type_pointer(storage::INPUT, uvec3);

    let arr = b.type_runtime_array(u32t);
    b.decorate(arr, decoration::ARRAY_STRIDE, &[4]);
    let block = b.type_struct(&[arr]);
    b.decorate(block, decoration::BLOCK, &[]);
    b.member_decorate(block, 0, decoration::OFFSET, &[0]);
    let p_block = b.type_pointer(storage::STORAGE_BUFFER, block);
    let regs_buf = b.global_variable(p_block, storage::STORAGE_BUFFER);
    b.decorate(regs_buf, decoration::DESCRIPTOR_SET, &[0]);
    b.decorate(regs_buf, decoration::BINDING, &[0]);
    b.name(regs_buf, "regs");
    let mem_var = opts.memory.map(|win| {
        let v = b.global_variable(p_block, storage::STORAGE_BUFFER);
        b.decorate(v, decoration::DESCRIPTOR_SET, &[0]);
        b.decorate(v, decoration::BINDING, &[1]);
        b.name(v, "memory");
        (v, p_u32_sb, win)
    });
    let gid = b.global_variable(p_uvec3_in, storage::INPUT);
    b.decorate(gid, decoration::BUILT_IN, &[builtin::GLOBAL_INVOCATION_ID]);
    let compute = match opts.io {
        Io::Compute(c) => Some(c),
        Io::RegisterDump | Io::Pixel(_) | Io::Vertex(_) => None,
    };
    let p_u32_in = b.type_pointer(storage::INPUT, u32t);
    let pixel_io = match opts.io {
        Io::Pixel(p) => Some(p),
        _ => None,
    };
    let builtin_var = |b: &mut Builder, ptr: Id, which: u32| {
        let v = b.global_variable(ptr, storage::INPUT);
        b.decorate(v, decoration::BUILT_IN, &[which]);
        v
    };
    let cs_vars = compute.map(|_| {
        let wg = builtin_var(&mut b, p_uvec3_in, builtin::WORKGROUP_ID);
        let lid = builtin_var(&mut b, p_uvec3_in, builtin::LOCAL_INVOCATION_ID);
        let lane = if opts.cross_lane == CrossLane::Subgroup {
            builtin_var(&mut b, p_u32_in, builtin::SUBGROUP_LOCAL_INVOCATION_ID)
        } else {
            builtin_var(&mut b, p_u32_in, builtin::LOCAL_INVOCATION_INDEX)
        };
        let n = b.constant_u32(MAX_USER_SGPRS);
        let arr = b.type_array(u32t, n);
        b.decorate(arr, decoration::ARRAY_STRIDE, &[4]);
        let blk = b.type_struct(&[arr]);
        b.decorate(blk, decoration::BLOCK, &[]);
        b.member_decorate(blk, 0, decoration::OFFSET, &[0]);
        let p_blk = b.type_pointer(storage::PUSH_CONSTANT, blk);
        let pc_var = b.global_variable(p_blk, storage::PUSH_CONSTANT);
        b.name(pc_var, "user_sgprs");
        let p_u32_pc = b.type_pointer(storage::PUSH_CONSTANT, u32t);
        (wg, lid, lane, pc_var, p_u32_pc)
    });
    // Pixel shader: FragCoord, subgroup lane id, MRT outputs, push constants.
    let px = pixel_io.map(|pio| {
        let f32t_ = f32t;
        let v4 = b.type_vector(f32t_, 4);
        let p_v4_in = b.type_pointer(storage::INPUT, v4);
        let p_v4_out = b.type_pointer(storage::OUTPUT, v4);
        let fc = b.global_variable(p_v4_in, storage::INPUT);
        b.decorate(fc, decoration::BUILT_IN, &[builtin::FRAG_COORD]);
        let lane = b.global_variable(p_u32_in, storage::INPUT);
        b.decorate(lane, decoration::BUILT_IN, &[builtin::SUBGROUP_LOCAL_INVOCATION_ID]);
        let mut outs = [None; 8];
        for (_, i) in &insts {
            if let Fields::Exp { target, .. } = i.fields {
                if (target as usize) < 8 && outs[target as usize].is_none() {
                    let o = b.global_variable(p_v4_out, storage::OUTPUT);
                    b.decorate(o, decoration::LOCATION, &[target as u32]);
                    outs[target as usize] = Some(o);
                }
            }
        }
        let n = b.constant_u32(MAX_USER_SGPRS);
        let arr = b.type_array(u32t, n);
        b.decorate(arr, decoration::ARRAY_STRIDE, &[4]);
        let blk = b.type_struct(&[arr]);
        b.decorate(blk, decoration::BLOCK, &[]);
        b.member_decorate(blk, 0, decoration::OFFSET, &[0]);
        let p_blk = b.type_pointer(storage::PUSH_CONSTANT, blk);
        let pc_var = b.global_variable(p_blk, storage::PUSH_CONSTANT);
        let p_u32_pc = b.type_pointer(storage::PUSH_CONSTANT, u32t);
        let exported = b.global_variable(p_bool_priv, storage::PRIVATE);
        b.name(exported, "exported");
        // Barycentrics and per-vertex attributes
        // (VK_KHR_fragment_shader_barycentric).
        let mut attrs: Vec<u8> = insts
            .iter()
            .filter_map(|(_, i)| match i.fields {
                Fields::Vintrp { attr, .. } => Some(attr),
                _ => None,
            })
            .collect();
        attrs.sort_unstable();
        attrs.dedup();
        let wants_bary = pio.ena >> crate::rdna::ps_input::PERSP_CENTER & 1 != 0;
        if wants_bary || !attrs.is_empty() {
            b.capability(capability::FRAGMENT_BARYCENTRIC_KHR);
            b.extension("SPV_KHR_fragment_shader_barycentric");
        }
        let bary = wants_bary.then(|| {
            let v3 = b.type_vector(f32t_, 3);
            let p = b.type_pointer(storage::INPUT, v3);
            let v = b.global_variable(p, storage::INPUT);
            b.decorate(v, decoration::BUILT_IN, &[builtin::BARY_COORD_KHR]);
            (v, v3)
        });
        let front = (pio.ena >> crate::rdna::ps_input::FRONT_FACE & 1 != 0).then(|| {
            let p = b.type_pointer(storage::INPUT, boolt);
            let v = b.global_variable(p, storage::INPUT);
            b.decorate(v, decoration::BUILT_IN, &[builtin::FRONT_FACING]);
            v
        });
        let three = b.constant_u32(3);
        let arr3 = b.type_array(v4, three);
        let p_arr3 = b.type_pointer(storage::INPUT, arr3);
        let attr_vars: Vec<(u8, Id)> = attrs
            .iter()
            .map(|&a| {
                let v = b.global_variable(p_arr3, storage::INPUT);
                b.decorate(v, decoration::LOCATION, &[a as u32]);
                b.decorate(v, decoration::PER_VERTEX_KHR, &[]);
                (a, v)
            })
            .collect();
        (pio, fc, lane, outs, pc_var, p_u32_pc, exported, v4, attr_vars, p_v4_in, bary, front)
    });
    // Vertex shader: draw parameters, lane id, Position and parameters.
    let vx = match opts.io {
        Io::Vertex(vio) => {
            b.capability(capability::DRAW_PARAMETERS);
            let mut bi = |which: u32| {
                let v = b.global_variable(p_u32_in, storage::INPUT);
                b.decorate(v, decoration::BUILT_IN, &[which]);
                v
            };
            let ins = [
                bi(builtin::VERTEX_INDEX),
                bi(builtin::BASE_VERTEX),
                bi(builtin::INSTANCE_INDEX),
                bi(builtin::BASE_INSTANCE),
                bi(builtin::SUBGROUP_LOCAL_INVOCATION_ID),
            ];
            let v4 = b.type_vector(f32t, 4);
            let p_v4_out = b.type_pointer(storage::OUTPUT, v4);
            let mut outs: Vec<(u8, Id)> = Vec::new();
            for (_, i) in &insts {
                if let Fields::Exp { target, .. } = i.fields {
                    if outs.iter().any(|(t, _)| *t == target) || target == crate::rdna::exp::NULL {
                        continue;
                    }
                    let o = b.global_variable(p_v4_out, storage::OUTPUT);
                    if target == crate::rdna::exp::POS0 {
                        b.decorate(o, decoration::BUILT_IN, &[builtin::POSITION]);
                    } else if (crate::rdna::exp::PARAM0..=crate::rdna::exp::PARAM_LAST).contains(&target) {
                        b.decorate(o, decoration::LOCATION, &[(target - crate::rdna::exp::PARAM0) as u32]);
                    } else {
                        continue; // refused when translated
                    }
                    outs.push((target, o));
                }
            }
            let n = b.constant_u32(MAX_USER_SGPRS);
            let arr = b.type_array(u32t, n);
            b.decorate(arr, decoration::ARRAY_STRIDE, &[4]);
            let blk = b.type_struct(&[arr]);
            b.decorate(blk, decoration::BLOCK, &[]);
            b.member_decorate(blk, 0, decoration::OFFSET, &[0]);
            let p_blk = b.type_pointer(storage::PUSH_CONSTANT, blk);
            let pc_var = b.global_variable(p_blk, storage::PUSH_CONSTANT);
            let p_u32_pc = b.type_pointer(storage::PUSH_CONSTANT, u32t);
            Some((vio, ins, outs, pc_var, p_u32_pc, v4))
        }
        _ => None,
    };
    // Multi-wave work-group state: (SubgroupId, LocalInvocationIndex,
    // shared wave-state u32[waves + threads], pointer, waiting, all_done,
    // waves).
    let multi = compute.filter(|c| c.threads() > opts.lanes).map(|c| {
        let waves = c.threads().div_ceil(opts.lanes);
        let sgid = builtin_var(&mut b, p_u32_in, builtin::SUBGROUP_ID);
        let lidx = builtin_var(&mut b, p_u32_in, builtin::LOCAL_INVOCATION_INDEX);
        let n = b.constant_u32(waves + c.threads());
        let arr_t = b.type_array(u32t, n);
        let p_arr = b.type_pointer(storage::WORKGROUP, arr_t);
        let p_elem = b.type_pointer(storage::WORKGROUP, u32t);
        let states = b.global_variable(p_arr, storage::WORKGROUP);
        b.name(states, "wave_state");
        let waiting = b.global_variable(p_bool_priv, storage::PRIVATE);
        b.name(waiting, "waiting");
        let all_done = b.global_variable(p_bool_priv, storage::PRIVATE);
        b.name(all_done, "all_done");
        (sgid, lidx, states, p_elem, waiting, all_done, waves)
    });
    let uses_images = insts.iter().any(|(_, i)| i.format == Format::Mimg);
    let lut = uses_images.then(|| {
        let n = b.constant_u32(256);
        let arr_t = b.type_array(u32t, n);
        let p_arr = b.type_pointer(storage::PRIVATE, arr_t);
        let v = b.global_variable(p_arr, storage::PRIVATE);
        b.name(v, "unorm8_lut");
        let parts: Vec<Id> = (0..256).map(|k| b.constant_u32(crate::rdna::unorm8_to_f32(k))).collect();
        let value = b.constant_composite(arr_t, &parts);
        (v, p_u32_priv, value)
    });
    let lds = (opts.lds_words > 0).then(|| {
        let n = b.constant_u32(opts.lds_words + 64);
        let arr_t = b.type_array(u32t, n);
        let p_arr = b.type_pointer(storage::WORKGROUP, arr_t);
        let p_elem = b.type_pointer(storage::WORKGROUP, u32t);
        let v = b.global_variable(p_arr, storage::WORKGROUP);
        b.name(v, "lds");
        (v, p_elem, opts.lds_words)
    });
    let xlane = (opts.cross_lane == CrossLane::Workgroup).then(|| {
        let n = b.constant_u32(64);
        let arr_t = b.type_array(u32t, n);
        let p_arr = b.type_pointer(storage::WORKGROUP, arr_t);
        let p_elem = b.type_pointer(storage::WORKGROUP, u32t);
        let v = b.global_variable(p_arr, storage::WORKGROUP);
        b.name(v, "xlane");
        (v, p_elem)
    });

    let private_u32 = |b: &mut Builder, name: &str| {
        let v = b.global_variable(p_u32_priv, storage::PRIVATE);
        b.name(v, name);
        v
    };
    let sgpr: Vec<Id> = (0..opts.num_sgprs)
        .map(|n| private_u32(&mut b, &format!("s{n}")))
        .collect();
    let vgpr: Vec<Id> = (0..opts.num_vgprs)
        .map(|n| private_u32(&mut b, &format!("v{n}")))
        .collect();
    let exec_lo = private_u32(&mut b, "exec_lo");
    let exec_hi = private_u32(&mut b, "exec_hi");
    let vcc_lo = private_u32(&mut b, "vcc_lo");
    let vcc_hi = private_u32(&mut b, "vcc_hi");
    let m0 = private_u32(&mut b, "m0");
    let pc = private_u32(&mut b, "block");
    let scc = b.global_variable(p_bool_priv, storage::PRIVATE);
    b.name(scc, "scc");

    let fn_ty = b.type_function(void, &[]);
    let main = b.begin_function(void, fn_ty);
    b.name(main, "main");
    let subgroup_scope = b.constant_u32(scope::SUBGROUP);
    let zero = b.constant_u32(0);
    let lane = match (cs_vars, &px) {
        (Some((_, _, lane_var, _, _)), _) => b.load(u32t, lane_var),
        (None, Some(p)) => b.load(u32t, p.2),
        (None, None) if vx.is_some() => b.load(u32t, vx.as_ref().unwrap().1[4]),
        (None, None) => {
            let g = b.load(uvec3, gid);
            b.op(u32t, op::COMPOSITE_EXTRACT, &[g, 0])
        }
    };
    let k31 = b.constant_u32(31);
    let k32 = b.constant_u32(32);
    let lane_shift = b.op(u32t, op::BITWISE_AND, &[lane, k31]);
    let lane_is_hi = b.op(boolt, op::U_GREATER_THAN_EQUAL, &[lane, k32]);

    let mut tr = Tr {
        code_base: opts.code_base,
        b,
        t: Types {
            u32t,
            i32t,
            f32t,
            boolt,
            uvec4,
        },
        wave64,
        sgpr,
        vgpr,
        exec_lo,
        exec_hi,
        vcc_lo,
        vcc_hi,
        m0,
        scc,
        lane,
        lane_is_hi,
        lane_shift,
        subgroup_scope,
        mem: mem_var,
        xlane,
        lds,
        waiting: multi.map(|m| m.4),
        dpp_src0: None,
        dpp_src1: None,
        dpp_enable: None,
        unorm_lut: None,
        glsl: None,
        pixel: None,
        vertex_outs: None,
        attr_vars: px.as_ref().map(|p| p.8.clone()).unwrap_or_default(),
        p_v4_in: px.as_ref().map_or(0, |p| p.9),
        at: 0,
        text: String::new(),
    };

    // Register dump slots (Io::RegisterDump only).
    let dump = matches!(opts.io, Io::RegisterDump).then_some(());
    let ns = opts.num_sgprs;
    let vbase = ns + SPECIALS;
    let slot = |tr: &mut Tr, index: Id| tr.b.op(p_u32_sb, op::ACCESS_CHAIN, &[regs_buf, zero, index]);
    let fixed_slot = |tr: &mut Tr, n: u32| {
        let i = tr.b.constant_u32(n);
        slot(tr, i)
    };
    let specials = [
        (SPECIAL_EXEC_LO, exec_lo),
        (SPECIAL_EXEC_HI, exec_hi),
        (SPECIAL_VCC_LO, vcc_lo),
        (SPECIAL_VCC_HI, vcc_hi),
        (SPECIAL_M0, m0),
    ];
    let mut vslots: Vec<Id> = Vec::new();
    if dump.is_some() {
        let nv = tr.b.constant_u32(opts.num_vgprs);
        let vb = tr.b.constant_u32(vbase);
        let lane_off = tr.u(op::I_MUL, &[tr.lane, nv]);
        let lane_base = tr.u(op::I_ADD, &[lane_off, vb]);
        vslots = (0..opts.num_vgprs)
            .map(|n| {
                let k = tr.c(n);
                let i = tr.u(op::I_ADD, &[lane_base, k]);
                slot(&mut tr, i)
            })
            .collect();
        for n in 0..ns {
            let p = fixed_slot(&mut tr, n);
            let v = tr.load_u(p);
            tr.b.store(tr.sgpr[n as usize], v);
        }
        for (k, var) in specials {
            let p = fixed_slot(&mut tr, ns + k);
            let v = tr.load_u(p);
            tr.b.store(var, v);
        }
        let p = fixed_slot(&mut tr, ns + SPECIAL_SCC);
        let v = tr.load_u(p);
        let s = tr.nonzero(v);
        tr.b.store(scc, s);
        for (n, &p) in vslots.iter().enumerate() {
            let v = tr.load_u(p);
            tr.b.store(tr.vgpr[n], v);
        }
    }
    if let (Some(cio), Some((wg, lid, _, pc_var, p_u32_pc))) = (compute, cs_vars) {
        // Everything starts at 0 except what the contract initializes.
        for r in tr.sgpr.clone().into_iter().chain(tr.vgpr.clone()) {
            tr.b.store(r, zero);
        }
        for r in [vcc_lo, vcc_hi, m0] {
            tr.b.store(r, zero);
        }
        let f = tr.b.constant_bool(false);
        tr.b.store(scc, f);
        let threads = cio.threads();
        let wave_id = multi.map(|m| tr.b.load(u32t, m.0));
        match wave_id {
            None => {
                let mask: u64 = if threads >= 64 { u64::MAX } else { (1u64 << threads) - 1 };
                let (elo, ehi) = (tr.c(mask as u32), tr.c((mask >> 32) as u32));
                tr.b.store(exec_lo, elo);
                tr.b.store(exec_hi, ehi);
            }
            Some(wid) => {
                // Lanes of this wave: min(wave size, threads - wave * size).
                let (lanes_c, threads_c) = (tr.c(opts.lanes), tr.c(threads));
                let first = tr.u(op::I_MUL, &[wid, lanes_c]);
                let left = tr.u(op::I_SUB, &[threads_c, first]);
                let lt = tr.bl(op::U_LESS_THAN, &[left, lanes_c]);
                let here = tr.select_u(lt, left, lanes_c);
                let lo = tr.low_mask(here);
                let k32 = tr.c(32);
                let ge32 = tr.bl(op::U_GREATER_THAN_EQUAL, &[here, k32]);
                let rest = tr.u(op::I_SUB, &[here, k32]);
                let rest = tr.select_u(ge32, rest, zero);
                let hi = tr.low_mask(rest);
                tr.b.store(exec_lo, lo);
                tr.b.store(exec_hi, hi);
            }
        }
        let mut next = 0usize;
        for i in 0..cio.user_sgprs {
            let ic = tr.c(i);
            let p = tr.b.op(p_u32_pc, op::ACCESS_CHAIN, &[pc_var, zero, ic]);
            let v = tr.load_u(p);
            tr.b.store(tr.sgpr[next], v);
            next += 1;
        }
        let wgv = tr.b.load(uvec3, wg);
        for d in 0..3 {
            if cio.tgid[d] {
                let v = tr.u(op::COMPOSITE_EXTRACT, &[wgv, d as u32]);
                tr.b.store(tr.sgpr[next], v);
                next += 1;
            }
        }
        if cio.tg_size {
            let v = match (wave_id, multi) {
                (Some(wid), Some(m)) => {
                    let is0 = tr.bl(op::I_EQUAL, &[wid, zero]);
                    let (first, other) = (
                        tr.c(ComputeIo::workgroup_info(0, m.6)),
                        tr.c(ComputeIo::workgroup_info(1, m.6)),
                    );
                    tr.select_u(is0, first, other)
                }
                _ => tr.c(ComputeIo::WORKGROUP_INFO_ONE_WAVE),
            };
            tr.b.store(tr.sgpr[next], v);
        }
        let lidv = tr.b.load(uvec3, lid);
        for d in 0..=cio.tidig_comp_cnt.min(2) as usize {
            let v = tr.u(op::COMPOSITE_EXTRACT, &[lidv, d as u32]);
            tr.b.store(tr.vgpr[d], v);
        }
    }
    if let Some((pio, fc, _, outs, pc_var, p_u32_pc, exported, v4, _, _, bary, front)) = px.clone() {
        for r in tr.sgpr.clone().into_iter().chain(tr.vgpr.clone()) {
            tr.b.store(r, zero);
        }
        for r in [vcc_lo, vcc_hi, m0] {
            tr.b.store(r, zero);
        }
        let f = tr.b.constant_bool(false);
        tr.b.store(scc, f);
        tr.b.store(exported, f);
        // exec: every invocation of the subgroup is a pixel of this wave.
        let t = tr.b.constant_bool(true);
        let all = tr.ballot(t);
        tr.b.store(exec_lo, all.lo);
        tr.b.store(exec_hi, all.hi);
        for i in 0..pio.user_sgprs {
            let ic = tr.c(i);
            let p = tr.b.op(p_u32_pc, op::ACCESS_CHAIN, &[pc_var, zero, ic]);
            let v = tr.load_u(p);
            tr.b.store(tr.sgpr[i as usize], v);
        }
        let fcv = tr.b.load(v4, fc);
        for (k, bit) in (crate::rdna::ps_input::POS_X_FLOAT..=crate::rdna::ps_input::POS_W_FLOAT).enumerate() {
            if pio.ena >> bit & 1 == 0 {
                continue;
            }
            let r = crate::rdna::ps_input::vgpr(pio.addr, bit).unwrap() as usize;
            let x = tr.b.op(f32t, op::COMPOSITE_EXTRACT, &[fcv, k as u32]);
            let u = tr.bits_of(x);
            tr.b.store(tr.vgpr[r], u);
        }
        if let Some((bv, v3)) = bary {
            // PERSP_CENTER (i, j) = barycentric weights of vertices 1 and 2.
            // ASSUMPTION: Vulkan's vertex order matches the hardware's.
            let r = crate::rdna::ps_input::vgpr(pio.addr, crate::rdna::ps_input::PERSP_CENTER).unwrap() as usize;
            let b3 = tr.b.load(v3, bv);
            for k in 0..2 {
                let x = tr.b.op(f32t, op::COMPOSITE_EXTRACT, &[b3, k as u32 + 1]);
                let u = tr.bits_of(x);
                tr.b.store(tr.vgpr[r + k], u);
            }
        }
        if let Some(fv) = front {
            // FRONT_FACE: +1.0 front, -1.0 back (see rdna::ps_input).
            let r = crate::rdna::ps_input::vgpr(pio.addr, crate::rdna::ps_input::FRONT_FACE).unwrap() as usize;
            let ff = tr.b.load(tr.t.boolt, fv);
            let (pos, neg) = (tr.c(1.0f32.to_bits()), tr.c((-1.0f32).to_bits()));
            let u = tr.select_u(ff, pos, neg);
            tr.b.store(tr.vgpr[r], u);
        }
        let z4 = tr.b.constant_f32(0.0);
        let zero4 = tr.b.constant_composite(v4, &[z4, z4, z4, z4]);
        for o in outs.into_iter().flatten() {
            tr.b.store(o, zero4);
        }
        tr.pixel = Some((outs, exported));
    }
    if let Some((vio, ins, outs, pc_var, p_u32_pc, v4)) = vx.clone() {
        for r in tr.sgpr.clone().into_iter().chain(tr.vgpr.clone()) {
            tr.b.store(r, zero);
        }
        for r in [vcc_lo, vcc_hi, m0] {
            tr.b.store(r, zero);
        }
        let f = tr.b.constant_bool(false);
        tr.b.store(scc, f);
        let t = tr.b.constant_bool(true);
        let all = tr.ballot(t);
        tr.b.store(exec_lo, all.lo);
        tr.b.store(exec_hi, all.hi);
        for i in 0..vio.user_sgprs {
            let ic = tr.c(i);
            let p = tr.b.op(p_u32_pc, op::ACCESS_CHAIN, &[pc_var, zero, ic]);
            let v = tr.load_u(p);
            tr.b.store(tr.sgpr[i as usize], v);
        }
        let ld = |tr: &mut Tr, v: Id| tr.load_u(v);
        let (vi, bv, ii, bi) = (ld(&mut tr, ins[0]), ld(&mut tr, ins[1]), ld(&mut tr, ins[2]), ld(&mut tr, ins[3]));
        let vid = tr.u(op::I_SUB, &[vi, bv]);
        let iid = tr.u(op::I_SUB, &[ii, bi]);
        use crate::rdna::vs_input;
        let vals = [(vs_input::VERTEX_ID, vid), (vs_input::INSTANCE_ID, iid)];
        for (r, v) in vals {
            if r <= vio.vgpr_comp_cnt as usize {
                tr.b.store(tr.vgpr[r], v);
            }
        }
        let z = tr.b.constant_f32(0.0);
        let zero4 = tr.b.constant_composite(v4, &[z, z, z, z]);
        for (_, o) in &outs {
            tr.b.store(*o, zero4);
        }
        tr.vertex_outs = Some(outs);
    }
    if let Some((v, p, value)) = lut {
        tr.b.store(v, value);
        tr.unorm_lut = Some((v, p));
    }
    tr.b.store(pc, zero);

    // Dispatcher loop:
    //   header: LoopMerge(exit, cont); pc != END ? dispatch : exit
    //   dispatch: chain of `if pc == k { block k; pc = next; goto cont }`
    //   cont: goto header
    //
    // Multi-wave work-groups: control flow is only uniform per wave, so the
    // loop runs while any wave is unfinished, a wave executes a block only
    // while it is neither done nor waiting at s_barrier, and the continue
    // block (reached by every invocation every iteration, so uniform)
    // publishes each wave's state, and releases the waiting waves once no
    // wave is running.
    let (header, dispatch, cont, exit) = (tr.b.id(), tr.b.id(), tr.b.id(), tr.b.id());
    let endc = tr.c(end_index);
    if let Some(m) = multi {
        let f = tr.b.constant_bool(false);
        tr.b.store(m.4, f);
        tr.b.store(m.5, f);
    }
    tr.b.branch(header);
    tr.b.label(header);
    let running = match multi {
        Some(m) => {
            let d = tr.b.load(boolt, m.5);
            tr.bl(op::LOGICAL_NOT, &[d])
        }
        None => {
            let cur = tr.load_u(pc);
            tr.bl(op::I_NOT_EQUAL, &[cur, endc])
        }
    };
    tr.b.op_void(op::LOOP_MERGE, &[exit, cont, 0]);
    tr.b.branch_conditional(running, dispatch, exit);
    tr.b.label(dispatch);
    let cur = tr.load_u(pc);
    if let Some(m) = multi {
        // Skip the blocks this iteration if this wave is done or waiting.
        let wait = tr.b.load(boolt, m.4);
        let live = tr.bl(op::I_NOT_EQUAL, &[cur, endc]);
        let nw = tr.bl(op::LOGICAL_NOT, &[wait]);
        let go = tr.bl(op::LOGICAL_AND, &[live, nw]);
        let (chain, skip_merge) = (tr.b.id(), tr.b.id());
        tr.b.op_void(op::SELECTION_MERGE, &[skip_merge, 0]);
        tr.b.branch_conditional(go, chain, cont);
        tr.b.label(skip_merge);
        tr.b.unreachable();
        tr.b.label(chain);
    }

    for (k, (start, body)) in blocks.iter().enumerate() {
        let (arm, next, merge) = (tr.b.id(), tr.b.id(), tr.b.id());
        let kc = tr.c(k as u32);
        let is_k = tr.bl(op::I_EQUAL, &[cur, kc]);
        tr.b.op_void(op::SELECTION_MERGE, &[merge, 0]);
        tr.b.branch_conditional(is_k, arm, next);
        tr.b.label(arm);

        let mut target: Option<Id> = None; // computed next-block id
        let mut fell_through = true;
        for (at, inst) in body {
            tr.at = *at;
            tr.text = inst.to_string();
            let fallthrough = index_of(at + inst.len()).map_or(end_index, |i| i as u32);
            if inst.is_endpgm() {
                target = Some(tr.c(end_index));
                fell_through = false;
                break;
            }
            if inst.format == Format::Sopp && inst.branch_target(*at).is_some() {
                let t = inst.branch_target(*at).unwrap();
                let ti = index_of(t).ok_or_else(|| tr.err("branch into the middle of a block"))?;
                let (tc, fc) = (tr.c(ti as u32), tr.c(fallthrough));
                let taken = match inst.opcode {
                    sopp::S_BRANCH => None,
                    sopp::S_CBRANCH_SCC0 => {
                        let s = tr.b.load(boolt, scc);
                        Some(tr.bl(op::LOGICAL_NOT, &[s]))
                    }
                    sopp::S_CBRANCH_SCC1 => Some(tr.b.load(boolt, scc)),
                    sopp::S_CBRANCH_VCCZ => Some(tr.mask_is_zero(vcc_lo, vcc_hi)),
                    sopp::S_CBRANCH_VCCNZ => {
                        let z = tr.mask_is_zero(vcc_lo, vcc_hi);
                        Some(tr.bl(op::LOGICAL_NOT, &[z]))
                    }
                    sopp::S_CBRANCH_EXECZ => Some(tr.mask_is_zero(exec_lo, exec_hi)),
                    sopp::S_CBRANCH_EXECNZ => {
                        let z = tr.mask_is_zero(exec_lo, exec_hi);
                        Some(tr.bl(op::LOGICAL_NOT, &[z]))
                    }
                    _ => return Err(tr.err("branch kind")),
                };
                target = Some(match taken {
                    None => tc,
                    Some(c) => tr.select_u(c, tc, fc),
                });
                fell_through = false;
                break;
            }
            tr.inst(inst)?;
        }
        if fell_through {
            let last_end = body.last().map(|(a, i)| a + i.len()).unwrap_or(*start);
            let f = index_of(last_end).map_or(end_index, |i| i as u32);
            target = Some(tr.c(f));
        }
        tr.b.store(pc, target.unwrap());
        tr.b.branch(cont);
        tr.b.label(merge);
        tr.b.unreachable();
        tr.b.label(next);
    }
    // No block matched (cannot happen while pc < END).
    tr.b.branch(cont);
    tr.b.label(cont);
    if let Some((_, lidx, states, p_elem, waiting, all_done, waves)) = multi {
        // state: 0 running, 1 waiting, 2 done; lane 0 of each wave writes it,
        // the other lanes write their own sink word.
        let cur = tr.load_u(pc);
        let done = tr.bl(op::I_EQUAL, &[cur, endc]);
        let wait = tr.b.load(boolt, waiting);
        let (one, two) = (tr.c(1), tr.c(2));
        let st0 = tr.select_u(wait, one, zero);
        let st = tr.select_u(done, two, st0);
        let wid = tr.b.load(u32t, multi.unwrap().0);
        let li = tr.b.load(u32t, lidx);
        let wc = tr.c(waves);
        let sink = tr.u(op::I_ADD, &[wc, li]);
        let lane0 = tr.bl(op::I_EQUAL, &[tr.lane, zero]);
        let slot = tr.select_u(lane0, wid, sink);
        let p = tr.b.op(p_elem, op::ACCESS_CHAIN, &[states, slot]);
        tr.b.store(p, st);
        tr.barrier();
        let t = tr.b.constant_bool(true);
        let (mut any_running, mut all) = (tr.b.constant_bool(false), t);
        for w in 0..waves {
            let wcst = tr.c(w);
            let p = tr.b.op(p_elem, op::ACCESS_CHAIN, &[states, wcst]);
            let v = tr.load_u(p);
            let r = tr.bl(op::I_EQUAL, &[v, zero]);
            let d = tr.bl(op::I_EQUAL, &[v, two]);
            any_running = tr.bl(op::LOGICAL_OR, &[any_running, r]);
            all = tr.bl(op::LOGICAL_AND, &[all, d]);
        }
        tr.barrier();
        tr.b.store(all_done, all);
        // Release: nobody runs, so every unfinished wave is waiting.
        let f = tr.b.constant_bool(false);
        let w2 = tr.b.op(boolt, op::SELECT, &[any_running, wait, f]);
        tr.b.store(waiting, w2);
    }
    tr.b.branch(header);
    tr.b.label(exit);

    // Epilogue (register dump only).
    if dump.is_some() {
        for n in 0..ns {
            let p = fixed_slot(&mut tr, n);
            let v = tr.load_u(tr.sgpr[n as usize]);
            tr.b.store(p, v);
        }
        for (k, var) in specials {
            let p = fixed_slot(&mut tr, ns + k);
            let v = tr.load_u(var);
            tr.b.store(p, v);
        }
        let p = fixed_slot(&mut tr, ns + SPECIAL_SCC);
        let s = tr.b.load(boolt, scc);
        let v = tr.bool_to_u(s);
        tr.b.store(p, v);
        for (n, &p) in vslots.iter().enumerate() {
            let v = tr.load_u(tr.vgpr[n]);
            tr.b.store(p, v);
        }
    }
    if let Some((_, exported)) = tr.pixel {
        let e = tr.b.load(boolt, exported);
        let (keep, kill) = (tr.b.id(), tr.b.id());
        tr.b.op_void(op::SELECTION_MERGE, &[keep, 0]);
        tr.b.branch_conditional(e, keep, kill);
        tr.b.label(kill);
        tr.b.kill();
        tr.b.label(keep);
    }
    tr.b.ret();
    tr.b.end_function();

    let mut b = tr.b;
    let mut interface = vec![regs_buf, gid];
    interface.extend(mem_var.map(|m| m.0));
    if let Some((wg, lid, lane_var, _, _)) = cs_vars {
        interface.extend([wg, lid, lane_var]);
    }
    if let Some(m) = multi {
        interface.extend([m.0, m.1]);
    }
    if let Some(v) = &vx {
        interface.extend(v.1);
        interface.extend(v.2.iter().map(|(_, o)| *o));
        b.entry_point(execution_model::VERTEX, main, "main", &interface);
        return Ok(b.finish());
    }
    if let Some(p) = &px {
        interface.extend([p.1, p.2]);
        interface.extend(p.3.iter().flatten());
        interface.extend(p.8.iter().map(|(_, v)| *v));
        interface.extend(p.10.map(|(v, _)| v));
        b.entry_point(execution_model::FRAGMENT, main, "main", &interface);
        b.execution_mode(main, execution_mode::ORIGIN_UPPER_LEFT, &[]);
        return Ok(b.finish());
    }
    b.entry_point(execution_model::GL_COMPUTE, main, "main", &interface);
    let local = match compute {
        Some(c) => c.workgroup,
        None => [opts.lanes, 1, 1],
    };
    b.execution_mode(main, execution_mode::LOCAL_SIZE, &local);
    Ok(b.finish())
}
