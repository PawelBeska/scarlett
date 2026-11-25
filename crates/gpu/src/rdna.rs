//! RDNA2 (GFX10.3) shader instruction decoding.
//!
//! Decodes the instruction *formats* completely: which encoding a dword
//! starts, how many dwords the instruction takes (literal constants, DPP/
//! SDWA words, MIMG NSA address words), every field, and operands
//! (SGPR/VGPR/special registers/inline constants/literals). Mnemonics are
//! given for a conservative subset of opcodes; others print as
//! `<format>_op<N>`, which is enough to split shaders into instructions and
//! basic blocks and to find what a translator still has to cover.
//!
//! Encodings follow AMD's public "RDNA 2 Instruction Set Architecture"
//! reference. Every mnemonic named here is checked against reference
//! encodings from LLVM's GFX10 assembler tests (`tests/llvm_encodings.rs`);
//! extend that table whenever a mnemonic is added.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Sop2,
    Sopk,
    Sop1,
    Sopc,
    Sopp,
    Smem,
    Vop2,
    Vop1,
    Vopc,
    Vop3,
    Vop3p,
    Vintrp,
    Ds,
    Flat,
    Mubuf,
    Mtbuf,
    Mimg,
    Exp,
}

impl Format {
    /// Format of the instruction whose first dword is `w`.
    pub fn of(w: u32) -> Option<Format> {
        Some(match w >> 23 {
            0x17D => Format::Sop1,
            0x17E => Format::Sopc,
            0x17F => Format::Sopp,
            _ if w >> 28 == 0xB => Format::Sopk,
            _ if w >> 30 == 0b10 => Format::Sop2,
            _ => match w >> 26 {
                0x3D => Format::Smem,
                0x35 => Format::Vop3,
                0x33 => Format::Vop3p,
                0x32 => Format::Vintrp,
                0x36 => Format::Ds,
                0x37 => Format::Flat,
                0x38 => Format::Mubuf,
                0x3A => Format::Mtbuf,
                0x3C => Format::Mimg,
                0x3E => Format::Exp,
                _ => match w >> 25 {
                    0x3F => Format::Vop1,
                    0x3E => Format::Vopc,
                    _ if w >> 31 == 0 => Format::Vop2,
                    _ => return None,
                },
            },
        })
    }

    /// Dwords of the base encoding (without literal/DPP/SDWA/NSA words).
    pub fn base_len(self) -> usize {
        match self {
            Format::Sop2 | Format::Sopk | Format::Sop1 | Format::Sopc | Format::Sopp => 1,
            Format::Vop2 | Format::Vop1 | Format::Vopc | Format::Vintrp => 1,
            _ => 2,
        }
    }

    fn prefix(self) -> &'static str {
        match self {
            Format::Sop2 => "sop2",
            Format::Sopk => "sopk",
            Format::Sop1 => "sop1",
            Format::Sopc => "sopc",
            Format::Sopp => "sopp",
            Format::Smem => "smem",
            Format::Vop2 => "vop2",
            Format::Vop1 => "vop1",
            Format::Vopc => "vopc",
            Format::Vop3 => "vop3",
            Format::Vop3p => "vop3p",
            Format::Vintrp => "vintrp",
            Format::Ds => "ds",
            Format::Flat => "flat",
            Format::Mubuf => "mubuf",
            Format::Mtbuf => "mtbuf",
            Format::Mimg => "mimg",
            Format::Exp => "exp",
        }
    }
}

/// A 9-bit (VALU) or 8-bit (SALU) source/destination operand.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Operand {
    Sgpr(u16),
    Vgpr(u16),
    VccLo,
    VccHi,
    Ttmp(u16),
    M0,
    Null,
    ExecLo,
    ExecHi,
    /// Inline integer constant (0, 1..64, -1..-16).
    Int(i32),
    /// Inline float constant.
    Float(f32),
    /// 1 / (2π)
    InvTwoPi,
    Vccz,
    Execz,
    Scc,
    /// Literal constant from the extra dword.
    Literal(u32),
    /// DPP / SDWA marker values (the real operand is in the extra dword).
    Dpp16,
    Dpp8,
    Sdwa,
    /// Encodings without a meaning here (shared/private base, pops, ...).
    Other(u16),
}

pub const SRC_LITERAL: u16 = 255;
pub const SRC_SDWA: u16 = 249;
pub const SRC_DPP16: u16 = 250;
pub const SRC_DPP8: u16 = 233;
pub const SRC_DPP8_FI: u16 = 234;

/// Decode a source operand value (`literal` used if it is 255).
pub fn operand(v: u16, literal: u32) -> Operand {
    match v {
        0..=105 => Operand::Sgpr(v),
        106 => Operand::VccLo,
        107 => Operand::VccHi,
        108..=123 => Operand::Ttmp(v - 108),
        124 => Operand::M0,
        125 => Operand::Null,
        126 => Operand::ExecLo,
        127 => Operand::ExecHi,
        128 => Operand::Int(0),
        129..=192 => Operand::Int(v as i32 - 128),
        193..=208 => Operand::Int(192 - v as i32),
        SRC_DPP8 | SRC_DPP8_FI => Operand::Dpp8,
        240 => Operand::Float(0.5),
        241 => Operand::Float(-0.5),
        242 => Operand::Float(1.0),
        243 => Operand::Float(-1.0),
        244 => Operand::Float(2.0),
        245 => Operand::Float(-2.0),
        246 => Operand::Float(4.0),
        247 => Operand::Float(-4.0),
        248 => Operand::InvTwoPi,
        SRC_SDWA => Operand::Sdwa,
        SRC_DPP16 => Operand::Dpp16,
        251 => Operand::Vccz,
        252 => Operand::Execz,
        253 => Operand::Scc,
        SRC_LITERAL => Operand::Literal(literal),
        256..=511 => Operand::Vgpr(v - 256),
        _ => Operand::Other(v),
    }
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Operand::Sgpr(n) => write!(f, "s{n}"),
            Operand::Vgpr(n) => write!(f, "v{n}"),
            Operand::VccLo => f.write_str("vcc_lo"),
            Operand::VccHi => f.write_str("vcc_hi"),
            Operand::Ttmp(n) => write!(f, "ttmp{n}"),
            Operand::M0 => f.write_str("m0"),
            Operand::Null => f.write_str("null"),
            Operand::ExecLo => f.write_str("exec_lo"),
            Operand::ExecHi => f.write_str("exec_hi"),
            Operand::Int(i) => write!(f, "{i}"),
            Operand::Float(x) => write!(f, "{x:?}"),
            Operand::InvTwoPi => f.write_str("0.15915494"),
            Operand::Vccz => f.write_str("vccz"),
            Operand::Execz => f.write_str("execz"),
            Operand::Scc => f.write_str("scc"),
            Operand::Literal(x) => write!(f, "{x:#x}"),
            Operand::Dpp16 => f.write_str("dpp"),
            Operand::Dpp8 => f.write_str("dpp8"),
            Operand::Sdwa => f.write_str("sdwa"),
            Operand::Other(v) => write!(f, "src{v}"),
        }
    }
}

/// One decoded instruction.
#[derive(Debug, Clone, PartialEq)]
pub struct Inst {
    pub format: Format,
    pub opcode: u16,
    /// All dwords of the instruction (base + literal/DPP/SDWA/NSA words).
    pub words: Vec<u32>,
    pub fields: Fields,
}

/// Format-specific fields. Register numbers are raw encodings; use
/// [`operand`] for sources.
#[derive(Debug, Clone, PartialEq)]
pub enum Fields {
    Sop2 {
        sdst: u8,
        ssrc0: u16,
        ssrc1: u16,
    },
    Sopk {
        sdst: u8,
        simm16: u16,
    },
    Sop1 {
        sdst: u8,
        ssrc0: u16,
    },
    Sopc {
        ssrc0: u16,
        ssrc1: u16,
    },
    Sopp {
        simm16: u16,
    },
    Smem {
        sdata: u8,
        sbase: u8,
        offset: i32,
        soffset: u8,
        glc: bool,
        dlc: bool,
    },
    Vop2 {
        vdst: u8,
        src0: u16,
        vsrc1: u8,
    },
    Vop1 {
        vdst: u8,
        src0: u16,
    },
    Vopc {
        src0: u16,
        vsrc1: u8,
    },
    Vop3 {
        vdst: u8,
        src: [u16; 3],
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
        op_sel: u8,
    },
    Vop3p {
        vdst: u8,
        src: [u16; 3],
        neg: u8,
        neg_hi: u8,
        op_sel: u8,
        op_sel_hi: u8,
        clamp: bool,
    },
    Vintrp {
        vdst: u8,
        vsrc: u8,
        attr: u8,
        chan: u8,
    },
    Ds {
        vdst: u8,
        addr: u8,
        data0: u8,
        data1: u8,
        offset0: u8,
        offset1: u8,
        gds: bool,
    },
    /// FLAT/GLOBAL/SCRATCH: `seg` 0 flat, 1 scratch, 2 global.
    Flat {
        vdst: u8,
        addr: u8,
        data: u8,
        saddr: u8,
        offset: i16,
        seg: u8,
        glc: bool,
        slc: bool,
        dlc: bool,
    },
    Mubuf {
        vdata: u8,
        vaddr: u8,
        srsrc: u8,
        soffset: u8,
        offset: u16,
        offen: bool,
        idxen: bool,
        glc: bool,
    },
    Mtbuf {
        vdata: u8,
        vaddr: u8,
        srsrc: u8,
        soffset: u8,
        offset: u16,
        format: u8,
        offen: bool,
        idxen: bool,
    },
    Mimg {
        vdata: u8,
        vaddr: u8,
        srsrc: u8,
        ssamp: u8,
        dmask: u8,
        dim: u8,
        nsa_addrs: Vec<u8>,
    },
    Exp {
        target: u8,
        en: u8,
        vsrc: [u8; 4],
        done: bool,
        compr: bool,
        vm: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    UnknownFormat { at: usize, word: u32 },
    Truncated { at: usize, need: usize },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::UnknownFormat { at, word } => {
                write!(f, "unknown encoding {word:#010x} at dword {at}")
            }
            DecodeError::Truncated { at, need } => {
                write!(f, "instruction at dword {at} needs {need} dwords")
            }
        }
    }
}

impl std::error::Error for DecodeError {}

fn bits(w: u32, lo: u32, n: u32) -> u32 {
    (w >> lo) & ((1 << n) - 1)
}

/// Decode the instruction starting at `code[at]`.
pub fn decode(code: &[u32], at: usize) -> Result<Inst, DecodeError> {
    let w0 = *code.get(at).ok_or(DecodeError::Truncated { at, need: 1 })?;
    let format = Format::of(w0).ok_or(DecodeError::UnknownFormat { at, word: w0 })?;
    let base = format.base_len();
    let w1 = if base == 2 {
        *code
            .get(at + 1)
            .ok_or(DecodeError::Truncated { at, need: 2 })?
    } else {
        0
    };
    let (opcode, fields, extra): (u16, Fields, usize) = match format {
        Format::Sop2 => {
            let (s0, s1) = (bits(w0, 0, 8) as u16, bits(w0, 8, 8) as u16);
            let lit = (s0 == SRC_LITERAL || s1 == SRC_LITERAL) as usize;
            (
                bits(w0, 23, 7) as u16,
                Fields::Sop2 {
                    sdst: bits(w0, 16, 7) as u8,
                    ssrc0: s0,
                    ssrc1: s1,
                },
                lit,
            )
        }
        Format::Sopk => {
            let op = bits(w0, 23, 5) as u16;
            // s_setreg_imm32_b32 carries a 32-bit immediate.
            let lit = (op == sopk::S_SETREG_IMM32_B32) as usize;
            (
                op,
                Fields::Sopk {
                    sdst: bits(w0, 16, 7) as u8,
                    simm16: w0 as u16,
                },
                lit,
            )
        }
        Format::Sop1 => {
            let s0 = bits(w0, 0, 8) as u16;
            (
                bits(w0, 8, 8) as u16,
                Fields::Sop1 {
                    sdst: bits(w0, 16, 7) as u8,
                    ssrc0: s0,
                },
                (s0 == SRC_LITERAL) as usize,
            )
        }
        Format::Sopc => {
            let (s0, s1) = (bits(w0, 0, 8) as u16, bits(w0, 8, 8) as u16);
            let lit = (s0 == SRC_LITERAL || s1 == SRC_LITERAL) as usize;
            (
                bits(w0, 16, 7) as u16,
                Fields::Sopc {
                    ssrc0: s0,
                    ssrc1: s1,
                },
                lit,
            )
        }
        Format::Sopp => (
            bits(w0, 16, 7) as u16,
            Fields::Sopp { simm16: w0 as u16 },
            0,
        ),
        Format::Smem => {
            let off = bits(w1, 0, 21);
            let offset = ((off << 11) as i32) >> 11; // sign-extend 21 bits
            (
                bits(w0, 18, 8) as u16,
                Fields::Smem {
                    sbase: (bits(w0, 0, 6) * 2) as u8,
                    sdata: bits(w0, 6, 7) as u8,
                    dlc: bits(w0, 14, 1) != 0,
                    glc: bits(w0, 16, 1) != 0,
                    offset,
                    soffset: bits(w1, 25, 7) as u8,
                },
                0,
            )
        }
        Format::Vop2 | Format::Vop1 | Format::Vopc => {
            let src0 = bits(w0, 0, 9) as u16;
            let extra = matches!(
                src0,
                SRC_LITERAL | SRC_SDWA | SRC_DPP16 | SRC_DPP8 | SRC_DPP8_FI
            ) as usize;
            // v_fmamk/v_fmaak carry the constant K in a trailing dword
            // (shared with a literal src0).
            let madk = format == Format::Vop2 && vop2::is_madk(bits(w0, 25, 6) as u16);
            let extra = if madk { 1 } else { extra };
            match format {
                Format::Vop2 => (
                    bits(w0, 25, 6) as u16,
                    Fields::Vop2 {
                        vdst: bits(w0, 17, 8) as u8,
                        src0,
                        vsrc1: bits(w0, 9, 8) as u8,
                    },
                    extra,
                ),
                Format::Vop1 => (
                    bits(w0, 9, 8) as u16,
                    Fields::Vop1 {
                        vdst: bits(w0, 17, 8) as u8,
                        src0,
                    },
                    extra,
                ),
                _ => (
                    bits(w0, 17, 8) as u16,
                    Fields::Vopc {
                        src0,
                        vsrc1: bits(w0, 9, 8) as u8,
                    },
                    extra,
                ),
            }
        }
        Format::Vop3 => {
            let src = [
                bits(w1, 0, 9) as u16,
                bits(w1, 9, 9) as u16,
                bits(w1, 18, 9) as u16,
            ];
            (
                bits(w0, 16, 10) as u16,
                Fields::Vop3 {
                    vdst: bits(w0, 0, 8) as u8,
                    abs: bits(w0, 8, 3) as u8,
                    op_sel: bits(w0, 11, 4) as u8,
                    clamp: bits(w0, 15, 1) != 0,
                    src,
                    omod: bits(w1, 27, 2) as u8,
                    neg: bits(w1, 29, 3) as u8,
                },
                src.contains(&SRC_LITERAL) as usize,
            )
        }
        Format::Vop3p => {
            let src = [
                bits(w1, 0, 9) as u16,
                bits(w1, 9, 9) as u16,
                bits(w1, 18, 9) as u16,
            ];
            (
                bits(w0, 16, 7) as u16,
                Fields::Vop3p {
                    vdst: bits(w0, 0, 8) as u8,
                    neg_hi: bits(w0, 8, 3) as u8,
                    op_sel: bits(w0, 11, 3) as u8,
                    op_sel_hi: (bits(w0, 14, 1) << 2 | bits(w1, 27, 2)) as u8,
                    clamp: bits(w0, 15, 1) != 0,
                    src,
                    neg: bits(w1, 29, 3) as u8,
                },
                src.contains(&SRC_LITERAL) as usize,
            )
        }
        Format::Vintrp => (
            bits(w0, 16, 2) as u16,
            Fields::Vintrp {
                vsrc: bits(w0, 0, 8) as u8,
                chan: bits(w0, 8, 2) as u8,
                attr: bits(w0, 10, 6) as u8,
                vdst: bits(w0, 18, 8) as u8,
            },
            0,
        ),
        Format::Ds => (
            bits(w0, 18, 8) as u16,
            Fields::Ds {
                offset0: bits(w0, 0, 8) as u8,
                offset1: bits(w0, 8, 8) as u8,
                gds: bits(w0, 17, 1) != 0,
                addr: bits(w1, 0, 8) as u8,
                data0: bits(w1, 8, 8) as u8,
                data1: bits(w1, 16, 8) as u8,
                vdst: bits(w1, 24, 8) as u8,
            },
            0,
        ),
        Format::Flat => (
            bits(w0, 18, 7) as u16,
            Fields::Flat {
                offset: ((bits(w0, 0, 12) << 20) as i32 >> 20) as i16,
                dlc: bits(w0, 12, 1) != 0,
                seg: bits(w0, 14, 2) as u8,
                glc: bits(w0, 16, 1) != 0,
                slc: bits(w0, 17, 1) != 0,
                addr: bits(w1, 0, 8) as u8,
                data: bits(w1, 8, 8) as u8,
                saddr: bits(w1, 16, 7) as u8,
                vdst: bits(w1, 24, 8) as u8,
            },
            0,
        ),
        Format::Mubuf => (
            bits(w0, 18, 7) as u16,
            Fields::Mubuf {
                offset: bits(w0, 0, 12) as u16,
                offen: bits(w0, 12, 1) != 0,
                idxen: bits(w0, 13, 1) != 0,
                glc: bits(w0, 14, 1) != 0,
                vaddr: bits(w1, 0, 8) as u8,
                vdata: bits(w1, 8, 8) as u8,
                srsrc: (bits(w1, 16, 5) * 4) as u8,
                soffset: bits(w1, 24, 8) as u8,
            },
            0,
        ),
        Format::Mtbuf => (
            (bits(w0, 16, 3) | bits(w1, 21, 1) << 3) as u16,
            Fields::Mtbuf {
                offset: bits(w0, 0, 12) as u16,
                offen: bits(w0, 12, 1) != 0,
                idxen: bits(w0, 13, 1) != 0,
                format: bits(w0, 19, 7) as u8,
                vaddr: bits(w1, 0, 8) as u8,
                vdata: bits(w1, 8, 8) as u8,
                srsrc: (bits(w1, 16, 5) * 4) as u8,
                soffset: bits(w1, 24, 8) as u8,
            },
            0,
        ),
        Format::Mimg => {
            // NSA: dword0 [2:1] = number of extra address dwords (4 VGPRs each).
            let nsa_words = bits(w0, 1, 2) as usize;
            let mut nsa = Vec::new();
            for k in 0..nsa_words {
                let w = *code.get(at + 2 + k).ok_or(DecodeError::Truncated {
                    at,
                    need: 2 + nsa_words,
                })?;
                nsa.extend(w.to_le_bytes());
            }
            (
                (bits(w0, 18, 7) | bits(w0, 0, 1) << 7) as u16,
                Fields::Mimg {
                    dim: bits(w0, 3, 3) as u8,
                    dmask: bits(w0, 8, 4) as u8,
                    vaddr: bits(w1, 0, 8) as u8,
                    vdata: bits(w1, 8, 8) as u8,
                    srsrc: (bits(w1, 16, 5) * 4) as u8,
                    ssamp: (bits(w1, 21, 5) * 4) as u8,
                    nsa_addrs: nsa,
                },
                nsa_words,
            )
        }
        Format::Exp => (
            0,
            Fields::Exp {
                en: bits(w0, 0, 4) as u8,
                target: bits(w0, 4, 6) as u8,
                compr: bits(w0, 10, 1) != 0,
                done: bits(w0, 11, 1) != 0,
                vm: bits(w0, 12, 1) != 0,
                vsrc: w1.to_le_bytes(),
            },
            0,
        ),
    };
    let len = base + extra;
    let words = code
        .get(at..at + len)
        .ok_or(DecodeError::Truncated { at, need: len })?
        .to_vec();
    Ok(Inst {
        format,
        opcode,
        words,
        fields,
    })
}

/// Mnemonics known to this decoder (a conservative subset).
pub mod sopp {
    pub const S_NOP: u16 = 0;
    pub const S_ENDPGM: u16 = 1;
    pub const S_BRANCH: u16 = 2;
    pub const S_WAKEUP: u16 = 3;
    pub const S_CBRANCH_SCC0: u16 = 4;
    pub const S_CBRANCH_SCC1: u16 = 5;
    pub const S_CBRANCH_VCCZ: u16 = 6;
    pub const S_CBRANCH_VCCNZ: u16 = 7;
    pub const S_CBRANCH_EXECZ: u16 = 8;
    pub const S_CBRANCH_EXECNZ: u16 = 9;
    pub const S_BARRIER: u16 = 10;
    pub const S_WAITCNT: u16 = 12;
    pub const S_SLEEP: u16 = 14;
    pub const S_SENDMSG: u16 = 16;
    pub const S_CODE_END: u16 = 31;
}

// Opcode numbers below were each read from an encoding in LLVM's GFX10 MC
// tests (llvm/test/MC/AMDGPU/gfx10_asm_{sop,smem,vop1,vop2,vopc,vop3}.s).

pub mod sop2 {
    /// 64-bit shifts by S1[5:0] (LLVM `cshl_64`/`csrl_64`/`csra_64`), SCC
    /// = result != 0 as the 32-bit shifts; GFX10 numbers from
    /// `SOP2_Real_gfx6_gfx7_gfx10_gfx13`.
    pub const S_LSHL_B64: u16 = 0x1F;
    pub const S_LSHR_B64: u16 = 0x21;
    pub const S_ASHR_I64: u16 = 0x23;
    /// `mulhu` / `mulhs`, no SCC (`SOP2_Real_gfx10_gfx13`).
    pub const S_MUL_HI_U32: u16 = 0x35;
    pub const S_MUL_HI_I32: u16 = 0x36;
    pub const S_ADD_U32: u16 = 0;
    pub const S_SUB_U32: u16 = 1;
    pub const S_ADD_I32: u16 = 2;
    pub const S_SUB_I32: u16 = 3;
    pub const S_ADDC_U32: u16 = 4;
    pub const S_SUBB_U32: u16 = 5;
    pub const S_CSELECT_B32: u16 = 10;
    pub const S_CSELECT_B64: u16 = 11;
    pub const S_AND_B32: u16 = 14;
    pub const S_AND_B64: u16 = 15;
    pub const S_OR_B32: u16 = 16;
    pub const S_OR_B64: u16 = 17;
    pub const S_XOR_B32: u16 = 18;
    pub const S_XOR_B64: u16 = 19;
    pub const S_ANDN2_B32: u16 = 20;
    pub const S_ANDN2_B64: u16 = 21;
    pub const S_ORN2_B32: u16 = 22;
    pub const S_NAND_B32: u16 = 24;
    pub const S_NOR_B32: u16 = 26;
    pub const S_XNOR_B32: u16 = 28;
    pub const S_LSHL_B32: u16 = 30;
    pub const S_LSHR_B32: u16 = 32;
    pub const S_ASHR_I32: u16 = 34;
    pub const S_BFM_B32: u16 = 36;
    pub const S_MUL_I32: u16 = 38;
    pub const S_BFE_U32: u16 = 39;
    pub const S_BFE_I32: u16 = 40;
    // From AMD's machine-readable RDNA2 ISA specification (see sop1).
    /// min/max: SCC = 1 iff S0 is the selected value.
    pub const S_MIN_I32: u16 = 6;
    pub const S_MIN_U32: u16 = 7;
    pub const S_MAX_I32: u16 = 8;
    pub const S_MAX_U32: u16 = 9;
    pub const S_ORN2_B64: u16 = 0x17;
    pub const S_NAND_B64: u16 = 0x19;
    pub const S_NOR_B64: u16 = 0x1B;
    pub const S_XNOR_B64: u16 = 0x1D;
    /// ((1 << S0[5:0]) - 1) << S1[5:0], no SCC.
    pub const S_BFM_B64: u16 = 0x25;
    /// |S0 - S1| (signed), SCC = result != 0.
    pub const S_ABSDIFF_I32: u16 = 0x2C;
    /// D = {S1 half, S0 half}: ll = S0[15:0] | S1[15:0] << 16, lh = S0[15:0]
    /// | S1[31:16] << 16, hh = S0[31:16] | S1[31:16] << 16 (LLVM's
    /// build_vector patterns in SIInstructions.td), no SCC.
    pub const S_PACK_LL_B32_B16: u16 = 0x32;
    pub const S_PACK_LH_B32_B16: u16 = 0x33;
    pub const S_PACK_HH_B32_B16: u16 = 0x34;
}

pub mod sop1 {
    /// The 32-bit result of a 64-bit-source s_bcnt1/ff1/flbit (see
    /// S_BCNT1_I32_B64).
    pub fn bit64(op: u16, x: u64) -> Option<u32> {
        let none = |n: u32, z: bool| if z { u32::MAX } else { n };
        Some(match op {
            S_BCNT1_I32_B64 => x.count_ones(),
            S_BCNT0_I32_B64 => x.count_zeros(),
            S_FF0_I32_B64 => none((!x).trailing_zeros(), x == u64::MAX),
            S_FF1_I32_B64 => none(x.trailing_zeros(), x == 0),
            S_FLBIT_I32_B64 => none(x.leading_zeros(), x == 0),
            S_FLBIT_I32_I64 => {
                let y = if (x as i64) < 0 { !x } else { x };
                none(y.leading_zeros(), y == 0)
            }
            _ => return None,
        })
    }
    /// Each bit of `x` twice (see S_BITREPLICATE_B64_B32).
    pub fn bitreplicate(x: u32) -> u64 {
        (0..32).fold(0u64, |r, i| r | ((x as u64 >> i & 1) * 3) << (2 * i))
    }

    /// D = address of the next instruction (LLVM `SIInstrInfo.cpp`: the
    /// value s_getpc returns "is the address of the s_add_u32
    /// instruction" after it). GFX10 opcode from `SOPInstructions.td`.
    pub const S_GETPC_B64: u16 = 0x1F;
    /// D = D & ~(1 << S0[4:0]) / D | (1 << S0[4:0]), no SCC: Mesa ACO
    /// rewrites s_and/s_or with a one-bit constant into these, dropping
    /// the SCC definition (`aco_register_allocation.cpp`).
    pub const S_BITSET0_B32: u16 = 0x1B;
    pub const S_BITSET1_B32: u16 = 0x1D;
    pub const S_MOV_B32: u16 = 3;
    pub const S_MOV_B64: u16 = 4;
    pub const S_CMOV_B32: u16 = 5;
    /// Opcodes and semantics below from AMD's machine-readable RDNA2 ISA
    /// specification (amdgpu_isa_rdna2.xml, GPUOpen): its descriptions
    /// give the operand order and which result sets SCC.
    pub const S_CMOV_B64: u16 = 6;
    pub const S_BCNT0_I32_B64: u16 = 0x0E;
    /// Trailing ones before the first 0; -1 if there is no 0 bit.
    pub const S_FF0_I32_B32: u16 = 0x11;
    pub const S_FF0_I32_B64: u16 = 0x12;
    /// saveexec: D = old EXEC; EXEC = op; SCC = EXEC != 0. "n2" negates
    /// EXEC, "n1" negates S0. wrexec: EXEC = D = op.
    pub const S_XOR_SAVEEXEC_B64: u16 = 0x26;
    pub const S_ORN2_SAVEEXEC_B64: u16 = 0x28;
    pub const S_NAND_SAVEEXEC_B64: u16 = 0x29;
    pub const S_NOR_SAVEEXEC_B64: u16 = 0x2A;
    pub const S_XNOR_SAVEEXEC_B64: u16 = 0x2B;
    pub const S_ANDN1_SAVEEXEC_B64: u16 = 0x37;
    pub const S_ORN1_SAVEEXEC_B64: u16 = 0x38;
    pub const S_ANDN1_WREXEC_B64: u16 = 0x39;
    pub const S_ANDN2_WREXEC_B64: u16 = 0x3A;
    pub const S_XOR_SAVEEXEC_B32: u16 = 0x3E;
    pub const S_ORN2_SAVEEXEC_B32: u16 = 0x40;
    pub const S_NAND_SAVEEXEC_B32: u16 = 0x41;
    pub const S_NOR_SAVEEXEC_B32: u16 = 0x42;
    pub const S_XNOR_SAVEEXEC_B32: u16 = 0x43;
    pub const S_ANDN1_SAVEEXEC_B32: u16 = 0x44;
    pub const S_ORN1_SAVEEXEC_B32: u16 = 0x45;
    pub const S_ANDN1_WREXEC_B32: u16 = 0x46;
    pub const S_ANDN2_WREXEC_B32: u16 = 0x47;

    /// New EXEC from (S0, old EXEC).
    pub type ExecFn = fn(u64, u64) -> u64;

    /// For a saveexec/wrexec opcode: (is the 64-bit form, writes the new
    /// EXEC to D instead of the old, new EXEC from (S0, old EXEC)).
    pub fn exec_op(op: u16) -> Option<(bool, bool, ExecFn)> {
        let f: ExecFn = match op {
            S_AND_SAVEEXEC_B64 | S_AND_SAVEEXEC_B32 => |s, e| s & e,
            S_OR_SAVEEXEC_B64 | S_OR_SAVEEXEC_B32 => |s, e| s | e,
            S_XOR_SAVEEXEC_B64 | S_XOR_SAVEEXEC_B32 => |s, e| s ^ e,
            S_ANDN2_SAVEEXEC_B64 | S_ANDN2_SAVEEXEC_B32 | S_ANDN2_WREXEC_B64 | S_ANDN2_WREXEC_B32 => |s, e| s & !e,
            S_ORN2_SAVEEXEC_B64 | S_ORN2_SAVEEXEC_B32 => |s, e| s | !e,
            S_NAND_SAVEEXEC_B64 | S_NAND_SAVEEXEC_B32 => |s, e| !(s & e),
            S_NOR_SAVEEXEC_B64 | S_NOR_SAVEEXEC_B32 => |s, e| !(s | e),
            S_XNOR_SAVEEXEC_B64 | S_XNOR_SAVEEXEC_B32 => |s, e| !(s ^ e),
            S_ANDN1_SAVEEXEC_B64 | S_ANDN1_SAVEEXEC_B32 | S_ANDN1_WREXEC_B64 | S_ANDN1_WREXEC_B32 => |s, e| e & !s,
            S_ORN1_SAVEEXEC_B64 | S_ORN1_SAVEEXEC_B32 => |s, e| e | !s,
            _ => return None,
        };
        let wide = matches!(
            op,
            S_AND_SAVEEXEC_B64 | S_OR_SAVEEXEC_B64 | S_XOR_SAVEEXEC_B64 | S_ANDN2_SAVEEXEC_B64 | S_ORN2_SAVEEXEC_B64
                | S_NAND_SAVEEXEC_B64 | S_NOR_SAVEEXEC_B64 | S_XNOR_SAVEEXEC_B64 | S_ANDN1_SAVEEXEC_B64
                | S_ORN1_SAVEEXEC_B64 | S_ANDN1_WREXEC_B64 | S_ANDN2_WREXEC_B64
        );
        let wr = matches!(op, S_ANDN1_WREXEC_B64 | S_ANDN2_WREXEC_B64 | S_ANDN1_WREXEC_B32 | S_ANDN2_WREXEC_B32);
        Some((wide, wr, f))
    }
    pub const S_NOT_B32: u16 = 7;
    pub const S_NOT_B64: u16 = 8;
    /// 64-bit scalar bit ops (GFX10 opcodes from `SOP1_Real_gfx6_gfx7_gfx10*`
    /// in `SOPInstructions.td`; ACO selects them for NIR bitfield_reverse,
    /// bit_count, find_lsb, ufind_msb(_rev) and ifind_msb on 64 bits, with
    /// -1 for "no bit"). Only s_bcnt1 writes SCC (`Defs = [SCC]`).
    pub const S_BREV_B64: u16 = 0x0C;
    pub const S_BCNT1_I32_B64: u16 = 0x10;
    pub const S_FF1_I32_B64: u16 = 0x14;
    pub const S_FLBIT_I32_B64: u16 = 0x16;
    pub const S_FLBIT_I32_I64: u16 = 0x18;
    /// D[S0[5:0]] = 0 / 1, no SCC (ACO's s_and_b64/s_or_b64 rewrite, as for
    /// the b32 forms).
    pub const S_BITSET0_B64: u16 = 0x1C;
    pub const S_BITSET1_B64: u16 = 0x1E;
    /// D[2i] = D[2i+1] = S0[i], no SCC (Mesa ACO `copy_constant` builds
    /// a wave64 mask from its de-replicated half this way; opcode from
    /// ACO `aco_opcodes.py` and LLVM `SOP1_Real_gfx10` 0x03b).
    pub const S_BITREPLICATE_B64_B32: u16 = 0x3B;
    pub const S_BREV_B32: u16 = 11;
    pub const S_BCNT0_I32_B32: u16 = 13;
    pub const S_BCNT1_I32_B32: u16 = 15;
    pub const S_FLBIT_I32_B32: u16 = 21;
    /// Scalar `v_ffbh_i32`, no SCC (NIR `ifind_msb_rev`; GFX10 opcode
    /// 0x017 from `SOP1_Real_gfx6_gfx7_gfx10` in `SOPInstructions.td`).
    pub const S_FLBIT_I32: u16 = 0x17;
    pub const S_SEXT_I32_I8: u16 = 25;
    pub const S_SEXT_I32_I16: u16 = 26;
    pub const S_OR_SAVEEXEC_B64: u16 = 37;
    pub const S_ANDN2_SAVEEXEC_B64: u16 = 39;
    pub const S_OR_SAVEEXEC_B32: u16 = 61;
    pub const S_ANDN2_SAVEEXEC_B32: u16 = 63;
    pub const S_FF1_I32_B32: u16 = 19;
    pub const S_AND_SAVEEXEC_B64: u16 = 36;
    pub const S_ABS_I32: u16 = 52;
    pub const S_AND_SAVEEXEC_B32: u16 = 60;
}

pub mod sopc {
    pub const S_CMP_EQ_I32: u16 = 0;
    pub const S_CMP_LG_I32: u16 = 1;
    pub const S_CMP_GT_I32: u16 = 2;
    pub const S_CMP_GE_I32: u16 = 3;
    pub const S_CMP_LT_I32: u16 = 4;
    pub const S_CMP_LE_I32: u16 = 5;
    pub const S_CMP_EQ_U32: u16 = 6;
    pub const S_CMP_LG_U32: u16 = 7;
    pub const S_CMP_GT_U32: u16 = 8;
    pub const S_CMP_GE_U32: u16 = 9;
    pub const S_CMP_LT_U32: u16 = 10;
    pub const S_CMP_LE_U32: u16 = 11;
    /// SCC = bit S1[4:0] of S0 is 0 / 1: Mesa ACO selects these for NIR
    /// `bitz` / `bitnz` (`aco_select_nir_alu.cpp`).
    pub const S_BITCMP0_B32: u16 = 12;
    pub const S_BITCMP1_B32: u16 = 13;
    /// SCC = S0[S1[5:0]] == 0 / 1 (AMD RDNA2 ISA specification).
    pub const S_BITCMP0_B64: u16 = 14;
    pub const S_BITCMP1_B64: u16 = 15;
    pub const S_CMP_EQ_U64: u16 = 18;
    pub const S_CMP_LG_U64: u16 = 19;
}

pub mod vop1 {
    /// f32 -> f16 into D[15:0], D[31:16] kept (LLVM GCNSubtarget::
    /// zeroesHigh16BitsOfDest: "On gfx10, all 16-bit instructions preserve
    /// the high bits"), and f16 (S0[15:0]) -> f32
    /// (`gfx10_asm_vop1.s`: `v_cvt_f16_f32_e32 v5, v1` = 0x7E0A1501).
    pub const V_CVT_F16_F32: u16 = 10;
    pub const V_CVT_F32_F16: u16 = 11;
    pub const V_NOP: u16 = 0;
    pub const V_MOV_B32: u16 = 1;
    /// Swap D and S0 (both VGPRs) in active lanes (AMD RDNA2 ISA spec).
    pub const V_SWAP_B32: u16 = 0x65;
    pub const V_READFIRSTLANE_B32: u16 = 2;
    pub const V_CVT_F32_I32: u16 = 5;
    pub const V_CVT_F32_U32: u16 = 6;
    pub const V_CVT_U32_F32: u16 = 7;
    pub const V_CVT_I32_F32: u16 = 8;
    /// fp_to_sint(floor(x + 0.5)) / fp_to_sint(floor(x)): LLVM's
    /// `cvt_rpi_i32_f32` / `cvt_flr_i32_f32` PatFrags (opcodes from
    /// `gfx10_asm_vop1.s`: `v_cvt_rpi_i32_f32_e32 v5, v1` = 0x7E0A1901).
    pub const V_CVT_RPI_I32_F32: u16 = 0x0C;
    pub const V_CVT_FLR_I32_F32: u16 = 0x0D;
    /// frexp exponent / significand (LLVM's constant folding of
    /// `amdgcn_frexp_exp/mant`: APFloat frexp, exponent 0 for Inf/NaN).
    pub const V_FREXP_EXP_I32_F32: u16 = 0x3F;
    pub const V_FREXP_MANT_F32: u16 = 0x40;
    pub const V_CVT_F32_UBYTE0: u16 = 17;
    pub const V_CVT_F32_UBYTE1: u16 = 18;
    pub const V_CVT_F32_UBYTE2: u16 = 19;
    pub const V_CVT_F32_UBYTE3: u16 = 20;
    pub const V_TRUNC_F32: u16 = 33;
    /// f64 rounding, VOP1 only (opcodes from LLVM `gfx10_asm_vop1.s`:
    /// `v_trunc_f64_e32 v[5:6], v[1:2]` = 0x7E0A2F01, ...).
    /// f64 conversions, VOP1 (opcodes from `VOP1_Real_gfx6_gfx7_gfx10*`,
    /// encodings from `gfx10_asm_vop1.s`). LLVM selects them for
    /// `fp_to_sint_sat`/`fp_to_uint_sat` ("these i32 conversions naturally
    /// saturate"), `sint_to_fp`/`uint_to_fp`, `fpround` and `fpextend`.
    pub const V_CVT_I32_F64: u16 = 0x03;
    pub const V_CVT_F64_I32: u16 = 0x04;
    pub const V_CVT_F32_F64: u16 = 0x0F;
    pub const V_CVT_F64_F32: u16 = 0x10;
    pub const V_CVT_U32_F64: u16 = 0x15;
    pub const V_CVT_F64_U32: u16 = 0x16;
    pub const V_TRUNC_F64: u16 = 0x17;
    pub const V_CEIL_F64: u16 = 0x18;
    pub const V_RNDNE_F64: u16 = 0x19;
    pub const V_FLOOR_F64: u16 = 0x1A;
    pub const V_CEIL_F32: u16 = 34;
    pub const V_RNDNE_F32: u16 = 35;
    pub const V_FLOOR_F32: u16 = 36;
    pub const V_NOT_B32: u16 = 55;
    pub const V_BFREV_B32: u16 = 56;
    pub const V_FFBH_U32: u16 = 57;
    /// Index from the top of the first bit unlike the sign bit, -1 for 0
    /// and -1 (NIR `ifind_msb_rev`, which ACO selects as this; opcode from
    /// `gfx10_asm_vop1.s`: `v_ffbh_i32_e32 v5, v1` = 0x7E0A7701).
    pub const V_FFBH_I32: u16 = 0x3B;
    pub const V_FFBL_B32: u16 = 58;
}

pub mod vop2 {
    /// D = {f16_rtz(S1), f16_rtz(S0)} (LLVM `VOP2_Real_gfx6_gfx7_gfx10<0x02f>`,
    /// `AMDGPUpkrtz_f16_f32`; `gfx10_asm_vop2.s` 0x5E0A0501).
    pub const V_CVT_PKRTZ_F16_F32: u16 = 0x2F;
    /// GFX10 `v_fmamk_f32` D = S0 * K + S1 and `v_fmaak_f32` D = S0 * S1 +
    /// K, K the dword after the instruction (LLVM `VOP_MADMK`/`VOP_MADAK`;
    /// `gfx10_asm_vop2.s`: `v_fmaak_f32 v5, v1, v2, 0x11213141` =
    /// 0x5A0A0501 0x11213141).
    pub const V_FMAMK_F32: u16 = 0x2C;
    pub const V_FMAAK_F32: u16 = 0x2D;

    pub fn is_madk(op: u16) -> bool {
        op == V_FMAMK_F32 || op == V_FMAAK_F32
    }
    /// Carry in from vcc, carry/borrow out to vcc (LLVM `wave32.s`:
    /// `v_add_co_ci_u32_e32 v3, vcc_lo, v3, v4, vcc_lo` = 0x50060903).
    pub const V_ADD_CO_CI_U32: u16 = 40;
    pub const V_SUB_CO_CI_U32: u16 = 41;
    pub const V_SUBREV_CO_CI_U32: u16 = 42;
    pub const V_CNDMASK_B32: u16 = 1;
    pub const V_ADD_F32: u16 = 3;
    pub const V_SUB_F32: u16 = 4;
    pub const V_SUBREV_F32: u16 = 5;
    pub const V_MUL_F32: u16 = 8;
    /// 24-bit multiplies: only S0[23:0] and S1[23:0] are read (LLVM's
    /// simplifyMul24 demands the low 24 bits), sign-extended for i24; lo =
    /// product[31:0], hi = product[63:32] (`mul24_64_Pat` builds the
    /// 64-bit product from the pair). Opcodes from `VOP2_Real_gfx6_gfx7_gfx10*`.
    pub const V_MUL_I32_I24: u16 = 9;
    pub const V_MUL_HI_I32_I24: u16 = 10;
    pub const V_MUL_U32_U24: u16 = 11;
    pub const V_MUL_HI_U32_U24: u16 = 12;
    pub const V_MIN_F32: u16 = 15;
    pub const V_MAX_F32: u16 = 16;
    pub const V_MIN_I32: u16 = 17;
    pub const V_MAX_I32: u16 = 18;
    pub const V_MIN_U32: u16 = 19;
    pub const V_MAX_U32: u16 = 20;
    pub const V_LSHRREV_B32: u16 = 22;
    pub const V_ASHRREV_I32: u16 = 24;
    pub const V_LSHLREV_B32: u16 = 26;
    pub const V_AND_B32: u16 = 27;
    pub const V_OR_B32: u16 = 28;
    pub const V_XOR_B32: u16 = 29;
    /// !(S0 ^ S1) (AMD RDNA2 ISA specification).
    pub const V_XNOR_B32: u16 = 0x1E;
    pub const V_ADD_NC_U32: u16 = 37;
    pub const V_SUB_NC_U32: u16 = 38;
    pub const V_SUBREV_NC_U32: u16 = 39;
    pub const V_FMAC_F32: u16 = 43;
}

pub mod vopc {
    pub const V_CMP_LT_F32: u16 = 1;
    pub const V_CMP_EQ_F32: u16 = 2;
    pub const V_CMP_GT_F32: u16 = 4;
    pub const V_CMP_LT_I32: u16 = 129;
    pub const V_CMP_EQ_I32: u16 = 130;
    pub const V_CMP_GT_I32: u16 = 132;
    pub const V_CMP_NE_I32: u16 = 133;
}

/// VOP3 opcodes. A VOP2 instruction's VOP3 (e64) form is its opcode +
/// [`vop3::FROM_VOP2`], a VOP1's + [`vop3::FROM_VOP1`]; both offsets are
/// confirmed by `v_add_f32_e64`, `v_cndmask_b32_e64` and `v_mov_b32_e64`.
pub mod vop3 {
    /// f64 arithmetic, IEEE (LLVM selects them for `any_fadd`,
    /// `any_fmul`, `any_fma`; opcodes from `gfx10_asm_vop3.s`:
    /// `v_add_f64 v[5:6], v[1:2], v[2:3]` = 0xD5640005 0x02020501).
    pub const V_FMA_F64: u16 = 0x14C;
    pub const V_ADD_F64: u16 = 0x164;
    pub const V_MUL_F64: u16 = 0x165;
    /// IEEE minNum/maxNum like the f32 forms (LLVM `fminnum_like`).
    pub const V_MIN_F64: u16 = 0x166;
    pub const V_MAX_F64: u16 = 0x167;
    /// VOP3b: {carry, D[63:0]} = S0 * S1 (u32) + S2 (u64); the carry is
    /// bit 64 of the sum (LLVM `AMDGPULegalizerInfo::buildMultiply` chains
    /// it into the next partial product). `gfx10_asm_vop3.s`:
    /// `v_mad_u64_u32 v[5:6], s12, v1, v2, v[3:4]` = 0xD5760C05 0x040E0501.
    pub const V_MAD_U64_U32: u16 = 0x176;
    /// D = S0 ^ S1 ^ S2 (`v_xor3_b32 v5, v1, v2, v3` = 0xD5780005 0x040E0501).
    pub const V_XOR3_B32: u16 = 0x178;
    /// ((1 << S0[4:0]) - 1) << S1[4:0] (NIR `bfm`, which ACO selects as
    /// this; `v_bfm_b32 v5, v1, v2` = 0xD7630005 0x00020501).
    pub const V_BFM_B32: u16 = 0x363;
    /// S0 * 2^S1 (S1 a signed integer; LLVM `fldexp`), VOP3 only on
    /// GFX10 (`v_ldexp_f32 v5, v1, v2` = 0xD7620005 0x00020501).
    pub const V_LDEXP_F32: u16 = 0x362;
    /// fminnum(fminnum(a, b), c) / fmaxnum(...): LLVM forms them from such
    /// chains (AMDGPUfmin3/fmax3); `gfx10_asm_vop3.s`: `v_min3_f32 v5, v1,
    /// v2, v3` = 0xD5510005 0x040E0501.
    pub const V_MIN3_F32: u16 = 0x151;
    pub const V_MAX3_F32: u16 = 0x154;
    /// Median, as LLVM's `fmed3AMDGCN` constant folding (no NaNs).
    pub const V_MED3_F32: u16 = 0x157;
    /// (S0 + S1) << S2[4:0] (LLVM `ThreeOp_i32_Pats<add, cshl_32>`).
    pub const V_ADD_LSHL_U32: u16 = 0x347;
    pub const FROM_VOP2: u16 = 0x100;
    pub const FROM_VOP1: u16 = 0x180;
    pub const V_BFE_U32: u16 = 328;
    pub const V_BFI_B32: u16 = 330;
    pub const V_FMA_F32: u16 = 331;
    pub const V_MED3_U32: u16 = 345;
    pub const V_MUL_LO_U32: u16 = 361;
    pub const V_LSHL_ADD_U32: u16 = 838;
    pub const V_ADD3_U32: u16 = 877;
    pub const V_AND_OR_B32: u16 = 881;
    pub const V_OR3_B32: u16 = 882;
    pub const V_MBCNT_LO_U32_B32: u16 = 869;
    pub const V_MBCNT_HI_U32_B32: u16 = 870;
    /// VOP3b: carry-out lane mask in dword0 bits [14:8] (`gfx10_asm_vop3.s`
    /// W32: `v_add_co_u32 v5, s0, v1, v2` = 0xD70F0005).
    /// 16-bit integer VOP3 (`gfx10_asm_vop3.s`): sources S0[15:0],
    /// S1[15:0]; D[15:0] written, D[31:16] kept (GFX10, see V_CVT_F16_F32);
    /// the *rev shifts shift S1 by S0[3:0].
    pub const V_ADD_NC_U16: u16 = 0x303;
    // 16-bit, from AMD's RDNA2 ISA specification (low halves, D[31:16]
    // kept as the other GFX10 16-bit ops)
    pub const V_ADD_NC_I16: u16 = 0x30D;
    pub const V_SUB_NC_I16: u16 = 0x30E;
    pub const V_MAD_U16: u16 = 0x340;
    pub const V_MAD_I16: u16 = 0x35E;
    pub const V_MIN3_I16: u16 = 0x352;
    pub const V_MIN3_U16: u16 = 0x353;
    pub const V_MAX3_I16: u16 = 0x355;
    pub const V_MAX3_U16: u16 = 0x356;
    pub const V_MED3_I16: u16 = 0x358;
    pub const V_MED3_U16: u16 = 0x359;
    pub const V_SUB_NC_U16: u16 = 0x304;
    pub const V_MUL_LO_U16: u16 = 0x305;
    pub const V_LSHRREV_B16: u16 = 0x307;
    pub const V_ASHRREV_I16: u16 = 0x308;
    pub const V_MAX_U16: u16 = 0x309;
    pub const V_MAX_I16: u16 = 0x30a;
    pub const V_MIN_U16: u16 = 0x30b;
    pub const V_MIN_I16: u16 = 0x30c;
    pub const V_LSHLREV_B16: u16 = 0x314;
    /// Byte permute of the 64 bits S0:S1 (S0 high), D byte i from selector
    /// byte i of S2: 0..7 = that byte, 8..11 = the sign bit of byte
    /// 2 (sel - 8) + 1 replicated, 12 = 0x00, 13.. = 0xFF (Mesa NIR
    /// `byte_perm_amd`, which ACO selects v_perm_b32 for; LLVM
    /// computeKnownBits for AMDGPUISD::PERM agrees). VOP3 0x344.
    pub const V_PERM_B32: u16 = 0x344;
    /// D = (S0:S1 >> 8 (S2 & 3))[31:0] (NIR `alignbyte_amd`; VOP3 0x14F).
    pub const V_ALIGNBYTE_B32: u16 = 0x14F;
    /// D = sat16(S0) | sat16(S1) << 16, unsigned / signed clamp (NIR
    /// `pack_uint_2x16` / `pack_sint_2x16`, which ACO selects these for;
    /// `gfx10_asm_vop3.s` 0x36A / 0x36B).
    pub const V_CVT_PK_U16_U32: u16 = 0x36A;
    pub const V_CVT_PK_I16_I32: u16 = 0x36B;
    /// Masked SAD: S2 + sum over bytes with S1's byte != 0 of
    /// |S0 byte - S1 byte| (NIR `msad_4x8` with the reference as its src0,
    /// which ACO swaps into S1; VOP3 0x171).
    pub const V_MSAD_U8: u16 = 0x171;
    // From AMD's machine-readable RDNA2 ISA specification:
    /// D = S0 + S1 / S0 - S1, wrapping (no carry; clamp refused).
    pub const V_ADD_NC_I32: u16 = 0x37F;
    pub const V_SUB_NC_I32: u16 = 0x376;
    /// S0[15:0] * S1[15:0] (unsigned / signed) + S2, 32-bit.
    pub const V_MAD_U32_U16: u16 = 0x373;
    pub const V_MAD_I32_I16: u16 = 0x375;
    /// Sums of absolute differences of bytes / halves / words plus S2;
    /// sad_hi_u8 shifts the sum left by 16 first.
    pub const V_SAD_U8: u16 = 0x15A;
    pub const V_SAD_HI_U8: u16 = 0x15B;
    pub const V_SAD_U16: u16 = 0x15C;
    pub const V_SAD_U32: u16 = 0x15D;
    /// Per byte (S0 + S1 + (S2 & 1)) >> 1.
    pub const V_LERP_U8: u16 = 0x14D;
    pub const V_ADD_CO_U32: u16 = 783;
    /// D[63:0] = S1[63:0] shifted by S0[5:0] (LLVM `clshl_rev_64` etc.;
    /// `gfx10_asm_vop3.s`: `v_lshlrev_b64 v[5:6], v1, v[2:3]` = 0xD6FF0005).
    pub const V_LSHLREV_B64: u16 = 0x2FF;
    /// `v_readlane_b32 sdst, vsrc, lane` / `v_writelane_b32 vdst, ssrc,
    /// lane` (VOP3 0x360 / 0x361, `gfx10_asm_vop3.s`). Neither depends on
    /// exec: Mesa ACO notes "writelane ignores exec" and "VALU which
    /// writes SGPRs are always executed on GFX10+".
    pub const V_READLANE_B32: u16 = 0x360;
    pub const V_WRITELANE_B32: u16 = 0x361;
    pub const V_LSHRREV_B64: u16 = 0x300;
    pub const V_ASHRREV_I64: u16 = 0x301;
    pub const V_SUB_CO_U32: u16 = 784;
    pub const V_SUBREV_CO_U32: u16 = 793;
    // Opcodes from `gfx10_asm_vop3.s`; semantics from LLVM's selection
    // patterns: mul_hi = mulhu/mulhs, bcnt = add_ctpop, alignbit = fshr,
    // xad = (xor, add), lshl_or = (shl, or), mad_*24 = MAD_U24/I24,
    // bfe_i32 as AMDGPUISelLowering's constantFoldBFE.
    pub const V_MUL_HI_U32: u16 = 362;
    pub const V_MUL_LO_I32: u16 = 363;
    pub const V_MUL_HI_I32: u16 = 364;
    pub const V_MAD_I32_I24: u16 = 322;
    pub const V_MAD_U32_U24: u16 = 323;
    pub const V_BCNT_U32_B32: u16 = 868;
    pub const V_ALIGNBIT_B32: u16 = 334;
    pub const V_BFE_I32: u16 = 329;
    pub const V_MIN3_I32: u16 = 338;
    pub const V_MIN3_U32: u16 = 339;
    pub const V_MAX3_I32: u16 = 341;
    pub const V_MAX3_U32: u16 = 342;
    pub const V_MED3_I32: u16 = 344;
    pub const V_XAD_U32: u16 = 837;
    pub const V_LSHL_OR_B32: u16 = 879;
}

/// SMEM opcodes (`s_buffer_load_*` confirmed by gfx10_asm_smem.s, which
/// also confirms the operand encoding: `soffset` 125 = null, 124 = m0, and
/// the immediate offset and `soffset` are added together).
pub mod smem {
    pub const S_LOAD_DWORD: u16 = 0;
    pub const S_LOAD_DWORDX2: u16 = 1;
    pub const S_LOAD_DWORDX4: u16 = 2;
    pub const S_LOAD_DWORDX8: u16 = 3;
    pub const S_LOAD_DWORDX16: u16 = 4;
    pub const S_BUFFER_LOAD_DWORD: u16 = 8;
    pub const S_BUFFER_LOAD_DWORDX2: u16 = 9;
    pub const S_BUFFER_LOAD_DWORDX4: u16 = 10;
    /// `gfx10_asm_smem.s`: `s_buffer_load_dwordx8 s[20:27], s[4:7], s0` =
    /// 0xF42C0502, x16 = 0xF4300502.
    pub const S_BUFFER_LOAD_DWORDX8: u16 = 11;
    pub const S_BUFFER_LOAD_DWORDX16: u16 = 12;
    pub const SOFFSET_NULL: u8 = 125;

    /// Dwords loaded by `s_buffer_load_dword*` opcode `op`.
    pub fn buffer_load_dwords(op: u16) -> Option<usize> {
        Some(match op {
            S_BUFFER_LOAD_DWORD => 1,
            S_BUFFER_LOAD_DWORDX2 => 2,
            S_BUFFER_LOAD_DWORDX4 => 4,
            S_BUFFER_LOAD_DWORDX8 => 8,
            S_BUFFER_LOAD_DWORDX16 => 16,
            _ => return None,
        })
    }

    /// Dwords loaded by `s_load_dword*` opcode `op`.
    pub fn load_dwords(op: u16) -> Option<usize> {
        Some(match op {
            S_LOAD_DWORD => 1,
            S_LOAD_DWORDX2 => 2,
            S_LOAD_DWORDX4 => 4,
            S_LOAD_DWORDX8 => 8,
            S_LOAD_DWORDX16 => 16,
            _ => return None,
        })
    }
}

/// GLOBAL (FLAT segment 2) opcodes and operand encoding, from LLVM's
/// `flat-global.s` GFX10 checks: a 12-bit signed byte offset, `saddr` 125
/// meaning `off` (64-bit address in a VGPR pair), otherwise an SGPR-pair
/// base plus a 32-bit VGPR offset.
pub mod global {
    pub const SEG_GLOBAL: u8 = 2;
    pub const SADDR_OFF: u8 = 125;
    pub const LOAD_DWORD: u16 = 12;
    pub const LOAD_DWORDX2: u16 = 13;
    pub const LOAD_DWORDX4: u16 = 14;
    pub const STORE_DWORD: u16 = 28;
    pub const STORE_DWORDX2: u16 = 29;
    pub const STORE_DWORDX4: u16 = 30;
    pub const LOAD_DWORDX3: u16 = 15;
    pub const STORE_DWORDX3: u16 = 31;
    /// Sub-dword loads (zero / sign extended; LLVM `FLATInstructions.td`).
    pub const LOAD_UBYTE: u16 = 8;
    pub const LOAD_SBYTE: u16 = 9;
    pub const LOAD_USHORT: u16 = 10;
    pub const LOAD_SSHORT: u16 = 11;

    /// Sub-dword stores and d16 loads (AMD's RDNA2 ISA specification):
    /// stores write the low 8/16 bits of data (the `_d16_hi` forms the
    /// high half's); d16 loads extend to 16 bits and write the low or
    /// (`_hi`) high half of vdst, keeping the other half.
    pub const STORE_BYTE: u16 = 0x18;
    pub const STORE_BYTE_D16_HI: u16 = 0x19;
    pub const STORE_SHORT: u16 = 0x1A;
    pub const STORE_SHORT_D16_HI: u16 = 0x1B;
    pub const LOAD_UBYTE_D16: u16 = 0x20;
    pub const LOAD_UBYTE_D16_HI: u16 = 0x21;
    pub const LOAD_SBYTE_D16: u16 = 0x22;
    pub const LOAD_SBYTE_D16_HI: u16 = 0x23;
    pub const LOAD_SHORT_D16: u16 = 0x24;
    pub const LOAD_SHORT_D16_HI: u16 = 0x25;

    /// (bytes, from the high half of data) of a sub-dword store.
    pub fn store_sub(op: u16) -> Option<(u32, bool)> {
        Some(match op {
            STORE_BYTE => (1, false),
            STORE_BYTE_D16_HI => (1, true),
            STORE_SHORT => (2, false),
            STORE_SHORT_D16_HI => (2, true),
            _ => return None,
        })
    }

    /// (bytes, sign-extended, into the high half) of a d16 load.
    pub fn load_d16(op: u16) -> Option<(u32, bool, bool)> {
        Some(match op {
            LOAD_UBYTE_D16 => (1, false, false),
            LOAD_UBYTE_D16_HI => (1, false, true),
            LOAD_SBYTE_D16 => (1, true, false),
            LOAD_SBYTE_D16_HI => (1, true, true),
            LOAD_SHORT_D16 => (2, false, false),
            LOAD_SHORT_D16_HI => (2, false, true),
            _ => return None,
        })
    }

    /// (bytes, sign-extended) of a sub-dword load.
    pub fn sub_dword(op: u16) -> Option<(u32, bool)> {
        Some(match op {
            LOAD_UBYTE => (1, false),
            LOAD_SBYTE => (1, true),
            LOAD_USHORT => (2, false),
            LOAD_SSHORT => (2, true),
            _ => return None,
        })
    }

    /// Value of a sub-dword load at byte offset `byte` (0..4) of `dword`.
    pub fn extract(dword: u32, byte: u32, bytes: u32, signed: bool) -> u32 {
        let x = dword >> (8 * byte);
        let bits = 8 * bytes;
        let m = (1u32 << bits) - 1;
        if signed {
            (((x & m) << (32 - bits)) as i32 >> (32 - bits)) as u32
        } else {
            x & m
        }
    }

    /// GFX10 global atomics (LLVM `FLATInstructions.td`,
    /// `FLAT_Real_GlblAtomics_gfx10`; encodings checked in `flat-global.s`).
    /// With `glc` the pre-operation value is returned in `vdst`.
    pub const ATOMIC_SWAP: u16 = 0x30;
    /// data = {src, cmp}: LLVM builds the operand as `{New, Old}`
    /// (`SIISelLowering.cpp`, atomic cmpxchg lowering).
    pub const ATOMIC_CMPSWAP: u16 = 0x31;
    pub const ATOMIC_ADD: u16 = 0x32;
    pub const ATOMIC_SUB: u16 = 0x33;
    pub const ATOMIC_SMIN: u16 = 0x35;
    pub const ATOMIC_UMIN: u16 = 0x36;
    pub const ATOMIC_SMAX: u16 = 0x37;
    pub const ATOMIC_UMAX: u16 = 0x38;
    pub const ATOMIC_AND: u16 = 0x39;
    pub const ATOMIC_OR: u16 = 0x3A;
    pub const ATOMIC_XOR: u16 = 0x3B;

    /// New memory value of a 32-bit atomic, or None if `op` is not one.
    pub fn atomic(op: u16, old: u32, data: u32, cmp: u32) -> Option<u32> {
        Some(match op {
            ATOMIC_SWAP => data,
            ATOMIC_CMPSWAP => {
                if old == cmp {
                    data
                } else {
                    old
                }
            }
            ATOMIC_ADD => old.wrapping_add(data),
            ATOMIC_SUB => old.wrapping_sub(data),
            ATOMIC_SMIN => (old as i32).min(data as i32) as u32,
            ATOMIC_UMIN => old.min(data),
            ATOMIC_SMAX => (old as i32).max(data as i32) as u32,
            ATOMIC_UMAX => old.max(data),
            ATOMIC_AND => old & data,
            ATOMIC_OR => old | data,
            ATOMIC_XOR => old ^ data,
            _ => return None,
        })
    }

    /// The final memory value does not depend on the order in which
    /// lanes apply the op.
    pub fn atomic_commutes(op: u16) -> bool {
        !matches!(op, ATOMIC_SWAP | ATOMIC_CMPSWAP)
    }

    /// (dwords, is_store) for a global opcode.
    pub fn access(op: u16) -> Option<(usize, bool)> {
        Some(match op {
            LOAD_DWORD => (1, false),
            LOAD_DWORDX2 => (2, false),
            LOAD_DWORDX4 => (4, false),
            LOAD_DWORDX3 => (3, false),
            STORE_DWORDX3 => (3, true),
            STORE_DWORD => (1, true),
            STORE_DWORDX2 => (2, true),
            STORE_DWORDX4 => (4, true),
            _ => return None,
        })
    }
}

/// LDS (DS) opcodes from LLVM's `gfx10_asm_ds.s`. Single-address ops take
/// a 16-bit byte offset (`offset1:offset0`); `ds_read2_b32` takes two 8-bit
/// offsets in dwords.
pub mod ds {
    pub const WRITE_B32: u16 = 13;
    pub const READ_B32: u16 = 54;
    pub const READ2_B32: u16 = 55;
    /// Two dwords at offset0 * 4 and offset1 * 4 (data0, data1); the ST64
    /// forms scale the offsets by 64 elements (LLVM SILoadStoreOptimizer:
    /// `Offset = EltOffset / 64`), i.e. 256 bytes for b32.
    pub const WRITE2_B32: u16 = 14;
    /// Cross-lane read within 32-lane groups, no LDS access (offset =
    /// offset1:offset0). QUAD_PERM (offset[15:8] = 0x80): lane reads lane
    /// (l & ~3) | sel[l & 3], sel 2-bit fields of offset[7:0]; BITMASK
    /// (offset[15] = 0): lane reads ((l & and) | or) ^ xor within its 32,
    /// and/or/xor = offset[4:0]/[9:5]/[14:10] (LLVM AMDGPUModifierSyntax
    /// and the assembler's encodeBitmaskPerm). FFT/ROTATE not modelled.
    pub const SWIZZLE_B32: u16 = 53;
    /// Wave-level LDS counter: one atomic add of +1 (append) or -1
    /// (consume) per active lane on the dword at M0 + offset, every active
    /// lane gets the uniform pre-op value (NIR `shared_append_amd` /
    /// `shared_consume_amd`; ACO `ds_append` / `ds_consume`; LLVM puts the
    /// base pointer in M0, `SelectDSAppendConsume`). Opcodes from LLVM
    /// `DS_Real_gfx6_gfx7_gfx10*` (0x3d) and `gfx10_ds.txt` encodings.
    pub const CONSUME: u16 = 0x3D;
    pub const APPEND: u16 = 0x3E;

    /// Source lane of ds_swizzle `offset` for `lane`; None where not
    /// modelled (FFT/ROTATE, or overlapping or/xor bits whose order the
    /// sources do not fix).
    pub fn swizzle_source(offset: u32, lane: usize) -> Option<usize> {
        let l = lane as u32;
        if offset & 0xFF00 == 0x8000 {
            let sel = offset >> (2 * (l & 3)) & 3;
            return Some(((l & !3) | sel) as usize);
        }
        if offset & 0x8000 != 0 {
            return None;
        }
        let (and, or, xor) = (offset & 31, offset >> 5 & 31, offset >> 10 & 31);
        if or & xor != 0 {
            return None;
        }
        Some(((l & !31) | (((l & 31 & and) | or) ^ xor)) as usize)
    }
    pub const WRITE2ST64_B32: u16 = 15;
    pub const READ2ST64_B32: u16 = 56;
    pub const WRITE_B64: u16 = 77;
    pub const READ_B64: u16 = 118;
    // More loads/stores (GFX10 opcodes from `gfx10_asm_ds.s` encodings).
    pub const WRITE_B8: u16 = 0x1E;
    pub const WRITE_B16: u16 = 0x1F;
    pub const READ_I8: u16 = 0x39;
    pub const READ_U8: u16 = 0x3A;
    pub const READ_I16: u16 = 0x3B;
    pub const READ_U16: u16 = 0x3C;
    /// Two qwords at offset0 * 8 and offset1 * 8 (ST64: * 512), as the
    /// b32 forms scale by the element size (`SILoadStoreOptimizer`).
    pub const WRITE2_B64: u16 = 0x4E;
    pub const WRITE2ST64_B64: u16 = 0x4F;
    pub const READ2_B64: u16 = 0x77;
    pub const READ2ST64_B64: u16 = 0x78;
    pub const WRITE_B96: u16 = 0xDE;
    pub const WRITE_B128: u16 = 0xDF;
    pub const READ_B96: u16 = 0xFE;
    pub const READ_B128: u16 = 0xFF;

    /// A dword-granular load/store: per dword its byte offset from the
    /// lane address, the data register as (operand 0/1, sub-register) for
    /// writes (reads go to vdst + k), and the alignment the interpreter
    /// requires of that dword's address. b96/b128 need 16 bytes and the
    /// b64 halves of read2/write2 8 bytes: what LLVM requires without
    /// unaligned access mode, which is a configuration the shader does not
    /// show.
    pub struct Layout {
        pub dwords: Vec<(u32, (u8, u8), u32)>,
        pub write: bool,
    }

    pub fn layout(op: u16, offset0: u8, offset1: u8) -> Option<Layout> {
        let off16 = offset0 as u32 | (offset1 as u32) << 8;
        let (o0, o1) = (offset0 as u32, offset1 as u32);
        let seq = |n: u32, align: u32| (0..n).map(|k| (off16 + 4 * k, (0, k as u8), if k == 0 { align } else { 4 })).collect::<Vec<_>>();
        let pair = |a: u32, b: u32, size: u32| {
            let mut v = Vec::new();
            for (sel, base) in [(0u8, a), (1u8, b)] {
                for k in 0..size / 4 {
                    v.push((base + 4 * k, (sel, k as u8), if k == 0 { size } else { 4 }));
                }
            }
            v
        };
        let (dwords, write) = match op {
            WRITE_B32 => (seq(1, 4), true),
            WRITE_B64 => (seq(2, 4), true),
            WRITE_B96 => (seq(3, 16), true),
            WRITE_B128 => (seq(4, 16), true),
            READ_B32 => (seq(1, 4), false),
            READ_B64 => (seq(2, 4), false),
            READ_B96 => (seq(3, 16), false),
            READ_B128 => (seq(4, 16), false),
            READ2_B32 => (pair(o0 * 4, o1 * 4, 4), false),
            READ2ST64_B32 => (pair(o0 * 256, o1 * 256, 4), false),
            WRITE2_B32 => (pair(o0 * 4, o1 * 4, 4), true),
            WRITE2ST64_B32 => (pair(o0 * 256, o1 * 256, 4), true),
            READ2_B64 => (pair(o0 * 8, o1 * 8, 8), false),
            READ2ST64_B64 => (pair(o0 * 512, o1 * 512, 8), false),
            WRITE2_B64 => (pair(o0 * 8, o1 * 8, 8), true),
            WRITE2ST64_B64 => (pair(o0 * 512, o1 * 512, 8), true),
            _ => return None,
        };
        Some(Layout { dwords, write })
    }

    /// Sub-dword access: (bytes, sign-extending load, is a write).
    pub fn sub_dword(op: u16) -> Option<(u32, bool, bool)> {
        Some(match op {
            READ_U8 => (1, false, false),
            READ_I8 => (1, true, false),
            READ_U16 => (2, false, false),
            READ_I16 => (2, true, false),
            WRITE_B8 => (1, false, true),
            WRITE_B16 => (2, false, true),
            _ => return None,
        })
    }

    // LDS atomics (LLVM `DSInstructions.td`, `DS_Real_gfx6_gfx7_gfx10*`);
    // the `_RTN` forms (+0x20) return the old value in vdst.
    pub const ADD_U32: u16 = 0x00;
    pub const SUB_U32: u16 = 0x01;
    pub const MIN_I32: u16 = 0x05;
    pub const MAX_I32: u16 = 0x06;
    pub const MIN_U32: u16 = 0x07;
    pub const MAX_U32: u16 = 0x08;
    pub const AND_B32: u16 = 0x09;
    pub const OR_B32: u16 = 0x0A;
    pub const XOR_B32: u16 = 0x0B;
    /// Up to GFX10 data0 is the comparand and data1 the new value, the
    /// opposite of BUFFER/GLOBAL cmpswap (LLVM `DSAtomicCmpXChgSwapped`).
    pub const CMPST_B32: u16 = 0x10;
    pub const RTN: u16 = 0x20;
    pub const WRXCHG_RTN_B32: u16 = 0x2D;
    pub const CMPST_RTN_B32: u16 = 0x30;

    /// (base op, returns the old value) of an LDS atomic. rsub, inc, dec
    /// and mskor are not modelled.
    pub fn atomic(op: u16) -> Option<(u16, bool)> {
        Some(match op {
            ADD_U32 | SUB_U32 | MIN_I32..=XOR_B32 | CMPST_B32 => (op, false),
            0x20 | 0x21 | 0x25..=0x2B | CMPST_RTN_B32 => (op - RTN, true),
            WRXCHG_RTN_B32 => (op, true),
            _ => return None,
        })
    }

    /// New LDS value for base op `base`.
    pub fn atomic_apply(base: u16, old: u32, data0: u32, data1: u32) -> u32 {
        match base {
            ADD_U32 => old.wrapping_add(data0),
            SUB_U32 => old.wrapping_sub(data0),
            MIN_I32 => (old as i32).min(data0 as i32) as u32,
            MAX_I32 => (old as i32).max(data0 as i32) as u32,
            MIN_U32 => old.min(data0),
            MAX_U32 => old.max(data0),
            AND_B32 => old & data0,
            OR_B32 => old | data0,
            XOR_B32 => old ^ data0,
            CMPST_B32 => {
                if old == data0 {
                    data1
                } else {
                    old
                }
            }
            WRXCHG_RTN_B32 => data0,
            _ => unreachable!("not an LDS atomic"),
        }
    }

    /// The final value does not depend on the lanes' order.
    pub fn atomic_commutes(base: u16) -> bool {
        !matches!(base, CMPST_B32 | WRXCHG_RTN_B32)
    }
}

/// DPP16 control word (the dword after a VOP1/VOP2/VOPC with src0 = 250).
/// Layout from LLVM's GFX10 `_dpp` encodings: src0 [7:0], dpp_ctrl [16:8],
/// bound_ctrl 19, src0 neg/abs 20/21, src1 neg/abs 22/23, bank_mask
/// [27:24], row_mask [31:28]. Bit 18 (fetch-inactive) was not seen in the
/// tests and is reported, not interpreted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dpp16 {
    pub src0: u8,
    pub ctrl: u16,
    pub fi: bool,
    pub bound_ctrl: bool,
    pub src0_neg: bool,
    pub src0_abs: bool,
    pub src1_neg: bool,
    pub src1_abs: bool,
    pub bank_mask: u8,
    pub row_mask: u8,
}

/// Source-lane rule of a DPP16 control, per Mesa ACO's `dpp_ctrl` values
/// (`aco_builder_h.py`) and its use of `row_shr` as "shift rows right" in
/// scans: lane `i` of a 16-lane row reads lane `i - n` for `row_shr:n`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DppRule {
    /// quad_perm:[a,b,c,d] (2-bit selectors, lane 0 first).
    QuadPerm(u8),
    RowShl(u8),
    RowShr(u8),
    RowRor(u8),
    RowMirror,
    RowHalfMirror,
    RowShare(u8),
    RowXmask(u8),
}

impl Dpp16 {
    pub fn parse(w: u32) -> Self {
        let b = |n: u32| w >> n & 1 != 0;
        Dpp16 {
            src0: w as u8,
            ctrl: ((w >> 8) & 0x1FF) as u16,
            fi: b(18),
            bound_ctrl: b(19),
            src0_neg: b(20),
            src0_abs: b(21),
            src1_neg: b(22),
            src1_abs: b(23),
            bank_mask: ((w >> 24) & 0xF) as u8,
            row_mask: ((w >> 28) & 0xF) as u8,
        }
    }

    /// The rule, or None for controls GFX10 lacks or this crate does not
    /// model (wavefront shifts, row broadcasts).
    pub fn rule(&self) -> Option<DppRule> {
        let n = (self.ctrl & 0xF) as u8;
        Some(match self.ctrl {
            0x000..=0x0FF => DppRule::QuadPerm(self.ctrl as u8),
            0x101..=0x10F => DppRule::RowShl(n),
            0x111..=0x11F => DppRule::RowShr(n),
            0x121..=0x12F => DppRule::RowRor(n),
            0x140 => DppRule::RowMirror,
            0x141 => DppRule::RowHalfMirror,
            0x150..=0x15F => DppRule::RowShare(n),
            0x160..=0x16F => DppRule::RowXmask(n),
            _ => return None,
        })
    }
}

impl DppRule {
    /// Source lane for `lane`, or None when it falls outside the row.
    pub fn source(self, lane: usize) -> Option<usize> {
        let (row, r) = (lane & !15, lane & 15);
        Some(match self {
            DppRule::QuadPerm(sel) => (lane & !3) | ((sel as usize >> (2 * (lane & 3))) & 3),
            DppRule::RowShl(n) => {
                if r + n as usize >= 16 {
                    return None;
                }
                lane + n as usize
            }
            DppRule::RowShr(n) => {
                if r < n as usize {
                    return None;
                }
                lane - n as usize
            }
            DppRule::RowRor(n) => row | ((r + 16 - n as usize) & 15),
            DppRule::RowMirror => row | (15 - r),
            DppRule::RowHalfMirror => (lane & !7) | (7 - (lane & 7)),
            DppRule::RowShare(n) => row | n as usize,
            DppRule::RowXmask(n) => row | (r ^ n as usize),
        })
    }
}

/// MIMG opcodes and flags from LLVM's `gfx10_asm_mimg.s`.
pub mod mimg {
    pub const IMAGE_LOAD: u16 = 0;
    pub const IMAGE_STORE: u16 = 8;
    pub const DIM_2D: u8 = 1;
    /// dword0 bits.
    pub const R128: u32 = 1 << 15;
    pub const TFE: u32 = 1 << 16;
    pub const LWE: u32 = 1 << 17;
    /// dword1 bits.
    pub const A16: u32 = 1 << 30;
    pub const D16: u32 = 1 << 31;
}

/// GFX10 image formats (Mesa `gfx10-rsrc.json`, GFX10_FORMAT) this crate
/// reads: (bytes per texel, channels, float result).
pub fn image_format(fmt: u32) -> Option<(u32, u32, bool)> {
    Some(match fmt {
        20 | 21 => (4, 1, false), // 32_UINT, 32_SINT
        22 => (4, 1, true),       // 32_FLOAT
        56 => (4, 4, true),       // 8_8_8_8_UNORM
        62 => (8, 2, false),      // 32_32_UINT
        77 => (16, 4, true),      // 32_32_32_32_FLOAT
        _ => return None,
    })
}

pub const FORMAT_8_8_8_8_UNORM: u32 = 56;

/// UNORM8 → float: `b / 255`, correctly rounded. ASSUMPTION: the texture
/// unit converts exactly (the result is a correctly rounded quotient); to
/// be confirmed on hardware.
pub fn unorm8_to_f32(b: u32) -> u32 {
    ((b & 0xFF) as f32 / 255.0).to_bits()
}

/// Image resource descriptor ("T#", 8 dwords), GFX10 fields from Mesa's
/// `gfx10-rsrc.json` (SQ_IMG_RSRC_WORD0..7) and `ac_descriptors.c`
/// (WORD0 = address >> 8, WIDTH/HEIGHT store size - 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageDesc {
    pub base: u64,
    pub format: u32,
    pub width: u32,
    pub height: u32,
    /// DST_SEL_X..W: SQ_SEL 0 = 0, 1 = 1, 4..7 = X..W.
    pub dst_sel: [u32; 4],
    pub base_level: u32,
    /// WORD3[19:16] (Mesa `gfx10-rsrc.json`).
    pub last_level: u32,
    pub sw_mode: u32,
    /// WORD4.DEPTH; on GFX10.3 it may carry a custom linear pitch.
    pub depth: u32,
}

impl ImageDesc {
    pub fn parse(w: [u32; 8]) -> Self {
        ImageDesc {
            base: (w[0] as u64) << 8 | ((w[1] & 0xFF) as u64) << 40,
            format: (w[1] >> 20) & 0x1FF,
            width: ((w[1] >> 30) | (w[2] & 0xFFF) << 2) + 1,
            height: ((w[2] >> 14) & 0x3FFF) + 1,
            dst_sel: [w[3] & 7, (w[3] >> 3) & 7, (w[3] >> 6) & 7, (w[3] >> 9) & 7],
            base_level: (w[3] >> 12) & 0xF,
            last_level: (w[3] >> 16) & 0xF,
            sw_mode: (w[3] >> 20) & 0x1F,
            depth: w[4] & 0x1FFF,
        }
    }

    /// Linear pitch in texels: addrlib's GFX10 `HwlComputeSurfaceInfoLinear`
    /// aligns the width to 256 bytes' worth of elements.
    pub fn linear_pitch(&self, bytes: u32) -> u32 {
        let a = 256 / bytes;
        (self.width + a - 1) & !(a - 1)
    }
}

/// Pixel-shader input VGPRs: SPI_PS_INPUT_ENA/ADDR bits in register field
/// order (Mesa `gfx10.json`) with their VGPR counts (RADV
/// `radv_shader_args.c`). The hardware packs the inputs enabled in ADDR in
/// this order from v0; only those also enabled in ENA are loaded.
pub mod ps_input {
    pub const COUNTS: [u32; 16] = [2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    pub const PERSP_CENTER: u32 = 1;
    pub const POS_X_FLOAT: u32 = 8;
    pub const POS_W_FLOAT: u32 = 11;
    /// One VGPR, a float: per Mesa (`ac_nir_lower_intrinsics_to_args.c`
    /// lowers `load_front_face_fsign`, defined in `nir_intrinsics.py` as
    /// `front_face ? 1.0 : -1.0`, to the raw argument) +1.0 for a front
    /// face and -1.0 for a back face.
    pub const FRONT_FACE: u32 = 12;

    /// First VGPR of input `bit` under `addr`, if allocated.
    pub fn vgpr(addr: u32, bit: u32) -> Option<u32> {
        (addr >> bit & 1 != 0).then(|| {
            (0..bit).filter(|b| addr >> b & 1 != 0).map(|b| COUNTS[b as usize]).sum()
        })
    }
}

/// VINTRP opcodes (LLVM SIInstructions.td; GFX10 uses the SI encoding,
/// prefix 0x32, per SIInstrInfo.td's VINTRP_m) and v_interp_mov slots.
pub mod vintrp {
    pub const P1_F32: u16 = 0;
    pub const P2_F32: u16 = 1;
    pub const MOV_F32: u16 = 2;
    pub const SLOT_P10: u8 = 0;
    pub const SLOT_P20: u8 = 1;
    pub const SLOT_P0: u8 = 2;
}

/// Export targets (EXP `target`): 0..7 MRTs, 9 null, 12..15 positions,
/// 32..63 parameters.
pub mod exp {
    pub const MRT_LAST: u8 = 7;
    pub const NULL: u8 = 9;
    pub const POS0: u8 = 12;
    pub const POS3: u8 = 15;
    pub const PARAM0: u8 = 32;
    pub const PARAM_LAST: u8 = 63;
}

/// Legacy (non-NGG) GFX10 vertex-shader input VGPRs, per RADV's
/// `declare_vs_input_vgprs`: v0 vertex id (zero-based: VertexIndex -
/// BaseVertex, Mesa's `load_vertex_id_zero_base`), v1 unused, v2
/// primitive id, v3 instance id (zero-based). SPI_SHADER_PGM_RSRC1_VS.
/// VGPR_COMP_CNT (bits 25:24, Mesa `gfx10.json`) is the last one loaded.
pub mod vs_input {
    pub const VERTEX_ID: usize = 0;
    pub const PRIM_ID: usize = 2;
    pub const INSTANCE_ID: usize = 3;

    pub fn comp_cnt(rsrc1: u32) -> u32 {
        (rsrc1 >> 24) & 3
    }
}

/// MUBUF opcodes (LLVM `gfx10_asm_mubuf.s`): (dwords, is_store).
pub mod mubuf {
    pub const LOAD_DWORD: u16 = 12;
    pub const LOAD_DWORDX2: u16 = 13;
    pub const LOAD_DWORDX4: u16 = 14;
    pub const LOAD_DWORDX3: u16 = 15;
    pub const STORE_DWORD: u16 = 28;
    pub const STORE_DWORDX2: u16 = 29;
    pub const STORE_DWORDX4: u16 = 30;
    pub const STORE_DWORDX3: u16 = 31;
    /// Sub-dword loads: the GLOBAL numbers (`global::sub_dword`), checked on
    /// `gfx10_asm_mubuf.s` (`buffer_load_ubyte` = 0xE0200FFF).
    pub fn sub_dword(op: u16) -> Option<(u32, bool)> {
        super::global::sub_dword(op)
    }
    /// dword0 bit 14: return the pre-op value of an atomic in vdata.
    /// Atomic opcodes on GFX10 are those of GLOBAL (`global::ATOMIC_*`,
    /// LLVM `BUFInstructions.td`), cmpswap data = {src, cmp}.
    pub const GLC: u32 = 1 << 14;
    /// dword0 bit 16: load to LDS (not modelled).
    pub const LDS: u32 = 1 << 16;

    pub fn access(op: u16) -> Option<(usize, bool)> {
        Some(match op {
            LOAD_DWORD => (1, false),
            LOAD_DWORDX2 => (2, false),
            LOAD_DWORDX3 => (3, false),
            LOAD_DWORDX4 => (4, false),
            STORE_DWORD => (1, true),
            STORE_DWORDX2 => (2, true),
            STORE_DWORDX3 => (3, true),
            STORE_DWORDX4 => (4, true),
            _ => return None,
        })
    }
}

/// SDWA (the dword after a VOP1/VOP2 with src0 = 249). Layout from LLVM
/// `VOPInstructions.td` (VOP_SDWA9e, bit numbers relative to the SDWA
/// word): src0 [7:0], dst_sel [10:8], dst_unused [12:11], clamp 13,
/// src0_sel [18:16], src0_sext 19, src0 neg/abs [21:20], src0 is an SGPR
/// 23, src1_sel [26:24], src1_sext 27, src1 neg/abs [29:28], src1 SGPR 31;
/// selector and dst_unused values from `SIDefines.h` (BYTE_0..3 = 0..3,
/// WORD_0/1 = 4/5, DWORD = 6; PAD 0, SEXT 1, PRESERVE 2), checked on
/// `gfx10_asm_vop2.s` (`v_and_b32_sdwa v5, v1, v2 dst_sel:BYTE_1` has the
/// SDWA word 0x06060101). Meaning, as in LLVM's SIPeepholeSDWA: a source
/// selector reads that byte/word shifted down (sign-extended with sext),
/// dst_sel writes the result's low byte/word at that position, the other
/// bits being 0 (PAD), the field's sign (SEXT) or the old value (PRESERVE).
pub mod sdwa {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Sdwa {
        pub src0: u8,
        pub src0_sgpr: bool,
        pub dst_sel: u32,
        pub dst_unused: u32,
        pub clamp: bool,
        pub sel: [u32; 2],
        pub sext: [bool; 2],
        pub mods: [u32; 2],
        pub src1_sgpr: bool,
    }

    pub fn parse(w: u32) -> Sdwa {
        Sdwa {
            src0: w as u8,
            src0_sgpr: w >> 23 & 1 != 0,
            dst_sel: w >> 8 & 7,
            dst_unused: w >> 11 & 3,
            clamp: w >> 13 & 1 != 0,
            sel: [w >> 16 & 7, w >> 24 & 7],
            sext: [w >> 19 & 1 != 0, w >> 27 & 1 != 0],
            mods: [w >> 20 & 3, w >> 28 & 3],
            src1_sgpr: w >> 31 != 0,
        }
    }

    /// (shift, width) of a selector; None for undefined values.
    pub fn field(sel: u32) -> Option<(u32, u32)> {
        Some(match sel {
            0..=3 => (8 * sel, 8),
            4 | 5 => (16 * (sel - 4), 16),
            6 => (0, 32),
            _ => return None,
        })
    }

    pub fn select(x: u32, sel: u32, sext: bool) -> Option<u32> {
        let (sh, w) = field(sel)?;
        if w == 32 {
            return Some(x);
        }
        let v = x >> sh & ((1 << w) - 1);
        Some(if sext { ((v << (32 - w)) as i32 >> (32 - w)) as u32 } else { v })
    }

    /// The written register: None for SEXT on a field that does not start
    /// at bit 0 (what fills the bits below it is not confirmed) and for
    /// undefined values.
    pub fn place(r: u32, old: u32, sel: u32, unused: u32) -> Option<u32> {
        let (sh, w) = field(sel)?;
        if w == 32 {
            return Some(r);
        }
        let m = ((1u32 << w) - 1) << sh;
        let f = (r << sh) & m;
        Some(match unused {
            0 => f,
            1 if sh == 0 => ((f << (32 - w)) as i32 >> (32 - w)) as u32,
            2 => old & !m | f,
            _ => return None,
        })
    }
}

/// VOP3P integer dot products (GFX10.1+ opcodes from LLVM
/// `VOP3P_Real_gfx10`: dot2_i32_i16 0x14, dot2_u32_u16 0x15, dot4_i32_i8
/// 0x16, dot4_u32_u8 0x17; `gfx1011_dlops.s`). Semantics are Mesa NIR's
/// `sdot_4x8_iadd`, `udot_4x8_uadd`, `sdot_2x16_iadd`, `udot_2x16_uadd`
/// and their `_sat` forms, which ACO selects these instructions for with
/// clamp = sat, op_sel = 0, op_sel_hi = 7, no neg.
pub mod vop3p {
    pub const V_DOT2_I32_I16: u16 = 0x14;
    pub const V_DOT2_U32_U16: u16 = 0x15;
    pub const V_DOT4_I32_I8: u16 = 0x16;
    pub const V_DOT4_U32_U8: u16 = 0x17;

    /// None where NIR leaves the result open (udot2 sat with an overflowing
    /// product sum, which it marks poison) or the 32-bit sum of the
    /// sdot2 products overflows (only -32768 * -32768 twice).
    pub fn dot(op: u16, a: u32, b: u32, c: u32, sat: bool) -> Option<u32> {
        let parts = |x: u32, w: u32, signed: bool| -> Vec<i64> {
            (0..32 / w)
                .map(|k| {
                    let v = (x >> (k * w)) & ((1u64 << w) - 1) as u32;
                    if signed { ((v << (32 - w)) as i32 >> (32 - w)) as i64 } else { v as i64 }
                })
                .collect()
        };
        let (w, signed) = match op {
            V_DOT2_I32_I16 => (16, true),
            V_DOT2_U32_U16 => (16, false),
            V_DOT4_I32_I8 => (8, true),
            V_DOT4_U32_U8 => (8, false),
            _ => return None,
        };
        let dp: i64 = parts(a, w, signed).iter().zip(parts(b, w, signed)).map(|(x, y)| x * y).sum();
        if signed && w == 16 && dp > i32::MAX as i64 {
            return None;
        }
        if !signed && w == 16 && sat && dp > u32::MAX as i64 {
            return None;
        }
        let t = if signed { dp + c as i32 as i64 } else { dp + c as i64 };
        Some(match (sat, signed) {
            (false, _) => t as u32,
            (true, true) => t.clamp(i32::MIN as i64, i32::MAX as i64) as i32 as u32,
            (true, false) => t.min(u32::MAX as i64) as u32,
        })
    }
}

/// MTBUF typed buffer access (GFX10 opcodes from LLVM `BUFInstructions.td`
/// `MTBUF_Real_AllAddr_gfx10`: 0..3 load_format_x..xyzw, 4..7 stores,
/// 8.. d16; the format field holds the unified format numbers images use,
/// `mtbuf-gfx10.s`). Modelled as the equivalent MUBUF dword access for
/// formats without conversion (32-bit channels): `Some(mubuf opcode)`.
pub mod mtbuf {
    use super::mubuf;

    pub fn as_mubuf(op: u16, format: u32) -> Option<u16> {
        let (bytes, channels, _) = super::image_format(format)?;
        if format == super::FORMAT_8_8_8_8_UNORM || bytes / 4 != channels || op >= 8 {
            return None;
        }
        let n = (op & 3) as u32 + 1;
        let store = op & 4 != 0;
        // Loads may read fewer channels than the format has; stores must
        // write all of them (partial typed stores are not modelled).
        if n > channels || (store && n != channels) {
            return None;
        }
        Some(match (n, store) {
            (1, false) => mubuf::LOAD_DWORD,
            (2, false) => mubuf::LOAD_DWORDX2,
            (3, false) => mubuf::LOAD_DWORDX3,
            (4, false) => mubuf::LOAD_DWORDX4,
            (1, true) => mubuf::STORE_DWORD,
            (2, true) => mubuf::STORE_DWORDX2,
            (4, true) => mubuf::STORE_DWORDX4,
            _ => return None,
        })
    }
}

/// Buffer resource descriptor ("V#", 4 dwords), GFX10 layout from Mesa's
/// `src/amd/registers/gfx10-rsrc.json` (SQ_BUF_RSRC_WORD0..3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferDesc {
    /// WORD0 + WORD1[15:0]: 48-bit base address.
    pub base: u64,
    /// WORD1[29:16].
    pub stride: u32,
    /// WORD1[31].
    pub swizzle: bool,
    /// WORD2.
    pub num_records: u32,
    /// WORD3[28:29]: 0 structured with offset, 1 structured, 2 disabled
    /// (only NUM_RECORDS == 0 is out of bounds), 3 raw.
    pub oob_select: u32,
}

impl BufferDesc {
    /// WORD3[23]: ADD_TID_ENABLE.
    pub fn add_tid(w3: u32) -> bool {
        w3 >> 23 & 1 != 0
    }

    /// Is dword `k` of a MUBUF access in bounds? GFX10 rules from Mesa's
    /// `ac_descriptors.c` OOB_SELECT comment: 0 `(index >= NUM_RECORDS) ||
    /// (offset >= STRIDE)`, 1 `index >= NUM_RECORDS`, 2 only `NUM_RECORDS
    /// == 0`, 3 (raw, no swizzle) `offset >= NUM_RECORDS`. `offset` is the
    /// byte offset of the dword within the record (instruction offset +
    /// VGPR offset + 4k), applied per dword. Swizzled descriptors are not
    /// handled (None).
    pub fn in_bounds(&self, index: u32, offset: u64) -> Option<bool> {
        if self.swizzle {
            return None;
        }
        let nr = self.num_records as u64;
        Some(match self.oob_select {
            0 => (index as u64) < nr && offset < self.stride as u64,
            1 => (index as u64) < nr,
            2 => nr != 0,
            _ => offset < nr,
        })
    }

    pub fn parse(w: [u32; 4]) -> Self {
        BufferDesc {
            base: w[0] as u64 | ((w[1] & 0xFFFF) as u64) << 32,
            stride: (w[1] >> 16) & 0x3FFF,
            swizzle: w[1] >> 31 != 0,
            num_records: w[2],
            oob_select: (w[3] >> 28) & 3,
        }
    }
}

/// GFX10 2D tiling for the swizzle modes whose pattern does not depend on
/// the chip's pipe/bank configuration: SW_256B_S (1), SW_4KB_S (5),
/// SW_64KB_S (9) (`AddrSwizzleMode` numbers from addrlib's
/// `addrtypes.h`). Patterns from addrlib `gfx10SwizzlePattern.h`
/// (`GFX10_SW_*_S_PATINFO` + the nibble tables; non-XOR modes index by
/// element size only, and the RB+ tables agree). Address of element
/// (x, y), as addrlib's `ComputeSurfaceAddrFromCoordMacroTiled`:
/// `(yb * pitch_in_blocks + xb) << block_bits` plus the in-block offset
/// whose bit i is the parity of `x & xmask[i]` and `y & ymask[i]`; the
/// pitch is the width aligned to the block width.
pub mod tiling {
    /// (x mask, y mask) per address bit, for elem log2 0..=4.
    const SW_256_S: [[(u16, u16); 8]; 5] = [
        [(0x1, 0x0), (0x2, 0x0), (0x4, 0x0), (0x8, 0x0), (0x0, 0x1), (0x0, 0x2), (0x0, 0x4), (0x0, 0x8)],
        [(0x0, 0x0), (0x1, 0x0), (0x2, 0x0), (0x4, 0x0), (0x0, 0x1), (0x0, 0x2), (0x0, 0x4), (0x8, 0x0)],
        [(0x0, 0x0), (0x0, 0x0), (0x1, 0x0), (0x2, 0x0), (0x0, 0x1), (0x0, 0x2), (0x0, 0x4), (0x4, 0x0)],
        [(0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x1, 0x0), (0x0, 0x1), (0x0, 0x2), (0x2, 0x0), (0x4, 0x0)],
        [(0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x1), (0x0, 0x2), (0x1, 0x0), (0x2, 0x0)],
    ];
    const SW_4K_S: [[(u16, u16); 12]; 5] = [
        [(0x1, 0x0), (0x2, 0x0), (0x4, 0x0), (0x8, 0x0), (0x0, 0x1), (0x0, 0x2), (0x0, 0x4), (0x0, 0x8), (0x0, 0x10), (0x10, 0x0), (0x0, 0x20), (0x20, 0x0)],
        [(0x0, 0x0), (0x1, 0x0), (0x2, 0x0), (0x4, 0x0), (0x0, 0x1), (0x0, 0x2), (0x0, 0x4), (0x8, 0x0), (0x0, 0x8), (0x10, 0x0), (0x0, 0x10), (0x20, 0x0)],
        [(0x0, 0x0), (0x0, 0x0), (0x1, 0x0), (0x2, 0x0), (0x0, 0x1), (0x0, 0x2), (0x0, 0x4), (0x4, 0x0), (0x0, 0x8), (0x8, 0x0), (0x0, 0x10), (0x10, 0x0)],
        [(0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x1, 0x0), (0x0, 0x1), (0x0, 0x2), (0x2, 0x0), (0x4, 0x0), (0x0, 0x4), (0x8, 0x0), (0x0, 0x8), (0x10, 0x0)],
        [(0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x1), (0x0, 0x2), (0x1, 0x0), (0x2, 0x0), (0x0, 0x4), (0x4, 0x0), (0x0, 0x8), (0x8, 0x0)],
    ];
    const SW_64K_S: [[(u16, u16); 16]; 5] = [
        [(0x1, 0x0), (0x2, 0x0), (0x4, 0x0), (0x8, 0x0), (0x0, 0x1), (0x0, 0x2), (0x0, 0x4), (0x0, 0x8), (0x0, 0x10), (0x10, 0x0), (0x0, 0x20), (0x20, 0x0), (0x0, 0x40), (0x40, 0x0), (0x0, 0x80), (0x80, 0x0)],
        [(0x0, 0x0), (0x1, 0x0), (0x2, 0x0), (0x4, 0x0), (0x0, 0x1), (0x0, 0x2), (0x0, 0x4), (0x8, 0x0), (0x0, 0x8), (0x10, 0x0), (0x0, 0x10), (0x20, 0x0), (0x0, 0x20), (0x40, 0x0), (0x0, 0x40), (0x80, 0x0)],
        [(0x0, 0x0), (0x0, 0x0), (0x1, 0x0), (0x2, 0x0), (0x0, 0x1), (0x0, 0x2), (0x0, 0x4), (0x4, 0x0), (0x0, 0x8), (0x8, 0x0), (0x0, 0x10), (0x10, 0x0), (0x0, 0x20), (0x20, 0x0), (0x0, 0x40), (0x40, 0x0)],
        [(0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x1, 0x0), (0x0, 0x1), (0x0, 0x2), (0x2, 0x0), (0x4, 0x0), (0x0, 0x4), (0x8, 0x0), (0x0, 0x8), (0x10, 0x0), (0x0, 0x10), (0x20, 0x0), (0x0, 0x20), (0x40, 0x0)],
        [(0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x0), (0x0, 0x1), (0x0, 0x2), (0x1, 0x0), (0x2, 0x0), (0x0, 0x4), (0x4, 0x0), (0x0, 0x8), (0x8, 0x0), (0x0, 0x10), (0x10, 0x0), (0x0, 0x20), (0x20, 0x0)],
    ];

    /// Pattern for `sw_mode` and element size 1 << `elem_log2` bytes.
    pub fn pattern(sw_mode: u32, elem_log2: u32) -> Option<&'static [(u16, u16)]> {
        let e = elem_log2 as usize;
        if e > 4 {
            return None;
        }
        Some(match sw_mode {
            1 => &SW_256_S[e],
            5 => &SW_4K_S[e],
            9 => &SW_64K_S[e],
            _ => return None,
        })
    }

    /// log2 of the block width and height in elements: one past the
    /// highest x / y bit the pattern uses.
    pub fn block_dims(p: &[(u16, u16)]) -> (u32, u32) {
        let top = |f: fn(&(u16, u16)) -> u16| 16 - p.iter().map(f).fold(0, |a, m| a | m).leading_zeros();
        (top(|c| c.0), top(|c| c.1))
    }

    /// In-block byte offset of element (x, y).
    pub fn block_offset(p: &[(u16, u16)], x: u32, y: u32) -> u32 {
        p.iter()
            .enumerate()
            .map(|(i, &(mx, my))| (((x & mx as u32).count_ones() + (y & my as u32).count_ones()) & 1) << i)
            .sum()
    }

    /// Byte offset of element (x, y) in a 2D surface `width` elements wide.
    pub fn offset(sw_mode: u32, elem_log2: u32, width: u32, x: u32, y: u32) -> Option<u64> {
        let p = pattern(sw_mode, elem_log2)?;
        let (bw, bh) = block_dims(p);
        let pitch_blocks = (width as u64).div_ceil(1 << bw);
        let blk = (y as u64 >> bh) * pitch_blocks + (x as u64 >> bw);
        Some((blk << p.len()) + block_offset(p, x, y) as u64)
    }
}

/// SOPK opcodes (GFX10 numbers from LLVM `SOPInstructions.td`, checked
/// against `sopk.s`). Signed forms sign-extend SIMM16, the `_u32`
/// compares zero-extend it (`s16imm` / `u16imm` operands).
pub mod sopk {
    use super::{sop1, sop2, Fields, Format, Inst, SRC_LITERAL};

    pub const S_MOVK_I32: u16 = 0;
    pub const S_CMOVK_I32: u16 = 2;
    pub const S_CMPK_EQ_I32: u16 = 3;
    pub const S_CMPK_LE_U32: u16 = 0xE;
    pub const S_ADDK_I32: u16 = 0xF;
    pub const S_MULK_I32: u16 = 0x10;
    pub const S_SETREG_IMM32_B32: u16 = 21;
    pub const S_WAITCNT_VSCNT: u16 = 0x17;
    pub const S_WAITCNT_LGKMCNT: u16 = 0x1A;

    /// The equivalent SOP1/SOP2/SOPC with SIMM16 as a literal: movk ->
    /// s_mov_b32, cmovk -> s_cmov_b32, addk -> s_add_i32 (SCC as there),
    /// mulk -> s_mul_i32, cmpk_* -> s_cmp_* (SOPC opcode = SOPK - 3, the
    /// same order of conditions in both tables).
    pub fn lower(inst: &Inst) -> Option<Inst> {
        let Fields::Sopk { sdst, simm16 } = inst.fields else {
            return None;
        };
        let op = inst.opcode;
        let sext = simm16 as i16 as i32 as u32;
        let zext = simm16 as u32;
        let d = sdst as u16;
        let (format, opcode, fields, lit) = match op {
            S_MOVK_I32 => (Format::Sop1, sop1::S_MOV_B32, Fields::Sop1 { sdst, ssrc0: SRC_LITERAL }, sext),
            S_CMOVK_I32 => (Format::Sop1, sop1::S_CMOV_B32, Fields::Sop1 { sdst, ssrc0: SRC_LITERAL }, sext),
            S_ADDK_I32 => (
                Format::Sop2,
                sop2::S_ADD_I32,
                Fields::Sop2 { sdst, ssrc0: d, ssrc1: SRC_LITERAL },
                sext,
            ),
            S_MULK_I32 => (
                Format::Sop2,
                sop2::S_MUL_I32,
                Fields::Sop2 { sdst, ssrc0: d, ssrc1: SRC_LITERAL },
                sext,
            ),
            S_CMPK_EQ_I32..=S_CMPK_LE_U32 => (
                Format::Sopc,
                op - 3,
                Fields::Sopc { ssrc0: d, ssrc1: SRC_LITERAL },
                if op >= 9 { zext } else { sext },
            ),
            _ => return None,
        };
        Some(Inst { format, opcode, words: vec![inst.words[0], lit], fields })
    }

    /// `s_waitcnt_{vscnt,vmcnt,expcnt,lgkmcnt}`: no effect in a model
    /// where memory operations complete in order.
    pub fn is_waitcnt(op: u16) -> bool {
        (S_WAITCNT_VSCNT..=S_WAITCNT_LGKMCNT).contains(&op)
    }
}

impl Inst {
    pub fn len(&self) -> usize {
        self.words.len()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// The literal constant, if the instruction has one.
    pub fn literal(&self) -> Option<u32> {
        let has = match &self.fields {
            Fields::Sop2 { ssrc0, ssrc1, .. } | Fields::Sopc { ssrc0, ssrc1 } => {
                *ssrc0 == SRC_LITERAL || *ssrc1 == SRC_LITERAL
            }
            Fields::Sop1 { ssrc0, .. } => *ssrc0 == SRC_LITERAL,
            Fields::Sopk { .. } => self.opcode == sopk::S_SETREG_IMM32_B32,
            Fields::Vop2 { .. } if vop2::is_madk(self.opcode) => true,
            Fields::Vop2 { src0, .. } | Fields::Vop1 { src0, .. } | Fields::Vopc { src0, .. } => {
                *src0 == SRC_LITERAL
            }
            Fields::Vop3 { src, .. } | Fields::Vop3p { src, .. } => src.contains(&SRC_LITERAL),
            _ => false,
        };
        has.then(|| *self.words.last().unwrap())
    }

    pub fn is_endpgm(&self) -> bool {
        self.format == Format::Sopp && matches!(self.opcode, sopp::S_ENDPGM | sopp::S_CODE_END)
    }

    /// For SOPP branches: target dword index, given this instruction's
    /// index (`simm16` counts dwords from the next instruction).
    pub fn branch_target(&self, at: usize) -> Option<usize> {
        let Fields::Sopp { simm16 } = self.fields else {
            return None;
        };
        if !(sopp::S_BRANCH..=sopp::S_CBRANCH_EXECNZ).contains(&self.opcode)
            || self.opcode == sopp::S_WAKEUP
        {
            return None;
        }
        (at as i64 + 1 + simm16 as i16 as i64).try_into().ok()
    }

    /// Unconditional end of a basic block (no fall-through).
    pub fn ends_block(&self) -> bool {
        self.is_endpgm() || (self.format == Format::Sopp && self.opcode == sopp::S_BRANCH)
    }

    pub fn mnemonic(&self) -> String {
        // VOPC compares (also in VOP3 form, opcode < 0x100): every kind and
        // condition `interp::Compare` decodes, named as LLVM's gfx10_asm_vopc.s
        let vopc = self.format == Format::Vopc || (self.format == Format::Vop3 && self.opcode < 0x100);
        if let Some(c) = vopc.then(|| crate::interp::Compare::decode(self.opcode)).flatten() {
            use crate::interp::CmpKind;
            const F: [&str; 16] = ["f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru"];
            const I: [&str; 8] = ["f", "lt", "eq", "le", "gt", "ne", "ge", "t"];
            let (cond, ty) = match c.kind {
                CmpKind::F32 => (F[c.cond as usize], "f32"),
                CmpKind::F64 => (F[c.cond as usize], "f64"),
                CmpKind::I32 => (I[c.cond as usize], "i32"),
                CmpKind::U32 => (I[c.cond as usize], "u32"),
                CmpKind::I64 => (I[c.cond as usize], "i64"),
                CmpKind::U64 => (I[c.cond as usize], "u64"),
                CmpKind::I16 => (I[c.cond as usize], "i16"),
                CmpKind::U16 => (I[c.cond as usize], "u16"),
            };
            return format!("v_cmp{}_{cond}_{ty}", if c.cmpx { "x" } else { "" });
        }
        let known = match (self.format, self.opcode) {
            (Format::Sopp, sopp::S_NOP) => "s_nop",
            (Format::Sopp, sopp::S_ENDPGM) => "s_endpgm",
            (Format::Sopp, sopp::S_BRANCH) => "s_branch",
            (Format::Sopp, sopp::S_WAKEUP) => "s_wakeup",
            (Format::Sopp, sopp::S_CBRANCH_SCC0) => "s_cbranch_scc0",
            (Format::Sopp, sopp::S_CBRANCH_SCC1) => "s_cbranch_scc1",
            (Format::Sopp, sopp::S_CBRANCH_VCCZ) => "s_cbranch_vccz",
            (Format::Sopp, sopp::S_CBRANCH_VCCNZ) => "s_cbranch_vccnz",
            (Format::Sopp, sopp::S_CBRANCH_EXECZ) => "s_cbranch_execz",
            (Format::Sopp, sopp::S_CBRANCH_EXECNZ) => "s_cbranch_execnz",
            (Format::Sopp, sopp::S_BARRIER) => "s_barrier",
            (Format::Sopp, sopp::S_WAITCNT) => "s_waitcnt",
            (Format::Sopp, sopp::S_SLEEP) => "s_sleep",
            (Format::Sopp, sopp::S_SENDMSG) => "s_sendmsg",
            (Format::Sopp, sopp::S_CODE_END) => "s_code_end",
            (Format::Sopk, sopk::S_MOVK_I32) => "s_movk_i32",
            (Format::Sopk, sopk::S_CMOVK_I32) => "s_cmovk_i32",
            (Format::Sopk, 3) => "s_cmpk_eq_i32",
            (Format::Sopk, 4) => "s_cmpk_lg_i32",
            (Format::Sopk, 5) => "s_cmpk_gt_i32",
            (Format::Sopk, 6) => "s_cmpk_ge_i32",
            (Format::Sopk, 7) => "s_cmpk_lt_i32",
            (Format::Sopk, 8) => "s_cmpk_le_i32",
            (Format::Sopk, 9) => "s_cmpk_eq_u32",
            (Format::Sopk, 10) => "s_cmpk_lg_u32",
            (Format::Sopk, 11) => "s_cmpk_gt_u32",
            (Format::Sopk, 12) => "s_cmpk_ge_u32",
            (Format::Sopk, 13) => "s_cmpk_lt_u32",
            (Format::Sopk, 14) => "s_cmpk_le_u32",
            (Format::Sopk, sopk::S_ADDK_I32) => "s_addk_i32",
            (Format::Sopk, sopk::S_MULK_I32) => "s_mulk_i32",
            (Format::Sopk, 0x17) => "s_waitcnt_vscnt",
            (Format::Sopk, 0x18) => "s_waitcnt_vmcnt",
            (Format::Sopk, 0x19) => "s_waitcnt_expcnt",
            (Format::Sopk, 0x1A) => "s_waitcnt_lgkmcnt",
            (Format::Sopk, sopk::S_SETREG_IMM32_B32) => "s_setreg_imm32_b32",
            (Format::Sop1, sop1::S_ABS_I32) => "s_abs_i32",
            (Format::Sop1, sop1::S_MOV_B32) => "s_mov_b32",
            (Format::Sop1, sop1::S_MOV_B64) => "s_mov_b64",
            (Format::Sop1, sop1::S_FF1_I32_B32) => "s_ff1_i32_b32",
            (Format::Sop1, sop1::S_AND_SAVEEXEC_B64) => "s_and_saveexec_b64",
            (Format::Sop1, sop1::S_AND_SAVEEXEC_B32) => "s_and_saveexec_b32",
            (Format::Sop2, sop2::S_ADD_U32) => "s_add_u32",
            (Format::Sop2, sop2::S_SUB_U32) => "s_sub_u32",
            (Format::Sop2, sop2::S_ADD_I32) => "s_add_i32",
            (Format::Sop2, sop2::S_ADDC_U32) => "s_addc_u32",
            (Format::Sop2, sop2::S_CSELECT_B32) => "s_cselect_b32",
            (Format::Sop2, sop2::S_AND_B32) => "s_and_b32",
            (Format::Sop2, sop2::S_AND_B64) => "s_and_b64",
            (Format::Sopc, sopc::S_CMP_LG_I32) => "s_cmp_lg_i32",
            (Format::Sopc, sopc::S_CMP_LT_I32) => "s_cmp_lt_i32",
            (Format::Sopc, sopc::S_CMP_LE_I32) => "s_cmp_le_i32",
            (Format::Sopc, sopc::S_CMP_LG_U32) => "s_cmp_lg_u32",
            (Format::Sopc, sopc::S_CMP_LT_U32) => "s_cmp_lt_u32",
            (Format::Sopc, sopc::S_CMP_LE_U32) => "s_cmp_le_u32",
            (Format::Sopc, sopc::S_BITCMP0_B32) => "s_bitcmp0_b32",
            (Format::Sopc, sopc::S_BITCMP1_B32) => "s_bitcmp1_b32",
            (Format::Vop2, vop2::V_FMAMK_F32) => "v_fmamk_f32",
            (Format::Vop2, vop2::V_CVT_PKRTZ_F16_F32) => "v_cvt_pkrtz_f16_f32",
            (Format::Vop1, vop1::V_CVT_F16_F32) => "v_cvt_f16_f32",
            (Format::Vop1, vop1::V_CVT_F32_F16) => "v_cvt_f32_f16",
            (Format::Vop2, vop2::V_FMAAK_F32) => "v_fmaak_f32",
            (Format::Sop2, sop2::S_SUB_I32) => "s_sub_i32",
            (Format::Sop2, sop2::S_SUBB_U32) => "s_subb_u32",
            (Format::Sop2, sop2::S_CSELECT_B64) => "s_cselect_b64",
            (Format::Sop2, sop2::S_OR_B32) => "s_or_b32",
            (Format::Sop2, sop2::S_OR_B64) => "s_or_b64",
            (Format::Sop2, sop2::S_XOR_B32) => "s_xor_b32",
            (Format::Sop2, sop2::S_XOR_B64) => "s_xor_b64",
            (Format::Sop2, sop2::S_ANDN2_B32) => "s_andn2_b32",
            (Format::Sop2, sop2::S_ANDN2_B64) => "s_andn2_b64",
            (Format::Sop2, sop2::S_ORN2_B32) => "s_orn2_b32",
            (Format::Sop2, sop2::S_NAND_B32) => "s_nand_b32",
            (Format::Sop2, sop2::S_NOR_B32) => "s_nor_b32",
            (Format::Sop2, sop2::S_XNOR_B32) => "s_xnor_b32",
            (Format::Sop2, sop2::S_LSHL_B32) => "s_lshl_b32",
            (Format::Sop2, sop2::S_LSHR_B32) => "s_lshr_b32",
            (Format::Sop2, sop2::S_ASHR_I32) => "s_ashr_i32",
            (Format::Sop2, sop2::S_BFM_B32) => "s_bfm_b32",
            (Format::Sop2, sop2::S_MUL_I32) => "s_mul_i32",
            (Format::Sop2, sop2::S_LSHL_B64) => "s_lshl_b64",
            (Format::Sop2, sop2::S_LSHR_B64) => "s_lshr_b64",
            (Format::Sop2, sop2::S_ASHR_I64) => "s_ashr_i64",
            (Format::Sop2, sop2::S_MUL_HI_U32) => "s_mul_hi_u32",
            (Format::Sop2, sop2::S_MUL_HI_I32) => "s_mul_hi_i32",
            (Format::Sop1, sop1::S_BITSET0_B32) => "s_bitset0_b32",
            (Format::Sop1, sop1::S_GETPC_B64) => "s_getpc_b64",
            (Format::Sop1, sop1::S_BITSET1_B32) => "s_bitset1_b32",
            (Format::Sop2, sop2::S_BFE_U32) => "s_bfe_u32",
            (Format::Sop2, sop2::S_BFE_I32) => "s_bfe_i32",
            (Format::Sop1, sop1::S_CMOV_B32) => "s_cmov_b32",
            (Format::Sop1, sop1::S_CMOV_B64) => "s_cmov_b64",
            (Format::Sop1, sop1::S_BCNT0_I32_B64) => "s_bcnt0_i32_b64",
            (Format::Sop1, sop1::S_FF0_I32_B32) => "s_ff0_i32_b32",
            (Format::Sop1, sop1::S_FF0_I32_B64) => "s_ff0_i32_b64",
            (Format::Sop1, sop1::S_XOR_SAVEEXEC_B64) => "s_xor_saveexec_b64",
            (Format::Sop1, sop1::S_ORN2_SAVEEXEC_B64) => "s_orn2_saveexec_b64",
            (Format::Sop1, sop1::S_NAND_SAVEEXEC_B64) => "s_nand_saveexec_b64",
            (Format::Sop1, sop1::S_NOR_SAVEEXEC_B64) => "s_nor_saveexec_b64",
            (Format::Sop1, sop1::S_XNOR_SAVEEXEC_B64) => "s_xnor_saveexec_b64",
            (Format::Sop1, sop1::S_ANDN1_SAVEEXEC_B64) => "s_andn1_saveexec_b64",
            (Format::Sop1, sop1::S_ORN1_SAVEEXEC_B64) => "s_orn1_saveexec_b64",
            (Format::Sop1, sop1::S_ANDN1_WREXEC_B64) => "s_andn1_wrexec_b64",
            (Format::Sop1, sop1::S_ANDN2_WREXEC_B64) => "s_andn2_wrexec_b64",
            (Format::Sop1, sop1::S_XOR_SAVEEXEC_B32) => "s_xor_saveexec_b32",
            (Format::Sop1, sop1::S_ORN2_SAVEEXEC_B32) => "s_orn2_saveexec_b32",
            (Format::Sop1, sop1::S_NAND_SAVEEXEC_B32) => "s_nand_saveexec_b32",
            (Format::Sop1, sop1::S_NOR_SAVEEXEC_B32) => "s_nor_saveexec_b32",
            (Format::Sop1, sop1::S_XNOR_SAVEEXEC_B32) => "s_xnor_saveexec_b32",
            (Format::Sop1, sop1::S_ANDN1_SAVEEXEC_B32) => "s_andn1_saveexec_b32",
            (Format::Sop1, sop1::S_ORN1_SAVEEXEC_B32) => "s_orn1_saveexec_b32",
            (Format::Sop1, sop1::S_ANDN1_WREXEC_B32) => "s_andn1_wrexec_b32",
            (Format::Sop1, sop1::S_ANDN2_WREXEC_B32) => "s_andn2_wrexec_b32",
            (Format::Sop2, sop2::S_MIN_I32) => "s_min_i32",
            (Format::Sop2, sop2::S_MIN_U32) => "s_min_u32",
            (Format::Sop2, sop2::S_MAX_I32) => "s_max_i32",
            (Format::Sop2, sop2::S_MAX_U32) => "s_max_u32",
            (Format::Sop2, sop2::S_ORN2_B64) => "s_orn2_b64",
            (Format::Sop2, sop2::S_NAND_B64) => "s_nand_b64",
            (Format::Sop2, sop2::S_NOR_B64) => "s_nor_b64",
            (Format::Sop2, sop2::S_XNOR_B64) => "s_xnor_b64",
            (Format::Sop2, sop2::S_BFM_B64) => "s_bfm_b64",
            (Format::Sop2, sop2::S_ABSDIFF_I32) => "s_absdiff_i32",
            (Format::Sop2, sop2::S_PACK_LL_B32_B16) => "s_pack_ll_b32_b16",
            (Format::Sop2, sop2::S_PACK_LH_B32_B16) => "s_pack_lh_b32_b16",
            (Format::Sop2, sop2::S_PACK_HH_B32_B16) => "s_pack_hh_b32_b16",
            (Format::Sopc, sopc::S_BITCMP0_B64) => "s_bitcmp0_b64",
            (Format::Sopc, sopc::S_BITCMP1_B64) => "s_bitcmp1_b64",
            (Format::Sop1, sop1::S_NOT_B32) => "s_not_b32",
            (Format::Sop1, sop1::S_NOT_B64) => "s_not_b64",
            (Format::Sop1, sop1::S_BREV_B64) => "s_brev_b64",
            (Format::Sop1, sop1::S_BCNT1_I32_B64) => "s_bcnt1_i32_b64",
            (Format::Sop1, sop1::S_FF1_I32_B64) => "s_ff1_i32_b64",
            (Format::Sop1, sop1::S_FLBIT_I32_B64) => "s_flbit_i32_b64",
            (Format::Sop1, sop1::S_FLBIT_I32_I64) => "s_flbit_i32_i64",
            (Format::Sop1, sop1::S_BITSET0_B64) => "s_bitset0_b64",
            (Format::Sop1, sop1::S_BITSET1_B64) => "s_bitset1_b64",
            (Format::Sop1, sop1::S_BITREPLICATE_B64_B32) => "s_bitreplicate_b64_b32",
            (Format::Sop1, sop1::S_BREV_B32) => "s_brev_b32",
            (Format::Sop1, sop1::S_BCNT0_I32_B32) => "s_bcnt0_i32_b32",
            (Format::Sop1, sop1::S_BCNT1_I32_B32) => "s_bcnt1_i32_b32",
            (Format::Sop1, sop1::S_FLBIT_I32_B32) => "s_flbit_i32_b32",
            (Format::Sop1, sop1::S_FLBIT_I32) => "s_flbit_i32",
            (Format::Sop1, sop1::S_SEXT_I32_I8) => "s_sext_i32_i8",
            (Format::Sop1, sop1::S_SEXT_I32_I16) => "s_sext_i32_i16",
            (Format::Sop1, sop1::S_OR_SAVEEXEC_B64) => "s_or_saveexec_b64",
            (Format::Sop1, sop1::S_ANDN2_SAVEEXEC_B64) => "s_andn2_saveexec_b64",
            (Format::Sop1, sop1::S_OR_SAVEEXEC_B32) => "s_or_saveexec_b32",
            (Format::Sop1, sop1::S_ANDN2_SAVEEXEC_B32) => "s_andn2_saveexec_b32",
            (Format::Sopc, sopc::S_CMP_EQ_I32) => "s_cmp_eq_i32",
            (Format::Sopc, sopc::S_CMP_GT_I32) => "s_cmp_gt_i32",
            (Format::Sopc, sopc::S_CMP_GE_I32) => "s_cmp_ge_i32",
            (Format::Sopc, sopc::S_CMP_EQ_U32) => "s_cmp_eq_u32",
            (Format::Sopc, sopc::S_CMP_GT_U32) => "s_cmp_gt_u32",
            (Format::Sopc, sopc::S_CMP_GE_U32) => "s_cmp_ge_u32",
            (Format::Sopc, sopc::S_CMP_EQ_U64) => "s_cmp_eq_u64",
            (Format::Sopc, sopc::S_CMP_LG_U64) => "s_cmp_lg_u64",
            (Format::Vop1, vop1::V_NOP) => "v_nop",
            (Format::Vop1, vop1::V_MOV_B32) => "v_mov_b32",
            (Format::Vop1, vop1::V_READFIRSTLANE_B32) => "v_readfirstlane_b32",
            (Format::Vop1, vop1::V_CVT_F32_I32) => "v_cvt_f32_i32",
            (Format::Vop1, vop1::V_CVT_F32_U32) => "v_cvt_f32_u32",
            (Format::Vop1, vop1::V_CVT_U32_F32) => "v_cvt_u32_f32",
            (Format::Vop1, vop1::V_CVT_I32_F32) => "v_cvt_i32_f32",
            (Format::Vop2, vop2::V_SUBREV_F32) => "v_subrev_f32",
            (Format::Vop3, vop3::V_BFE_U32) => "v_bfe_u32",
            (Format::Vop3, vop3::V_BFI_B32) => "v_bfi_b32",
            (Format::Vop3, vop3::V_MED3_U32) => "v_med3_u32",
            (Format::Vop3, vop3::V_MUL_LO_U32) => "v_mul_lo_u32",
            (Format::Vop3, vop3::V_LSHL_ADD_U32) => "v_lshl_add_u32",
            (Format::Vop3, vop3::V_ADD3_U32) => "v_add3_u32",
            (Format::Vop3, vop3::V_ADD_NC_U16) => "v_add_nc_u16",
            (Format::Vop3, vop3::V_SUB_NC_U16) => "v_sub_nc_u16",
            (Format::Vop3, vop3::V_MUL_LO_U16) => "v_mul_lo_u16",
            (Format::Vop3, vop3::V_LSHRREV_B16) => "v_lshrrev_b16",
            (Format::Vop3, vop3::V_ASHRREV_I16) => "v_ashrrev_i16",
            (Format::Vop3, vop3::V_MAX_U16) => "v_max_u16",
            (Format::Vop3, vop3::V_MAX_I16) => "v_max_i16",
            (Format::Vop3, vop3::V_MIN_U16) => "v_min_u16",
            (Format::Vop3, vop3::V_MIN_I16) => "v_min_i16",
            (Format::Vop3, vop3::V_LSHLREV_B16) => "v_lshlrev_b16",
            (Format::Vop3, vop3::V_PERM_B32) => "v_perm_b32",
            (Format::Vop3p, vop3p::V_DOT2_I32_I16) => "v_dot2_i32_i16",
            (Format::Vop3p, vop3p::V_DOT2_U32_U16) => "v_dot2_u32_u16",
            (Format::Vop3p, vop3p::V_DOT4_I32_I8) => "v_dot4_i32_i8",
            (Format::Vop3p, vop3p::V_DOT4_U32_U8) => "v_dot4_u32_u8",
            (Format::Vop3, vop3::V_ALIGNBYTE_B32) => "v_alignbyte_b32",
            (Format::Vop3, vop3::V_CVT_PK_U16_U32) => "v_cvt_pk_u16_u32",
            (Format::Vop3, vop3::V_CVT_PK_I16_I32) => "v_cvt_pk_i16_i32",
            (Format::Vop3, vop3::V_MSAD_U8) => "v_msad_u8",
            (Format::Ds, ds::SWIZZLE_B32) => "ds_swizzle_b32",
            (Format::Ds, ds::APPEND) => "ds_append",
            (Format::Ds, ds::CONSUME) => "ds_consume",
            (Format::Vop3, vop3::V_ADD_CO_U32) => "v_add_co_u32",
            (Format::Vop3, vop3::V_LSHLREV_B64) => "v_lshlrev_b64",
            (Format::Vop3, vop3::V_READLANE_B32) => "v_readlane_b32",
            (Format::Vop3, vop3::V_WRITELANE_B32) => "v_writelane_b32",
            (Format::Vop3, vop3::V_FMA_F64) => "v_fma_f64",
            (Format::Vop3, vop3::V_ADD_F64) => "v_add_f64",
            (Format::Vop3, vop3::V_MUL_F64) => "v_mul_f64",
            (Format::Vop3, vop3::V_MIN_F64) => "v_min_f64",
            (Format::Vop3, vop3::V_MAX_F64) => "v_max_f64",
            (Format::Vop3, vop3::V_MAD_U64_U32) => "v_mad_u64_u32",
            (Format::Vop3, vop3::V_XOR3_B32) => "v_xor3_b32",
            (Format::Vop3, vop3::V_BFM_B32) => "v_bfm_b32",
            (Format::Vop3, vop3::V_ADD_NC_I16) => "v_add_nc_i16",
            (Format::Vop3, vop3::V_SUB_NC_I16) => "v_sub_nc_i16",
            (Format::Vop3, vop3::V_MAD_U16) => "v_mad_u16",
            (Format::Vop3, vop3::V_MAD_I16) => "v_mad_i16",
            (Format::Vop3, vop3::V_MIN3_I16) => "v_min3_i16",
            (Format::Vop3, vop3::V_MIN3_U16) => "v_min3_u16",
            (Format::Vop3, vop3::V_MAX3_I16) => "v_max3_i16",
            (Format::Vop3, vop3::V_MAX3_U16) => "v_max3_u16",
            (Format::Vop3, vop3::V_MED3_I16) => "v_med3_i16",
            (Format::Vop3, vop3::V_MED3_U16) => "v_med3_u16",
            (Format::Vop2, vop2::V_XNOR_B32) => "v_xnor_b32",
            (Format::Vop1, vop1::V_SWAP_B32) => "v_swap_b32",
            (Format::Vop3, vop3::V_ADD_NC_I32) => "v_add_nc_i32",
            (Format::Vop3, vop3::V_SUB_NC_I32) => "v_sub_nc_i32",
            (Format::Vop3, vop3::V_MAD_U32_U16) => "v_mad_u32_u16",
            (Format::Vop3, vop3::V_MAD_I32_I16) => "v_mad_i32_i16",
            (Format::Vop3, vop3::V_SAD_U8) => "v_sad_u8",
            (Format::Vop3, vop3::V_SAD_HI_U8) => "v_sad_hi_u8",
            (Format::Vop3, vop3::V_SAD_U16) => "v_sad_u16",
            (Format::Vop3, vop3::V_SAD_U32) => "v_sad_u32",
            (Format::Vop3, vop3::V_LERP_U8) => "v_lerp_u8",
            (Format::Vop3, vop3::V_LDEXP_F32) => "v_ldexp_f32",
            (Format::Vop3, vop3::V_MIN3_F32) => "v_min3_f32",
            (Format::Vop3, vop3::V_MAX3_F32) => "v_max3_f32",
            (Format::Vop3, vop3::V_MED3_F32) => "v_med3_f32",
            (Format::Vop3, vop3::V_ADD_LSHL_U32) => "v_add_lshl_u32",
            (Format::Vop3, vop3::V_LSHRREV_B64) => "v_lshrrev_b64",
            (Format::Vop3, vop3::V_ASHRREV_I64) => "v_ashrrev_i64",
            (Format::Vop3, vop3::V_SUB_CO_U32) => "v_sub_co_u32",
            (Format::Vop3, vop3::V_SUBREV_CO_U32) => "v_subrev_co_u32",
            (Format::Vop2, vop2::V_ADD_CO_CI_U32) => "v_add_co_ci_u32",
            (Format::Vop2, vop2::V_SUB_CO_CI_U32) => "v_sub_co_ci_u32",
            (Format::Vop2, vop2::V_SUBREV_CO_CI_U32) => "v_subrev_co_ci_u32",
            (Format::Vop3, vop3::V_MUL_HI_U32) => "v_mul_hi_u32",
            (Format::Vop3, vop3::V_MUL_LO_I32) => "v_mul_lo_i32",
            (Format::Vop3, vop3::V_MUL_HI_I32) => "v_mul_hi_i32",
            (Format::Vop3, vop3::V_MAD_I32_I24) => "v_mad_i32_i24",
            (Format::Vop3, vop3::V_MAD_U32_U24) => "v_mad_u32_u24",
            (Format::Vop3, vop3::V_BCNT_U32_B32) => "v_bcnt_u32_b32",
            (Format::Vop3, vop3::V_ALIGNBIT_B32) => "v_alignbit_b32",
            (Format::Vop3, vop3::V_BFE_I32) => "v_bfe_i32",
            (Format::Vop3, vop3::V_MIN3_I32) => "v_min3_i32",
            (Format::Vop3, vop3::V_MIN3_U32) => "v_min3_u32",
            (Format::Vop3, vop3::V_MAX3_I32) => "v_max3_i32",
            (Format::Vop3, vop3::V_MAX3_U32) => "v_max3_u32",
            (Format::Vop3, vop3::V_MED3_I32) => "v_med3_i32",
            (Format::Vop3, vop3::V_XAD_U32) => "v_xad_u32",
            (Format::Vop3, vop3::V_LSHL_OR_B32) => "v_lshl_or_b32",
            (Format::Vop3, vop3::V_AND_OR_B32) => "v_and_or_b32",
            (Format::Vop3, vop3::V_OR3_B32) => "v_or3_b32",
            (Format::Vop1, vop1::V_NOT_B32) => "v_not_b32",
            (Format::Vop1, vop1::V_CVT_F32_UBYTE0) => "v_cvt_f32_ubyte0",
            (Format::Vop1, vop1::V_CVT_F32_UBYTE1) => "v_cvt_f32_ubyte1",
            (Format::Vop1, vop1::V_CVT_F32_UBYTE2) => "v_cvt_f32_ubyte2",
            (Format::Vop1, vop1::V_CVT_F32_UBYTE3) => "v_cvt_f32_ubyte3",
            (Format::Vop1, vop1::V_TRUNC_F32) => "v_trunc_f32",
            (Format::Vop1, vop1::V_TRUNC_F64) => "v_trunc_f64",
            (Format::Vop1, vop1::V_CVT_RPI_I32_F32) => "v_cvt_rpi_i32_f32",
            (Format::Vop1, vop1::V_CVT_FLR_I32_F32) => "v_cvt_flr_i32_f32",
            (Format::Vop1, vop1::V_FREXP_EXP_I32_F32) => "v_frexp_exp_i32_f32",
            (Format::Vop1, vop1::V_FREXP_MANT_F32) => "v_frexp_mant_f32",
            (Format::Vop1, vop1::V_CVT_I32_F64) => "v_cvt_i32_f64",
            (Format::Vop1, vop1::V_CVT_F64_I32) => "v_cvt_f64_i32",
            (Format::Vop1, vop1::V_CVT_F32_F64) => "v_cvt_f32_f64",
            (Format::Vop1, vop1::V_CVT_F64_F32) => "v_cvt_f64_f32",
            (Format::Vop1, vop1::V_CVT_U32_F64) => "v_cvt_u32_f64",
            (Format::Vop1, vop1::V_CVT_F64_U32) => "v_cvt_f64_u32",
            (Format::Vop1, vop1::V_CEIL_F64) => "v_ceil_f64",
            (Format::Vop1, vop1::V_RNDNE_F64) => "v_rndne_f64",
            (Format::Vop1, vop1::V_FLOOR_F64) => "v_floor_f64",
            (Format::Vop1, vop1::V_CEIL_F32) => "v_ceil_f32",
            (Format::Vop1, vop1::V_RNDNE_F32) => "v_rndne_f32",
            (Format::Vop1, vop1::V_FLOOR_F32) => "v_floor_f32",
            (Format::Vop1, vop1::V_BFREV_B32) => "v_bfrev_b32",
            (Format::Vop1, vop1::V_FFBH_U32) => "v_ffbh_u32",
            (Format::Vop1, vop1::V_FFBH_I32) => "v_ffbh_i32",
            (Format::Vop1, vop1::V_FFBL_B32) => "v_ffbl_b32",
            (Format::Vop2, vop2::V_MUL_U32_U24) => "v_mul_u32_u24",
            (Format::Vop2, vop2::V_MUL_I32_I24) => "v_mul_i32_i24",
            (Format::Vop2, vop2::V_MUL_HI_I32_I24) => "v_mul_hi_i32_i24",
            (Format::Vop2, vop2::V_MUL_HI_U32_U24) => "v_mul_hi_u32_u24",
            (Format::Vop2, vop2::V_MIN_F32) => "v_min_f32",
            (Format::Vop2, vop2::V_MAX_F32) => "v_max_f32",
            (Format::Vop2, vop2::V_FMAC_F32) => "v_fmac_f32",
            (Format::Vop2, vop2::V_MIN_I32) => "v_min_i32",
            (Format::Vop2, vop2::V_MAX_I32) => "v_max_i32",
            (Format::Vop2, vop2::V_MIN_U32) => "v_min_u32",
            (Format::Vop2, vop2::V_MAX_U32) => "v_max_u32",
            (Format::Vop2, vop2::V_LSHRREV_B32) => "v_lshrrev_b32",
            (Format::Vop2, vop2::V_ASHRREV_I32) => "v_ashrrev_i32",
            (Format::Vop2, vop2::V_LSHLREV_B32) => "v_lshlrev_b32",
            (Format::Vop2, vop2::V_AND_B32) => "v_and_b32",
            (Format::Vop2, vop2::V_OR_B32) => "v_or_b32",
            (Format::Vop2, vop2::V_XOR_B32) => "v_xor_b32",
            (Format::Vop2, vop2::V_ADD_NC_U32) => "v_add_nc_u32",
            (Format::Vop2, vop2::V_SUB_NC_U32) => "v_sub_nc_u32",
            (Format::Vop2, vop2::V_SUBREV_NC_U32) => "v_subrev_nc_u32",
            (Format::Vop2, vop2::V_CNDMASK_B32) => "v_cndmask_b32",
            (Format::Vop2, vop2::V_ADD_F32) => "v_add_f32",
            (Format::Vop2, vop2::V_SUB_F32) => "v_sub_f32",
            (Format::Vop2, vop2::V_MUL_F32) => "v_mul_f32",
            (Format::Vopc, vopc::V_CMP_LT_F32) => "v_cmp_lt_f32",
            (Format::Vopc, vopc::V_CMP_EQ_F32) => "v_cmp_eq_f32",
            (Format::Vopc, vopc::V_CMP_GT_F32) => "v_cmp_gt_f32",
            (Format::Vopc, vopc::V_CMP_LT_I32) => "v_cmp_lt_i32",
            (Format::Vopc, vopc::V_CMP_EQ_I32) => "v_cmp_eq_i32",
            (Format::Vopc, vopc::V_CMP_GT_I32) => "v_cmp_gt_i32",
            (Format::Vopc, vopc::V_CMP_NE_I32) => "v_cmp_ne_i32",
            (Format::Vop3, 0x101) => "v_cndmask_b32_e64",
            (Format::Vop3, 0x103) => "v_add_f32_e64",
            (Format::Vop3, 0x181) => "v_mov_b32_e64",
            (Format::Vop3, vop3::V_FMA_F32) => "v_fma_f32",
            (Format::Vop3, vop3::V_MBCNT_LO_U32_B32) => "v_mbcnt_lo_u32_b32",
            (Format::Vop3, vop3::V_MBCNT_HI_U32_B32) => "v_mbcnt_hi_u32_b32",
            (Format::Smem, smem::S_BUFFER_LOAD_DWORD) => "s_buffer_load_dword",
            (Format::Smem, smem::S_BUFFER_LOAD_DWORDX2) => "s_buffer_load_dwordx2",
            (Format::Smem, smem::S_BUFFER_LOAD_DWORDX4) => "s_buffer_load_dwordx4",
            (Format::Smem, smem::S_BUFFER_LOAD_DWORDX8) => "s_buffer_load_dwordx8",
            (Format::Smem, smem::S_BUFFER_LOAD_DWORDX16) => "s_buffer_load_dwordx16",
            (Format::Flat, global::LOAD_DWORD) => "global_load_dword",
            (Format::Flat, global::LOAD_DWORDX2) => "global_load_dwordx2",
            (Format::Flat, global::LOAD_DWORDX4) => "global_load_dwordx4",
            (Format::Flat, global::STORE_DWORD) => "global_store_dword",
            (Format::Flat, global::STORE_DWORDX2) => "global_store_dwordx2",
            (Format::Flat, global::STORE_DWORDX4) => "global_store_dwordx4",
            (Format::Ds, ds::WRITE_B32) => "ds_write_b32",
            (Format::Ds, ds::READ_B32) => "ds_read_b32",
            (Format::Ds, ds::READ2_B32) => "ds_read2_b32",
            (Format::Ds, ds::WRITE_B64) => "ds_write_b64",
            (Format::Ds, ds::READ_B64) => "ds_read_b64",
            (Format::Ds, ds::WRITE_B8) => "ds_write_b8",
            (Format::Ds, ds::WRITE_B16) => "ds_write_b16",
            (Format::Ds, ds::READ_I8) => "ds_read_i8",
            (Format::Ds, ds::READ_U8) => "ds_read_u8",
            (Format::Ds, ds::READ_I16) => "ds_read_i16",
            (Format::Ds, ds::READ_U16) => "ds_read_u16",
            (Format::Ds, ds::WRITE2_B64) => "ds_write2_b64",
            (Format::Ds, ds::WRITE2ST64_B64) => "ds_write2st64_b64",
            (Format::Ds, ds::READ2_B64) => "ds_read2_b64",
            (Format::Ds, ds::READ2ST64_B64) => "ds_read2st64_b64",
            (Format::Ds, ds::WRITE_B96) => "ds_write_b96",
            (Format::Ds, ds::WRITE_B128) => "ds_write_b128",
            (Format::Ds, ds::READ_B96) => "ds_read_b96",
            (Format::Ds, ds::READ_B128) => "ds_read_b128",
            (Format::Mimg, mimg::IMAGE_LOAD) => "image_load",
            (Format::Mimg, mimg::IMAGE_STORE) => "image_store",
            (Format::Vintrp, vintrp::P1_F32) => "v_interp_p1_f32",
            (Format::Vintrp, vintrp::P2_F32) => "v_interp_p2_f32",
            (Format::Vintrp, vintrp::MOV_F32) => "v_interp_mov_f32",
            (Format::Ds, ds::WRITE2_B32) => "ds_write2_b32",
            (Format::Ds, ds::WRITE2ST64_B32) => "ds_write2st64_b32",
            (Format::Ds, ds::READ2ST64_B32) => "ds_read2st64_b32",
            (Format::Ds, ds::ADD_U32) => "ds_add_u32",
            (Format::Ds, ds::SUB_U32) => "ds_sub_u32",
            (Format::Ds, ds::MIN_I32) => "ds_min_i32",
            (Format::Ds, ds::MAX_I32) => "ds_max_i32",
            (Format::Ds, ds::MIN_U32) => "ds_min_u32",
            (Format::Ds, ds::MAX_U32) => "ds_max_u32",
            (Format::Ds, ds::AND_B32) => "ds_and_b32",
            (Format::Ds, ds::OR_B32) => "ds_or_b32",
            (Format::Ds, ds::XOR_B32) => "ds_xor_b32",
            (Format::Ds, ds::CMPST_B32) => "ds_cmpst_b32",
            (Format::Ds, 0x20) => "ds_add_rtn_u32",
            (Format::Ds, 0x21) => "ds_sub_rtn_u32",
            (Format::Ds, 0x25) => "ds_min_rtn_i32",
            (Format::Ds, 0x26) => "ds_max_rtn_i32",
            (Format::Ds, 0x27) => "ds_min_rtn_u32",
            (Format::Ds, 0x28) => "ds_max_rtn_u32",
            (Format::Ds, 0x29) => "ds_and_rtn_b32",
            (Format::Ds, 0x2A) => "ds_or_rtn_b32",
            (Format::Ds, 0x2B) => "ds_xor_rtn_b32",
            (Format::Ds, ds::WRXCHG_RTN_B32) => "ds_wrxchg_rtn_b32",
            (Format::Ds, ds::CMPST_RTN_B32) => "ds_cmpst_rtn_b32",
            (Format::Flat, global::LOAD_UBYTE) => "global_load_ubyte",
            (Format::Flat, global::STORE_BYTE) => "global_store_byte",
            (Format::Flat, global::STORE_BYTE_D16_HI) => "global_store_byte_d16_hi",
            (Format::Flat, global::STORE_SHORT) => "global_store_short",
            (Format::Flat, global::STORE_SHORT_D16_HI) => "global_store_short_d16_hi",
            (Format::Flat, global::LOAD_UBYTE_D16) => "global_load_ubyte_d16",
            (Format::Flat, global::LOAD_UBYTE_D16_HI) => "global_load_ubyte_d16_hi",
            (Format::Flat, global::LOAD_SBYTE_D16) => "global_load_sbyte_d16",
            (Format::Flat, global::LOAD_SBYTE_D16_HI) => "global_load_sbyte_d16_hi",
            (Format::Flat, global::LOAD_SHORT_D16) => "global_load_short_d16",
            (Format::Flat, global::LOAD_SHORT_D16_HI) => "global_load_short_d16_hi",
            (Format::Flat, global::LOAD_SBYTE) => "global_load_sbyte",
            (Format::Flat, global::LOAD_USHORT) => "global_load_ushort",
            (Format::Flat, global::LOAD_SSHORT) => "global_load_sshort",
            (Format::Flat, global::LOAD_DWORDX3) => "global_load_dwordx3",
            (Format::Flat, global::STORE_DWORDX3) => "global_store_dwordx3",
            (Format::Flat, global::ATOMIC_SWAP) => "global_atomic_swap",
            (Format::Flat, global::ATOMIC_CMPSWAP) => "global_atomic_cmpswap",
            (Format::Flat, global::ATOMIC_ADD) => "global_atomic_add",
            (Format::Flat, global::ATOMIC_SUB) => "global_atomic_sub",
            (Format::Flat, global::ATOMIC_SMIN) => "global_atomic_smin",
            (Format::Flat, global::ATOMIC_UMIN) => "global_atomic_umin",
            (Format::Flat, global::ATOMIC_SMAX) => "global_atomic_smax",
            (Format::Flat, global::ATOMIC_UMAX) => "global_atomic_umax",
            (Format::Flat, global::ATOMIC_AND) => "global_atomic_and",
            (Format::Flat, global::ATOMIC_OR) => "global_atomic_or",
            (Format::Flat, global::ATOMIC_XOR) => "global_atomic_xor",
            (Format::Mubuf, global::ATOMIC_SWAP) => "buffer_atomic_swap",
            (Format::Mubuf, global::ATOMIC_CMPSWAP) => "buffer_atomic_cmpswap",
            (Format::Mubuf, global::ATOMIC_ADD) => "buffer_atomic_add",
            (Format::Mubuf, global::ATOMIC_SUB) => "buffer_atomic_sub",
            (Format::Mubuf, global::ATOMIC_SMIN) => "buffer_atomic_smin",
            (Format::Mubuf, global::ATOMIC_UMIN) => "buffer_atomic_umin",
            (Format::Mubuf, global::ATOMIC_SMAX) => "buffer_atomic_smax",
            (Format::Mubuf, global::ATOMIC_UMAX) => "buffer_atomic_umax",
            (Format::Mubuf, global::ATOMIC_AND) => "buffer_atomic_and",
            (Format::Mubuf, global::ATOMIC_OR) => "buffer_atomic_or",
            (Format::Mubuf, global::ATOMIC_XOR) => "buffer_atomic_xor",
            (Format::Mtbuf, 0) => "tbuffer_load_format_x",
            (Format::Mtbuf, 1) => "tbuffer_load_format_xy",
            (Format::Mtbuf, 2) => "tbuffer_load_format_xyz",
            (Format::Mtbuf, 3) => "tbuffer_load_format_xyzw",
            (Format::Mtbuf, 4) => "tbuffer_store_format_x",
            (Format::Mtbuf, 5) => "tbuffer_store_format_xy",
            (Format::Mtbuf, 6) => "tbuffer_store_format_xyz",
            (Format::Mtbuf, 7) => "tbuffer_store_format_xyzw",
            (Format::Mubuf, 8) => "buffer_load_ubyte",
            (Format::Mubuf, 9) => "buffer_load_sbyte",
            (Format::Mubuf, 10) => "buffer_load_ushort",
            (Format::Mubuf, 11) => "buffer_load_sshort",
            (Format::Mubuf, mubuf::STORE_DWORDX3) => "buffer_store_dwordx3",
            (Format::Mubuf, mubuf::LOAD_DWORD) => "buffer_load_dword",
            (Format::Mubuf, mubuf::LOAD_DWORDX2) => "buffer_load_dwordx2",
            (Format::Mubuf, mubuf::LOAD_DWORDX3) => "buffer_load_dwordx3",
            (Format::Mubuf, mubuf::LOAD_DWORDX4) => "buffer_load_dwordx4",
            (Format::Mubuf, mubuf::STORE_DWORD) => "buffer_store_dword",
            (Format::Mubuf, mubuf::STORE_DWORDX2) => "buffer_store_dwordx2",
            (Format::Mubuf, mubuf::STORE_DWORDX4) => "buffer_store_dwordx4",
            (Format::Smem, 0) => "s_load_dword",
            (Format::Smem, 1) => "s_load_dwordx2",
            (Format::Smem, 2) => "s_load_dwordx4",
            (Format::Smem, 3) => "s_load_dwordx8",
            (Format::Smem, 4) => "s_load_dwordx16",
            (Format::Exp, _) => "exp",
            _ => "",
        };
        if known.is_empty() {
            format!("{}_op{}", self.format.prefix(), self.opcode)
        } else {
            known.to_string()
        }
    }
}

impl fmt::Display for Inst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lit = self.literal().unwrap_or(0);
        let src = |v: u16| operand(v, lit);
        let sdst = |v: u8| operand(v as u16, 0);
        write!(f, "{}", self.mnemonic())?;
        match &self.fields {
            Fields::Sop2 {
                sdst: d,
                ssrc0,
                ssrc1,
            } => write!(f, " {}, {}, {}", sdst(*d), src(*ssrc0), src(*ssrc1)),
            Fields::Sopk { sdst: d, simm16 } => write!(f, " {}, {:#x}", sdst(*d), simm16),
            Fields::Sop1 { sdst: d, ssrc0 } => write!(f, " {}, {}", sdst(*d), src(*ssrc0)),
            Fields::Sopc { ssrc0, ssrc1 } => write!(f, " {}, {}", src(*ssrc0), src(*ssrc1)),
            Fields::Sopp { simm16 } if self.is_endpgm() || *simm16 == 0 => Ok(()),
            Fields::Sopp { simm16 } => write!(f, " {:#x}", simm16),
            Fields::Smem {
                sdata,
                sbase,
                offset,
                soffset,
                ..
            } => {
                write!(f, " s{sdata}, s[{sbase}:{}], {offset:#x}", sbase + 1)?;
                if *soffset != 125 {
                    write!(f, ", {}", sdst(*soffset))?;
                }
                Ok(())
            }
            Fields::Vop2 { vdst, src0, vsrc1 } => write!(f, " v{vdst}, {}, v{vsrc1}", src(*src0)),
            Fields::Vop1 { vdst, src0 } => write!(f, " v{vdst}, {}", src(*src0)),
            Fields::Vopc { src0, vsrc1 } => write!(f, " vcc, {}, v{vsrc1}", src(*src0)),
            Fields::Vop3 { vdst, src: s, .. } | Fields::Vop3p { vdst, src: s, .. } => {
                write!(f, " v{vdst}, {}, {}, {}", src(s[0]), src(s[1]), src(s[2]))
            }
            Fields::Vintrp {
                vdst,
                vsrc,
                attr,
                chan,
            } => {
                write!(
                    f,
                    " v{vdst}, v{vsrc}, attr{attr}.{}",
                    ['x', 'y', 'z', 'w'][*chan as usize]
                )
            }
            Fields::Exp {
                target,
                en,
                vsrc,
                done,
                ..
            } => {
                let t = match target {
                    0..=7 => format!("mrt{target}"),
                    8 => "mrtz".into(),
                    9 => "null".into(),
                    12..=15 => format!("pos{}", target - 12),
                    32..=63 => format!("param{}", target - 32),
                    _ => format!("target{target}"),
                };
                write!(f, " {t}")?;
                for (i, v) in vsrc.iter().enumerate() {
                    if en >> i & 1 == 1 {
                        write!(f, ", v{v}")?;
                    } else {
                        f.write_str(", off")?;
                    }
                }
                if *done {
                    f.write_str(" done")?;
                }
                Ok(())
            }
            other => write!(f, " {other:?}"),
        }
    }
}

/// Decode a whole shader, stopping after `s_endpgm`/`s_code_end` (or at
/// the end of `code`). Returns `(dword index, instruction)` pairs.
pub fn decode_shader(code: &[u32]) -> Result<Vec<(usize, Inst)>, DecodeError> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < code.len() {
        let i = decode(code, at)?;
        let (len, end) = (i.len(), i.is_endpgm());
        out.push((at, i));
        at += len;
        if end {
            break;
        }
    }
    Ok(out)
}

/// Basic-block leaders (dword indices): the entry, every branch target and
/// every instruction after a branch.
pub fn block_starts(insts: &[(usize, Inst)]) -> Vec<usize> {
    let mut v = vec![0];
    for (at, i) in insts {
        if let Some(t) = i.branch_target(*at) {
            v.push(t);
            v.push(at + i.len());
        }
    }
    v.sort_unstable();
    v.dedup();
    v
}

/// Text listing, one instruction per line with its dword offset.
pub fn disassemble(code: &[u32]) -> String {
    let mut s = String::new();
    match decode_shader(code) {
        Ok(insts) => {
            for (at, i) in insts {
                s.push_str(&format!("{:5}: {i}\n", at * 4));
            }
        }
        Err(e) => s.push_str(&format!("error: {e}\n")),
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_lengths_and_text() {
        let code = [
            0xBE80_03FF,
            0x1234_5678, // s_mov_b32 s0, 0x12345678   (SOP1 + literal)
            0xF408_0100,
            0xFA00_0010, // s_load_dwordx4 s4, s[0:1], 0x10
            0x7E02_0202, //              v_mov_b32 v1, s2            (VOP1)
            0x0600_02F2, //              v_add_f32 v0, 1.0, v1       (VOP2, inline float)
            0xBF84_FFFE, //              s_cbranch_scc0 back to v_mov
            0xF800_080F,
            0x0302_0100, // exp mrt0, v0, v1, v2, v3 done
            0xBF81_0000, //              s_endpgm
            0xDEAD_BEEF, //              (never decoded)
        ];
        let insts = decode_shader(&code).unwrap();
        let text: Vec<String> = insts.iter().map(|(_, i)| i.to_string()).collect();
        assert_eq!(
            text,
            [
                "s_mov_b32 s0, 0x12345678",
                "s_load_dwordx4 s4, s[0:1], 0x10",
                "v_mov_b32 v1, s2",
                "v_add_f32 v0, 1.0, v1",
                "s_cbranch_scc0 0xfffe",
                "exp mrt0, v0, v1, v2, v3 done",
                "s_endpgm",
            ]
        );
        let at: Vec<usize> = insts.iter().map(|(a, _)| *a).collect();
        assert_eq!(at, [0, 2, 4, 5, 6, 7, 9]);
        assert_eq!(
            insts[4].1.branch_target(6),
            Some(5),
            "simm16 counts dwords from the next instruction"
        );
        assert_eq!(block_starts(&insts), [0, 5, 7]);
        assert!(insts[6].1.is_endpgm());
    }

    #[test]
    fn operands_and_extra_words() {
        assert_eq!(operand(129, 0), Operand::Int(1));
        assert_eq!(operand(192, 0), Operand::Int(64));
        assert_eq!(operand(193, 0), Operand::Int(-1));
        assert_eq!(operand(208, 0), Operand::Int(-16));
        assert_eq!(operand(256 + 7, 0), Operand::Vgpr(7));
        assert_eq!(operand(106, 0), Operand::VccLo);
        assert_eq!(operand(126, 0), Operand::ExecLo);
        // VOP2 with DPP16 marker in src0: one extra dword.
        let dpp = [0x0600_02FA, 0x0000_00E4];
        assert_eq!(decode(&dpp, 0).unwrap().len(), 2);
        // VOP3 (v_op with a literal in src1) → 3 dwords.
        let vop3 = [0xD540_0000, 255 << 9, 0x3F80_0000];
        let i = decode(&vop3, 0).unwrap();
        assert_eq!(
            (i.format, i.len(), i.literal()),
            (Format::Vop3, 3, Some(0x3F80_0000))
        );
        // MIMG with NSA=1: one extra address dword.
        let mimg = [0xF000_0002 | (1 << 8), 0, 0x0403_0201];
        let i = decode(&mimg, 0).unwrap();
        assert_eq!(i.len(), 3);
        assert!(
            matches!(&i.fields, Fields::Mimg { nsa_addrs, dmask: 1, .. } if nsa_addrs == &[1, 2, 3, 4])
        );
        // Truncation is an error, not a panic.
        assert!(matches!(
            decode(&[0xBE80_03FF], 0),
            Err(DecodeError::Truncated { .. })
        ));
        assert!(matches!(
            decode(&[0xFFFF_FFFF], 0),
            Err(DecodeError::UnknownFormat { .. })
        ));
    }
}
