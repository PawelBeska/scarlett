//! PM4 packet decoding (the command format of AMD's command processor).
//!
//! A stream is a sequence of little-endian dwords. Each packet starts with
//! a header whose bits 31:30 give the type:
//!
//! | type | layout |
//! |------|--------|
//! | 0 | `[15:0]` first register (dword index), `[29:16]` count−1; body = values for consecutive registers |
//! | 2 | filler, no body |
//! | 3 | `[0]` predicate, `[1]` shader type (compute), `[15:8]` opcode, `[29:16]` body length−1 |
//!
//! Opcode values follow AMD's public GFX9/GFX10 documentation (the same
//! numbers Mesa's `sid.h` uses). They have not yet been checked against
//! streams from real titles.

use std::fmt;

/// Type-3 opcodes.
pub mod op {
    pub const NOP: u8 = 0x10;
    pub const SET_BASE: u8 = 0x11;
    pub const CLEAR_STATE: u8 = 0x12;
    pub const INDEX_BUFFER_SIZE: u8 = 0x13;
    pub const DISPATCH_DIRECT: u8 = 0x15;
    pub const DISPATCH_INDIRECT: u8 = 0x16;
    pub const ATOMIC_MEM: u8 = 0x1E;
    pub const OCCLUSION_QUERY: u8 = 0x1F;
    pub const SET_PREDICATION: u8 = 0x20;
    pub const COND_EXEC: u8 = 0x22;
    pub const PRED_EXEC: u8 = 0x23;
    pub const DRAW_INDIRECT: u8 = 0x24;
    pub const DRAW_INDEX_INDIRECT: u8 = 0x25;
    pub const INDEX_BASE: u8 = 0x26;
    pub const DRAW_INDEX_2: u8 = 0x27;
    pub const CONTEXT_CONTROL: u8 = 0x28;
    pub const INDEX_TYPE: u8 = 0x2A;
    pub const DRAW_INDIRECT_MULTI: u8 = 0x2C;
    pub const DRAW_INDEX_AUTO: u8 = 0x2D;
    pub const NUM_INSTANCES: u8 = 0x2F;
    pub const DRAW_INDEX_MULTI_AUTO: u8 = 0x30;
    pub const INDIRECT_BUFFER_CONST: u8 = 0x33;
    pub const STRMOUT_BUFFER_UPDATE: u8 = 0x34;
    pub const DRAW_INDEX_OFFSET_2: u8 = 0x35;
    pub const WRITE_DATA: u8 = 0x37;
    pub const DRAW_INDEX_INDIRECT_MULTI: u8 = 0x38;
    pub const MEM_SEMAPHORE: u8 = 0x39;
    pub const COPY_DW: u8 = 0x3B;
    pub const WAIT_REG_MEM: u8 = 0x3C;
    pub const INDIRECT_BUFFER: u8 = 0x3F;
    pub const COPY_DATA: u8 = 0x40;
    pub const PFP_SYNC_ME: u8 = 0x42;
    pub const SURFACE_SYNC: u8 = 0x43;
    pub const COND_WRITE: u8 = 0x45;
    pub const EVENT_WRITE: u8 = 0x46;
    pub const EVENT_WRITE_EOP: u8 = 0x47;
    pub const EVENT_WRITE_EOS: u8 = 0x48;
    pub const RELEASE_MEM: u8 = 0x49;
    pub const PREAMBLE_CNTL: u8 = 0x4A;
    pub const DMA_DATA: u8 = 0x50;
    pub const ACQUIRE_MEM: u8 = 0x58;
    pub const REWIND: u8 = 0x59;
    pub const LOAD_UCONFIG_REG: u8 = 0x5E;
    pub const LOAD_SH_REG: u8 = 0x5F;
    pub const LOAD_CONFIG_REG: u8 = 0x60;
    pub const LOAD_CONTEXT_REG: u8 = 0x61;
    pub const SET_CONFIG_REG: u8 = 0x68;
    pub const SET_CONTEXT_REG: u8 = 0x69;
    pub const SET_CONTEXT_REG_INDIRECT: u8 = 0x73;
    pub const SET_SH_REG: u8 = 0x76;
    pub const SET_SH_REG_OFFSET: u8 = 0x77;
    pub const SET_UCONFIG_REG: u8 = 0x79;
    pub const SET_UCONFIG_REG_INDEX: u8 = 0x7A;
    pub const LOAD_CONST_RAM: u8 = 0x80;
    pub const WRITE_CONST_RAM: u8 = 0x81;
    pub const DUMP_CONST_RAM: u8 = 0x83;
    pub const INCREMENT_CE_COUNTER: u8 = 0x84;
    pub const INCREMENT_DE_COUNTER: u8 = 0x85;
    pub const WAIT_ON_CE_COUNTER: u8 = 0x86;
    pub const SWITCH_BUFFER: u8 = 0x8B;
    pub const SET_SH_REG_INDEX: u8 = 0x9B;

    /// Human-readable name for logs.
    pub fn name(op: u8) -> &'static str {
        match op {
            NOP => "NOP",
            SET_BASE => "SET_BASE",
            CLEAR_STATE => "CLEAR_STATE",
            INDEX_BUFFER_SIZE => "INDEX_BUFFER_SIZE",
            DISPATCH_DIRECT => "DISPATCH_DIRECT",
            DISPATCH_INDIRECT => "DISPATCH_INDIRECT",
            DRAW_INDIRECT => "DRAW_INDIRECT",
            DRAW_INDEX_INDIRECT => "DRAW_INDEX_INDIRECT",
            INDEX_BASE => "INDEX_BASE",
            DRAW_INDEX_2 => "DRAW_INDEX_2",
            CONTEXT_CONTROL => "CONTEXT_CONTROL",
            INDEX_TYPE => "INDEX_TYPE",
            DRAW_INDEX_AUTO => "DRAW_INDEX_AUTO",
            NUM_INSTANCES => "NUM_INSTANCES",
            DRAW_INDEX_OFFSET_2 => "DRAW_INDEX_OFFSET_2",
            WRITE_DATA => "WRITE_DATA",
            WAIT_REG_MEM => "WAIT_REG_MEM",
            INDIRECT_BUFFER => "INDIRECT_BUFFER",
            COPY_DATA => "COPY_DATA",
            PFP_SYNC_ME => "PFP_SYNC_ME",
            EVENT_WRITE => "EVENT_WRITE",
            EVENT_WRITE_EOP => "EVENT_WRITE_EOP",
            RELEASE_MEM => "RELEASE_MEM",
            DMA_DATA => "DMA_DATA",
            ACQUIRE_MEM => "ACQUIRE_MEM",
            SET_CONFIG_REG => "SET_CONFIG_REG",
            SET_CONTEXT_REG => "SET_CONTEXT_REG",
            SET_SH_REG => "SET_SH_REG",
            SET_UCONFIG_REG => "SET_UCONFIG_REG",
            SET_UCONFIG_REG_INDEX => "SET_UCONFIG_REG_INDEX",
            SET_SH_REG_INDEX => "SET_SH_REG_INDEX",
            _ => "?",
        }
    }
}

/// Build a type-3 header (`body_len` ≥ 1 dwords).
pub const fn pkt3(opcode: u8, body_len: u32) -> u32 {
    (3 << 30) | ((body_len - 1) & 0x3FFF) << 16 | (opcode as u32) << 8
}

/// Build a type-0 header writing `count` consecutive registers from `reg`.
pub const fn pkt0(reg: u16, count: u32) -> u32 {
    ((count - 1) & 0x3FFF) << 16 | reg as u32
}

pub const PKT2: u32 = 2 << 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Packet<'a> {
    /// Write `values` to consecutive registers starting at dword `reg`.
    Type0 {
        reg: u16,
        values: &'a [u32],
    },
    Type2,
    Type3 {
        opcode: u8,
        predicate: bool,
        compute: bool,
        body: &'a [u32],
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Packet at dword `at` needs `need` dwords, only `have` remain.
    Truncated { at: usize, need: usize, have: usize },
    /// Type-1 headers do not exist on GFX9+.
    BadType { at: usize, header: u32 },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Truncated { at, need, have } => {
                write!(f, "packet at dword {at} needs {need} dwords, {have} left")
            }
            Error::BadType { at, header } => {
                write!(f, "bad packet type at dword {at}: {header:#010x}")
            }
        }
    }
}

impl std::error::Error for Error {}

/// Iterator over the packets of a dword stream. Yields `(dword offset, packet)`.
pub struct Packets<'a> {
    dw: &'a [u32],
    pos: usize,
    failed: bool,
}

pub fn packets(dw: &[u32]) -> Packets<'_> {
    Packets {
        dw,
        pos: 0,
        failed: false,
    }
}

impl<'a> Iterator for Packets<'a> {
    type Item = Result<(usize, Packet<'a>), Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed || self.pos >= self.dw.len() {
            return None;
        }
        let at = self.pos;
        let h = self.dw[at];
        let ty = h >> 30;
        let len = match ty {
            0 | 3 => (h >> 16 & 0x3FFF) as usize + 1,
            2 => 0,
            _ => {
                self.failed = true;
                return Some(Err(Error::BadType { at, header: h }));
            }
        };
        let have = self.dw.len() - at - 1;
        if len > have {
            self.failed = true;
            return Some(Err(Error::Truncated {
                at,
                need: len + 1,
                have: have + 1,
            }));
        }
        self.pos = at + 1 + len;
        let body = &self.dw[at + 1..at + 1 + len];
        Some(Ok((
            at,
            match ty {
                0 => Packet::Type0 {
                    reg: h as u16,
                    values: body,
                },
                2 => Packet::Type2,
                _ => Packet::Type3 {
                    opcode: (h >> 8) as u8,
                    predicate: h & 1 != 0,
                    compute: h & 2 != 0,
                    body,
                },
            },
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_all_types_and_reports_truncation() {
        let s = [
            pkt0(0x1234, 2),
            7,
            8,
            PKT2,
            pkt3(op::DRAW_INDEX_AUTO, 2) | 2,
            3,
            0x2,
            pkt3(op::NOP, 4),
            1,
        ];
        let v: Vec<_> = packets(&s).collect();
        assert_eq!(
            v[0],
            Ok((
                0,
                Packet::Type0 {
                    reg: 0x1234,
                    values: &[7, 8]
                }
            ))
        );
        assert_eq!(v[1], Ok((3, Packet::Type2)));
        assert_eq!(
            v[2],
            Ok((
                4,
                Packet::Type3 {
                    opcode: op::DRAW_INDEX_AUTO,
                    predicate: false,
                    compute: true,
                    body: &[3, 2]
                }
            ))
        );
        assert_eq!(
            v[3],
            Err(Error::Truncated {
                at: 7,
                need: 5,
                have: 2
            })
        );
        assert_eq!(v.len(), 4, "iteration stops after an error");
        assert!(matches!(
            packets(&[1 << 30]).next(),
            Some(Err(Error::BadType { .. }))
        ));
    }
}
