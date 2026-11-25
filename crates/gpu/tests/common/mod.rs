//! Hand encoders for RDNA2 instructions, shared by the GPU tests. Checked
//! against LLVM's GFX10 encodings in `interp.rs`.
#![allow(dead_code)]

pub const VCC: u16 = 106;
pub const EXEC: u16 = 126;
pub const LIT: u16 = 255;
pub const ONE_F: u16 = 242; // 1.0
pub const fn int(i: i32) -> u16 {
    if i >= 0 {
        128 + i as u16
    } else {
        (192 - i) as u16
    }
}
pub const fn v(n: u16) -> u16 {
    256 + n
}

pub fn sop2e(op: u16, d: u16, s0: u16, s1: u16) -> u32 {
    0x8000_0000 | (op as u32) << 23 | (d as u32) << 16 | (s1 as u32) << 8 | s0 as u32
}
pub fn sop1e(op: u16, d: u16, s0: u16) -> u32 {
    0xBE80_0000 | (d as u32) << 16 | (op as u32) << 8 | s0 as u32
}
pub fn sopce(op: u16, s0: u16, s1: u16) -> u32 {
    0xBF00_0000 | (op as u32) << 16 | (s1 as u32) << 8 | s0 as u32
}
pub fn soppe(op: u16, simm: i16) -> u32 {
    0xBF80_0000 | (op as u32) << 16 | simm as u16 as u32
}
pub fn vop1e(op: u16, vdst: u16, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | src0 as u32
}
pub fn vop2e(op: u16, vdst: u16, src0: u16, vsrc1: u16) -> u32 {
    (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | src0 as u32
}
pub fn vopce(op: u16, src0: u16, vsrc1: u16) -> u32 {
    0x7C00_0000 | (op as u32) << 17 | (vsrc1 as u32) << 9 | src0 as u32
}
pub fn vop3e(op: u16, vdst: u16, s: [u16; 3]) -> [u32; 2] {
    [
        0xD400_0000 | (op as u32) << 16 | vdst as u32,
        s[0] as u32 | (s[1] as u32) << 9 | (s[2] as u32) << 18,
    ]
}
pub const ENDPGM: u32 = 0xBF81_0000;

pub fn le(b: [u8; 4]) -> u32 {
    u32::from_le_bytes(b)
}

