//! Decoder checked against reference GFX10 encodings from LLVM's AMDGPU
//! assembler tests (llvm/test/MC/AMDGPU/gfx10_asm_{sop,smem,vop1,vop2,exp}.s).
//! Each case: encoding bytes as LLVM prints them, the assembly LLVM gives
//! (for reference), and what this decoder must make of them.

use scarlett_gpu::rdna::{decode, Fields, Format};

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

/// (llvm assembly, bytes, format, mnemonic, length in dwords)
const CASES: &[(&str, &[u8], Format, &str, usize)] = &[
    (
        "s_nop 0x0",
        &[0x00, 0x00, 0x80, 0xbf],
        Format::Sopp,
        "s_nop",
        1,
    ),
    (
        "s_endpgm",
        &[0x00, 0x00, 0x81, 0xbf],
        Format::Sopp,
        "s_endpgm",
        1,
    ),
    (
        "s_branch 0x0",
        &[0x00, 0x00, 0x82, 0xbf],
        Format::Sopp,
        "s_branch",
        1,
    ),
    (
        "s_wakeup",
        &[0x00, 0x00, 0x83, 0xbf],
        Format::Sopp,
        "s_wakeup",
        1,
    ),
    (
        "s_cbranch_scc0 0x0",
        &[0x00, 0x00, 0x84, 0xbf],
        Format::Sopp,
        "s_cbranch_scc0",
        1,
    ),
    (
        "s_cbranch_scc1 0x0",
        &[0x00, 0x00, 0x85, 0xbf],
        Format::Sopp,
        "s_cbranch_scc1",
        1,
    ),
    (
        "s_cbranch_vccz 0x0",
        &[0x00, 0x00, 0x86, 0xbf],
        Format::Sopp,
        "s_cbranch_vccz",
        1,
    ),
    (
        "s_cbranch_vccnz 0x0",
        &[0x00, 0x00, 0x87, 0xbf],
        Format::Sopp,
        "s_cbranch_vccnz",
        1,
    ),
    (
        "s_cbranch_execz 0x0",
        &[0x00, 0x00, 0x88, 0xbf],
        Format::Sopp,
        "s_cbranch_execz",
        1,
    ),
    (
        "s_cbranch_execnz 0x0",
        &[0x00, 0x00, 0x89, 0xbf],
        Format::Sopp,
        "s_cbranch_execnz",
        1,
    ),
    (
        "s_barrier",
        &[0x00, 0x00, 0x8a, 0xbf],
        Format::Sopp,
        "s_barrier",
        1,
    ),
    (
        "s_waitcnt 0x0",
        &[0x00, 0x00, 0x8c, 0xbf],
        Format::Sopp,
        "s_waitcnt",
        1,
    ),
    (
        "s_sleep 0x0",
        &[0x00, 0x00, 0x8e, 0xbf],
        Format::Sopp,
        "s_sleep",
        1,
    ),
    (
        "s_sendmsg 0x0",
        &[0x00, 0x00, 0x90, 0xbf],
        Format::Sopp,
        "s_sendmsg",
        1,
    ),
    (
        "s_code_end",
        &[0x00, 0x00, 0x9f, 0xbf],
        Format::Sopp,
        "s_code_end",
        1,
    ),
    (
        "s_movk_i32 exec_hi, 0x1234",
        &[0x34, 0x12, 0x7f, 0xb0],
        Format::Sopk,
        "s_movk_i32",
        1,
    ),
    (
        "s_setreg_imm32_b32 0x1234, 0x11213141",
        &[0x34, 0x12, 0x80, 0xba, 0x41, 0x31, 0x21, 0x11],
        Format::Sopk,
        "s_setreg_imm32_b32",
        2,
    ),
    (
        "s_mov_b32 exec_hi, s1",
        &[0x01, 0x03, 0xff, 0xbe],
        Format::Sop1,
        "s_mov_b32",
        1,
    ),
    (
        "s_mov_b64 exec, s[2:3]",
        &[0x02, 0x04, 0xfe, 0xbe],
        Format::Sop1,
        "s_mov_b64",
        1,
    ),
    (
        "s_abs_i32 s0, 0xaf123456",
        &[0xff, 0x34, 0x80, 0xbe, 0x56, 0x34, 0x12, 0xaf],
        Format::Sop1,
        "s_abs_i32",
        2,
    ),
    (
        "s_add_u32 s0, s1, s2",
        &[0x01, 0x02, 0x00, 0x80],
        Format::Sop2,
        "s_add_u32",
        1,
    ),
    (
        "s_sub_u32 exec_hi, s1, s2",
        &[0x01, 0x02, 0xff, 0x80],
        Format::Sop2,
        "s_sub_u32",
        1,
    ),
    (
        "v_nop_e32",
        &[0x00, 0x00, 0x00, 0x7e],
        Format::Vop1,
        "v_nop",
        1,
    ),
    (
        "v_mov_b32_e32 v5, v1",
        &[0x01, 0x03, 0x0a, 0x7e],
        Format::Vop1,
        "v_mov_b32",
        1,
    ),
    (
        "v_mov_b32_e32 v5, 0xaf123456",
        &[0xff, 0x02, 0x0a, 0x7e, 0x56, 0x34, 0x12, 0xaf],
        Format::Vop1,
        "v_mov_b32",
        2,
    ),
    (
        "v_cndmask_b32_e32 v5, v1, v2, vcc",
        &[0x01, 0x05, 0x0a, 0x02],
        Format::Vop2,
        "v_cndmask_b32",
        1,
    ),
    (
        "v_add_f32_e32 v5, v1, v2",
        &[0x01, 0x05, 0x0a, 0x06],
        Format::Vop2,
        "v_add_f32",
        1,
    ),
    (
        "v_sub_f32_e32 v5, v1, v2",
        &[0x01, 0x05, 0x0a, 0x08],
        Format::Vop2,
        "v_sub_f32",
        1,
    ),
    (
        "v_mul_f32_e32 v5, v1, v2",
        &[0x01, 0x05, 0x0a, 0x10],
        Format::Vop2,
        "v_mul_f32",
        1,
    ),
    (
        "s_load_dword s5, s[2:3], s0",
        &[0x41, 0x01, 0x00, 0xf4, 0, 0, 0, 0],
        Format::Smem,
        "s_load_dword",
        2,
    ),
    (
        "s_load_dwordx2 s[10:11], s[2:3], s0",
        &[0x81, 0x02, 0x04, 0xf4, 0, 0, 0, 0],
        Format::Smem,
        "s_load_dwordx2",
        2,
    ),
    (
        "s_load_dwordx4 s[20:23], s[2:3], s0",
        &[0x01, 0x05, 0x08, 0xf4, 0, 0, 0, 0],
        Format::Smem,
        "s_load_dwordx4",
        2,
    ),
    (
        "s_load_dwordx8 s[20:27], s[2:3], s0",
        &[0x01, 0x05, 0x0c, 0xf4, 0, 0, 0, 0],
        Format::Smem,
        "s_load_dwordx8",
        2,
    ),
    (
        "s_load_dwordx16 s[20:35], s[2:3], s0",
        &[0x01, 0x05, 0x10, 0xf4, 0, 0, 0, 0],
        Format::Smem,
        "s_load_dwordx16",
        2,
    ),
    (
        "exp mrt0 v0, v0, v0, v0",
        &[0x0f, 0x00, 0x00, 0xf8, 0, 0, 0, 0],
        Format::Exp,
        "exp",
        2,
    ),
];

#[test]
fn formats_mnemonics_and_lengths_match_llvm() {
    for (asm, bytes, format, mnemonic, len) in CASES {
        let w = words(bytes);
        let i = decode(&w, 0).unwrap_or_else(|e| panic!("{asm}: {e}"));
        assert_eq!(i.format, *format, "{asm}");
        assert_eq!(i.mnemonic(), *mnemonic, "{asm}");
        assert_eq!(i.len(), *len, "{asm}: length");
    }
}

#[test]
fn operand_fields_match_llvm() {
    let d = |b: &[u8]| decode(&words(b), 0).unwrap();
    // v_mov_b32_e32 v5, v1: VGPR source is 256 + n.
    assert_eq!(
        d(&[0x01, 0x03, 0x0a, 0x7e]).fields,
        Fields::Vop1 { vdst: 5, src0: 257 }
    );
    // v_add_f32_e32 v5, v1, v2
    assert_eq!(
        d(&[0x01, 0x05, 0x0a, 0x06]).fields,
        Fields::Vop2 {
            vdst: 5,
            src0: 257,
            vsrc1: 2
        }
    );
    // s_sub_u32 exec_hi, s1, s2 (exec_hi = 127)
    assert_eq!(
        d(&[0x01, 0x02, 0xff, 0x80]).fields,
        Fields::Sop2 {
            sdst: 127,
            ssrc0: 1,
            ssrc1: 2
        }
    );
    // s_load_dwordx4 s[20:23], s[2:3], s0
    match d(&[0x01, 0x05, 0x08, 0xf4, 0, 0, 0, 0]).fields {
        Fields::Smem {
            sdata,
            sbase,
            offset,
            soffset,
            ..
        } => assert_eq!((sdata, sbase, offset, soffset), (20, 2, 0, 0)),
        f => panic!("{f:?}"),
    }
    // literals
    assert_eq!(
        d(&[0xff, 0x02, 0x0a, 0x7e, 0x56, 0x34, 0x12, 0xaf]).literal(),
        Some(0xaf12_3456)
    );
    assert_eq!(
        d(&[0x34, 0x12, 0x80, 0xba, 0x41, 0x31, 0x21, 0x11]).literal(),
        Some(0x1121_3141)
    );
    // exp mrt0 v0, v0, v0, v0: all four channels enabled, target 0
    assert!(matches!(
        d(&[0x0f, 0x00, 0x00, 0xf8, 0, 0, 0, 0]).fields,
        Fields::Exp {
            target: 0,
            en: 0xF,
            vsrc: [0, 0, 0, 0],
            done: false,
            ..
        }
    ));
}
