//! The reference interpreter on small hand-encoded programs, in both wave
//! sizes. The encoders are first checked against LLVM's GFX10 encodings.

use scarlett_gpu::interp::{run, ExecError, Wave, WaveSize};
use scarlett_gpu::rdna::{sop1, sop2, sopc, sopp, vop1, vop2, vop3, vopc};

mod common;
use common::*;

#[test]
fn encoders_match_llvm() {
    // Each pair: our encoder, the bytes from llvm/test/MC/AMDGPU/gfx10_asm_*.s.
    assert_eq!(sop2e(sop2::S_ADD_U32, 0, 1, 2), le([0x01, 0x02, 0x00, 0x80]));
    assert_eq!(vop2e(vop2::V_ADD_NC_U32, 5, v(1), 2), le([0x01, 0x05, 0x0a, 0x4a]));
    assert_eq!(vop2e(vop2::V_LSHRREV_B32, 5, v(1), 2), le([0x01, 0x05, 0x0a, 0x2c]));
    assert_eq!(vop1e(vop1::V_FFBH_U32, 5, v(1)), le([0x01, 0x73, 0x0a, 0x7e]));
    assert_eq!(
        vop3e(vop3::V_ADD3_U32, 5, [v(1), v(2), v(3)]),
        [le([0x05, 0x00, 0x6d, 0xd7]), le([0x01, 0x05, 0x0e, 0x04])]
    );
    assert_eq!(sop2e(sop2::S_AND_B64, 0, 2, 4), le([0x02, 0x04, 0x80, 0x87]));
    assert_eq!(sop1e(sop1::S_AND_SAVEEXEC_B64, 0, 2), le([0x02, 0x24, 0x80, 0xbe]));
    assert_eq!(sopce(sopc::S_CMP_LE_I32, 105, 1), le([0x69, 0x01, 0x05, 0xbf]));
    assert_eq!(vop1e(vop1::V_MOV_B32, 5, v(1)), le([0x01, 0x03, 0x0a, 0x7e]));
    assert_eq!(vop2e(vop2::V_ADD_F32, 5, v(1), 2), le([0x01, 0x05, 0x0a, 0x06]));
    assert_eq!(vopce(vopc::V_CMP_NE_I32, v(1), 2), le([0x01, 0x05, 0x0a, 0x7d]));
    assert_eq!(
        vop3e(vop3::V_FMA_F32, 5, [v(1), v(2), v(3)]),
        [le([0x05, 0x00, 0x4b, 0xd5]), le([0x01, 0x05, 0x0e, 0x04])]
    );
    assert_eq!(
        vop3e(vop3::FROM_VOP2 + vop2::V_CNDMASK_B32, 5, [v(1), v(2), VCC]),
        [le([0x05, 0x00, 0x01, 0xd5]), le([0x01, 0x05, 0xaa, 0x01])]
    );
}

#[test]
fn scalar_loop_with_scc_branch() {
    // s0 = 0; s1 = 5; do { s0 += s1; s1 += -1 } while (s1 != 0)
    let code = [
        sop1e(sop1::S_MOV_B32, 0, int(0)),
        sop1e(sop1::S_MOV_B32, 1, int(5)),
        sop2e(sop2::S_ADD_U32, 0, 0, 1), // loop:
        sop2e(sop2::S_ADD_I32, 1, 1, int(-1)),
        sopce(sopc::S_CMP_LG_U32, 1, int(0)),
        soppe(sopp::S_CBRANCH_SCC1, -4),
        ENDPGM,
    ];
    let mut w = Wave::new(WaveSize::W64, 0);
    let steps = run(&code, &mut w, 100).unwrap();
    assert_eq!(w.sgpr[0], 15);
    assert_eq!(steps, 2 + 5 * 4 + 1);
}

/// if (lane < k) v1 = 1.0, with exec saved and restored around it.
fn divergent_if(size: WaveSize, k: u32) -> Wave {
    let saveexec = match size {
        WaveSize::W64 => sop1::S_AND_SAVEEXEC_B64,
        WaveSize::W32 => sop1::S_AND_SAVEEXEC_B32,
    };
    let restore = match size {
        WaveSize::W64 => sop1e(sop1::S_MOV_B64, EXEC, 2),
        WaveSize::W32 => sop1e(sop1::S_MOV_B32, EXEC, 2),
    };
    let code = [
        vopce(vopc::V_CMP_GT_I32, LIT, 0), // vcc = k > v0
        k,
        sop1e(saveexec, 2, VCC),
        vop1e(vop1::V_MOV_B32, 1, ONE_F),
        restore,
        ENDPGM,
    ];
    let mut w = Wave::new(size, 2);
    for lane in 0..64 {
        w.vgpr[0][lane] = lane as u32;
    }
    run(&code, &mut w, 100).unwrap();
    w
}

#[test]
fn divergent_if_wave64() {
    let w = divergent_if(WaveSize::W64, 40);
    for lane in 0..64 {
        let want = if lane < 40 { 1.0f32.to_bits() } else { 0 };
        assert_eq!(w.vgpr[1][lane], want, "lane {lane}");
    }
    assert_eq!(w.exec, u64::MAX, "exec restored");
    assert_eq!(w.vcc, (1 << 40) - 1);
}

#[test]
fn divergent_if_wave32() {
    let w = divergent_if(WaveSize::W32, 20);
    for lane in 0..32 {
        let want = if lane < 20 { 1.0f32.to_bits() } else { 0 };
        assert_eq!(w.vgpr[1][lane], want, "lane {lane}");
    }
    // Lanes 32..63 do not exist in wave32.
    assert!(w.vgpr[1][32..].iter().all(|&x| x == 0));
    assert_eq!(w.exec, 0xFFFF_FFFF);
    assert_eq!(w.vcc, (1 << 20) - 1, "vcc_hi untouched");
}

#[test]
fn mbcnt_gives_lane_index_among_active_lanes() {
    let lo = vop3e(vop3::V_MBCNT_LO_U32_B32, 1, [int(-1), int(0), 0]);
    let hi = vop3e(vop3::V_MBCNT_HI_U32_B32, 1, [int(-1), v(1), 0]);
    let code = [lo[0], lo[1], hi[0], hi[1], ENDPGM];
    let mut w = Wave::new(WaveSize::W64, 2);
    run(&code, &mut w, 10).unwrap();
    assert!((0..64).all(|l| w.vgpr[1][l] == l as u32));

    // With src0 = exec, each active lane gets its rank among active lanes.
    let mut w = Wave::new(WaveSize::W64, 2);
    w.exec = 0xAAAA_AAAA_AAAA_AAAA;
    let lo_e = vop3e(vop3::V_MBCNT_LO_U32_B32, 1, [EXEC, int(0), 0]);
    let hi_e = vop3e(vop3::V_MBCNT_HI_U32_B32, 1, [EXEC + 1, v(1), 0]);
    let code = [lo_e[0], lo_e[1], hi_e[0], hi_e[1], ENDPGM];
    run(&code, &mut w, 10).unwrap();
    for l in (1..64).step_by(2) {
        assert_eq!(w.vgpr[1][l], (l / 2) as u32, "lane {l}");
    }
    assert!((0..64).step_by(2).all(|l| w.vgpr[1][l] == 0), "inactive lanes untouched");
}

#[test]
fn fma_cndmask_and_readfirstlane() {
    // v2 = fma(v0, 2.0, -v1) ; vcc = v2 > 0 ; v3 = vcc ? v2 : 0 ;
    // s4 = readfirstlane(v3) with exec starting at lane 3
    let mut fma = vop3e(vop3::V_FMA_F32, 2, [v(0), 244, v(1)]);
    fma[1] |= 1 << 31; // neg on src2 (bit 29 + 2)
    let cnd = vop3e(vop3::FROM_VOP2 + vop2::V_CNDMASK_B32, 3, [int(0), v(2), VCC]);
    let code = [
        fma[0],
        fma[1],
        vopce(vopc::V_CMP_LT_F32, int(0), 2), // 0 < v2
        cnd[0],
        cnd[1],
        vop1e(vop1::V_READFIRSTLANE_B32, 4, v(3)),
        ENDPGM,
    ];
    let mut w = Wave::new(WaveSize::W32, 4);
    w.exec = !0b111 & 0xFFFF_FFFF;
    for l in 0..32 {
        w.vgpr[0][l] = (l as f32).to_bits();
        w.vgpr[1][l] = 5.0f32.to_bits();
    }
    run(&code, &mut w, 10).unwrap();
    for l in 3..32 {
        let x = 2.0 * l as f32 - 5.0;
        assert_eq!(f32::from_bits(w.vgpr[2][l]), x);
        assert_eq!(f32::from_bits(w.vgpr[3][l]), if x > 0.0 { x } else { 0.0 });
    }
    assert_eq!(f32::from_bits(w.sgpr[4]), 1.0, "lane 3: 2*3-5");
}

#[test]
fn refuses_what_it_does_not_model() {
    // VOP2 opcode 2 is not in the table.
    let code = [vop2e(2, 1, v(0), 0), ENDPGM];
    let mut w = Wave::new(WaveSize::W64, 2);
    assert!(matches!(
        run(&code, &mut w, 10),
        Err(ExecError::Unsupported { at: 0, .. })
    ));
    // Saveexec of the wrong width for the wave.
    let code = [sop1e(sop1::S_AND_SAVEEXEC_B32, 2, VCC), ENDPGM];
    assert!(run(&code, &mut Wave::new(WaveSize::W64, 0), 10).is_err());
    // No s_endpgm.
    let code = [soppe(sopp::S_NOP, 0)];
    assert_eq!(
        run(&code, &mut Wave::new(WaveSize::W64, 0), 10),
        Err(ExecError::PcOutOfRange(1))
    );
}

#[test]
fn smem_load_faults_on_unmapped_memory() {
    use scarlett_gpu::cp::Memory;
    use scarlett_gpu::interp::run_with_memory;
    // s_load_dwordx2 s[4:5], s[2:3], 0x10 (soffset null)
    let code = [0xF404_0101, 125 << 25 | 0x10, ENDPGM];
    let mut mem = scarlett_gpu::SparseMemory::new();
    mem.map(0x1000, 0x1000);
    assert!(mem.write(0x1010, &[7, 9]));
    let mut w = Wave::new(WaveSize::W64, 0);
    w.sgpr[2] = 0x1000;
    run_with_memory(&code, &mut w, &mut mem, 10).unwrap();
    assert_eq!(&w.sgpr[4..6], &[7, 9]);
    w.pc = 0;
    w.sgpr[2] = 0x9000;
    assert_eq!(
        run_with_memory(&code, &mut w, &mut mem, 10),
        Err(ExecError::MemoryFault { at: 0, addr: 0x9010 })
    );
}

/// Hand-computed values for the integer semantics shared by the
/// interpreter and (through the differential tests) the translator.
#[test]
fn integer_semantics_by_hand() {
    use scarlett_gpu::interp::{vop1_int, vop2_int, vop3_int};
    assert_eq!(vop1_int(vop1::V_FFBH_U32, 1), Some(31));
    assert_eq!(vop1_int(vop1::V_FFBH_U32, 0), Some(u32::MAX));
    assert_eq!(vop1_int(vop1::V_FFBL_B32, 8), Some(3));
    assert_eq!(vop1_int(vop1::V_FFBL_B32, 0), Some(u32::MAX));
    assert_eq!(vop1_int(vop1::V_BFREV_B32, 1), Some(0x8000_0000));
    assert_eq!(vop1_int(vop1::V_NOT_B32, 0x0F0F_0000), Some(0xF0F0_FFFF));
    // *rev shifts: S1 shifted by S0[4:0]
    assert_eq!(vop2_int(vop2::V_LSHRREV_B32, 33, 8), Some(4));
    assert_eq!(vop2_int(vop2::V_ASHRREV_I32, 4, 0x8000_0000), Some(0xF800_0000));
    assert_eq!(vop2_int(vop2::V_LSHLREV_B32, 1, 3), Some(6));
    assert_eq!(vop2_int(vop2::V_SUBREV_NC_U32, 1, 5), Some(4));
    assert_eq!(vop2_int(vop2::V_MIN_I32, 0xFFFF_FFFF, 1), Some(0xFFFF_FFFF));
    assert_eq!(vop2_int(vop2::V_MIN_U32, 0xFFFF_FFFF, 1), Some(1));
    assert_eq!(vop2_int(vop2::V_MUL_U32_U24, 0x0100_0003, 2), Some(6));
    assert_eq!(vop3_int(vop3::V_LSHL_ADD_U32, 1, 4, 1), Some(17));
    assert_eq!(vop3_int(vop3::V_BFE_U32, 0xABCD, 4, 8), Some(0xBC));
    assert_eq!(vop3_int(vop3::V_BFI_B32, 0xFF00, 0x1234, 0xABCD), Some(0x12CD));
    assert_eq!(vop3_int(vop3::V_MED3_U32, 5, 1, 3), Some(3));
    // second batch, by hand
    assert_eq!(vop3_int(vop3::V_MUL_HI_U32, 0xFFFF_FFFF, 0xFFFF_FFFF, 0), Some(0xFFFF_FFFE));
    assert_eq!(vop3_int(vop3::V_MUL_HI_I32, 0xFFFF_FFFF, 0xFFFF_FFFF, 0), Some(0)); // -1 * -1
    assert_eq!(vop3_int(vop3::V_MUL_HI_I32, 0x8000_0000, 2, 0), Some(0xFFFF_FFFF)); // -2^32
    assert_eq!(vop3_int(vop3::V_MAD_U32_U24, 0x0100_0002, 3, 1), Some(7));
    assert_eq!(vop3_int(vop3::V_MAD_I32_I24, 0x00FF_FFFF, 3, 1), Some((-2i32) as u32)); // -1 * 3 + 1
    assert_eq!(vop3_int(vop3::V_BCNT_U32_B32, 0xF0, 10, 0), Some(14));
    assert_eq!(vop3_int(vop3::V_ALIGNBIT_B32, 0x1234_5678, 0x9ABC_DEF0, 8), Some(0x789A_BCDE));
    assert_eq!(vop3_int(vop3::V_ALIGNBIT_B32, 1, 2, 32), Some(2)); // shift & 31
    assert_eq!(vop3_int(vop3::V_BFE_I32, 0xF0, 4, 4), Some(u32::MAX)); // field 0xF -> -1
    assert_eq!(vop3_int(vop3::V_BFE_I32, 0x70, 4, 4), Some(7));
    assert_eq!(vop3_int(vop3::V_BFE_I32, 0x8000_0000, 28, 8), Some(0xFFFF_FFF8)); // a >> 28 arithmetic
    assert_eq!(vop3_int(vop3::V_BFE_I32, 0xFF, 0, 0), Some(0));
    assert_eq!(vop3_int(vop3::V_MIN3_I32, 1, 0xFFFF_FFFF, 2), Some(0xFFFF_FFFF));
    assert_eq!(vop3_int(vop3::V_MAX3_U32, 1, 0xFFFF_FFFF, 2), Some(0xFFFF_FFFF));
    assert_eq!(vop3_int(vop3::V_MED3_I32, 0xFFFF_FFFF, 5, 0), Some(0));
    assert_eq!(vop3_int(vop3::V_XAD_U32, 0b1100, 0b1010, 1), Some(0b0111));
    assert_eq!(vop3_int(vop3::V_LSHL_OR_B32, 1, 36, 1), Some(0x11));
}

#[test]
fn s_buffer_load_refuses_what_it_does_not_model() {
    use scarlett_gpu::cp::Memory;
    use scarlett_gpu::interp::run_with_memory;
    let mut mem = scarlett_gpu::SparseMemory::new();
    mem.map(0x1000, 0x1000);
    // s_buffer_load_dwordx2 s[4:5], s[8:11], s0
    let code = [0xF424_0104, 0, ENDPGM];
    let run = |desc: [u32; 4], s0: u32, mem: &mut scarlett_gpu::SparseMemory| {
        let mut w = Wave::new(WaveSize::W64, 0);
        w.sgpr[8..12].copy_from_slice(&desc);
        w.sgpr[0] = s0;
        run_with_memory(&code, &mut w, mem, 10).map(|_| (w.sgpr[4], w.sgpr[5]))
    };
    let raw = [0x1000, 0, 64, 3 << 28];
    assert!(run(raw, 0, &mut mem).is_ok());
    // In bounds then out: the second dword reads 0.
    assert!(mem.write(0x1038, &[7]));
    assert_eq!(run(raw, 0x38, &mut mem), Ok((7, 0)));
    // Strided, swizzled, non-raw OOB_SELECT, wrapping offset, straddling.
    assert!(run([0x1000, 16 << 16, 64, 3 << 28], 0, &mut mem).is_err());
    assert!(run([0x1000, 1 << 31, 64, 3 << 28], 0, &mut mem).is_err());
    assert!(run([0x1000, 0, 64, 1 << 28], 0, &mut mem).is_err());
    assert!(run(raw, 0xFFFF_FFFC, &mut mem).is_err());
    assert!(run([0x1000, 0, 62, 3 << 28], 0x38, &mut mem).is_err());
}

/// DPP16 source lanes by hand, and the control word against LLVM.
#[test]
fn dpp16_rules_by_hand() {
    use scarlett_gpu::rdna::{Dpp16, DppRule};
    // v_mov_b32_dpp v5, v1 row_shl:1 row_mask:0x0 bank_mask:0x0
    let d = Dpp16::parse(u32::from_le_bytes([0x01, 0x01, 0x01, 0x00]));
    assert_eq!((d.src0, d.rule()), (1, Some(DppRule::RowShl(1))));
    // ... quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0x0 bound_ctrl:1
    let d = Dpp16::parse(u32::from_le_bytes([0x01, 0xe4, 0x08, 0xf0]));
    assert!(d.bound_ctrl && d.row_mask == 0xF && d.bank_mask == 0);
    assert_eq!(d.rule(), Some(DppRule::QuadPerm(0xE4)));
    // v_add_f32_dpp v5, -v1, |v2| ...: neg src0 bit 20, abs src1 bit 23
    let d = Dpp16::parse(u32::from_le_bytes([0x01, 0xe4, 0x10 | 0x80, 0x00]));
    assert!(d.src0_neg && d.src1_abs && !d.src0_abs && !d.src1_neg);
    // row_shr:1 is ACO's "shift rows right": lane i reads i - 1.
    assert_eq!(DppRule::RowShr(1).source(17), Some(16));
    assert_eq!(DppRule::RowShr(1).source(16), None);
    assert_eq!(DppRule::RowShl(3).source(12), Some(15));
    assert_eq!(DppRule::RowShl(3).source(13), None);
    assert_eq!(DppRule::RowRor(1).source(16), Some(31));
    assert_eq!(DppRule::RowMirror.source(33), Some(46));
    assert_eq!(DppRule::RowHalfMirror.source(9), Some(14));
    assert_eq!(DppRule::RowShare(5).source(40), Some(37));
    assert_eq!(DppRule::RowXmask(1).source(6), Some(7));
    assert_eq!(DppRule::QuadPerm(0b00_01_10_11).source(5), Some(6)); // [3,2,1,0]: lane 1 of quad reads 2
}

#[test]
fn image_load_by_hand() {
    use scarlett_gpu::cp::Memory;
    use scarlett_gpu::interp::run_with_memory;
    use scarlett_gpu::rdna::ImageDesc;
    // 32_32_UINT, 20x3: pitch = align(20, 256 / 8) = 32 texels.
    let base = 0x1_0000_0000u64;
    let t = [
        (base >> 8) as u32,
        ((base >> 40) & 0xFF) as u32 | 62 << 20 | (19 & 3) << 30,
        19 >> 2 | 2 << 14 | 1 << 31,
        5 | 4 << 3, // dst_sel = (y, x, 0, 0)
        0, 0, 0, 0,
    ];
    let d = ImageDesc::parse(t);
    assert_eq!((d.base, d.width, d.height, d.format), (base, 20, 3, 62));
    assert_eq!(d.linear_pitch(8), 32);
    let mut mem = scarlett_gpu::SparseMemory::new();
    mem.map(base, 0x1000);
    // texel (x=3, y=2) at (2 * 32 + 3) * 8
    assert!(mem.write(base + (2 * 32 + 3) * 8, &[0xAAAA, 0xBBBB]));
    // image_load v[2:3], v[0:1], s[8:15] dmask:0x3 dim:2D
    let code = [0xF000_0308, 2 << 16 | 2 << 8, ENDPGM];
    let mut w = Wave::new(WaveSize::W32, 4);
    w.sgpr[8..16].copy_from_slice(&t);
    w.vgpr[0][0] = 3;
    w.vgpr[1][0] = 2;
    w.exec = 1;
    run_with_memory(&code, &mut w, &mut mem, 10).unwrap();
    assert_eq!((w.vgpr[2][0], w.vgpr[3][0]), (0xBBBB, 0xAAAA), "swizzled (g, r)");
    // A chip-dependent tiled mode is refused.
    let mut tiled = t;
    tiled[3] |= 10 << 20; // SW_64KB_D: chip-dependent under RB+, refused
    w.sgpr[8..16].copy_from_slice(&tiled);
    w.pc = 0;
    assert!(run_with_memory(&code, &mut w, &mut mem, 10).is_err());
}

#[test]
fn unorm8_conversion_by_hand() {
    use scarlett_gpu::rdna::unorm8_to_f32;
    assert_eq!(f32::from_bits(unorm8_to_f32(0)), 0.0);
    assert_eq!(f32::from_bits(unorm8_to_f32(255)), 1.0);
    assert_eq!(f32::from_bits(unorm8_to_f32(51)), 0.2);
    assert_eq!(unorm8_to_f32(128), (128.0f32 / 255.0).to_bits());
    // Only the low byte counts.
    assert_eq!(unorm8_to_f32(0x1FF), unorm8_to_f32(0xFF));
}

#[test]
fn salu_semantics_by_hand() {
    use scarlett_gpu::interp::{sop1_b32, sop2_b32, sop2_b64};
    // s_bfe: S1 = offset | width << 16
    assert_eq!(sop2_b32(sop2::S_BFE_U32, 0xABCD, 4 | 8 << 16), Some((0xBC, true)));
    assert_eq!(sop2_b32(sop2::S_BFE_I32, 0xF0, 4 | 4 << 16), Some((u32::MAX, true)));
    assert_eq!(sop2_b32(sop2::S_BFE_I32, 0x70, 4 | 4 << 16), Some((7, true)));
    assert_eq!(sop2_b32(sop2::S_BFE_U32, 0xFF, 0), Some((0, true)), "width 0");
    // s_bfm(bits, offset)
    assert_eq!(sop2_b32(sop2::S_BFM_B32, 4, 8), Some((0xF00, false)));
    assert_eq!(sop2_b32(sop2::S_LSHL_B32, 1, 33), Some((2, true)), "shift uses S1[4:0]");
    assert_eq!(sop2_b32(sop2::S_ASHR_I32, 0x8000_0000, 31), Some((u32::MAX, true)));
    assert_eq!(sop2_b32(sop2::S_ANDN2_B32, 0xFF, 0x0F), Some((0xF0, true)));
    assert_eq!(sop2_b32(sop2::S_XNOR_B32, 0, 0), Some((u32::MAX, true)));
    assert_eq!(sop2_b32(sop2::S_MUL_I32, 0xFFFF_FFFF, 3), Some((0xFFFF_FFFD, false)));
    assert_eq!(sop2_b64(sop2::S_ANDN2_B64, u64::MAX, 1), Some(u64::MAX - 1));
    assert_eq!(sop1_b32(sop1::S_BCNT0_I32_B32, 0xF), Some((28, true)));
    assert_eq!(sop1_b32(sop1::S_FLBIT_I32_B32, 1), Some((31, false)));
    assert_eq!(sop1_b32(sop1::S_FLBIT_I32_B32, 0), Some((u32::MAX, false)));
    assert_eq!(sop1_b32(sop1::S_SEXT_I32_I8, 0x80), Some((0xFFFF_FF80, false)));
    assert_eq!(sop1_b32(sop1::S_SEXT_I32_I16, 0x1_7FFF), Some((0x7FFF, false)));
    // s_subb_u32 with SCC: 5 - 5 - 1 borrows
    let code = [
        sop1e(sop1::S_MOV_B32, 0, int(5)),
        sopce(sopc::S_CMP_EQ_U32, 0, int(5)), // scc = 1
        sop2e(sop2::S_SUBB_U32, 1, 0, 0),
        ENDPGM,
    ];
    let mut w = Wave::new(WaveSize::W64, 0);
    scarlett_gpu::interp::run(&code, &mut w, 10).unwrap();
    assert_eq!((w.sgpr[1], w.scc), (u32::MAX, true));
}

#[test]
fn compare_truth_table_by_hand() {
    use scarlett_gpu::interp::{CmpKind, Compare};
    let nan = f32::NAN.to_bits();
    let (one, two) = (1.0f32.to_bits(), 2.0f32.to_bits());
    // f lt eq le gt lg ge o u nge nlg ngt nle neq nlt tru
    let lt12 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
    let with_nan = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
    for cond in 0..16u16 {
        let c = Compare::decode(cond).unwrap();
        assert_eq!(c.eval(one, two) as u8, lt12[cond as usize], "1 vs 2, cond {cond}");
        assert_eq!(c.eval(one, nan) as u8, with_nan[cond as usize], "1 vs NaN, cond {cond}");
        assert!(Compare::decode(cond + 16).unwrap().cmpx);
    }
    let lt_i = Compare::decode(129).unwrap();
    let lt_u = Compare::decode(193).unwrap();
    assert_eq!((lt_i.kind, lt_u.kind), (CmpKind::I32, CmpKind::U32));
    assert!(lt_i.eval(u32::MAX, 0), "-1 < 0 signed");
    assert!(!lt_u.eval(u32::MAX, 0), "0xffffffff > 0 unsigned");
    // 136 = 0x88 is v_cmp_class_f32 (AMD RDNA2 ISA spec); u32 has no ninth slot
    assert_eq!(Compare::decode(136).unwrap().kind, CmpKind::Class32);
    assert!(Compare::decode(0xC8).is_none(), "no ninth u32 condition");
}

#[test]
fn rounding_by_hand() {
    use scarlett_gpu::interp::vop1_float;
    let f = |op, x: f32| f32::from_bits(vop1_float(op, x.to_bits()).unwrap());
    assert_eq!(f(vop1::V_RNDNE_F32, 2.5), 2.0, "ties to even");
    assert_eq!(f(vop1::V_RNDNE_F32, 3.5), 4.0);
    assert_eq!(f(vop1::V_RNDNE_F32, -0.5).to_bits(), (-0.0f32).to_bits());
    assert_eq!(f(vop1::V_FLOOR_F32, -0.5), -1.0);
    assert_eq!(f(vop1::V_CEIL_F32, -0.5).to_bits(), (-0.0f32).to_bits());
    assert_eq!(f(vop1::V_TRUNC_F32, -1.5), -1.0);
    assert_eq!(vop1_float(vop1::V_CVT_F32_UBYTE2, 0x00AB_0000), Some(171.0f32.to_bits()));
    assert_eq!(vop1_float(vop1::V_CVT_F32_UBYTE3, 0xFF00_0000), Some(255.0f32.to_bits()));
}

#[test]
fn min_max_by_hand() {
    use scarlett_gpu::interp::min_max_f32;
    let b = |x: f32| x.to_bits();
    let (qnan, snan) = (0x7FC0_0000, 0x7F80_0001);
    assert_eq!(min_max_f32(false, qnan, b(2.0)), Some(b(2.0)), "quiet NaN -> other");
    assert_eq!(min_max_f32(true, b(-1.0), qnan), Some(b(-1.0)));
    assert_eq!(min_max_f32(false, b(0.0), b(-0.0)), Some(b(-0.0)), "-0 < +0");
    assert_eq!(min_max_f32(true, b(-0.0), b(0.0)), Some(b(0.0)));
    assert_eq!(min_max_f32(false, b(1.0), b(2.0)), Some(b(1.0)));
    assert_eq!(min_max_f32(true, b(1.0), b(2.0)), Some(b(2.0)));
    assert_eq!(min_max_f32(false, snan, b(1.0)), None, "signaling NaN refused");
    assert_eq!(min_max_f32(false, qnan, qnan), None, "two NaNs refused");
}

/// GFX10 OOB_SELECT rules (Mesa `ac_descriptors.c`), by hand: stride 16,
/// NUM_RECORDS 3.
#[test]
fn buffer_oob_select_rules() {
    use scarlett_gpu::rdna::BufferDesc;
    let d = |oob: u32, nr: u32| BufferDesc::parse([0, 16 << 16, nr, oob << 28]);
    // 0: index < NUM_RECORDS and offset < STRIDE
    assert_eq!(d(0, 3).in_bounds(2, 12), Some(true));
    assert_eq!(d(0, 3).in_bounds(3, 0), Some(false));
    assert_eq!(d(0, 3).in_bounds(0, 16), Some(false));
    // 1: index only
    assert_eq!(d(1, 3).in_bounds(2, 400), Some(true));
    assert_eq!(d(1, 3).in_bounds(3, 0), Some(false));
    // 2: only NUM_RECORDS == 0
    assert_eq!(d(2, 0).in_bounds(0, 0), Some(false));
    assert_eq!(d(2, 1).in_bounds(99, 9999), Some(true));
    // 3 (raw): offset < NUM_RECORDS, index ignored
    assert_eq!(d(3, 3).in_bounds(99, 2), Some(true));
    assert_eq!(d(3, 3).in_bounds(0, 3), Some(false));
    // swizzled: not modelled
    assert_eq!(BufferDesc::parse([0, 1 << 31, 3, 3 << 28]).in_bounds(0, 0), None);
}
