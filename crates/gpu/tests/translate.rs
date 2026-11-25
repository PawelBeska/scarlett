//! Differential check: each program runs through the RDNA2 reference
//! interpreter and, translated, through the SPIR-V evaluator; every SGPR
//! and every lane's VGPRs must match bit for bit, in wave32 and wave64.

mod common;
use common::*;

use scarlett_gpu::interp::{self, Wave, WaveSize};
use scarlett_gpu::rdna::{ds, global, smem, sop1, sop2, sopc, sopp, vop1, vop2, vop3, vopc};
use scarlett_gpu::spirv;
use scarlett_gpu::spirv_eval::{run_compute, run_dispatch, run_dispatch_subgroups, Buffers};
use scarlett_gpu::cp::Memory;
use scarlett_gpu::SparseMemory;
use scarlett_gpu::translate::{
    translate, ComputeIo, CrossLane, Io, Options, TranslateError, Window, SPECIALS, SPECIAL_EXEC_HI, SPECIAL_EXEC_LO,
    SPECIAL_SCC, SPECIAL_VCC_HI, SPECIAL_VCC_LO,
};

const NS: u32 = 16;
/// Where every differential test's code sits in GPU memory (s_getpc_b64).
const CODE_BASE: u64 = 0x7_FFFF_FF00;
const LDS_WORDS: u32 = 256;
const NV: u32 = 8;

/// Deterministic inputs: a mix of ordinary floats, integers, zero,
/// negative zero and NaN.
fn inputs(seed: u32, lanes: usize) -> (Vec<u32>, Vec<[u32; 64]>) {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        match x % 8 {
            0 => 0,
            1 => 0x8000_0000,
            2 => f32::NAN.to_bits(),
            3 => x % 100,
            4 => (x % 7) as i32 as u32 ^ 0xFFFF_FFF0,
            _ => ((x % 2000) as f32 / 16.0 - 60.0).to_bits(),
        }
    };
    let sgprs: Vec<u32> = (0..NS).map(|_| next()).collect();
    let mut vgprs = vec![[0u32; 64]; NV as usize];
    for lane in 0..lanes {
        for r in vgprs.iter_mut() {
            r[lane] = next();
        }
    }
    (sgprs, vgprs)
}

/// Initial exec for a run: all lanes, or a sparse pattern.
fn exec_for(size: WaveSize, sparse: bool) -> u64 {
    let m = size.mask();
    if sparse {
        0x9E37_79B9_7F4A_7C15 & m
    } else {
        m
    }
}

fn check(code: &[u32], size: WaveSize, seed: u32, sparse: bool) {
    check_mem(code, size, seed, sparse, None)
}

/// GPU memory for a run: base address and contents.
type Mem<'a> = Option<(u64, &'a [u32])>;

fn check_mem(code: &[u32], size: WaveSize, seed: u32, sparse: bool, mem: Mem) {
    let lanes = size.lanes();
    let (sgprs, vgprs) = inputs(seed, lanes);
    let exec = exec_for(size, sparse);

    let mut w = Wave::new(size, NV as usize);
    w.lds = vec![0; LDS_WORDS as usize];
    w.sgpr[..NS as usize].copy_from_slice(&sgprs);
    w.vgpr.clone_from(&vgprs);
    w.exec = exec;
    w.code_base = CODE_BASE;
    let mut memory = SparseMemory::new();
    if let Some((base, data)) = mem {
        memory.map(base, 4 * data.len() as u64);
        assert!(memory.write(base, data));
    }
    interp::run_with_memory(code, &mut w, &mut memory, 10_000).expect("interpreter");

    for cross_lane in [CrossLane::Subgroup, CrossLane::Workgroup] {
    let opts = Options {
        io: Io::RegisterDump,
        num_sgprs: NS,
        num_vgprs: NV,
        lanes: lanes as u32,
        memory: mem.map(|(base, data)| Window {
            base,
            words: data.len() as u32,
        }),
        cross_lane,
        lds_words: LDS_WORDS,
        code_base: CODE_BASE,
    };
    let module = translate(code, &opts).expect("translate");
    spirv::validate(&module).expect("structurally valid SPIR-V");
    let mut buf = sgprs.clone();
    let mut specials = [0u32; SPECIALS as usize];
    specials[SPECIAL_EXEC_LO as usize] = exec as u32;
    specials[SPECIAL_EXEC_HI as usize] = (exec >> 32) as u32;
    buf.extend_from_slice(&specials);
    for lane in 0..lanes {
        for r in &vgprs {
            buf.push(r[lane]);
        }
    }
    let mut buffers = Buffers::new();
    buffers.insert((0, 0), buf);
    if let Some((_, data)) = mem {
        let mut window = data.to_vec();
        window.push(0); // the store sink
        buffers.insert((0, 1), window);
    }
    run_compute(&module, lanes as u32, &mut buffers, 1_000_000).expect("evaluate");
    let out = &buffers[&(0, 0)];

    let ctx = format!("{size:?}, {cross_lane:?}, seed {seed}, sparse {sparse}");
    for (n, (got, want)) in out.iter().zip(&w.sgpr[..NS as usize]).enumerate() {
        assert_eq!(got, want, "s{n} ({ctx})");
    }
    let sp = |k: u32| out[(NS + k) as usize];
    let got_exec = sp(SPECIAL_EXEC_LO) as u64 | (sp(SPECIAL_EXEC_HI) as u64) << 32;
    let got_vcc = sp(SPECIAL_VCC_LO) as u64 | (sp(SPECIAL_VCC_HI) as u64) << 32;
    assert_eq!(got_exec, w.exec, "exec ({ctx})");
    assert_eq!(got_vcc, w.vcc, "vcc ({ctx})");
    assert_eq!(sp(SPECIAL_SCC) != 0, w.scc, "scc ({ctx})");
    if let Some((base, data)) = mem {
        let mut want = vec![0u32; data.len()];
        assert!(memory.read(base, &mut want));
        assert_eq!(&buffers[&(0, 1)][..data.len()], &want[..], "memory ({ctx})");
    }
    let out = &buffers[&(0, 0)];
    let vbase = (NS + SPECIALS) as usize;
    for lane in 0..lanes {
        for n in 0..NV as usize {
            let got = out[vbase + lane * NV as usize + n];
            assert_eq!(
                got, w.vgpr[n][lane],
                "v{n} lane {lane} ({ctx}): {:?} vs {:?}",
                f32::from_bits(got),
                f32::from_bits(w.vgpr[n][lane])
            );
        }
    }
    }
}

fn check_all(code: &[u32]) {
    for seed in 0..4 {
        for sparse in [false, true] {
            check(code, WaveSize::W32, seed, sparse);
            check(code, WaveSize::W64, seed, sparse);
        }
    }
}

/// Wave-size specific programs (saveexec and exec restores differ).
fn check_sized(make: impl Fn(WaveSize) -> Vec<u32>) {
    for size in [WaveSize::W32, WaveSize::W64] {
        let code = make(size);
        for seed in 0..4 {
            for sparse in [false, true] {
                check(&code, size, seed, sparse);
            }
        }
    }
}

#[test]
fn float_alu_and_scalar_ops() {
    let sub_neg = vop3e(vop3::FROM_VOP2 + vop2::V_SUB_F32, 4, [v(3), v(0), 0]);
    let code = [
        vop2e(vop2::V_ADD_F32, 2, v(0), 1),
        vop2e(vop2::V_MUL_F32, 3, 244, 2), // 2.0 * v2
        sub_neg[0],
        sub_neg[1] | 1 << 29, // -v3 - v0
        vop1e(vop1::V_CVT_F32_I32, 5, v(1)),
        sop2e(sop2::S_ADD_U32, 2, 0, 1),
        sop2e(sop2::S_CSELECT_B32, 3, 0, 1), // scc = carry of the add
        vop2e(vop2::V_ADD_F32, 6, 2, 0),     // s2 + v0
        vop1e(vop1::V_MOV_B32, 7, LIT),
        0x4049_0FDB, // pi
        sop1e(sop1::S_MOV_B32, 4, int(-3)),
        ENDPGM,
    ];
    check_all(&code);
}

#[test]
fn per_lane_compare_and_select() {
    let cnd64 = vop3e(vop3::FROM_VOP2 + vop2::V_CNDMASK_B32, 4, [v(2), ONE_F, VCC]);
    let code = [
        vopce(vopc::V_CMP_LT_F32, v(0), 1),
        vop2e(vop2::V_CNDMASK_B32, 2, v(0), 1), // min-ish of v0, v1
        vopce(vopc::V_CMP_NE_I32, int(5), 3),
        cnd64[0],
        cnd64[1],
        vopce(vopc::V_CMP_GT_I32, v(5), 6),
        vop2e(vop2::V_CNDMASK_B32, 7, int(-1), 6),
        sop2e(sop2::S_AND_B32, 5, 0, LIT),
        0xFF,
        sop2e(sop2::S_CSELECT_B32, 6, 1, 2), // scc = (s0 & 0xff) != 0
        ENDPGM,
    ];
    check_all(&code);
}

#[test]
fn conversions_and_bit_ops() {
    let subrev64 = vop3e(vop3::FROM_VOP2 + vop2::V_SUBREV_F32, 4, [v(0), v(1), 0]);
    let bfe = vop3e(vop3::V_BFE_U32, 5, [v(1), int(3), int(9)]);
    let bfi = vop3e(vop3::V_BFI_B32, 6, [v(2), v(0), v(1)]);
    let med3 = vop3e(vop3::V_MED3_U32, 7, [v(0), v(1), v(2)]);
    // Scale v0 so some lanes land outside the i32/u32 range.
    let big = vop2e(vop2::V_MUL_F32, 3, LIT, 0);
    let code = [
        vop1e(vop1::V_CVT_U32_F32, 2, v(0)),
        vop1e(vop1::V_CVT_I32_F32, 3, v(1)),
        big,
        0x4F80_0000, // 2^32
        vop1e(vop1::V_CVT_I32_F32, 3, v(3)),
        vop1e(vop1::V_CVT_F32_U32, 2, v(2)),
        subrev64[0],
        subrev64[1] | 1 << 29, // v1 - (-v0)
        vop2e(vop2::V_SUBREV_F32, 0, v(2), 0), // v0 = v0 - v2
        bfe[0],
        bfe[1],
        bfi[0],
        bfi[1],
        med3[0],
        med3[1],
        ENDPGM,
    ];
    check_all(&code);
}

#[test]
fn integer_alu() {
    let e64 = |op: u16, d: u16, s: [u16; 3]| vop3e(op, d, s);
    let mut code = vec![
        vop2e(vop2::V_ADD_NC_U32, 2, v(0), 1),
        vop2e(vop2::V_SUB_NC_U32, 3, v(0), 1),
        vop2e(vop2::V_SUBREV_NC_U32, 4, int(-7), 1),
        vop2e(vop2::V_LSHLREV_B32, 5, v(0), 1), // shift by v0 & 31
        vop2e(vop2::V_LSHRREV_B32, 6, v(1), 0),
        vop2e(vop2::V_ASHRREV_I32, 7, int(3), 0),
        vop2e(vop2::V_MIN_I32, 2, v(2), 3),
        vop2e(vop2::V_MAX_U32, 3, v(3), 4),
        vop2e(vop2::V_MIN_U32, 4, v(4), 5),
        vop2e(vop2::V_MAX_I32, 5, v(5), 6),
        vop2e(vop2::V_MUL_U32_U24, 6, v(6), 7),
        vop2e(vop2::V_XOR_B32, 7, v(7), 0),
        vop2e(vop2::V_AND_B32, 0, LIT, 1),
        0x00FF_F0F0,
        vop2e(vop2::V_OR_B32, 1, v(1), 2),
        vop1e(vop1::V_NOT_B32, 2, v(2)),
        vop1e(vop1::V_BFREV_B32, 3, v(3)),
        vop1e(vop1::V_FFBL_B32, 4, v(4)),
        vop1e(vop1::V_FFBH_U32, 5, v(0)),
    ];
    for (op, d, s) in [
        (vop3::V_MUL_LO_U32, 6, [v(6), v(1), 0]),
        (vop3::V_LSHL_ADD_U32, 7, [v(7), v(2), v(3)]),
        (vop3::V_ADD3_U32, 0, [v(0), v(4), v(5)]),
        (vop3::V_AND_OR_B32, 1, [v(1), v(6), v(7)]),
        (vop3::V_OR3_B32, 2, [v(2), v(3), int(1)]),
        (vop3::FROM_VOP2 + vop2::V_XOR_B32, 3, [2, v(3), 0]), // s2 ^ v3 in e64 form
    ] {
        code.extend(e64(op, d, s));
    }
    code.push(ENDPGM);
    check_all(&code);
}

#[test]
fn scalar_loop() {
    // s0 = 0; s1 = 5; do { s0 += s1; s1 += -1 } while (s1 != 0); v1 = s0
    let code = [
        sop1e(sop1::S_MOV_B32, 0, int(0)),
        sop1e(sop1::S_MOV_B32, 1, int(5)),
        sop2e(sop2::S_ADD_U32, 0, 0, 1),
        sop2e(sop2::S_ADD_I32, 1, 1, int(-1)),
        sopce(sopc::S_CMP_LG_U32, 1, int(0)),
        soppe(sopp::S_CBRANCH_SCC1, -4),
        vop1e(vop1::V_MOV_B32, 1, 0),
        ENDPGM,
    ];
    check_all(&code);
}

/// if (v0 < v1) { v2 = v0 * 2.0 } else { v2 = 1.0 } with saveexec, the
/// else arm under the inverted mask, and an execz skip of each arm.
fn if_else(size: WaveSize) -> Vec<u32> {
    let w64 = size == WaveSize::W64;
    let saveexec = if w64 { sop1::S_AND_SAVEEXEC_B64 } else { sop1::S_AND_SAVEEXEC_B32 };
    let mov = if w64 { sop1::S_MOV_B64 } else { sop1::S_MOV_B32 };
    vec![
        vopce(vopc::V_CMP_LT_F32, v(0), 1), // 0
        sop1e(saveexec, 4, VCC),            // 1: s[4:5] = exec; exec &= vcc
        soppe(sopp::S_CBRANCH_EXECZ, 1),    // 2: skip the then arm
        vop2e(vop2::V_MUL_F32, 2, 244, 0),  // 3: v2 = 2.0 * v0
        sop1e(mov, EXEC, 4),               // 4: restore exec
        vopce(vopc::V_CMP_GT_F32, v(0), 1), // 5: vcc = v0 > v1
        sop1e(saveexec, 6, VCC),            // 6
        soppe(sopp::S_CBRANCH_EXECZ, 1),    // 7
        vop1e(vop1::V_MOV_B32, 2, ONE_F),  // 8
        sop1e(mov, EXEC, 6),               // 9: restore exec
        ENDPGM,
    ]
}

#[test]
fn divergent_if_else_with_saveexec_and_execz() {
    check_sized(if_else);
}

/// A per-lane loop: while (v3 > 0) { v3 -= 4.0; v4 += 1.0 }, lanes
/// leaving as their condition fails, the wave leaving when exec is empty.
/// v3 starts as the input float of each lane (NaN and negatives never
/// enter; the inputs are small, so trip counts stay bounded).
fn per_lane_loop(size: WaveSize) -> Vec<u32> {
    let w64 = size == WaveSize::W64;
    let saveexec = if w64 { sop1::S_AND_SAVEEXEC_B64 } else { sop1::S_AND_SAVEEXEC_B32 };
    let mov = if w64 { sop1::S_MOV_B64 } else { sop1::S_MOV_B32 };
    vec![
        sop1e(mov, 6, EXEC),                  // 0: save exec
        vopce(vopc::V_CMP_LT_F32, int(0), 3), // 1 loop: vcc = 0.0 < v3
        sop1e(saveexec, 2, VCC),              // 2: exec &= vcc
        soppe(sopp::S_CBRANCH_EXECZ, 3),      // 3: -> 7
        vop2e(vop2::V_ADD_F32, 3, 247, 3),    // 4: v3 += -4.0
        vop2e(vop2::V_ADD_F32, 4, ONE_F, 4),  // 5: v4 += 1.0
        soppe(sopp::S_BRANCH, -6),            // 6: -> 1
        sop1e(mov, EXEC, 6),                  // 7: restore exec
        ENDPGM,
    ]
}

#[test]
fn per_lane_loop_with_exec_shrinking() {
    check_sized(per_lane_loop);
}

/// Every conditional branch kind, with exec and vcc deliberately
/// different (exec emptied while vcc is not) so execz and vccz disagree.
fn branch_kinds(size: WaveSize) -> Vec<u32> {
    let mov = if size == WaveSize::W64 { sop1::S_MOV_B64 } else { sop1::S_MOV_B32 };
    vec![
        sop1e(mov, 6, EXEC),                 // 0: save exec
        vopce(vopc::V_CMP_LT_F32, v(0), 1),  // 1
        soppe(sopp::S_CBRANCH_VCCZ, 1),      // 2: -> 4
        vop1e(vop1::V_MOV_B32, 2, ONE_F),    // 3
        sop1e(mov, EXEC, int(0)),            // 4: exec = 0
        soppe(sopp::S_CBRANCH_EXECZ, 1),     // 5: -> 7
        sop1e(sop1::S_MOV_B32, 5, int(7)),   // 6
        sop1e(mov, EXEC, 6),                 // 7: restore exec
        soppe(sopp::S_CBRANCH_VCCNZ, 1),     // 8: -> 10
        sop1e(sop1::S_MOV_B32, 4, int(9)),   // 9
        sopce(sopc::S_CMP_LT_U32, 0, 1),     // 10
        soppe(sopp::S_CBRANCH_SCC0, 1),      // 11: -> 13
        sop1e(sop1::S_MOV_B32, 3, int(11)),  // 12
        soppe(sopp::S_CBRANCH_EXECNZ, 1),    // 13: -> 15
        sop1e(sop1::S_MOV_B32, 2, int(13)),  // 14
        sopce(sopc::S_CMP_LE_I32, 1, 0),     // 15
        soppe(sopp::S_CBRANCH_SCC1, 1),      // 16: -> 18
        sop1e(sop1::S_MOV_B32, 1, int(15)),  // 17
        ENDPGM,                              // 18
    ]
}

#[test]
fn every_branch_kind() {
    check_sized(branch_kinds);
}

#[test]
fn mbcnt_and_readfirstlane_under_partial_exec() {
    let lo = vop3e(vop3::V_MBCNT_LO_U32_B32, 1, [EXEC, int(0), 0]);
    let hi = vop3e(vop3::V_MBCNT_HI_U32_B32, 1, [EXEC + 1, v(1), 0]);
    let code = [
        lo[0],
        lo[1],
        hi[0],
        hi[1],
        vop1e(vop1::V_READFIRSTLANE_B32, 3, v(0)),
        vop2e(vop2::V_ADD_F32, 2, 3, 0), // v2 = s3 + v0
        sop1e(sop1::S_FF1_I32_B32, 4, EXEC),
        ENDPGM,
    ];
    check_all(&code);
}

/// SMEM: `s_load_dword{,x2,x4}` with an immediate offset, an SGPR
/// offset and m0, then a pointer walk across a 4 GiB boundary (the window
/// starts just below one, so the 64-bit carry matters), feeding VALU.
fn smem_ops(_: WaveSize) -> Vec<u32> {
    // SMEM: w0 = 0xF4000000 | op<<18 | sdata<<6 | sbase/2; w1 = soffset<<25 | offset
    let smeme = |op: u16, sdata: u32, sbase: u32, offset: u32, soffset: u32| {
        [
            0xF400_0000 | (op as u32) << 18 | sdata << 6 | (sbase / 2),
            soffset << 25 | (offset & 0x1F_FFFF),
        ]
    };
    let null = smem::SOFFSET_NULL as u32;
    let mut c = Vec::new();
    // s[2:3] = window base (set below via literals), pointer into the table.
    c.extend([sop1e(sop1::S_MOV_B32, 2, LIT), BASE as u32]);
    c.extend([sop1e(sop1::S_MOV_B32, 3, LIT), (BASE >> 32) as u32]);
    c.extend(smeme(smem::S_LOAD_DWORD, 4, 2, 8, null)); // s4 = mem[2]
    c.extend(smeme(smem::S_LOAD_DWORDX2, 6, 2, 0, 0)); // s[6:7] = mem[s0/4 ..] (s0 = byte offset)
    c.extend([sop1e(sop1::S_MOV_B32, 124, int(16))]); // m0 = 16
    c.extend(smeme(smem::S_LOAD_DWORDX4, 8, 2, 4, 124)); // s[8:11] = mem[5..9]
    // advance the pointer by 0x100 bytes with carry into the high half
    c.extend([sop2e(sop2::S_ADD_U32, 2, 2, LIT), 0x100]);
    c.push(sop2e(sop2::S_ADDC_U32, 3, 3, int(0)));
    c.extend(smeme(smem::S_LOAD_DWORD, 12, 2, 0, null)); // s12 = mem[64]
    c.push(soppe(sopp::S_WAITCNT, 0));
    c.push(vop2e(vop2::V_ADD_F32, 1, 4, 0)); // v1 = s4 + v0
    c.push(vop2e(vop2::V_MUL_F32, 2, 9, 0)); // v2 = s9 * v0
    c.push(vop1e(vop1::V_MOV_B32, 3, 12)); // v3 = s12
    c.push(ENDPGM);
    c
}

/// Just below 2^33, so pointer + 0x100 crosses into the next 4 GiB.
const BASE: u64 = 0x1_FFFF_FFC0;

#[test]
fn smem_loads_through_the_memory_window() {
    let table: Vec<u32> = (0..80u32).map(|k| (k as f32 * 1.5 - 7.0).to_bits()).collect();
    for size in [WaveSize::W32, WaveSize::W64] {
        let code = smem_ops(size);
        for seed in 0..4 {
            for sparse in [false, true] {
                // s0 is the byte offset of the x2 load: keep it in range and
                // dword aligned for this test.
                let mut code = code.clone();
                code.splice(0..0, [sop1e(sop1::S_MOV_B32, 0, int(12))]);
                check_mem(&code, size, seed, sparse, Some((BASE, &table)));
            }
        }
    }
}

/// `s_buffer_load_dword{,x2,x4}` through a raw V# (descriptor in
/// s[8:11]) whose NUM_RECORDS cuts a load in half, plus an SGPR offset
/// from the random inputs (in or out of bounds).
#[test]
fn s_buffer_load_with_bounds() {
    let table: Vec<u32> = (0..80u32).map(|k| 0x1000 + k).collect();
    let null = smem::SOFFSET_NULL as u32;
    let smeme = |op: u16, sdata: u32, sbase: u32, offset: u32, soffset: u32| {
        [
            0xF400_0000 | (op as u32) << 18 | sdata << 6 | (sbase / 2),
            soffset << 25 | (offset & 0x1F_FFFF),
        ]
    };
    let desc_base = BASE + 0x20;
    let mut code = vec![
        sop1e(sop1::S_MOV_B32, 8, LIT),
        desc_base as u32,
        sop1e(sop1::S_MOV_B32, 9, LIT),
        (desc_base >> 32) as u32 & 0xFFFF, // stride 0, no swizzle
        sop1e(sop1::S_MOV_B32, 10, int(24)), // NUM_RECORDS = 24 bytes
        sop1e(sop1::S_MOV_B32, 11, LIT),
        3 << 28, // OOB_SELECT raw
    ];
    code.extend(smeme(smem::S_BUFFER_LOAD_DWORDX4, 12, 8, 16, null)); // 2 in, 2 out
    code.extend(smeme(smem::S_BUFFER_LOAD_DWORDX2, 4, 8, 0, 0)); // at s0 (random)
    code.extend(smeme(smem::S_BUFFER_LOAD_DWORD, 6, 8, 4, 124)); // m0 = 0 -> offset 4
    code.extend(smeme(smem::S_BUFFER_LOAD_DWORDX8, 0, 8, 0, null)); // s[0:7]: 6 in, 2 out
    code.extend(smeme(smem::S_BUFFER_LOAD_DWORDX4, 12, 8, 16, null)); // s12..s15 again, so s[0:7] lands
    code.push(vop1e(vop1::V_MOV_B32, 2, 7)); // out-of-bounds dword -> v2
    code.extend(smeme(smem::S_BUFFER_LOAD_DWORDX16, 0, 8, 4, null)); // s[0:15] at 4: 5 in, 11 out
    code.push(soppe(sopp::S_WAITCNT, 0));
    code.push(vop1e(vop1::V_MOV_B32, 1, 13));
    code.push(ENDPGM);
    for size in [WaveSize::W32, WaveSize::W64] {
        for seed in 0..4 {
            for s0 in [0u32, 8, 20, 22, 24, 0x7FFF_FFF0] {
                let mut c = code.clone();
                c.splice(0..0, [sop1e(sop1::S_MOV_B32, 0, LIT), s0]);
                check_mem(&c, size, seed, false, Some((BASE, &table)));
            }
        }
    }
}

/// GLOBAL loads and stores, both addressing forms, under full and sparse
/// exec: each lane reads its own slot, stores a result to another slot,
/// and the memory afterwards must match too.
#[test]
fn global_loads_and_stores() {
    let table: Vec<u32> = (0..66 * 8u32).map(|k| (k as f32 * 0.25).to_bits()).collect();
    // FLAT: w0 = 0xDC000000 | op<<18 | seg<<14 | offset (12 bits);
    //       w1 = vdst<<24 | saddr<<16 | data<<8 | addr
    let flat = |op: u16, vdst: u32, addr: u32, data: u32, saddr: u32, offset: i32| {
        [
            0xDC00_0000 | (op as u32) << 18 | 2 << 14 | (offset as u32 & 0xFFF),
            vdst << 24 | saddr << 16 | data << 8 | addr,
        ]
    };
    let off = global::SADDR_OFF as u32;
    // v6 = 32 + lane * 32 (byte offset of the lane's slot), v[6:7] its
    // 64-bit address, both computed with every lane on, so inactive lanes
    // hold valid addresses too and a store that ignored exec would land in
    // the window. lane comes from mbcnt.
    let mov = |size| if size == WaveSize::W64 { sop1::S_MOV_B64 } else { sop1::S_MOV_B32 };
    let mut code = vec![];
    code.extend(vop3e(vop3::V_MBCNT_LO_U32_B32, 6, [int(-1), int(0), 0]));
    code.extend(vop3e(vop3::V_MBCNT_HI_U32_B32, 6, [int(-1), v(6), 0]));
    code.push(vop2e(vop2::V_LSHLREV_B32, 6, int(5), 6));
    code.push(vop2e(vop2::V_ADD_NC_U32, 6, int(32), 6));
    code.extend([sop1e(sop1::S_MOV_B32, 2, LIT), GBASE as u32]);
    code.extend([sop1e(sop1::S_MOV_B32, 3, LIT), (GBASE >> 32) as u32]);
    code.push(vop1e(vop1::V_MOV_B32, 4, v(6))); // v4 = slot offset (saddr form)
    code.push(vop2e(vop2::V_ADD_NC_U32, 6, 2, 6)); // v6 = GBASE.lo + slot (no carry, see GBASE)
    code.push(vop1e(vop1::V_MOV_B32, 7, 3)); // v7 = GBASE.hi
    code.push(sop1e(SAVE_OR_RESTORE, EXEC, 8)); // restore the input exec
    code.extend(flat(global::LOAD_DWORDX4, 0, 4, 0, 2, 0)); // v[0:3] = mem[slot .. +16]
    code.extend(flat(global::LOAD_DWORD, 4, 4, 0, 2, 16)); // v4 = mem[slot + 16]
    code.push(vop2e(vop2::V_ADD_F32, 5, v(0), 1));
    code.extend(flat(global::STORE_DWORD, 0, 6, 5, off, 20)); // mem[slot + 20] = v0 + v1
    code.extend(flat(global::STORE_DWORDX2, 0, 6, 2, off, 24)); // mem[slot + 24..] = v2, v3
    code.extend(flat(global::LOAD_DWORD, 3, 6, 0, off, -4)); // v3 = previous slot's last word
    code.push(soppe(sopp::S_WAITCNT, 0));
    code.push(ENDPGM);
    for size in [WaveSize::W32, WaveSize::W64] {
        let mut c = vec![sop1e(mov(size), 8, EXEC), sop1e(mov(size), EXEC, int(-1))];
        c.extend(code.iter().map(|&w| {
            if w == sop1e(SAVE_OR_RESTORE, EXEC, 8) {
                sop1e(mov(size), EXEC, 8)
            } else {
                w
            }
        }));
        for seed in 0..3 {
            for sparse in [false, true] {
                check_mem(&c, size, seed, sparse, Some((GBASE, &table)));
            }
        }
    }
}

/// Low half far from a carry, so `v_add_nc_u32` above stays exact.
/// Placeholder opcode replaced by the wave-size specific s_mov.
const SAVE_OR_RESTORE: u16 = 0x7F;

const GBASE: u64 = 0x3_0000_1000;

/// DS: w0 = 0xD8000000 | op<<18 | offset1<<8 | offset0;
///     w1 = vdst<<24 | data1<<16 | data0<<8 | addr
fn dse(op: u16, vdst: u32, addr: u32, data0: u32, offset0: u32, offset1: u32) -> [u32; 2] {
    [
        0xD800_0000 | (op as u32) << 18 | offset1 << 8 | offset0,
        vdst << 24 | data0 << 8 | addr,
    ]
}

/// LDS: every lane writes its slots, a barrier, then reads its
/// neighbours' (b32, b64, read2). Inactive lanes write nothing, so their
/// slots read 0 on both sides.
#[test]
fn lds_write_barrier_read() {
    assert_eq!(
        dse(ds::WRITE_B32, 0, 0, 1, 0, 0),
        [0xD834_0000, 0x0000_0100],
        "ds_write_b32 v0, v1 from gfx10_asm_ds.s"
    );
    assert_eq!(dse(ds::READ2_B32, 5, 1, 0, 127, 255), [0xD8DC_FF7F, 0x0500_0001]);
    let mut code = vec![];
    code.extend(vop3e(vop3::V_MBCNT_LO_U32_B32, 6, [int(-1), int(0), 0]));
    code.extend(vop3e(vop3::V_MBCNT_HI_U32_B32, 6, [int(-1), v(6), 0]));
    code.push(vop2e(vop2::V_LSHLREV_B32, 7, int(2), 6)); // v7 = lane * 4
    code.push(vop2e(vop2::V_LSHLREV_B32, 6, int(3), 6)); // v6 = lane * 8
    code.extend(dse(ds::WRITE_B32, 0, 7, 0, 0, 0)); // lds[lane] = v0
    code.extend(dse(ds::WRITE_B64, 0, 6, 1, 0, 2)); // lds[128 + 2 lane ..] = v1, v2 (offset 512)
    code.push(soppe(sopp::S_BARRIER, 0));
    code.extend(dse(ds::READ_B32, 3, 7, 0, 4, 0)); // v3 = lds[lane + 1]
    code.extend(dse(ds::READ_B64, 4, 6, 0, 0xF8, 1)); // v[4:5] = previous lane's b64 (offset 504)
    code.extend(dse(ds::READ2_B32, 0, 7, 0, 2, 3)); // v0, v1 = lds[lane + 2], lds[lane + 3]
    code.push(soppe(sopp::S_WAITCNT, 0));
    code.push(ENDPGM);
    check_all(&code);
}

/// Outside the window a load reads 0 (by design of the memory window),
/// including addresses that alias the window in their low 32 bits.
#[test]
fn loads_outside_the_window_read_zero() {
    let table: Vec<u32> = (100..180).collect();
    let null = smem::SOFFSET_NULL as u32;
    let load = |sdata: u32, offset: u32| [0xF400_0000 | sdata << 6 | 1, null << 25 | offset];
    let mut code = Vec::new();
    for (k, addr) in [BASE + 8, BASE + (1 << 32) + 8, BASE - 4, BASE + 4 * 80].iter().enumerate() {
        code.extend([sop1e(sop1::S_MOV_B32, 2, LIT), *addr as u32]);
        code.extend([sop1e(sop1::S_MOV_B32, 3, LIT), (*addr >> 32) as u32]);
        code.extend(load(4 + k as u32, 0));
    }
    code.push(ENDPGM);
    let opts = Options {
        io: Io::RegisterDump,
        num_sgprs: NS,
        num_vgprs: NV,
        lanes: 32,
        memory: Some(Window { base: BASE, words: 80 }),
        cross_lane: CrossLane::Subgroup,
        lds_words: 0,
        code_base: 0,
    };
    let module = translate(&code, &opts).unwrap();
    spirv::validate(&module).unwrap();
    let mut buf = vec![0u32; (NS + SPECIALS) as usize + 32 * NV as usize];
    buf[(NS + SPECIAL_EXEC_LO) as usize] = u32::MAX;
    let mut buffers = Buffers::new();
    buffers.insert((0, 0), buf);
    buffers.insert((0, 1), table);
    run_compute(&module, 32, &mut buffers, 100_000).unwrap();
    let out = &buffers[&(0, 0)];
    assert_eq!(&out[4..8], &[102, 0, 0, 0], "in window, +4 GiB, below, past the end");
}

#[test]
fn refuses_what_it_cannot_translate() {
    let opts = Options {
        io: Io::RegisterDump,
        num_sgprs: NS,
        num_vgprs: NV,
        lanes: 64,
        memory: None,
        cross_lane: CrossLane::Subgroup,
        lds_words: 0,
        code_base: 0,
    };
    let mut fma = vop3e(vop3::V_FMA_F32, 1, [v(0), v(1), v(2)]);
    fma[0] |= 1 << 15; // clamp: not translated
    for code in [
        vec![fma[0], fma[1], ENDPGM],
        vec![sop1e(sop1::S_AND_SAVEEXEC_B32, 2, VCC), ENDPGM], // wrong width for wave64
    ] {
        assert!(
            matches!(translate(&code, &opts), Err(TranslateError::Unsupported { .. })),
            "{}",
            scarlett_gpu::rdna::disassemble(&code)
        );
    }
    let bad = Options { lanes: 16, ..opts };
    assert!(matches!(
        translate(&[ENDPGM], &bad),
        Err(TranslateError::BadOptions(_))
    ));
}


/// The hardware compute contract end to end: user SGPRs (output pointer
/// and a scale) from push constants, the work-group id in the next SGPR,
/// work-item ids x, y in v0, v1, three 8x4 work-groups. Every thread
/// writes `float(index) * scale + in[index]`; memory must match.
#[test]
fn compute_dispatch_contract() {
    let flat = |op: u16, vdst: u32, addr: u32, data: u32, saddr: u32, offset: i32| {
        [
            0xDC00_0000 | (op as u32) << 18 | 2 << 14 | (offset as u32 & 0xFFF),
            vdst << 24 | saddr << 16 | data << 8 | addr,
        ]
    };
    // RSRC2: USER_SGPR = 3, TGID_X_EN, TIDIG_COMP_CNT = 1
    let rsrc2 = 3 << 1 | 1 << 7 | 1 << 11;
    let io = ComputeIo::from_rsrc2(rsrc2, [8, 4, 1]);
    assert_eq!(io.user_sgprs, 3);
    assert_eq!(io.tgid, [true, false, false]);
    assert_eq!(io.tidig_comp_cnt, 1);
    let mut code = vec![
        vop1e(vop1::V_MOV_B32, 3, 3),                // v3 = s3 (work-group id x)
        vop2e(vop2::V_LSHLREV_B32, 3, int(5), 3),    // v3 = wg * 32
        vop2e(vop2::V_LSHLREV_B32, 4, int(3), 1),    // v4 = y * 8
        vop2e(vop2::V_ADD_NC_U32, 4, v(4), 0),       // + x
        vop2e(vop2::V_ADD_NC_U32, 4, v(4), 3),       // + wg * 32 = index
        vop2e(vop2::V_LSHLREV_B32, 5, int(2), 4),    // v5 = index * 4
    ];
    code.extend(flat(global::LOAD_DWORD, 6, 5, 0, 0, 0)); // v6 = in[index]
    code.push(soppe(sopp::S_WAITCNT, 0));
    code.push(vop1e(vop1::V_CVT_F32_U32, 7, v(4)));
    code.push(vop2e(vop2::V_MUL_F32, 7, 2, 7)); // * s2
    code.push(vop2e(vop2::V_ADD_F32, 7, v(6), 7));
    code.extend(flat(global::STORE_DWORD, 0, 5, 7, 0, 1024)); // out[index] at +1 KiB
    code.push(ENDPGM);

    let table: Vec<u32> = (0..512u32).map(|k| (k as f32 * -0.5).to_bits()).collect();
    let user = [GBASE as u32, (GBASE >> 32) as u32, 1.25f32.to_bits()];
    let groups = [3, 1, 1];
    for size in [WaveSize::W32, WaveSize::W64] {
        let mut memory = SparseMemory::new();
        memory.map(GBASE, 4 * table.len() as u64);
        assert!(memory.write(GBASE, &table));
        interp::dispatch_compute(&code, &io, &user, groups, size, NV as usize, 0, &mut memory, 10_000)
            .expect("interpreter");
        let mut want = vec![0u32; table.len()];
        assert!(memory.read(GBASE, &mut want));
        // Sanity: thread (x=3, y=2) of group 1 wrote its value.
        let idx = 32 + 2 * 8 + 3;
        assert_eq!(f32::from_bits(want[256 + idx]), idx as f32 * 1.25 + idx as f32 * -0.5);

        for cross_lane in [CrossLane::Subgroup, CrossLane::Workgroup] {
            let opts = Options {
                io: Io::Compute(io),
                num_sgprs: NS,
                num_vgprs: NV,
                lanes: size.lanes() as u32,
                memory: Some(Window { base: GBASE, words: table.len() as u32 }),
                cross_lane,
                lds_words: 0,
                code_base: 0,
            };
            let module = translate(&code, &opts).expect("translate");
            spirv::validate(&module).expect("valid");
            let mut buffers = Buffers::new();
            let mut window = table.clone();
            window.push(0);
            buffers.insert((0, 1), window);
            run_dispatch(&module, groups, io.workgroup, &mut buffers, &user, 1_000_000)
                .expect("evaluate");
            assert_eq!(&buffers[&(0, 1)][..table.len()], &want[..], "{size:?} {cross_lane:?}");
        }
    }
}

/// A multi-wave work-group: 96 threads (3 waves of 32 or 2 of 64, the last
/// one half full). Each thread writes LDS, `s_barrier`, then reads a slot
/// written by another wave and stores it to memory.
#[test]
fn multi_wave_workgroup_with_s_barrier() {
    let flat = |op: u16, vdst: u32, addr: u32, data: u32, saddr: u32, offset: i32| {
        [
            0xDC00_0000 | (op as u32) << 18 | 2 << 14 | (offset as u32 & 0xFFF),
            vdst << 24 | saddr << 16 | data << 8 | addr,
        ]
    };
    // USER_SGPR = 2 (s[0:1] = output), TGID_X_EN (s2), TG_SIZE_EN (s3).
    let io = ComputeIo::from_rsrc2(2 << 1 | 1 << 7 | 1 << 10, [96, 1, 1]);
    let mut code = vec![
        vop2e(vop2::V_MUL_U32_U24, 1, int(3), 0), // v1 = 3 * id
        vop2e(vop2::V_ADD_NC_U32, 1, 2, 1),       // + group id
        vop2e(vop2::V_ADD_NC_U32, 1, 3, 1),       // + work-group info
        vop2e(vop2::V_LSHLREV_B32, 6, int(2), 0), // v6 = id * 4
        // Only the first wave (first_wavefront set in s3) spins for a while,
        // so the waves reach s_barrier in different dispatcher iterations.
        sopce(sopc::S_CMP_LT_U32, 3, LIT),
        0x8000_0000,
        soppe(sopp::S_CBRANCH_SCC1, 4), // -> skip
        sop1e(sop1::S_MOV_B32, 4, int(5)),
        sop2e(sop2::S_ADD_I32, 4, 4, int(-1)), // loop:
        sopce(sopc::S_CMP_LG_U32, 4, int(0)),
        soppe(sopp::S_CBRANCH_SCC1, -3),
    ];
    // skip:
    code.extend(dse(ds::WRITE_B32, 0, 6, 1, 0, 0)); // lds[id] = v1
    code.push(soppe(sopp::S_BARRIER, 0));
    code.extend([
        vop2e(vop2::V_ADD_NC_U32, 2, int(37), 0), // v2 = id + 37
        vop2e(vop2::V_SUBREV_NC_U32, 3, LIT, 2),  // v3 = v2 - 96
        96,
        vopce(vopc::V_CMP_GT_I32, LIT, 2), // vcc = 96 > v2
        96,
        vop2e(vop2::V_CNDMASK_B32, 2, v(3), 2),   // v2 = (id + 37) % 96
        vop2e(vop2::V_LSHLREV_B32, 2, int(2), 2),
    ]);
    code.extend(dse(ds::READ_B32, 4, 2, 0, 0, 0)); // v4 = lds[(id + 37) % 96]
    code.extend([
        vop1e(vop1::V_MOV_B32, 5, 2),          // v5 = group
        vop2e(vop2::V_MUL_U32_U24, 5, LIT, 5), // * 96
        96,
        vop2e(vop2::V_ADD_NC_U32, 5, v(5), 0),
        vop2e(vop2::V_LSHLREV_B32, 5, int(2), 5),
    ]);
    code.extend(flat(global::STORE_DWORD, 0, 5, 4, 0, 0));
    code.push(ENDPGM);

    let words = 2 * 96;
    let user = [GBASE as u32, (GBASE >> 32) as u32];
    let groups = [2, 1, 1];
    for size in [WaveSize::W32, WaveSize::W64] {
        let mut memory = SparseMemory::new();
        memory.map(GBASE, 4 * words as u64);
        interp::dispatch_compute(&code, &io, &user, groups, size, NV as usize, 96, &mut memory, 100_000)
            .expect("interpreter");
        let mut want = vec![0u32; words];
        assert!(memory.read(GBASE, &mut want));
        // By hand: thread 70 of group 1 reads slot 11 = 3*11 + 1 + info.
        let waves = 96usize.div_ceil(size.lanes()) as u32;
        assert_eq!(want[96 + 70], 3 * 11 + 1 + ComputeIo::workgroup_info(0, waves));
        // thread 0 reads slot 37, written by a later wave.
        let w37 = (37 / size.lanes()) as u32;
        assert_eq!(want[0], 3 * 37 + ComputeIo::workgroup_info(w37, waves));

        let opts = Options {
            io: Io::Compute(io),
            num_sgprs: NS,
            num_vgprs: NV,
            lanes: size.lanes() as u32,
            memory: Some(Window { base: GBASE, words: words as u32 }),
            cross_lane: CrossLane::Subgroup,
            lds_words: 96,
            code_base: 0,
        };
        let module = translate(&code, &opts).expect("translate");
        spirv::validate(&module).expect("valid");
        let mut buffers = Buffers::new();
        buffers.insert((0, 1), vec![0; words + 1]);
        run_dispatch_subgroups(&module, groups, io.workgroup, size.lanes() as u32, &mut buffers, &user, 10_000_000)
            .expect("evaluate");
        assert_eq!(&buffers[&(0, 1)][..words], &want[..], "{size:?}");
        // Workgroup emulation cannot span waves: refused.
        let bad = Options { cross_lane: CrossLane::Workgroup, ..opts };
        assert!(matches!(translate(&code, &bad), Err(TranslateError::BadOptions(_))));
    }
}

/// DPP16 word: src0 | ctrl << 8 | bound_ctrl << 19 | mods | bank << 24 | row << 28
fn dppw(src0: u32, ctrl: u32, bound: bool, mods: u32, bank: u32, row: u32) -> u32 {
    src0 | ctrl << 8 | (bound as u32) << 19 | mods << 20 | bank << 24 | row << 28
}

/// Every DPP16 rule on VOP1 and VOP2, with and without bound_ctrl, row
/// and bank masks, source modifiers, under full and sparse exec.
#[test]
fn dpp16_cross_lane() {
    const DPP: u16 = 250;
    let mut code = vec![];
    let ctrls = [
        0x1B,  // quad_perm:[3,2,1,0]
        0x101, // row_shl:1
        0x10F, // row_shl:15
        0x113, // row_shr:3
        0x125, // row_ror:5
        0x140, // row_mirror
        0x141, // row_half_mirror
        0x157, // row_share:7
        0x16A, // row_xmask:10
    ];
    for (k, ctrl) in ctrls.into_iter().enumerate() {
        let bound = k % 2 == 0;
        let (bank, row) = if k == 3 { (0b1011, 0b0110) } else { (0xF, 0xF) };
        // v2 = dpp(v0); v3 = dpp(v1) + v0 (float) with -src0 and |src1|
        code.push(vop1e(vop1::V_MOV_B32, 2, DPP));
        code.push(dppw(0, ctrl, bound, 0, bank, row));
        code.push(vop2e(vop2::V_ADD_F32, 3, DPP, 0));
        code.push(dppw(1, ctrl, bound, 0b1001, bank, row)); // src0 neg, src1 abs
        code.push(vop2e(vop2::V_ADD_NC_U32, 4, DPP, 4));
        code.push(dppw(4, ctrl, !bound, 0, 0xF, 0xF));
        // fold into v5..v7 so every step is observable
        code.push(vop2e(vop2::V_XOR_B32, 5, v(2), 5));
        code.push(vop2e(vop2::V_ADD_NC_U32, 6, v(3), 6));
        code.push(vop2e(vop2::V_XOR_B32, 7, v(4), 7));
    }
    code.push(ENDPGM);
    check_all(&code);
}

/// MIMG: w0 = 0xF0000000 | op<<18 | dmask<<8 | dim<<3; w1 = srsrc/4<<16 | vdata<<8 | vaddr
fn mimge(dmask: u32, vdata: u32, vaddr: u32, srsrc: u32) -> [u32; 2] {
    [0xF000_0000 | dmask << 8 | 1 << 3, (srsrc / 4) << 16 | vdata << 8 | vaddr]
}

/// A GFX10 T# for a linear 2D image.
fn tdesc(base: u64, fmt: u32, w: u32, h: u32, sel: [u32; 4]) -> [u32; 8] {
    [
        (base >> 8) as u32,
        ((base >> 40) & 0xFF) as u32 | fmt << 20 | ((w - 1) & 3) << 30,
        (w - 1) >> 2 | (h - 1) << 14 | 1 << 31,
        sel[0] | sel[1] << 3 | sel[2] << 6 | sel[3] << 9 | 9 << 28,
        0,
        0,
        0,
        0,
    ]
}

#[test]
fn image_load_linear_2d() {
    assert_eq!(
        mimge(1, 0, 2, 0),
        [0xF000_0108, 0x0000_0002],
        "image_load v0, v[2:3], s[0:7] dmask:0x1 dim:2D from gfx10_asm_mimg.s"
    );
    let words = 1024usize;
    let mut table = vec![0u32; words];
    for (k, t) in table.iter_mut().enumerate() {
        *t = (k as u32).wrapping_mul(2_654_435_761) ^ 0x3F80_0000;
    }
    let (x, y, z, w) = (4, 5, 6, 7);
    let images = [
        (tdesc(IBASE, 22, 37, 5, [x, 0, 1, w]), 0b1011),        // 32_FLOAT: r, 0, 1.0, 1.0
        (tdesc(IBASE + 1280, 62, 20, 3, [y, x, 0, 0]), 0b0011), // 32_32_UINT: g, r
        (tdesc(IBASE + 2048, 77, 9, 4, [w, z, y, x]), 0b1111),  // RGBA float, reversed
        (tdesc(IBASE + 3072, 56, 13, 2, [z, y, x, w]), 0b1111), // 8_8_8_8_UNORM, BGRA
    ];
    let mut code = vec![];
    code.extend(vop3e(vop3::V_MBCNT_LO_U32_B32, 0, [int(-1), int(0), 0]));
    code.extend(vop3e(vop3::V_MBCNT_HI_U32_B32, 0, [int(-1), v(0), 0]));
    code.push(vop2e(vop2::V_LSHRREV_B32, 1, int(3), 0)); // v1 = lane >> 3
    code.push(vop2e(vop2::V_AND_B32, 1, int(1), 1)); // y = v1 & 1
    code.push(vop2e(vop2::V_AND_B32, 0, int(7), 0)); // x = lane & 7
    for (desc, dmask) in images {
        for (k, dw) in desc.iter().enumerate() {
            code.extend([sop1e(sop1::S_MOV_B32, 8 + k as u16, LIT), *dw]);
        }
        code.extend(mimge(dmask, 2, 0, 8)); // v[2:5] = image_load(v[0:1])
        code.push(soppe(sopp::S_WAITCNT, 0));
        for r in 2..2 + dmask.count_ones() as u16 {
            code.push(vop2e(vop2::V_XOR_B32, 6, v(r), 6));
            code.push(vop2e(vop2::V_ADD_NC_U32, 7, v(r), 7));
        }
    }
    code.push(ENDPGM);
    for size in [WaveSize::W32, WaveSize::W64] {
        for seed in 0..2 {
            for sparse in [false, true] {
                check_mem(&code, size, seed, sparse, Some((IBASE, &table)));
            }
        }
    }
}

/// Above 2^40, so the T# BASE_ADDRESS_HI field is exercised.
const IBASE: u64 = 0x123_4567_8900;

/// image_store then image_load back, every lane its own texel; memory and
/// registers must match.
#[test]
fn image_store_linear_2d() {
    let st = |dmask: u32, vdata: u32, vaddr: u32, srsrc: u32| {
        let [w0, w1] = mimge(dmask, vdata, vaddr, srsrc);
        [w0 | (8 << 18), w1] // IMAGE_STORE
    };
    assert_eq!(
        st(0xF, 0, 0xFE, 96),
        [0xF020_0F08, 0x0018_00FE],
        "image_store v[0:3], v[254:255], s[96:103] dmask:0xf dim:2D from gfx10_asm_mimg.s"
    );
    let words = 1024usize;
    let table = vec![0u32; words];
    let images = [
        (tdesc(IBASE, 22, 37, 8, [4, 0, 0, 1]), 0b0001), // 32_FLOAT: r
        (tdesc(IBASE + 2048, 77, 9, 8, [4, 5, 6, 7]), 0b1111), // RGBA float
    ];
    let mut code = vec![];
    code.extend(vop3e(vop3::V_MBCNT_LO_U32_B32, 0, [int(-1), int(0), 0]));
    code.extend(vop3e(vop3::V_MBCNT_HI_U32_B32, 0, [int(-1), v(0), 0]));
    code.push(vop2e(vop2::V_LSHRREV_B32, 1, int(3), 0)); // y = lane >> 3
    code.push(vop2e(vop2::V_AND_B32, 0, int(7), 0)); // x = lane & 7
    code.push(vop1e(vop1::V_CVT_F32_U32, 2, v(1)));
    code.push(vop2e(vop2::V_ADD_F32, 3, ONE_F, 2));
    code.push(vop2e(vop2::V_MUL_F32, 4, 244, 3));
    code.push(vop2e(vop2::V_SUB_F32, 5, v(4), 2));
    for (desc, dmask) in images {
        for (k, dw) in desc.iter().enumerate() {
            code.extend([sop1e(sop1::S_MOV_B32, 8 + k as u16, LIT), *dw]);
        }
        code.extend(st(dmask, 2, 0, 8)); // store v[2:5]
        code.push(soppe(sopp::S_WAITCNT, 0));
        code.extend(mimge(dmask, 4, 0, 8)); // load back into v4..
        code.push(soppe(sopp::S_WAITCNT, 0));
    }
    code.push(ENDPGM);
    for size in [WaveSize::W32, WaveSize::W64] {
        for seed in 0..2 {
            for sparse in [false, true] {
                check_mem(&code, size, seed, sparse, Some((IBASE, &table)));
            }
        }
    }
}

/// The SALU batch on random SGPRs: logic, shifts, bit fields, compares,
/// selects, carries, 64-bit ops, then into VGPRs so lanes see them.
#[test]
fn salu_batch() {
    let mut code = vec![
        sop2e(sop2::S_OR_B32, 8, 0, 1),
        sop2e(sop2::S_XOR_B32, 9, 1, 2),
        sop2e(sop2::S_ANDN2_B32, 10, 2, 3),
        sop2e(sop2::S_ORN2_B32, 11, 3, 4),
        sop2e(sop2::S_NAND_B32, 12, 4, 5),
        sop2e(sop2::S_NOR_B32, 13, 5, 6),
        sop1e(sop1::S_CMOV_B32, 14, 7), // scc from the nor
        sop2e(sop2::S_XNOR_B32, 15, 6, 7),
        sop2e(sop2::S_LSHL_B32, 8, 8, 0),
        sop2e(sop2::S_LSHR_B32, 9, 9, 1),
        sop2e(sop2::S_ASHR_I32, 10, 10, 2),
        sop2e(sop2::S_MUL_I32, 11, 11, 3),
        sop2e(sop2::S_BFM_B32, 12, 0, 1),
        sop2e(sop2::S_BFE_U32, 13, 13, LIT),
        5 | 11 << 16,
        sop2e(sop2::S_BFE_I32, 14, 14, LIT),
        3 | 7 << 16,
        sop2e(sop2::S_SUB_I32, 15, 15, 0),
        sopce(sopc::S_CMP_EQ_U32, 0, 0), // scc = 1: borrow in
        sop2e(sop2::S_SUBB_U32, 15, 15, 1), // s15 survives to the end
        sop2e(sop2::S_OR_B64, 0, 0, 2),
        sop2e(sop2::S_XOR_B64, 2, 2, 4),
        sop2e(sop2::S_ANDN2_B64, 4, 4, 6),
        sop2e(sop2::S_CSELECT_B64, 6, 8, 10),
        sopce(sopc::S_CMP_EQ_U64, 0, 2),
        sop1e(sop1::S_CMOV_B32, 1, 9),
        sopce(sopc::S_CMP_GT_I32, 3, 5),
        sop2e(sop2::S_CSELECT_B32, 3, 11, 12),
        sopce(sopc::S_CMP_GE_U32, 4, 6),
        sop1e(sop1::S_NOT_B32, 5, 13),
        sop1e(sop1::S_NOT_B64, 6, 6),
        sop1e(sop1::S_BREV_B32, 7, 7),
        sop1e(sop1::S_BCNT1_I32_B32, 8, 8),
        sop1e(sop1::S_BCNT0_I32_B32, 9, 9),
        sop1e(sop1::S_FLBIT_I32_B32, 10, 10),
        sop1e(sop1::S_SEXT_I32_I8, 11, 11),
        sop1e(sop1::S_SEXT_I32_I16, 12, 12),
        sopce(sopc::S_CMP_LG_U64, 4, 6),
        sopce(sopc::S_CMP_EQ_I32, 13, 14),
    ];
    for r in 0..8u16 {
        code.push(vop2e(vop2::V_XOR_B32, r % 8, 8 + r, r % 8)); // vN ^= s(8+N)
    }
    code.push(ENDPGM);
    check_all(&code);
}

/// or/andn2 saveexec, then exec restored.
fn saveexec_variants(size: WaveSize) -> Vec<u32> {
    let w64 = size == WaveSize::W64;
    let (or_se, andn2_se, mov) = if w64 {
        (sop1::S_OR_SAVEEXEC_B64, sop1::S_ANDN2_SAVEEXEC_B64, sop1::S_MOV_B64)
    } else {
        (sop1::S_OR_SAVEEXEC_B32, sop1::S_ANDN2_SAVEEXEC_B32, sop1::S_MOV_B32)
    };
    vec![
        vopce(vopc::V_CMP_LT_F32, v(0), 1),
        sop1e(or_se, 8, VCC),      // exec |= vcc
        vop1e(vop1::V_MOV_B32, 2, ONE_F),
        sop1e(mov, EXEC, 8),       // restore
        sop1e(andn2_se, 10, VCC),  // exec = vcc & !exec
        vop1e(vop1::V_MOV_B32, 3, int(7)),
        sop1e(mov, EXEC, 10),      // restore
        ENDPGM,
    ]
}

#[test]
fn or_and_andn2_saveexec() {
    check_sized(saveexec_variants);
}

/// Every VOPC condition (f32, i32, u32) folded bit by bit into v4, the
/// VOP3 form into an SGPR mask with -|src0|, and v_cmpx shrinking exec.
fn all_compares(size: WaveSize) -> Vec<u32> {
    let w64 = size == WaveSize::W64;
    let mov = if w64 { sop1::S_MOV_B64 } else { sop1::S_MOV_B32 };
    let mut c = vec![];
    let ops: Vec<u16> = (0..16).chain(128..136).chain(192..200).collect();
    for (k, &opc) in ops.iter().enumerate() {
        let (a, b) = if k % 3 == 0 { (v(1), 0) } else { (v(0), 1) };
        c.push(vopce(opc, a, b));
        c.push(vop2e(vop2::V_CNDMASK_B32, 2, int(0), 7)); // v2 = vcc ? v7 : 0
        c.push(vop2e(vop2::V_LSHLREV_B32, 4, int(1), 4));
        c.push(vop2e(vop2::V_XOR_B32, 4, v(2), 4));
    }
    // e64: s[8:9] = -|v0| < v1 ; v5 = mask ? 1 : 0
    let mut e64 = vop3e(1, 8, [v(0), v(1), 0]);
    e64[0] |= 1 << 8; // abs src0
    e64[1] |= 1 << 29; // neg src0
    c.extend(e64);
    c.extend(vop3e(vop3::FROM_VOP2 + vop2::V_CNDMASK_B32, 5, [int(0), int(1), 8]));
    // cmpx: exec &= (v0 > v1 unsigned), v6 = 5 there, restore
    c.push(sop1e(mov, 10, EXEC));
    c.push(vopce(212, v(0), 1)); // v_cmpx_gt_u32
    c.push(vop1e(vop1::V_MOV_B32, 6, int(5)));
    c.push(sop1e(mov, EXEC, 10));
    c.push(ENDPGM);
    c
}

#[test]
fn every_compare_condition() {
    check_sized(all_compares);
}

#[test]
fn rounding_and_ubyte() {
    let mut code = vec![];
    for (k, opc) in [
        vop1::V_FLOOR_F32,
        vop1::V_CEIL_F32,
        vop1::V_TRUNC_F32,
        vop1::V_RNDNE_F32,
        vop1::V_CVT_F32_UBYTE0,
        vop1::V_CVT_F32_UBYTE1,
        vop1::V_CVT_F32_UBYTE2,
        vop1::V_CVT_F32_UBYTE3,
    ]
    .into_iter()
    .enumerate()
    {
        // scale v0 so halves and fractions show up: v2 = v0 * 0.5 + k
        code.push(vop2e(vop2::V_MUL_F32, 2, 240, 0)); // 0.5 * v0
        code.push(vop2e(vop2::V_ADD_F32, 2, int(k as i32), 2));
        code.push(vop1e(opc, 3, v(if k < 4 { 2 } else { 1 })));
        code.push(vop2e(vop2::V_XOR_B32, 4, v(3), 4));
        code.push(vop2e(vop2::V_ADD_NC_U32, 5, v(3), 5));
    }
    code.push(ENDPGM);
    check_all(&code);
}

#[test]
fn min_max_fma_fmac() {
    let mut code = vec![];
    code.extend(vop3e(vop3::V_MBCNT_LO_U32_B32, 6, [int(-1), int(0), 0]));
    code.extend(vop3e(vop3::V_MBCNT_HI_U32_B32, 6, [int(-1), v(6), 0]));
    code.push(vop1e(vop1::V_CVT_F32_U32, 6, v(6)));
    code.push(vop2e(vop2::V_ADD_F32, 6, 240, 6)); // v6 = lane + 0.5, never NaN or 0
    code.push(vop2e(vop2::V_MIN_F32, 2, v(0), 6));
    code.push(vop2e(vop2::V_MAX_F32, 3, v(6), 0));
    let mut e = vop3e(vop3::FROM_VOP2 + vop2::V_MIN_F32, 4, [v(6), v(0), 0]);
    e[1] |= 2 << 29; // -src1
    code.extend(e);
    code.push(vop1e(vop1::V_MOV_B32, 5, int(0))); // +0
    code.push(vop2e(vop2::V_MIN_F32, 5, LIT, 5)); // min(-0, +0) = -0
    code.push(0x8000_0000);
    code.extend(vop3e(vop3::V_FMA_F32, 7, [v(0), v(1), v(6)]));
    code.push(vop2e(vop2::V_FMAC_F32, 7, v(6), 0)); // v7 += v6 * v0
    let mut m = vop3e(vop3::FROM_VOP2 + vop2::V_MAX_F32, 1, [v(1), v(6), 0]);
    m[0] |= 1 << 8; // |src0|
    code.extend(m);
    code.push(ENDPGM);
    check_all(&code);
}

/// EXP: w0 = 0xF8000000 | done<<11 | target<<4 | en; w1 = v3<<24|v2<<16|v1<<8|v0
fn expe(target: u32, en: u32, v: [u32; 4], done: bool) -> [u32; 2] {
    [
        0xF800_0000 | (done as u32) << 11 | target << 4 | en,
        v[3] << 24 | v[2] << 16 | v[1] << 8 | v[0],
    ]
}

#[test]
fn pixel_shader_position_inputs_and_exports() {
    assert_eq!(expe(0, 0xF, [0; 4], true), [0xF800_080F, 0], "exp mrt0, v0, v0, v0, v0 done");
    use scarlett_gpu::interp::dispatch_pixels;
    use scarlett_gpu::rdna::ps_input;
    use scarlett_gpu::spirv_eval::run_fragment;
    use scarlett_gpu::translate::PixelIo;
    let pos = 0xF << ps_input::POS_X_FLOAT;
    let (ena, addr) = (pos, pos | 1 << 1); // PERSP_CENTER allocated, not loaded
    assert_eq!(ps_input::vgpr(addr, ps_input::POS_X_FLOAT), Some(2));
    let mut code = vec![
        vop2e(vop2::V_MUL_F32, 6, 240, 2),    // v6 = x * 0.5
        vop2e(vop2::V_ADD_F32, 7, 0, 3),      // v7 = s0 + y
        vop2e(vop2::V_SUB_F32, 1, v(4), 5),   // v1 = z - 1/w
        vop1e(vop1::V_FLOOR_F32, 0, v(2)),    // v0 = floor(x)
    ];
    // Two masks, so some pixels export only mrt1 and some only mrt0:
    // mrt1 where 3.0 > x, mrt0 where 5.0 < x; pixels with neither are
    // discarded.
    // s_mov_b64 of exec works for both wave sizes (wave32 also copies the
    // unused exec_hi).
    let w64 = (sop1::S_MOV_B64,);
    code.extend([sop1e(w64.0, 10, EXEC), vopce(20, LIT, 2), 3.0f32.to_bits()]); // v_cmpx_gt_f32
    code.extend(expe(1, 0xF, [2, 3, 4, 5], false)); // mrt1 = raw position
    code.extend([sop1e(w64.0, EXEC, 10), vopce(17, LIT, 2), 5.0f32.to_bits()]); // v_cmpx_lt_f32: 5.0 < x
    code.extend(expe(0, 0xF, [6, 7, 1, 0], true));
    code.push(ENDPGM);
    let user = [1.5f32.to_bits()];
    for size in [WaveSize::W32, WaveSize::W64] {
        let n = size.lanes() - 3;
        let fc: Vec<[f32; 4]> = (0..n)
            .map(|i| [i as f32 * 0.37 + 0.5, i as f32 * 2.0 + 0.5, 0.25, 1.0 / (i as f32 + 1.0)])
            .collect();
        let w = dispatch_pixels(&code, ena, addr, &user, &fc, size, NV as usize, 1000).expect("interpreter");
        let opts = Options {
            io: Io::Pixel(PixelIo { ena, addr, user_sgprs: 1 }),
            num_sgprs: NS,
            num_vgprs: NV,
            lanes: size.lanes() as u32,
            memory: None,
            cross_lane: CrossLane::Subgroup,
            lds_words: 0,
            code_base: 0,
        };
        let module = translate(&code, &opts).expect("translate");
        spirv::validate(&module).expect("valid");
        let out = run_fragment(&module, &fc, &mut Buffers::new(), &user, 100_000).expect("evaluate");
        let mut kept = 0;
        for (lane, o) in out.iter().enumerate() {
            let exports: Vec<_> = w.exports.iter().filter(|e| e.lane == lane).collect();
            match o {
                None => assert!(exports.is_empty(), "lane {lane} discarded but exported"),
                Some(locs) => {
                    kept += 1;
                    assert!(!exports.is_empty(), "lane {lane} kept without exports");
                    for t in [0u8, 1] {
                        // an MRT this pixel never exported stays 0 (the
                        // translator's convention; hardware leaves it alone)
                        let want = exports
                            .iter()
                            .rev()
                            .find(|e| e.target == t)
                            .map_or(vec![0; 4], |e| e.values.to_vec());
                        assert_eq!(locs[&(t as u32)], want, "lane {lane} mrt{t} ({size:?})");
                    }
                }
            }
        }
        let x = |i: usize| i as f32 * 0.37 + 0.5;
        assert_eq!(kept, (0..n).filter(|&i| x(i) < 3.0 || x(i) > 5.0).count());
        assert!(kept < n, "some pixels are discarded");
    }
}

/// VINTRP (LLVM VINTRPe): 0x32 << 26 | vdst << 18 | op << 16 | attr << 10 | chan << 8 | vsrc
fn vintrpe(op: u32, vdst: u32, vsrc: u32, attr: u32, chan: u32) -> u32 {
    0x32 << 26 | vdst << 18 | op << 16 | attr << 10 | chan << 8 | vsrc
}

#[test]
fn pixel_attribute_interpolation() {
    use scarlett_gpu::interp::dispatch_pixels_interp;
    use scarlett_gpu::rdna::{ps_input, vintrp};
    use scarlett_gpu::spirv_eval::{run_fragment_with, FragmentInput};
    use scarlett_gpu::translate::PixelIo;
    let ena = 1 << ps_input::PERSP_CENTER | 1 << ps_input::POS_X_FLOAT;
    let addr = ena;
    assert_eq!(ps_input::vgpr(addr, ps_input::POS_X_FLOAT), Some(2));
    let code = [
        vintrpe(0, 6, 0, 0, 0), // v_interp_p1_f32 v6, v0, attr0.x
        vintrpe(1, 6, 1, 0, 0), // v_interp_p2_f32 v6, v1, attr0.x
        vintrpe(0, 7, 0, 1, 3), // attr1.w
        vintrpe(1, 7, 1, 1, 3),
        vintrpe(2, 4, vintrp::SLOT_P0 as u32, 1, 1), // v_interp_mov_f32 v4, p0, attr1.y
        vintrpe(2, 5, vintrp::SLOT_P20 as u32, 0, 2), // v_interp_mov_f32 v5, p20, attr0.z
        expe(0, 0xF, [6, 7, 4, 5], true)[0],
        expe(0, 0xF, [6, 7, 4, 5], true)[1],
        ENDPGM,
    ];
    let attrs = [
        [[1.0, 2.0, 3.0, 4.0], [5.5, -6.0, 7.25, 8.0], [-9.0, 10.0, 0.125, 12.0]],
        [[0.3, 0.7, -1.1, 2.9], [4.4, 0.0, 6.6, -7.7], [8.8, 9.9, 1.25, 1e-3]],
    ];
    for size in [WaveSize::W32, WaveSize::W64] {
        let n = size.lanes();
        let ij: Vec<[f32; 2]> = (0..n)
            .map(|k| {
                let i = (k as f32 * 0.618_034).fract() * 0.6;
                let j = (k as f32 * 0.414_214).fract() * 0.4;
                [i, j]
            })
            .collect();
        let bary: Vec<[f32; 3]> = ij.iter().map(|&[i, j]| [1.0 - i - j, i, j]).collect();
        let fc: Vec<[f32; 4]> = (0..n).map(|k| [k as f32 + 0.5, 0.5, 0.0, 1.0]).collect();
        let w = dispatch_pixels_interp(&code, ena, addr, &[], &fc, &ij, &[], &attrs, size, NV as usize, 100)
            .expect("interpreter");
        // By hand for lane 0: attr0.x = 1 + i*(5.5-1) + j*(-9-1) (fused steps)
        let (i, j) = (ij[0][0], ij[0][1]);
        let p1 = 4.5f32.mul_add(i, 1.0);
        assert_eq!(w.exports[0].values[0], (-10.0f32).mul_add(j, p1).to_bits());
        assert_eq!(w.exports[0].values[2], 0.7f32.to_bits(), "mov p0 attr1.y");
        assert_eq!(w.exports[0].values[3], (0.125f32 - 3.0).to_bits(), "mov p20 attr0.z");
        let opts = Options {
            io: Io::Pixel(PixelIo { ena, addr, user_sgprs: 0 }),
            num_sgprs: NS,
            num_vgprs: NV,
            lanes: n as u32,
            memory: None,
            cross_lane: CrossLane::Subgroup,
            lds_words: 0,
            code_base: 0,
        };
        let module = translate(&code, &opts).expect("translate");
        spirv::validate(&module).expect("valid");
        let la: Vec<(u32, [[f32; 4]; 3])> = attrs.iter().enumerate().map(|(k, a)| (k as u32, *a)).collect();
        let input = FragmentInput { frag_coord: &fc, bary: &bary, front: &[], attrs: &la };
        let out = run_fragment_with(&module, &input, &mut Buffers::new(), &[], 100_000).expect("evaluate");
        for (lane, o) in out.iter().enumerate() {
            let e = w.exports.iter().find(|e| e.lane == lane).expect("exported");
            assert_eq!(o.as_ref().expect("kept")[&0], e.values.to_vec(), "lane {lane} ({size:?})");
        }
    }
}

/// FRONT_FACE (+1.0 / -1.0) next to POS_X_FLOAT, with PERSP_CENTER
/// allocated in ADDR but not enabled (so FRONT_FACE sits at v3): the face
/// value, its product with x, and a facing-dependent branch-free select.
#[test]
fn pixel_front_face() {
    use scarlett_gpu::interp::dispatch_pixels_interp;
    use scarlett_gpu::rdna::ps_input;
    use scarlett_gpu::spirv_eval::{run_fragment_with, FragmentInput};
    use scarlett_gpu::translate::PixelIo;
    let ena = 1 << ps_input::FRONT_FACE | 1 << ps_input::POS_X_FLOAT;
    let addr = ena | 1 << ps_input::PERSP_CENTER;
    assert_eq!(ps_input::vgpr(addr, ps_input::POS_X_FLOAT), Some(2));
    assert_eq!(ps_input::vgpr(addr, ps_input::FRONT_FACE), Some(3));
    let mut code = vec![
        vop2e(vop2::V_MUL_F32, 4, v(3), 2), // v4 = face * x
        vop2e(vop2::V_MAX_F32, 5, int(0), 3), // v5 = max(0, face): 1 front, 0 back
    ];
    code.extend(expe(0, 0xF, [3, 4, 5, 2], true));
    code.push(ENDPGM);
    for size in [WaveSize::W32, WaveSize::W64] {
        let n = size.lanes() - 3;
        let fc: Vec<[f32; 4]> = (0..n).map(|k| [k as f32 * 0.75 + 0.5, 0.5, 0.0, 1.0]).collect();
        let front: Vec<bool> = (0..n).map(|k| k % 3 != 1).collect();
        let w = dispatch_pixels_interp(&code, ena, addr, &[], &fc, &[], &front, &[], size, NV as usize, 100)
            .expect("interpreter");
        let e1 = w.exports.iter().find(|e| e.lane == 1).unwrap();
        assert_eq!(e1.values[..3], [(-1.0f32).to_bits(), (-1.25f32).to_bits(), 0], "back face by hand");
        let opts = Options {
            io: Io::Pixel(PixelIo { ena, addr, user_sgprs: 0 }),
            num_sgprs: NS,
            num_vgprs: NV,
            lanes: size.lanes() as u32,
            memory: None,
            cross_lane: CrossLane::Subgroup,
            lds_words: 0,
            code_base: 0,
        };
        let module = translate(&code, &opts).expect("translate");
        spirv::validate(&module).expect("valid");
        let input = FragmentInput { frag_coord: &fc, bary: &[], front: &front, attrs: &[] };
        let out = run_fragment_with(&module, &input, &mut Buffers::new(), &[], 100_000).expect("evaluate");
        assert_eq!(out.len(), n);
        for (lane, o) in out.iter().enumerate() {
            let e = w.exports.iter().find(|e| e.lane == lane).expect("exported");
            assert_eq!(o.as_ref().expect("kept")[&0], e.values.to_vec(), "lane {lane} ({size:?})");
        }
    }
}

#[test]
fn vertex_shader_inputs_and_exports() {
    use scarlett_gpu::interp::{dispatch_vertices, VertexIn};
    use scarlett_gpu::rdna::{exp, vs_input};
    use scarlett_gpu::spirv_eval::{run_vertex, VertexInput, POSITION_KEY};
    use scarlett_gpu::translate::VertexIo;
    let rsrc1 = 3 << 24; // VGPR_COMP_CNT = 3: v0..v3
    assert_eq!(vs_input::comp_cnt(rsrc1), 3);
    let mut code = vec![
        vop1e(vop1::V_CVT_F32_U32, 4, v(0)), // vertex id
        vop1e(vop1::V_CVT_F32_U32, 5, v(3)), // instance id
        vop2e(vop2::V_MUL_F32, 6, 0, 4),      // s0 * vertex id
        vop2e(vop2::V_ADD_F32, 7, v(5), 6),
    ];
    code.extend(expe(exp::POS0 as u32, 0xF, [6, 7, 4, 5], true));
    code.extend(expe(exp::PARAM0 as u32, 0xF, [4, 5, 6, 7], false));
    code.extend(expe(exp::PARAM0 as u32 + 2, 0xF, [7, 6, 5, 4], false));
    code.push(ENDPGM);
    let user = [0.75f32.to_bits()];
    for size in [WaveSize::W32, WaveSize::W64] {
        let n = size.lanes() - 5;
        let verts: Vec<VertexIn> =
            (0..n as u32).map(|k| VertexIn { vertex_id: k * 3 + 1, prim_id: 0, instance_id: 2 }).collect();
        let w = dispatch_vertices(&code, 3, &user, &verts, size, NV as usize, 100).expect("interpreter");
        let opts = Options {
            io: Io::Vertex(VertexIo { user_sgprs: 1, vgpr_comp_cnt: 3 }),
            num_sgprs: NS,
            num_vgprs: NV,
            lanes: size.lanes() as u32,
            memory: None,
            cross_lane: CrossLane::Subgroup,
            lds_words: 0,
            code_base: 0,
        };
        let module = translate(&code, &opts).expect("translate");
        spirv::validate(&module).expect("valid");
        // Vulkan's indices include the base vertex / instance; the
        // hardware's are zero-based.
        let vin: Vec<VertexInput> = verts
            .iter()
            .map(|v| VertexInput {
                vertex_index: v.vertex_id + 100,
                base_vertex: 100,
                instance_index: v.instance_id + 7,
                base_instance: 7,
            })
            .collect();
        let out = run_vertex(&module, &vin, &mut Buffers::new(), &user, 100_000).expect("evaluate");
        for (lane, o) in out.iter().enumerate() {
            for (target, key) in [(exp::POS0, POSITION_KEY), (exp::PARAM0, 0), (exp::PARAM0 + 2, 2)] {
                let e = w.exports.iter().find(|e| e.lane == lane && e.target == target).expect("exported");
                assert_eq!(o[&key], e.values.to_vec(), "lane {lane} target {target} ({size:?})");
            }
        }
        // by hand: lane 2 position.x = 0.75 * 7
        assert_eq!(f32::from_bits(out[2][&POSITION_KEY][0]), 0.75 * 7.0);
    }
}

/// MUBUF: w0 = 0xE0000000 | op<<18 | glc<<14 | idxen<<13 | offen<<12 |
/// offset; w1 = soffset<<24 | (srsrc/4)<<16 | vdata<<8 | vaddr (LLVM
/// `gfx10_asm_mubuf.s`).
#[allow(clippy::too_many_arguments)]
fn mubufe(op: u16, vdata: u32, vaddr: u32, srsrc: u32, soffset: u32, offset: u32, offen: bool, idxen: bool) -> [u32; 2] {
    [
        0xE000_0000 | (op as u32) << 18 | (idxen as u32) << 13 | (offen as u32) << 12 | (offset & 0xFFF),
        soffset << 24 | (srsrc / 4) << 16 | vdata << 8 | vaddr,
    ]
}

const MBASE: u64 = 0x1234_5600_0100;

/// MUBUF loads and stores through a V# in s[8:11], for each OOB_SELECT
/// with strides and NUM_RECORDS that put some lanes' dwords out of
/// bounds: idxen (v6 = lane), offen (v7 = 4 * lane), both, and strided
/// stores under sparse exec (addresses computed with every lane on).
#[test]
fn mubuf_loads_and_stores_with_bounds() {
    use scarlett_gpu::rdna::mubuf;
    let table: Vec<u32> = (0..768u32).map(|k| 0xA000 + k).collect();
    let mov = |size| if size == WaveSize::W64 { sop1::S_MOV_B64 } else { sop1::S_MOV_B32 };
    let z = int(0) as u32;
    for (stride, nr, oob) in [(16u32, 20u32, 0u32), (16, 40, 1), (8, 0, 2), (8, 64, 2), (16, 48, 3), (4, 100, 3), (16, 12, 0), (0x104, 8, 1)] {
        let mut code = vec![];
        code.extend(vop3e(vop3::V_MBCNT_LO_U32_B32, 6, [int(-1), int(0), 0]));
        code.extend(vop3e(vop3::V_MBCNT_HI_U32_B32, 6, [int(-1), v(6), 0]));
        code.push(vop2e(vop2::V_LSHLREV_B32, 7, int(2), 6));
        code.push(vop2e(vop2::V_LSHLREV_B32, 4, int(3), 6)); // v4 = 8 * lane
        code.extend([sop1e(sop1::S_MOV_B32, 8, LIT), MBASE as u32]);
        code.extend([sop1e(sop1::S_MOV_B32, 9, LIT), (MBASE >> 32) as u32 | stride << 16]);
        code.extend([sop1e(sop1::S_MOV_B32, 10, LIT), nr]);
        code.extend([sop1e(sop1::S_MOV_B32, 11, LIT), oob << 28]);
        code.push(sop1e(SAVE_OR_RESTORE, EXEC, 12));
        code.extend(mubufe(mubuf::LOAD_DWORDX2, 0, 6, 8, z, 4, false, true)); // idxen
        code.extend(mubufe(mubuf::LOAD_DWORDX3, 1, 7, 8, z, 0, true, false)); // offen, v[1:3]
        code.extend(mubufe(mubuf::LOAD_DWORD, 5, 6, 8, z, 8, true, true)); // index v6, offset v7
        code.extend(mubufe(mubuf::STORE_DWORD, 0, 6, 8, z, 0, false, true)); // record = v0
        code.extend(mubufe(mubuf::STORE_DWORDX2, 2, 4, 8, z, 0x600, true, false)); // v2, v3
        code.extend(mubufe(mubuf::LOAD_DWORDX4, 0, 6, 8, z, 0, false, true));
        code.extend(mubufe(mubuf::STORE_DWORD, 1, 6, 8, z, 0x400, true, true)); // index v6, offset v7
        code.push(soppe(sopp::S_WAITCNT, 0));
        code.push(ENDPGM);
        for size in [WaveSize::W32, WaveSize::W64] {
            let mut c = vec![sop1e(mov(size), 12, EXEC), sop1e(mov(size), EXEC, int(-1))];
            c.extend(code.iter().map(|&w| {
                if w == sop1e(SAVE_OR_RESTORE, EXEC, 12) {
                    sop1e(mov(size), EXEC, 12)
                } else {
                    w
                }
            }));
            for seed in 0..2 {
                for sparse in [false, true] {
                    check_mem(&c, size, seed, sparse, Some((MBASE, &table)));
                }
            }
        }
    }
}

/// GLOBAL atomics: each op with `glc` on a per-lane word (returned value
/// compared too), the commutative ones also without `glc` on one word
/// shared by every lane, and cmpswap whose comparand matches on even
/// lanes only. Addresses computed with every lane on.
#[test]
fn global_atomics() {
    let table: Vec<u32> = (0..512u32).map(|k| k.wrapping_mul(0x9E37_79B9) ^ 0x5555).collect();
    let flat = |op: u16, vdst: u32, addr: u32, data: u32, saddr: u32, offset: i32, glc: bool| {
        [
            0xDC00_0000 | (op as u32) << 18 | (glc as u32) << 16 | 2 << 14 | (offset as u32 & 0xFFF),
            vdst << 24 | saddr << 16 | data << 8 | addr,
        ]
    };
    let mov = |size| if size == WaveSize::W64 { sop1::S_MOV_B64 } else { sop1::S_MOV_B32 };
    use global::*;
    for aop in [
        ATOMIC_SWAP, ATOMIC_ADD, ATOMIC_SUB, ATOMIC_SMIN, ATOMIC_UMIN, ATOMIC_SMAX, ATOMIC_UMAX, ATOMIC_AND,
        ATOMIC_OR, ATOMIC_XOR,
    ] {
        let mut code = vec![];
        code.extend(vop3e(vop3::V_MBCNT_LO_U32_B32, 6, [int(-1), int(0), 0]));
        code.extend(vop3e(vop3::V_MBCNT_HI_U32_B32, 6, [int(-1), v(6), 0]));
        code.push(vop2e(vop2::V_LSHLREV_B32, 7, int(2), 6));
        code.push(vop2e(vop2::V_ADD_NC_U32, 7, int(64), 7)); // v7 = 64 + 4 * lane
        code.push(vop1e(vop1::V_MOV_B32, 4, int(0)));
        code.extend([sop1e(sop1::S_MOV_B32, 2, LIT), GBASE as u32]);
        code.extend([sop1e(sop1::S_MOV_B32, 3, LIT), (GBASE >> 32) as u32]);
        code.push(sop1e(SAVE_OR_RESTORE, EXEC, 8));
        code.extend(flat(aop, 2, 7, 1, 2, 1024, true)); // v2 = old, per-lane word
        if global::atomic_commutes(aop) {
            code.extend(flat(aop, 0, 4, 0, 2, 8, false)); // shared word 2
        }
        // cmpswap: v5 = current ^ (lane & 1), v4 = new value
        code.extend(flat(LOAD_DWORD, 5, 7, 0, 2, 0, false));
        code.push(vop2e(vop2::V_AND_B32, 3, int(1), 6));
        code.push(vop2e(vop2::V_XOR_B32, 5, v(3), 5));
        code.push(vop1e(vop1::V_MOV_B32, 4, v(0)));
        code.extend(flat(ATOMIC_CMPSWAP, 3, 7, 4, 2, 0, true));
        code.push(soppe(sopp::S_WAITCNT, 0));
        code.push(ENDPGM);
        for size in [WaveSize::W32, WaveSize::W64] {
            let mut c = vec![sop1e(mov(size), 8, EXEC), sop1e(mov(size), EXEC, int(-1))];
            c.extend(code.iter().map(|&w| {
                if w == sop1e(SAVE_OR_RESTORE, EXEC, 8) {
                    sop1e(mov(size), EXEC, 8)
                } else {
                    w
                }
            }));
            for seed in 0..2 {
                for sparse in [false, true] {
                    check_mem(&c, size, seed, sparse, Some((GBASE, &table)));
                }
            }
        }
    }
}

/// Atomic semantics by hand, and the interpreter's refusal of an
/// order-dependent case.
#[test]
fn global_atomic_rules() {
    use global::*;
    assert_eq!(atomic(ATOMIC_CMPSWAP, 5, 9, 5), Some(9));
    assert_eq!(atomic(ATOMIC_CMPSWAP, 5, 9, 4), Some(5));
    assert_eq!(atomic(ATOMIC_SMIN, 0xFFFF_FFFF, 3, 0), Some(0xFFFF_FFFF));
    assert_eq!(atomic(ATOMIC_UMIN, 0xFFFF_FFFF, 3, 0), Some(3));
    assert_eq!(atomic(ATOMIC_SUB, 1, 2, 0), Some(u32::MAX));
    // every lane swapping into one word: order-dependent, refused
    let code = [
        vop1e(vop1::V_MOV_B32, 4, int(0)),
        sop1e(sop1::S_MOV_B32, 2, LIT),
        GBASE as u32,
        sop1e(sop1::S_MOV_B32, 3, LIT),
        (GBASE >> 32) as u32,
        0xDC00_0000 | (ATOMIC_SWAP as u32) << 18 | 2 << 14,
        2 << 16 | 1 << 8 | 4,
        ENDPGM,
    ];
    let mut w = Wave::new(WaveSize::W32, NV as usize);
    w.exec = 0b11;
    let mut memory = SparseMemory::new();
    memory.map(GBASE, 64);
    let e = interp::run_with_memory(&code, &mut w, &mut memory, 100).unwrap_err();
    assert!(format!("{e:?}").contains("order-dependent"), "{e:?}");
}

/// LDS atomics: every `_rtn` op on a per-lane dword (seeded with v0), the
/// commutative ones also without return on one dword shared by all lanes,
/// then cmpst_rtn whose comparand matches on even lanes; results read
/// back into VGPRs. Addresses computed with every lane on.
#[test]
fn lds_atomics() {
    let mov = |size| if size == WaveSize::W64 { sop1::S_MOV_B64 } else { sop1::S_MOV_B32 };
    let ds3 = |op: u16, vdst: u32, addr: u32, data0: u32, data1: u32, offset: u32| {
        [
            0xD800_0000 | (op as u32) << 18 | offset,
            vdst << 24 | data1 << 16 | data0 << 8 | addr,
        ]
    };
    for (op, rtn_op) in [
        (ds::ADD_U32, 0x20),
        (ds::SUB_U32, 0x21),
        (ds::MIN_I32, 0x25),
        (ds::MAX_I32, 0x26),
        (ds::MIN_U32, 0x27),
        (ds::MAX_U32, 0x28),
        (ds::AND_B32, 0x29),
        (ds::OR_B32, 0x2A),
        (ds::XOR_B32, 0x2B),
        (ds::WRXCHG_RTN_B32, ds::WRXCHG_RTN_B32),
    ] {
        let mut code = vec![];
        code.extend(vop3e(vop3::V_MBCNT_LO_U32_B32, 6, [int(-1), int(0), 0]));
        code.extend(vop3e(vop3::V_MBCNT_HI_U32_B32, 6, [int(-1), v(6), 0]));
        code.push(vop2e(vop2::V_LSHLREV_B32, 7, int(2), 6)); // v7 = 4 * lane
        code.push(vop1e(vop1::V_MOV_B32, 4, int(0)));
        code.push(sop1e(SAVE_OR_RESTORE, EXEC, 8));
        code.extend(ds3(ds::WRITE_B32, 0, 7, 0, 0, 0)); // seed: lds[lane] = v0
        code.extend(ds3(rtn_op, 2, 7, 1, 0, 0)); // v2 = old
        if ds::atomic_commutes(op) {
            code.extend(ds3(op, 0, 4, 1, 0, 1020)); // shared dword 255
        }
        code.extend(ds3(ds::READ_B32, 5, 7, 0, 0, 0));
        code.push(vop2e(vop2::V_AND_B32, 3, int(1), 6));
        code.push(vop2e(vop2::V_XOR_B32, 5, v(3), 5)); // comparand
        code.extend(ds3(ds::CMPST_RTN_B32, 3, 7, 5, 0, 0)); // data1 = v0: new value
        code.extend(ds3(ds::READ_B32, 1, 7, 0, 0, 0));
        code.extend(ds3(ds::READ_B32, 4, 4, 0, 0, 1020));
        code.push(soppe(sopp::S_WAITCNT, 0));
        code.push(ENDPGM);
        for size in [WaveSize::W32, WaveSize::W64] {
            let mut c = vec![sop1e(mov(size), 8, EXEC), sop1e(mov(size), EXEC, int(-1))];
            c.extend(code.iter().map(|&w| {
                if w == sop1e(SAVE_OR_RESTORE, EXEC, 8) {
                    sop1e(mov(size), EXEC, 8)
                } else {
                    w
                }
            }));
            for seed in 0..2 {
                for sparse in [false, true] {
                    check(&c, size, seed, sparse);
                }
            }
        }
    }
}

/// MUBUF atomics: each op with glc on a per-lane record (all in bounds),
/// without glc through a second V# whose NUM_RECORDS drops lanes >= 20,
/// commutative ones on one shared dword, and cmpswap with a comparand
/// matching on even lanes only.
#[test]
fn mubuf_atomics() {
    use scarlett_gpu::rdna::mubuf;
    let table: Vec<u32> = (0..768u32).map(|k| k.wrapping_mul(0x9E37_79B9) ^ 0x1234).collect();
    let mov = |size| if size == WaveSize::W64 { sop1::S_MOV_B64 } else { sop1::S_MOV_B32 };
    let z = int(0) as u32;
    let glc = |mut w: [u32; 2]| {
        w[0] |= mubuf::GLC;
        w
    };
    use global::*;
    for aop in [
        ATOMIC_SWAP, ATOMIC_ADD, ATOMIC_SUB, ATOMIC_SMIN, ATOMIC_UMIN, ATOMIC_SMAX, ATOMIC_UMAX, ATOMIC_AND,
        ATOMIC_OR, ATOMIC_XOR,
    ] {
        let mut code = vec![];
        code.extend(vop3e(vop3::V_MBCNT_LO_U32_B32, 6, [int(-1), int(0), 0]));
        code.extend(vop3e(vop3::V_MBCNT_HI_U32_B32, 6, [int(-1), v(6), 0]));
        code.push(vop1e(vop1::V_MOV_B32, 4, int(0)));
        for (s, nr, oob) in [(8u32, 64u32, 1u32), (12, 20, 0)] {
            code.extend([sop1e(sop1::S_MOV_B32, s as u16, LIT), MBASE as u32]);
            code.extend([sop1e(sop1::S_MOV_B32, s as u16 + 1, LIT), (MBASE >> 32) as u32 | 16 << 16]);
            code.extend([sop1e(sop1::S_MOV_B32, s as u16 + 2, LIT), nr]);
            code.extend([sop1e(sop1::S_MOV_B32, s as u16 + 3, LIT), oob << 28]);
        }
        code.push(sop1e(SAVE_OR_RESTORE, EXEC, 2));
        code.push(vop1e(vop1::V_MOV_B32, 2, v(1)));
        code.extend(glc(mubufe(aop, 2, 6, 8, z, 0, false, true))); // v2 = old
        if atomic_commutes(aop) {
            code.extend(mubufe(aop, 0, 4, 8, z, 0x500, true, false)); // shared
        }
        code.extend(mubufe(aop, 1, 6, 12, z, 4, false, true)); // lanes >= 20 dropped
        code.extend(mubufe(mubuf::LOAD_DWORD, 5, 6, 8, z, 8, false, true));
        code.push(vop2e(vop2::V_AND_B32, 3, int(1), 6));
        code.push(vop2e(vop2::V_XOR_B32, 4, v(3), 5)); // comparand
        code.push(vop1e(vop1::V_MOV_B32, 3, v(0))); // new value
        code.extend(glc(mubufe(ATOMIC_CMPSWAP, 3, 6, 8, z, 8, false, true)));
        code.push(soppe(sopp::S_WAITCNT, 0));
        code.push(ENDPGM);
        for size in [WaveSize::W32, WaveSize::W64] {
            let mut c = vec![sop1e(mov(size), 2, EXEC), sop1e(mov(size), EXEC, int(-1))];
            c.extend(code.iter().map(|&w| {
                if w == sop1e(SAVE_OR_RESTORE, EXEC, 2) {
                    sop1e(mov(size), EXEC, 2)
                } else {
                    w
                }
            }));
            for seed in 0..2 {
                for sparse in [false, true] {
                    check_mem(&c, size, seed, sparse, Some((MBASE, &table)));
                }
            }
        }
    }
}

/// SOPK: w0 = 0xB0000000 | op << 23 | sdst << 16 | simm16 (LLVM `sopk.s`:
/// `s_cmpk_eq_i32 s2, 0x6` = 0xB1820006).
fn sopke(op: u32, sdst: u32, imm: u16) -> u32 {
    0xB000_0000 | op << 23 | sdst << 16 | imm as u32
}

/// Every SOPK compare against random SGPRs with immediates whose sign
/// matters, the SCC each leaves feeding s_cmovk, plus movk/addk/mulk with
/// negative immediates (s_addk overflow into SCC) and s_waitcnt_*.
#[test]
fn sopk_ops() {
    assert_eq!(sopke(3, 2, 6), 0xB182_0006);
    use scarlett_gpu::rdna::sopk;
    for imm in [0x0006u16, 0xFFFF, 0x8000, 0x7FFF, 0x0000] {
        let mut code = vec![sopke(sopk::S_WAITCNT_VSCNT as u32, 125, 0)];
        for (k, op) in (3u32..=14).enumerate() {
            code.push(sopke(op, k as u32 % 4, imm)); // compare s0..s3
            code.push(sopke(sopk::S_CMOVK_I32 as u32, 4 + k as u32 % 8, 0x100 + k as u16));
        }
        code.push(sopke(sopk::S_MOVK_I32 as u32, 12, imm));
        code.push(sopke(sopk::S_ADDK_I32 as u32, 1, imm)); // random s1 + imm, SCC = overflow
        code.push(sopke(sopk::S_CMOVK_I32 as u32, 13, 0xABCD));
        code.push(sopke(sopk::S_MULK_I32 as u32, 2, imm));
        code.extend([sop1e(sop1::S_MOV_B32, 3, LIT), 0x7FFF_FFF0]);
        code.push(sopke(sopk::S_ADDK_I32 as u32, 3, 0x7FFF)); // overflows: SCC = 1
        code.push(sopke(sopk::S_CMOVK_I32 as u32, 14, 0x1234));
        code.push(sopke(0x1A, 125, 0));
        code.push(ENDPGM);
        for size in [WaveSize::W32, WaveSize::W64] {
            for seed in 0..6 {
                check(&code, size, seed, false);
            }
        }
    }
    // by hand: sign/zero extension
    let code = [
        sopke(sopk::S_MOVK_I32 as u32, 0, 0x8000),
        sopke(3 + 6, 0, 0x8000), // s_cmpk_eq_u32 s0, 0x8000: 0xFFFF8000 != 0x8000
        sopke(sopk::S_CMOVK_I32 as u32, 1, 1),
        sopke(3, 0, 0x8000), // s_cmpk_eq_i32: equal
        sopke(sopk::S_CMOVK_I32 as u32, 2, 1),
        sopke(3 + 4, 0, 0xFFFF), // s_cmpk_lt_i32 s0, -1: -32768 < -1
        sopke(sopk::S_CMOVK_I32 as u32, 3, 7),
        sopke(sopk::S_MOVK_I32 as u32, 4, 5),
        sopke(sopk::S_MULK_I32 as u32, 4, 0xFFFD), // 5 * -3
        sopke(sopk::S_MOVK_I32 as u32, 5, 0x7FFF),
        sopke(sopk::S_ADDK_I32 as u32, 5, 0xFFFF), // 32767 - 1, no overflow
        ENDPGM,
    ];
    let mut w = Wave::new(WaveSize::W32, NV as usize);
    interp::run(&code, &mut w, 100).unwrap();
    assert_eq!(w.sgpr[..6], [0xFFFF_8000, 0, 1, 7, (-15i32) as u32, 0x7FFE]);
    assert!(!w.scc);
}

/// The second batch of integer VOP3 ops on random sources, plus bfe_i32 /
/// alignbit edge cases (width 0, offset + width >= 32, shift 0).
#[test]
fn integer_vop3_more() {
    let ops = [
        vop3::V_MUL_HI_U32,
        vop3::V_MUL_LO_I32,
        vop3::V_MUL_HI_I32,
        vop3::V_MAD_I32_I24,
        vop3::V_MAD_U32_U24,
        vop3::V_BCNT_U32_B32,
        vop3::V_ALIGNBIT_B32,
        vop3::V_BFE_I32,
        vop3::V_MIN3_I32,
        vop3::V_MIN3_U32,
        vop3::V_MAX3_I32,
        vop3::V_MAX3_U32,
        vop3::V_MED3_I32,
        vop3::V_XAD_U32,
        vop3::V_LSHL_OR_B32,
        vop3::V_ALIGNBYTE_B32,
        vop3::V_MSAD_U8,
        vop3::V_CVT_PK_U16_U32,
        vop3::V_CVT_PK_I16_I32,
    ];
    for op in ops {
        let mut code = vec![];
        for (d, s) in [(3u16, [v(0), v(1), v(2)]), (4, [v(2), v(0), v(1)]), (5, [v(1), v(2), v(0)])] {
            code.extend(vop3e(op, d, s));
        }
        code.push(ENDPGM);
        check_all(&code);
    }
    let mut code = vec![];
    for (d, s) in [
        (3u16, [v(0), int(0), int(0)]),   // bfe width 0
        (4, [v(0), int(20), int(12)]),     // off + width = 32
        (5, [v(0), int(4), int(9)]),       // a field
        (6, [v(1), int(31), int(1)]),      // top bit only, via the >= 32 path
    ] {
        code.extend(vop3e(vop3::V_BFE_I32, d, s));
    }
    code.extend(vop3e(vop3::V_ALIGNBIT_B32, 7, [v(0), v(1), int(0)]));
    code.push(ENDPGM);
    check_all(&code);
}

/// Carry arithmetic: VOP3b v_{add,sub,subrev}_co_u32 into an SGPR mask
/// and vcc, VOP2 v_*_co_ci_u32 chained through vcc (a 64-bit add and
/// subtract), and their VOP3 forms with a random SGPR mask as carry in.
#[test]
fn carry_ops() {
    let b3 = |op: u16, d: u16, sdst: u32, s: [u16; 3]| {
        let mut w = vop3e(op, d, s);
        w[0] |= sdst << 8;
        w
    };
    let e = |o: u16| vop3::FROM_VOP2 + o;
    for (lo_op, hi_op) in [
        (vop3::V_ADD_CO_U32, vop2::V_ADD_CO_CI_U32),
        (vop3::V_SUB_CO_U32, vop2::V_SUB_CO_CI_U32),
        (vop3::V_SUBREV_CO_U32, vop2::V_SUBREV_CO_CI_U32),
    ] {
        let mut code = vec![];
        code.extend(b3(lo_op, 2, 106, [v(0), v(1), 0])); // carry -> vcc
        code.push(vop2e(hi_op, 3, v(4), 5)); // v3 = v4 op v5 op vcc, carry -> vcc
        code.extend(b3(lo_op, 6, 8, [v(1), v(0), 0])); // carry -> s8 (s[8:9])
        code.extend(b3(e(hi_op), 7, 10, [v(5), v(4), 4])); // carry in s4/s[4:5], out s10
        code.extend(b3(lo_op, 1, 12, [v(0), LIT, 0]));
        code.push(0xFFFF_FFF8);
        code.extend(b3(e(hi_op), 0, 14, [int(-1), int(0), 4])); // carry only from carry in
        code.extend(b3(e(hi_op), 5, 6, [int(3), int(3), 4])); // borrow only from carry in
        code.push(ENDPGM);
        check_all(&code);
    }
}

#[test]
fn carry_op_rules() {
    use scarlett_gpu::interp::carry_op;
    assert_eq!(carry_op(vop3::V_ADD_CO_U32, u32::MAX, 1, true), Some((0, true))); // no carry in for VOP3b
    assert_eq!(carry_op(vop2::V_ADD_CO_CI_U32, u32::MAX, 0, true), Some((0, true)));
    assert_eq!(carry_op(vop2::V_ADD_CO_CI_U32, 1, 2, true), Some((4, false)));
    assert_eq!(carry_op(vop3::V_SUB_CO_U32, 1, 2, false), Some((u32::MAX, true)));
    assert_eq!(carry_op(vop2::V_SUB_CO_CI_U32, 2, 2, true), Some((u32::MAX, true)));
    assert_eq!(carry_op(vop2::V_SUB_CO_CI_U32, 3, 2, true), Some((0, false)));
    assert_eq!(carry_op(vop3::V_SUBREV_CO_U32, 1, 2, false), Some((1, false))); // 2 - 1
    assert_eq!(carry_op(vop3::FROM_VOP2 + vop2::V_SUBREV_CO_CI_U32, 2, 2, true), Some((u32::MAX, true)));
}

/// 64-bit scalar shifts by random and edge amounts (0, 31, 32, 33, 63,
/// 64 = 0), s_mul_hi_u32/i32 and s_bitset0/1_b32; SCC after each shift
/// lands in an SGPR through s_cselect.
#[test]
fn salu_shift64_mulhi_bitset() {
    let mut code = vec![];
    for (k, op) in [sop2::S_LSHL_B64, sop2::S_LSHR_B64, sop2::S_ASHR_I64].into_iter().enumerate() {
        let d = 4 + 2 * k as u16; // s[4:5], s[6:7], s[8:9]
        code.push(sop2e(op, d, 2, 1)); // s[2:3] by s1
        for amt in [0i32, 31, 32, 33, 63, 64] {
            code.push(sop2e(op, 10, 2, int(amt)));
            code.push(sop2e(sop2::S_XOR_B32, d, d, 10)); // fold into the result
            code.push(sop2e(sop2::S_XOR_B32, d + 1, d + 1, 11));
        }
        code.push(sop2e(op, 10, 0, 3)); // s[0:1] by s3, sets SCC
        code.push(sop2e(sop2::S_CSELECT_B32, 12 + k as u16, int(1), int(0)));
    }
    code.push(sop2e(sop2::S_MUL_HI_U32, 15, 0, 1));
    code.push(sop2e(sop2::S_MUL_HI_I32, 13, 2, 3));
    code.push(sop1e(sop1::S_BITSET1_B32, 0, 3)); // s0 |= 1 << s3
    code.push(sop1e(sop1::S_BITSET0_B32, 1, 2)); // s1 &= ~(1 << s2)
    code.push(sop1e(sop1::S_BITSET0_B32, 2, int(36))); // bit 4
    code.push(ENDPGM);
    check_all(&code);
}

#[test]
fn salu_shift64_by_hand() {
    use scarlett_gpu::interp::{shift64, sop2_b32};
    assert_eq!(shift64(sop2::S_LSHL_B64, 0x8000_0001, 33), Some(0x2_0000_0000));
    assert_eq!(shift64(sop2::S_LSHR_B64, 0x8000_0000_0000_0000, 63), Some(1));
    assert_eq!(shift64(sop2::S_ASHR_I64, 0x8000_0000_0000_0000, 32), Some(0xFFFF_FFFF_8000_0000));
    assert_eq!(shift64(sop2::S_LSHL_B64, 5, 64), Some(5)); // amount & 63
    assert_eq!(sop2_b32(sop2::S_MUL_HI_I32, u32::MAX, 2), Some((u32::MAX, false))); // -2 >> 32
    assert_eq!(sop2_b32(sop2::S_MUL_HI_U32, u32::MAX, 2), Some((1, false)));
}

/// v_{lshl,lshr}rev_b64 / v_ashrrev_i64: VGPR-pair, SGPR-pair and inline
/// 64-bit sources, per-lane random and constant shift amounts, results
/// written over a source pair.
#[test]
fn valu_shift64() {
    for op in [vop3::V_LSHLREV_B64, vop3::V_LSHRREV_B64, vop3::V_ASHRREV_I64] {
        let mut code = vec![];
        code.extend(vop3e(op, 4, [v(0), v(2), 0])); // v[4:5] = v[2:3] by v0
        code.extend(vop3e(op, 6, [int(33), 4, 0])); // v[6:7] = s[4:5] by 33
        code.extend(vop3e(op, 2, [v(1), int(-7), 0])); // v[2:3] = -7 (i64) by v1
        code.extend(vop3e(op, 0, [int(32), v(6), 0])); // v[0:1] = v[6:7] by 32
        code.extend(vop3e(op, 6, [3, v(6), 0])); // v[6:7] by s3, in place
        code.push(ENDPGM);
        check_all(&code);
    }
}

/// v_readlane_b32 / v_writelane_b32 with constant and SGPR lane selects
/// (s4 = s0 & 31), under sparse exec too: both ignore exec, so inactive
/// lanes are read and written.
#[test]
fn readlane_writelane() {
    let mut code = vec![sop2e(sop2::S_AND_B32, 4, 0, int(31))];
    code.extend(vop3e(vop3::V_READLANE_B32, 5, [v(1), 4, 0])); // s5 = v1[s4]
    code.extend(vop3e(vop3::V_READLANE_B32, 6, [v(2), int(3), 0])); // s6 = v2[3]
    code.extend(vop3e(vop3::V_WRITELANE_B32, 3, [1, 4, 0])); // v3[s4] = s1
    code.extend(vop3e(vop3::V_WRITELANE_B32, 4, [int(-5), int(0), 0])); // v4[0] = -5
    code.extend(vop3e(vop3::V_WRITELANE_B32, 4, [5, int(31), 0])); // v4[31] = s5
    code.extend(vop3e(vop3::V_READLANE_B32, 7, [v(4), int(31), 0])); // s7 = s5
    code.push(ENDPGM);
    check_all(&code);
    // by hand: exec = 0 still writes and reads
    let mut w = Wave::new(WaveSize::W32, NV as usize);
    w.exec = 0;
    w.sgpr[1] = 77;
    let mut c = vec![];
    c.extend(vop3e(vop3::V_WRITELANE_B32, 0, [1, int(9), 0]));
    c.extend(vop3e(vop3::V_READLANE_B32, 2, [v(0), int(9), 0]));
    c.push(ENDPGM);
    interp::run(&c, &mut w, 100).unwrap();
    assert_eq!((w.vgpr[0][9], w.sgpr[2]), (77, 77));
    // lane select past the wave: refused
    let mut w = Wave::new(WaveSize::W32, NV as usize);
    let mut c = vec![];
    c.extend(vop3e(vop3::V_READLANE_B32, 2, [v(0), int(40), 0]));
    c.push(ENDPGM);
    assert!(interp::run(&c, &mut w, 100).is_err());
}

/// v_fmamk_f32 / v_fmaak_f32 (K in the trailing dword) and
/// s_bitcmp0/1_b32 feeding s_cselect.
#[test]
fn fmak_and_bitcmp() {
    use scarlett_gpu::rdna::{decode_shader, sopc};
    // LLVM: v_fmaak_f32 v5, v1, v2, 0x11213141 = 0x5A0A0501 0x11213141
    let insts = decode_shader(&[0x5A0A_0501, 0x1121_3141, ENDPGM]).unwrap();
    assert_eq!((insts[0].1.opcode, insts[0].1.words.len()), (vop2::V_FMAAK_F32, 2));
    assert_eq!(insts[0].1.literal(), Some(0x1121_3141));
    let code = [
        vop2e(vop2::V_FMAMK_F32, 4, v(0), 1),
        1.5f32.to_bits(),
        vop2e(vop2::V_FMAAK_F32, 5, v(1), 2),
        (-0.375f32).to_bits(),
        vop2e(vop2::V_FMAAK_F32, 6, LIT, 3), // src0 literal = K
        3.0f32.to_bits(),
        sopce(sopc::S_BITCMP1_B32, 0, 1),
        sop2e(sop2::S_CSELECT_B32, 4, int(7), int(9)),
        sopce(sopc::S_BITCMP0_B32, 2, int(36)), // bit 4
        sop2e(sop2::S_CSELECT_B32, 5, int(7), int(9)),
        sopce(sopc::S_BITCMP1_B32, 3, 0),
        sop2e(sop2::S_CSELECT_B32, 6, int(7), int(9)),
        ENDPGM,
    ];
    check_all(&code);
}

/// s_getpc_b64 at several positions (the base is close to a 4 GiB
/// boundary, so the carry into the high dword shows), then relative
/// address arithmetic as in LLVM's `s_getpc` + `s_add_u32` + `s_addc_u32`.
#[test]
fn getpc() {
    use scarlett_gpu::rdna::sop1;
    let code = [
        sop1e(sop1::S_GETPC_B64, 0, 0),
        sop1e(sop1::S_MOV_B32, 4, int(1)),
        sop1e(sop1::S_GETPC_B64, 2, 0),
        sop2e(sop2::S_ADD_U32, 2, 2, LIT),
        0x100,
        sop2e(sop2::S_ADDC_U32, 3, 3, int(0)),
        sop1e(sop1::S_GETPC_B64, 6, 0),
        ENDPGM,
    ];
    check_all(&code);
    let mut w = Wave::new(WaveSize::W32, NV as usize);
    w.code_base = CODE_BASE;
    interp::run(&code, &mut w, 100).unwrap();
    let pc = |s: usize| w.sgpr[s] as u64 | (w.sgpr[s + 1] as u64) << 32;
    assert_eq!(pc(0), CODE_BASE + 4);
    assert_eq!(pc(2), CODE_BASE + 12 + 0x100);
    assert_eq!(pc(6), 0x8_0000_0000 - 0x100 + 28);
    let mut w = Wave::new(WaveSize::W32, NV as usize);
    w.code_base = 1 << 47;
    assert!(interp::run(&code, &mut w, 100).is_err());
}

/// rdna::tiling by hand: SW_4KB_S with 4-byte elements is a 32 x 32
/// block with bits [0, 0, X0, X1, Y0, Y1, Y2, X2, Y3, X3, Y4, X4].
#[test]
fn tiling_offsets_by_hand() {
    use scarlett_gpu::rdna::tiling;
    let p = tiling::pattern(5, 2).unwrap();
    assert_eq!(tiling::block_dims(p), (5, 5));
    assert_eq!(tiling::offset(5, 2, 64, 1, 0), Some(4));
    assert_eq!(tiling::offset(5, 2, 64, 0, 1), Some(16));
    assert_eq!(tiling::offset(5, 2, 64, 4, 0), Some(128));
    assert_eq!(tiling::offset(5, 2, 64, 0, 8), Some(256));
    assert_eq!(tiling::offset(5, 2, 64, 32, 0), Some(4096)); // next block
    assert_eq!(tiling::offset(5, 2, 40, 0, 32), Some(2 * 4096)); // pitch 2 blocks
    assert_eq!(tiling::block_dims(tiling::pattern(9, 2).unwrap()), (7, 7)); // 64 KiB
    assert_eq!(tiling::block_dims(tiling::pattern(1, 3).unwrap()), (3, 2)); // 256 B, 8-byte
    assert!(tiling::pattern(6, 2).is_none()); // SW_4KB_D: RB+-dependent
    // a tiled image with more than one mip level is refused
    let mut d = tdesc(IBASE, 22, 64, 36, [4, 0, 0, 1]);
    d[3] |= 5 << 20 | 1 << 16;
    let mut code = vec![];
    for (k, dw) in d.iter().enumerate() {
        code.extend([sop1e(sop1::S_MOV_B32, 8 + k as u16, LIT), *dw]);
    }
    code.extend(mimge(1, 2, 0, 8));
    code.push(ENDPGM);
    let mut w = Wave::new(WaveSize::W32, NV as usize);
    let mut memory = SparseMemory::new();
    memory.map(IBASE, 0x10000);
    let e = interp::run_with_memory(&code, &mut w, &mut memory, 100).unwrap_err();
    assert!(format!("{e:?}").contains("tiled mips"), "{e:?}");
}

/// image_load and image_store on tiled images: SW_4KB_S (4- and 16-byte
/// elements), SW_256B_S (8-byte), SW_64KB_S (4-byte); coordinates cross
/// block boundaries.
#[test]
fn image_tiled_2d() {
    let tiled = |mut d: [u32; 8], sw: u32| {
        d[3] |= sw << 20;
        d
    };
    let words = 0x2F000 / 4;
    let mut table = vec![0u32; words];
    for (k, t) in table.iter_mut().enumerate() {
        *t = (k as u32).wrapping_mul(2_654_435_761) ^ 0x3F80_0000;
    }
    let (x, y, z, w) = (4, 5, 6, 7);
    let images = [
        // width 70: not a multiple of any block width, so pitch rounding shows
        (tiled(tdesc(IBASE, 22, 70, 36, [x, 0, 0, 1]), 5), 0b0001),
        (tiled(tdesc(IBASE + 0x6000, 62, 70, 36, [x, y, 0, 0]), 1), 0b0011),
        (tiled(tdesc(IBASE + 0x10000, 22, 70, 36, [x, 0, 0, 1]), 9), 0b0001),
        (tiled(tdesc(IBASE + 0x20000, 77, 70, 36, [x, y, z, w]), 5), 0b1111),
    ];
    let mut code = vec![];
    code.extend(vop3e(vop3::V_MBCNT_LO_U32_B32, 6, [int(-1), int(0), 0]));
    code.extend(vop3e(vop3::V_MBCNT_HI_U32_B32, 6, [int(-1), v(6), 0]));
    code.push(vop2e(vop2::V_MUL_U32_U24, 0, int(5), 6));
    code.push(vop2e(vop2::V_AND_B32, 0, LIT, 0));
    code.push(63); // x = 5 lane & 63
    code.push(vop2e(vop2::V_MUL_U32_U24, 1, int(3), 6));
    code.push(vop2e(vop2::V_AND_B32, 1, int(31), 1));
    code.push(vop2e(vop2::V_AND_B32, 7, int(2), 6));
    code.push(vop2e(vop2::V_ADD_NC_U32, 1, v(7), 1)); // y = (3 lane & 31) + (lane & 2)
    code.push(vop1e(vop1::V_MOV_B32, 6, int(0)));
    code.push(vop1e(vop1::V_MOV_B32, 7, int(0)));
    for (desc, dmask) in images {
        for (k, dw) in desc.iter().enumerate() {
            code.extend([sop1e(sop1::S_MOV_B32, 8 + k as u16, LIT), *dw]);
        }
        code.extend(mimge(dmask, 2, 0, 8)); // v[2:5] = image_load(v[0:1])
        code.push(soppe(sopp::S_WAITCNT, 0));
        for r in 2..2 + dmask.count_ones() as u16 {
            code.push(vop2e(vop2::V_XOR_B32, 6, v(r), 6));
            code.push(vop2e(vop2::V_ADD_NC_U32, 7, v(r), 7));
        }
    }
    // store the running sum into the 4KB_S image at (x, y) and read it back
    let [s0, s1] = mimge(1, 7, 0, 8);
    let d0 = images[0].0;
    for (k, dw) in d0.iter().enumerate() {
        code.extend([sop1e(sop1::S_MOV_B32, 8 + k as u16, LIT), *dw]);
    }
    code.extend([s0 | 8 << 18, s1]); // image_store v7, v[0:1]
    code.push(soppe(sopp::S_WAITCNT, 0));
    code.extend(mimge(1, 2, 0, 8));
    code.push(soppe(sopp::S_WAITCNT, 0));
    code.push(ENDPGM);
    for size in [WaveSize::W32, WaveSize::W64] {
        for seed in 0..2 {
            for sparse in [false, true] {
                check_mem(&code, size, seed, sparse, Some((IBASE, &table)));
            }
        }
    }
}

/// MTBUF: w0 = 0xE8000000 | format << 19 | op[2:0] << 16 | idxen << 13 |
/// offen << 12 | offset; w1 = soffset << 24 | op[3] << 21 | srsrc/4 << 16
/// | vdata << 8 | vaddr (LLVM `mtbuf-gfx10.s`).
fn mtbufe(op: u32, format: u32, vdata: u32, vaddr: u32, srsrc: u32, offset: u32, idxen: bool) -> [u32; 2] {
    [
        0xE800_0000 | format << 19 | (op & 7) << 16 | (idxen as u32) << 13 | offset,
        (int(0) as u32) << 24 | (op >> 3) << 21 | (srsrc / 4) << 16 | vdata << 8 | vaddr,
    ]
}

/// Typed buffer loads and stores for the 32-bit-channel formats, with
/// identity DST_SEL, through a V# whose NUM_RECORDS drops lanes >= 40.
#[test]
fn mtbuf_typed_32bit() {
    assert_eq!(mtbufe(0, 77, 0, 0, 0, 0, false), [0xEA68_0000, 0x8000_0000], "LLVM tbuffer_load_format_x");
    let table: Vec<u32> = (0..768u32).map(|k| 0x5000 + k * 7).collect();
    let mov = |size| if size == WaveSize::W64 { sop1::S_MOV_B64 } else { sop1::S_MOV_B32 };
    let ident = 4 | 5 << 3 | 6 << 6 | 7 << 9;
    let mut code = vec![];
    code.extend(vop3e(vop3::V_MBCNT_LO_U32_B32, 6, [int(-1), int(0), 0]));
    code.extend(vop3e(vop3::V_MBCNT_HI_U32_B32, 6, [int(-1), v(6), 0]));
    code.extend([sop1e(sop1::S_MOV_B32, 8, LIT), MBASE as u32]);
    code.extend([sop1e(sop1::S_MOV_B32, 9, LIT), (MBASE >> 32) as u32 | 16 << 16]); // stride 16
    code.extend([sop1e(sop1::S_MOV_B32, 10, LIT), 40]);
    code.extend([sop1e(sop1::S_MOV_B32, 11, LIT), ident | 1 << 28]); // OOB_SELECT 1
    code.push(sop1e(SAVE_OR_RESTORE, EXEC, 12));
    code.extend(mtbufe(0, 22, 0, 6, 8, 0, true)); // v0 = 32_FLOAT x
    code.extend(mtbufe(1, 62, 1, 6, 8, 4, true)); // v[1:2] = 32_32_UINT xy at +4
    code.extend(mtbufe(2, 77, 2, 6, 8, 0, true)); // v[2:4] = 32x4 FLOAT xyz
    code.extend(mtbufe(0, 62, 5, 6, 8, 8, true)); // v5 = x of a 2-channel format
    code.extend(mtbufe(3, 77, 0, 6, 8, 0x400, true)); // v[0:3] = xyzw at +0x400
    code.extend(mtbufe(7, 77, 2, 6, 8, 0x600, true)); // store xyzw v[2:5]
    code.extend(mtbufe(5, 62, 0, 6, 8, 0x200, true)); // store xy v[0:1]
    code.extend(mtbufe(4, 21, 5, 6, 8, 0x208, true)); // store x (32_SINT)
    code.push(soppe(sopp::S_WAITCNT, 0));
    code.push(ENDPGM);
    for size in [WaveSize::W32, WaveSize::W64] {
        let mut c = vec![sop1e(mov(size), 12, EXEC), sop1e(mov(size), EXEC, int(-1))];
        c.extend(code.iter().map(|&w| if w == sop1e(SAVE_OR_RESTORE, EXEC, 12) { sop1e(mov(size), EXEC, 12) } else { w }));
        for seed in 0..2 {
            for sparse in [false, true] {
                check_mem(&c, size, seed, sparse, Some((MBASE, &table)));
            }
        }
    }
    // Refused: conversion formats, d16, a non-identity DST_SEL.
    use scarlett_gpu::rdna::mtbuf::as_mubuf;
    assert!(as_mubuf(0, 56).is_none()); // 8_8_8_8_UNORM
    assert!(as_mubuf(8, 22).is_none()); // d16
    assert!(as_mubuf(1, 22).is_none()); // xy of a 1-channel format
    assert!(as_mubuf(4, 77).is_none()); // partial store
    use scarlett_gpu::rdna::mubuf as mb;
    assert_eq!(as_mubuf(3, 77), Some(mb::LOAD_DWORDX4));
    assert_eq!(as_mubuf(2, 77), Some(mb::LOAD_DWORDX3));
    assert_eq!(as_mubuf(0, 62), Some(mb::LOAD_DWORD));
    assert_eq!(as_mubuf(5, 62), Some(mb::STORE_DWORDX2));
    assert_eq!(as_mubuf(7, 77), Some(mb::STORE_DWORDX4));
    let mut code = vec![];
    for (k, dw) in [MBASE as u32, (MBASE >> 32) as u32, 64, 5 | 4 << 3 | 3 << 28].iter().enumerate() {
        code.extend([sop1e(sop1::S_MOV_B32, 8 + k as u16, LIT), *dw]);
    }
    code.extend(mtbufe(0, 22, 0, 0, 8, 0, false));
    code.push(ENDPGM);
    let mut w = Wave::new(WaveSize::W32, NV as usize);
    let mut memory = SparseMemory::new();
    memory.map(MBASE, 0x1000);
    let e = interp::run_with_memory(&code, &mut w, &mut memory, 100).unwrap_err();
    assert!(format!("{e:?}").contains("DST_SEL"), "{e:?}");
}

/// ds_write2_b32 / ds_write2st64_b32 / ds_read2st64_b32: each lane writes
/// two dwords both ways, then reads other lanes' pairs.
#[test]
fn lds_write2_read2st64() {
    let ds = |op: u16, vdst: u32, addr: u32, d0: u32, d1: u32, o0: u32, o1: u32| {
        [0xD800_0000 | (op as u32) << 18 | o1 << 8 | o0, vdst << 24 | d1 << 16 | d0 << 8 | addr]
    };
    let mov = |size| if size == WaveSize::W64 { sop1::S_MOV_B64 } else { sop1::S_MOV_B32 };
    let mut code = vec![];
    code.extend(vop3e(vop3::V_MBCNT_LO_U32_B32, 6, [int(-1), int(0), 0]));
    code.extend(vop3e(vop3::V_MBCNT_HI_U32_B32, 6, [int(-1), v(6), 0]));
    code.push(vop2e(vop2::V_LSHLREV_B32, 7, int(3), 6)); // v7 = 8 lane
    code.push(vop2e(vop2::V_LSHLREV_B32, 3, int(2), 6)); // v3 = 4 lane
    code.push(vop2e(vop2::V_SUB_NC_U32, 4, int(63), 6));
    code.push(vop2e(vop2::V_LSHLREV_B32, 4, int(2), 4)); // v4 = 4 (63 - lane)
    code.push(vop2e(vop2::V_ADD_NC_U32, 5, int(1), 6));
    code.push(vop2e(vop2::V_AND_B32, 5, int(63), 5));
    code.push(vop2e(vop2::V_LSHLREV_B32, 5, int(3), 5)); // v5 = 8 ((lane + 1) & 63)
    code.push(sop1e(SAVE_OR_RESTORE, EXEC, 8));
    code.extend(ds(ds::WRITE2_B32, 0, 7, 0, 1, 0, 1)); // words 2l, 2l+1 = v0, v1
    code.extend(ds(ds::WRITE2ST64_B32, 0, 3, 2, 0, 2, 3)); // words 128+l, 192+l = v2, v0
    code.extend(ds(ds::READ2ST64_B32, 0, 4, 0, 0, 2, 3)); // v[0:1] = words 128+63-l, 192+63-l
    code.extend(ds(ds::READ2_B32, 2, 5, 0, 0, 0, 1)); // v[2:3] = next lane's pair
    code.push(ENDPGM);
    for size in [WaveSize::W32, WaveSize::W64] {
        let mut c = vec![sop1e(mov(size), 8, EXEC), sop1e(mov(size), EXEC, int(-1))];
        c.extend(code.iter().map(|&w| if w == sop1e(SAVE_OR_RESTORE, EXEC, 8) { sop1e(mov(size), EXEC, 8) } else { w }));
        for seed in 0..3 {
            for sparse in [false, true] {
                check(&c, size, seed, sparse);
            }
        }
    }
}

/// global_load_{u,s}byte / {u,s}short at every byte / half offset, and
/// global_load/store_dwordx3; addresses computed with every lane on.
#[test]
fn global_sub_dword_and_x3() {
    use scarlett_gpu::rdna::global::{self as g, extract};
    assert_eq!(extract(0x80FF_7F01, 1, 1, true), 0x7F);
    assert_eq!(extract(0x80FF_7F01, 2, 1, true), 0xFFFF_FFFF);
    assert_eq!(extract(0x80FF_7F01, 2, 2, false), 0x80FF);
    assert_eq!(extract(0x80FF_7F01, 2, 2, true), 0xFFFF_80FF);
    let table: Vec<u32> = (0..512u32).map(|k| k.wrapping_mul(0x9E37_79B9) ^ 0x8080_8080).collect();
    let flat = |op: u16, vdst: u32, addr: u32, data: u32, saddr: u32, offset: i32| {
        [0xDC00_0000 | (op as u32) << 18 | 2 << 14 | (offset as u32 & 0xFFF), vdst << 24 | saddr << 16 | data << 8 | addr]
    };
    let mov = |size| if size == WaveSize::W64 { sop1::S_MOV_B64 } else { sop1::S_MOV_B32 };
    let mut code = vec![];
    code.extend(vop3e(vop3::V_MBCNT_LO_U32_B32, 6, [int(-1), int(0), 0]));
    code.extend(vop3e(vop3::V_MBCNT_HI_U32_B32, 6, [int(-1), v(6), 0]));
    code.push(vop2e(vop2::V_MUL_U32_U24, 7, int(13), 6)); // v7 = 13 lane: every byte offset
    code.push(vop2e(vop2::V_AND_B32, 5, int(-2), 7)); // v5 = even (short) offset
    code.push(vop2e(vop2::V_MUL_U32_U24, 4, int(12), 6)); // v4 = 12 lane
    code.extend([sop1e(sop1::S_MOV_B32, 2, LIT), GBASE as u32]);
    code.extend([sop1e(sop1::S_MOV_B32, 3, LIT), (GBASE >> 32) as u32]);
    code.push(sop1e(SAVE_OR_RESTORE, EXEC, 8));
    code.extend(flat(g::LOAD_UBYTE, 0, 7, 0, 2, 0));
    code.extend(flat(g::LOAD_SBYTE, 1, 7, 0, 2, 0));
    code.extend(flat(g::LOAD_USHORT, 2, 5, 0, 2, 0));
    code.extend(flat(g::LOAD_SSHORT, 3, 5, 0, 2, 2));
    code.extend(flat(g::STORE_DWORDX3, 0, 4, 0, 2, 1024)); // mem[1024 + 12 lane ..] = v0..v2
    code.extend(flat(g::LOAD_DWORDX3, 1, 4, 0, 2, 4)); // v[1:3] = mem[4 + 12 lane ..]
    code.push(soppe(sopp::S_WAITCNT, 0));
    code.push(ENDPGM);
    for size in [WaveSize::W32, WaveSize::W64] {
        let mut c = vec![sop1e(mov(size), 8, EXEC), sop1e(mov(size), EXEC, int(-1))];
        c.extend(code.iter().map(|&w| if w == sop1e(SAVE_OR_RESTORE, EXEC, 8) { sop1e(mov(size), EXEC, 8) } else { w }));
        for seed in 0..2 {
            for sparse in [false, true] {
                check_mem(&c, size, seed, sparse, Some((GBASE, &table)));
            }
        }
    }
}

/// buffer_load_{u,s}byte / {u,s}short at every byte offset (offen), in
/// and out of a raw V#'s NUM_RECORDS, and buffer_store/load_dwordx3.
#[test]
fn mubuf_sub_dword_and_x3() {
    let table: Vec<u32> = (0..768u32).map(|k| k.wrapping_mul(0x9E37_79B9) ^ 0x8080_8080).collect();
    let mov = |size| if size == WaveSize::W64 { sop1::S_MOV_B64 } else { sop1::S_MOV_B32 };
    let z = int(0) as u32;
    let mut code = vec![];
    code.extend(vop3e(vop3::V_MBCNT_LO_U32_B32, 6, [int(-1), int(0), 0]));
    code.extend(vop3e(vop3::V_MBCNT_HI_U32_B32, 6, [int(-1), v(6), 0]));
    code.push(vop2e(vop2::V_MUL_U32_U24, 7, int(13), 6)); // byte offsets
    code.push(vop2e(vop2::V_AND_B32, 5, int(-2), 7)); // even offsets
    code.push(vop2e(vop2::V_MUL_U32_U24, 4, int(12), 6));
    for (s, nr) in [(8u32, 0x1800u32), (12, 300)] {
        code.extend([sop1e(sop1::S_MOV_B32, s as u16, LIT), MBASE as u32]);
        code.extend([sop1e(sop1::S_MOV_B32, s as u16 + 1, LIT), (MBASE >> 32) as u32]);
        code.extend([sop1e(sop1::S_MOV_B32, s as u16 + 2, LIT), nr]);
        code.extend([sop1e(sop1::S_MOV_B32, s as u16 + 3, LIT), 3 << 28]);
    }
    code.push(sop1e(SAVE_OR_RESTORE, EXEC, 2));
    code.extend(mubufe(8, 0, 7, 12, z, 0, true, false)); // ubyte, NUM_RECORDS 300
    code.extend(mubufe(9, 1, 7, 12, z, 0, true, false)); // sbyte
    code.extend(mubufe(10, 2, 5, 12, z, 0, true, false)); // ushort
    code.extend(mubufe(11, 3, 5, 8, z, 2, true, false)); // sshort, in bounds
    code.extend(mubufe(31, 0, 4, 8, z, 0x400, true, false)); // store_dwordx3 v[0:2]
    code.extend(mubufe(15, 1, 4, 8, z, 4, true, false)); // load_dwordx3 v[1:3]
    code.push(soppe(sopp::S_WAITCNT, 0));
    code.push(ENDPGM);
    for size in [WaveSize::W32, WaveSize::W64] {
        let mut c = vec![sop1e(mov(size), 2, EXEC), sop1e(mov(size), EXEC, int(-1))];
        c.extend(code.iter().map(|&w| if w == sop1e(SAVE_OR_RESTORE, EXEC, 2) { sop1e(mov(size), EXEC, 2) } else { w }));
        for seed in 0..2 {
            for sparse in [false, true] {
                check_mem(&c, size, seed, sparse, Some((MBASE, &table)));
            }
        }
    }
}

/// v_cvt_f16_f32 (RNE, D[31:16] kept), v_cvt_f32_f16 (S0[15:0]) and
/// v_cvt_pkrtz_f16_f32 (RTZ) on per-lane values with ties and arbitrary
/// mantissas in the normal f16 range.
#[test]
fn f16_conversions() {
    use scarlett_gpu::interp::{f16_to_f32, f32_to_f16};
    assert_eq!(f32_to_f16(1.0f32.to_bits(), false), Some(0x3C00));
    assert_eq!(f32_to_f16((-2.5f32).to_bits(), false), Some(0xC100));
    assert_eq!(f32_to_f16(1024.5f32.to_bits(), false), Some(0x6400)); // tie -> even (1024)
    assert_eq!(f32_to_f16(1025.5f32.to_bits(), false), Some(0x6402)); // tie -> even (1026)
    assert_eq!(f32_to_f16(1025.5f32.to_bits(), true), Some(0x6401)); // toward zero (1025)
    assert_eq!(f32_to_f16(65519.0f32.to_bits(), false), Some(0x7BFF));
    assert_eq!(f32_to_f16(65520.0f32.to_bits(), false), None); // overflow: refused
    assert_eq!(f32_to_f16(f32::INFINITY.to_bits(), true), Some(0x7C00));
    assert_eq!(f32_to_f16(1e-5f32.to_bits(), false), None); // f16 denormal range
    assert_eq!(f32_to_f16(f32::NAN.to_bits(), false), None);
    assert_eq!(f32_to_f16((-0.0f32).to_bits(), false), Some(0x8000));
    assert_eq!(f16_to_f32(0x3C00), Some(1.0f32.to_bits()));
    assert_eq!(f16_to_f32(0xFC00), Some(f32::NEG_INFINITY.to_bits()));
    assert_eq!(f16_to_f32(0x0001), None);
    let mut code = vec![];
    code.extend(vop3e(vop3::V_MBCNT_LO_U32_B32, 6, [int(-1), int(0), 0]));
    code.extend(vop3e(vop3::V_MBCNT_HI_U32_B32, 6, [int(-1), v(6), 0]));
    code.push(vop1e(vop1::V_CVT_F32_U32, 0, v(6)));
    code.extend([vop2e(vop2::V_MUL_F32, 0, LIT, 0), 37.5f32.to_bits()]);
    code.extend([vop2e(vop2::V_ADD_F32, 0, LIT, 0), (-1000.0f32).to_bits()]); // ties above 1024
    code.extend([vop2e(vop2::V_MUL_F32, 1, LIT, 0), 1.1f32.to_bits()]); // arbitrary mantissas
    code.push(vop1e(vop1::V_CVT_F16_F32, 4, v(0))); // v4[15:0], random v4[31:16] kept
    code.push(vop1e(vop1::V_CVT_F16_F32, 5, v(1)));
    code.push(vop2e(vop2::V_CVT_PKRTZ_F16_F32, 2, v(0), 1));
    code.push(vop1e(vop1::V_CVT_F32_F16, 3, v(4))); // back, ignoring v4[31:16]
    code.push(vop1e(vop1::V_CVT_F32_F16, 7, v(2)));
    code.push(ENDPGM);
    check_all(&code);
}

/// SDWA word: src0 | dst_sel << 8 | dst_unused << 11 | src0_sel << 16 |
/// src0_sext << 19 | src0_sgpr << 23 | src1_sel << 24 | src1_sext << 27.
#[allow(clippy::too_many_arguments)]
fn sdwaw(src0: u32, dst: u32, unused: u32, s0: u32, x0: bool, sgpr: bool, s1: u32, x1: bool) -> u32 {
    src0 | dst << 8 | unused << 11 | s0 << 16 | (x0 as u32) << 19 | (sgpr as u32) << 23 | s1 << 24 | (x1 as u32) << 27
}

/// SDWA on integer VOP1 / VOP2: every source selector (with and without
/// sext), every dst_sel with PAD and PRESERVE, SEXT on the low fields, an
/// SGPR src0.
#[test]
fn sdwa_integer() {
    use scarlett_gpu::rdna::sdwa;
    assert_eq!(sdwaw(1, 1, 0, 6, false, false, 6, false), 0x0606_0101, "LLVM v_and_b32_sdwa dst_sel:BYTE_1");
    assert_eq!(sdwa::select(0x8180_7F01, 2, true), Some(0xFFFF_FF80));
    assert_eq!(sdwa::select(0x8180_7F01, 5, false), Some(0x8180));
    assert_eq!(sdwa::place(0x1234_56F0, 0xAAAA_AAAA, 1, 2), Some(0xAAAA_F0AA));
    assert_eq!(sdwa::place(0x1234_56F0, 0, 0, 1), Some(0xFFFF_FFF0));
    assert_eq!(sdwa::place(0x1234_56F0, 0, 1, 1), None);
    const SDWA: u16 = 249;
    // Every result is folded into v2 so none is lost to a later overwrite.
    let fold = |r: u16| vop2e(vop2::V_ADD_NC_U32, 2, v(r), 2);
    let mut code = vec![vop1e(vop1::V_MOV_B32, 2, int(0))];
    for s0 in 0..7u32 {
        for s1 in [0u32, 3, 4, 5, 6] {
            let x = s0 % 2 == 0;
            code.push(vop2e(vop2::V_ADD_NC_U32, 3, SDWA, 1));
            code.push(sdwaw(0, 6, 0, s0, x, false, s1, !x));
            code.push(fold(3));
        }
    }
    for (dst, unused) in [(0u32, 0u32), (1, 0), (2, 2), (3, 2), (4, 1), (5, 2), (0, 1), (4, 0), (5, 0), (6, 0)] {
        code.push(vop2e(vop2::V_XOR_B32, 3, SDWA, 1));
        code.push(sdwaw(0, dst, unused, 1, false, false, 4, false));
        code.push(fold(3));
        code.push(vop1e(vop1::V_NOT_B32, 4, SDWA));
        code.push(sdwaw(1, dst, unused, 5, true, false, 6, false));
        code.push(fold(4));
        code.push(vop1e(vop1::V_MOV_B32, 5, SDWA));
        code.push(sdwaw(2, dst, unused, 3, false, true, 6, false)); // s2 BYTE_3
        code.push(fold(5));
        code.push(vop2e(vop2::V_LSHLREV_B32, 6, SDWA, 7));
        code.push(sdwaw(0, dst, unused, 0, false, false, 4, false));
        code.push(fold(6));
    }
    code.push(ENDPGM);
    check_all(&code);
}

/// 16-bit integer VOP3 on random sources: low halves in, D[15:0] out,
/// D[31:16] of the (random) destination kept.
#[test]
fn integer_16bit_vop3() {
    use scarlett_gpu::interp::vop3_16;
    assert_eq!(vop3_16(vop3::V_ADD_NC_U16, 0x1_FFFF, 0x2_0002), Some(1));
    assert_eq!(vop3_16(vop3::V_ASHRREV_I16, 17, 0x8000), Some(0xC000)); // 17 & 15 = 1
    assert_eq!(vop3_16(vop3::V_MIN_I16, 0xFFFF, 1), Some(0xFFFF));
    assert_eq!(vop3_16(vop3::V_MAX_U16, 0xFFFF, 1), Some(0xFFFF));
    let ops = [
        vop3::V_ADD_NC_U16,
        vop3::V_SUB_NC_U16,
        vop3::V_MUL_LO_U16,
        vop3::V_LSHLREV_B16,
        vop3::V_LSHRREV_B16,
        vop3::V_ASHRREV_I16,
        vop3::V_MAX_U16,
        vop3::V_MAX_I16,
        vop3::V_MIN_U16,
        vop3::V_MIN_I16,
    ];
    for op in ops {
        let mut code = vec![];
        for (d, s) in [(3u16, [v(0), v(1), 0]), (4, [v(1), v(2), 0]), (5, [v(2), v(0), 0]), (6, [int(-3), v(1), 0])] {
            code.extend(vop3e(op, d, s));
        }
        code.push(ENDPGM);
        check_all(&code);
    }
}

/// v_perm_b32 with literal selectors covering every defined selector
/// value (0..6, 0x0C, above 0x0C); 7..=11 refused.
#[test]
fn perm_b32() {
    use scarlett_gpu::interp::vop3_int;
    assert_eq!(vop3_int(vop3::V_PERM_B32, 0xAABB_CCDD, 0x1122_3344, 0x0605_0403), Some(0xBBCC_DD11));
    assert_eq!(vop3_int(vop3::V_PERM_B32, 0xAABB_CCDD, 0x1122_3344, 0x0C0D_0001), Some(0x00FF_4433));
    let mut code = vec![];
    for (d, sel) in [(3u16, 0x0605_0403u32), (4, 0x0C0D_0102), (5, 0x00FF_0E04), (6, 0x0500_0C01)] {
        code.extend(vop3e(vop3::V_PERM_B32, d, [v(0), v(1), LIT]));
        code.push(sel);
    }
    code.push(ENDPGM);
    check_all(&code);
    // 7..11 (NIR byte_perm_amd): byte 7, and sign bits of bytes 1, 3, 5, 7
    assert_eq!(vop3_int(vop3::V_PERM_B32, 0x80BB_00DD, 0x8022_0044, 0x0807_0A0B), Some(0x0080_00FF));
    let mut code = vec![];
    for (d, sel) in [(3u16, 0x0807_0A0Bu32), (4, 0x0908_0B07)] {
        code.extend(vop3e(vop3::V_PERM_B32, d, [v(0), v(1), LIT]));
        code.push(sel);
    }
    code.push(ENDPGM);
    check_all(&code);
}

/// Random straight-line programs over the supported SALU / VALU / VOPC /
/// VOP3 / SDWA / carry ops on random registers: wherever the interpreter
/// accepts a program, the translation must agree bit for bit.
#[test]
fn fuzz_straight_line() {
    let mut x = 0x2545_F491u32;
    let mut rnd = move |n: u32| {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        x % n
    };
    let (mut ran, mut skipped) = (0, 0);
    for prog in 0..160u32 {
        let mut code = vec![];
        for _ in 0..12 {
            let vd = rnd(8) as u16;
            let vs = |r: u32| -> u16 {
                match r % 5 {
                    0 => (r / 5 % 16) as u16,                       // SGPR
                    1 => int((r / 5 % 81) as i32 - 16),              // inline int
                    _ => v((r / 5 % 8) as u16),                      // VGPR
                }
            };
            let (a, b, c) = (vs(rnd(1000)), rnd(8) as u16, vs(rnd(1000)));
            let sd = (rnd(16) & !1) as u16; // even, so pairs fit
            let (s0, s1) = (rnd(16) as u16, rnd(16) as u16);
            match rnd(9) {
                0 => {
                    let ops = [
                        vop2::V_ADD_NC_U32, vop2::V_SUB_NC_U32, vop2::V_SUBREV_NC_U32, vop2::V_AND_B32, vop2::V_OR_B32,
                        vop2::V_XOR_B32, vop2::V_LSHLREV_B32, vop2::V_LSHRREV_B32, vop2::V_ASHRREV_I32, vop2::V_MIN_I32,
                        vop2::V_MAX_I32, vop2::V_MIN_U32, vop2::V_MAX_U32, vop2::V_MUL_U32_U24, vop2::V_CNDMASK_B32,
                        vop2::V_ADD_F32, vop2::V_MUL_F32, vop2::V_SUB_F32,
                    ];
                    code.push(vop2e(ops[rnd(ops.len() as u32) as usize], vd, a, b));
                }
                1 => {
                    let ops = [
                        vop1::V_NOT_B32, vop1::V_BFREV_B32, vop1::V_FFBH_U32, vop1::V_FFBL_B32, vop1::V_MOV_B32,
                        vop1::V_CVT_F32_U32, vop1::V_CVT_F32_I32, vop1::V_FLOOR_F32, vop1::V_CVT_F32_UBYTE2,
                    ];
                    code.push(vop1e(ops[rnd(ops.len() as u32) as usize], vd, a));
                }
                2 => {
                    let ops = [
                        vop3::V_MUL_HI_U32, vop3::V_MUL_HI_I32, vop3::V_BFE_U32, vop3::V_BFE_I32, vop3::V_ALIGNBIT_B32,
                        vop3::V_MED3_I32, vop3::V_MIN3_U32, vop3::V_LSHL_OR_B32, vop3::V_XAD_U32, vop3::V_ADD3_U32,
                        vop3::V_BFI_B32, vop3::V_MAD_U32_U24, vop3::V_ADD_NC_U16, vop3::V_ASHRREV_I16, vop3::V_MAX_I16,
                        vop3::V_LSHLREV_B64,
                    ];
                    let op = ops[rnd(ops.len() as u32) as usize];
                    let d = if op == vop3::V_LSHLREV_B64 { vd & !1 } else { vd };
                    let b2 = if op == vop3::V_LSHLREV_B64 { v(b & !1) } else { v(b) };
                    code.extend(vop3e(op, d, [a, b2, c]));
                }
                3 => {
                    let ops = [
                        sop2::S_AND_B32, sop2::S_OR_B32, sop2::S_XOR_B32, sop2::S_ADD_U32, sop2::S_SUB_U32, sop2::S_ADD_I32,
                        sop2::S_ADDC_U32, sop2::S_SUBB_U32, sop2::S_LSHL_B32, sop2::S_LSHR_B32, sop2::S_ASHR_I32,
                        sop2::S_MUL_I32, sop2::S_CSELECT_B32, sop2::S_NAND_B32, sop2::S_MUL_HI_U32, sop2::S_LSHR_B64,
                    ];
                    let op = ops[rnd(ops.len() as u32) as usize];
                    let (d, x0) = if op == sop2::S_LSHR_B64 { (sd, s0 & !1) } else { (s0, s1) };
                    code.push(sop2e(op, d, x0, rnd(16) as u16));
                }
                4 => {
                    let op = rnd(14) as u16; // every 32-bit SOPC compare and bitcmp
                    code.push(sopce(op, s0, s1));
                    code.push(sop2e(sop2::S_CSELECT_B32, rnd(16) as u16, s0, s1));
                }
                5 => {
                    let ops = [vopc::V_CMP_LT_I32, vopc::V_CMP_EQ_I32, vopc::V_CMP_GT_I32, vopc::V_CMP_NE_I32];
                    code.push(vopce(ops[rnd(4) as usize], a, b));
                    code.push(vop2e(vop2::V_CNDMASK_B32, vd, a, b));
                }
                6 => {
                    // carry chain through vcc, and an SGPR-mask carry
                    let mut w = vop3e(vop3::V_ADD_CO_U32, vd, [a, v(b), 0]);
                    w[0] |= 106 << 8;
                    code.extend(w);
                    code.push(vop2e(vop2::V_ADD_CO_CI_U32, rnd(8) as u16, a, b));
                }
                7 => {
                    // SDWA add with random selectors
                    let s = |r: u32| r % 7;
                    let word = rnd(8) | (rnd(7)) << 8 | [0, 2][rnd(2) as usize] << 11 | s(rnd(7)) << 16 | rnd(2) << 19 | s(rnd(7)) << 24 | rnd(2) << 27;
                    code.push(vop2e(vop2::V_ADD_NC_U32, vd, 249, b));
                    code.push(word);
                }
                _ => {
                    code.push(sop1e(sop1::S_MOV_B32, s0, if a < 255 { a } else { s1 }));
                }
            }
        }
        code.push(ENDPGM);
        for size in [WaveSize::W32, WaveSize::W64] {
            for sparse in [false, true] {
                let seed = prog;
                // Skip what the interpreter refuses (NaN rules, widths...).
                let lanes = size.lanes();
                let (sgprs, vgprs) = inputs(seed, lanes);
                let mut w = Wave::new(size, NV as usize);
                w.lds = vec![0; LDS_WORDS as usize];
                w.sgpr[..NS as usize].copy_from_slice(&sgprs);
                w.vgpr.clone_from(&vgprs);
                w.exec = exec_for(size, sparse);
                w.code_base = CODE_BASE;
                if interp::run(&code, &mut w, 10_000).is_err() {
                    skipped += 1;
                    continue;
                }
                ran += 1;
                check(&code, size, seed, sparse);
            }
        }
    }
    assert!(ran > 3 * skipped, "too many refused programs: ran {ran}, skipped {skipped}");
}

/// Random programs with control flow: uniform if (scc), divergent if/else
/// (saveexec, execz skips, exec restore) and counted scalar loops around
/// random straight-line VALU/SALU bodies; s13 (loop counter) and s[14:15]
/// (saved exec) are reserved. Compared wherever the interpreter accepts.
#[test]
fn fuzz_control_flow() {
    let mut x = 0x1234_5679u32;
    let mut rnd = move |n: u32| {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        x % n
    };
    let (mut ran, mut skipped) = (0, 0);
    for prog in 0..80u32 {
        // a size-independent description, assembled per wave size
        enum Blk {
            Straight(Vec<u32>),
            UIf(u16, u16, Vec<u32>),
            DIf(u16, u16, u16, Vec<u32>, Vec<u32>),
            Loop(u32, Vec<u32>),
        }
        let body = |rnd: &mut dyn FnMut(u32) -> u32, n: u32| {
            let mut c = vec![];
            for _ in 0..n {
                let vd = rnd(7) as u16; // v7 = lane id, reserved
                let a = match rnd(3) { 0 => rnd(16) as u16, 1 => int(rnd(40) as i32 - 8), _ => v(rnd(8) as u16) };
                let b = rnd(8) as u16;
                match rnd(5) {
                    0 | 1 => {
                        let ops = [vop2::V_ADD_NC_U32, vop2::V_XOR_B32, vop2::V_LSHLREV_B32, vop2::V_MAX_I32, vop2::V_SUB_NC_U32, vop2::V_AND_B32];
                        c.push(vop2e(ops[rnd(6) as usize], vd, a, b));
                    }
                    2 => c.extend(vop3e(vop3::V_MUL_HI_U32, vd, [a, v(b), 0])),
                    3 => {
                        let ops = [sop2::S_ADD_U32, sop2::S_XOR_B32, sop2::S_LSHR_B32, sop2::S_MUL_I32];
                        c.push(sop2e(ops[rnd(4) as usize], rnd(12) as u16, rnd(16) as u16, rnd(16) as u16));
                    }
                    _ => c.push(vop1e(vop1::V_NOT_B32, vd, a)),
                }
            }
            c
        };
        let mut blocks = vec![];
        for _ in 0..4 {
            let n = 1 + rnd(4);
            blocks.push(match rnd(4) {
                0 => Blk::Straight(body(&mut rnd, n)),
                1 => Blk::UIf(rnd(16) as u16, rnd(16) as u16, body(&mut rnd, n)),
                2 => {
                    let t = body(&mut rnd, n);
                    let ne = 1 + rnd(3);
                    let e = body(&mut rnd, ne);
                    // random data, or lane-id conditions: lanes >= 32 only
                    // (empty low half in wave64), lanes < 32 only
                    match rnd(3) {
                        0 => Blk::DIf(vopc::V_CMP_LT_I32, v(rnd(8) as u16), rnd(8) as u16, t, e),
                        1 => Blk::DIf(vopc::V_CMP_LT_I32, int(31), 7, t, e), // 31 < lane
                        _ => Blk::DIf(vopc::V_CMP_GT_I32, int(32), 7, t, e), // 32 > lane
                    }
                }
                _ => Blk::Loop(1 + rnd(3), body(&mut rnd, n)),
            });
        }
        for size in [WaveSize::W32, WaveSize::W64] {
            let w64 = size == WaveSize::W64;
            let (mov, andn2, saveexec) = if w64 {
                (sop1::S_MOV_B64, sop2::S_ANDN2_B64, sop1::S_AND_SAVEEXEC_B64)
            } else {
                (sop1::S_MOV_B32, sop2::S_ANDN2_B32, sop1::S_AND_SAVEEXEC_B32)
            };
            // v7 = lane id, computed with every lane on (then exec restored)
            let mut code: Vec<u32> = vec![sop1e(mov, 14, EXEC), sop1e(mov, EXEC, int(-1))];
            code.extend(vop3e(vop3::V_MBCNT_LO_U32_B32, 7, [int(-1), int(0), 0]));
            code.extend(vop3e(vop3::V_MBCNT_HI_U32_B32, 7, [int(-1), v(7), 0]));
            code.push(sop1e(mov, EXEC, 14));
            let br = |code: &mut Vec<u32>, op: u16| {
                code.push(soppe(op, 0));
                code.len() - 1
            };
            let patch = |code: &mut Vec<u32>, at: usize, target: usize| {
                let off = target as i32 - at as i32 - 1;
                code[at] = (code[at] & 0xFFFF_0000) | (off as i16 as u16) as u32;
            };
            for b in &blocks {
                match b {
                    Blk::Straight(c) => code.extend(c),
                    Blk::UIf(s0, s1, c) => {
                        code.push(sopce(sopc::S_CMP_LT_I32, *s0, *s1));
                        let j = br(&mut code, sopp::S_CBRANCH_SCC0);
                        code.extend(c);
                        let end = code.len();
                        patch(&mut code, j, end);
                    }
                    Blk::DIf(cmp, a, vb, t, e) => {
                        code.push(vopce(*cmp, *a, *vb));
                        code.push(sop1e(saveexec, 14, VCC));
                        let j1 = br(&mut code, sopp::S_CBRANCH_EXECZ);
                        code.extend(t);
                        let els = code.len();
                        patch(&mut code, j1, els);
                        code.push(sop2e(andn2, EXEC, 14, EXEC));
                        let j2 = br(&mut code, sopp::S_CBRANCH_EXECZ);
                        code.extend(e);
                        let end = code.len();
                        patch(&mut code, j2, end);
                        code.push(sop1e(mov, EXEC, 14));
                    }
                    Blk::Loop(k, c) => {
                        code.push(sop1e(sop1::S_MOV_B32, 13, int(*k as i32)));
                        let top = code.len();
                        code.extend(c);
                        code.push(sop2e(sop2::S_SUB_I32, 13, 13, int(1)));
                        code.push(sopce(sopc::S_CMP_LG_U32, 13, int(0)));
                        let j = br(&mut code, sopp::S_CBRANCH_SCC1);
                        patch(&mut code, j, top);
                    }
                }
            }
            code.push(ENDPGM);
            for sparse in [false, true] {
                let (sgprs, vgprs) = inputs(prog, size.lanes());
                let mut w = Wave::new(size, NV as usize);
                w.lds = vec![0; LDS_WORDS as usize];
                w.sgpr[..NS as usize].copy_from_slice(&sgprs);
                w.vgpr.clone_from(&vgprs);
                w.exec = exec_for(size, sparse);
                w.code_base = CODE_BASE;
                if interp::run(&code, &mut w, 10_000).is_err() {
                    skipped += 1;
                    continue;
                }
                ran += 1;
                check(&code, size, prog, sparse);
            }
        }
    }
    assert!(ran > 3 * skipped, "ran {ran}, skipped {skipped}");
}

/// Random programs mixing memory and ALU: GLOBAL loads/stores (dword,
/// x2..x4, bytes), MUBUF through a V# in s[8:11] (raw, NUM_RECORDS cutting
/// some lanes), LDS writes/reads, GLOBAL and LDS atomics. Each lane owns a
/// 64-byte slot (v7 = 64 * lane, reserved; s[2:3] = GBASE) so stores never
/// collide; v6 = 16 * lane its LDS words (also reserved). Memory is
/// compared too.
#[test]
fn fuzz_memory() {
    use scarlett_gpu::rdna::mubuf;
    let mut x = 0x0BAD_5EEDu32;
    let mut rnd = move |n: u32| {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        x % n
    };
    let table: Vec<u32> = (0..(64 * 16 + 64) as u32).map(|k| k.wrapping_mul(0x9E37_79B9) ^ 0x8080_8080).collect();
    let flat = |op: u16, vdst: u32, addr: u32, data: u32, offset: u32, glc: bool| {
        [0xDC00_0000 | (op as u32) << 18 | (glc as u32) << 16 | 2 << 14 | offset, vdst << 24 | 2 << 16 | data << 8 | addr]
    };
    let ds = |op: u16, vdst: u32, addr: u32, d0: u32, off: u32| [0xD800_0000 | (op as u32) << 18 | off, vdst << 24 | d0 << 8 | addr];
    let z = int(0) as u32;
    let (mut ran, mut skipped) = (0, 0);
    for prog in 0..60u32 {
        let mut body = vec![];
        for _ in 0..14 {
            let vd = rnd(6); // v6, v7: reserved addresses
            let o = 4 * rnd(13); // within the lane's slot, room for x4
            match rnd(12) {
                0 => body.extend(flat(global::STORE_DWORD, 0, 7, rnd(7), o, false)),
                1 => body.extend(flat([global::STORE_DWORDX2, global::STORE_DWORDX3, global::STORE_DWORDX4][rnd(3) as usize], 0, 7, rnd(4), o, false)),
                2 => body.extend(flat([global::LOAD_DWORD, global::LOAD_DWORDX2, global::LOAD_DWORDX4][rnd(3) as usize], rnd(3), 7, 0, o, false)),
                3 => body.extend(flat([global::LOAD_UBYTE, global::LOAD_SBYTE][rnd(2) as usize], vd, 7, 0, o + rnd(4), false)),
                4 => {
                    let ops = [global::ATOMIC_ADD, global::ATOMIC_XOR, global::ATOMIC_UMAX, global::ATOMIC_SMIN];
                    body.extend(flat(ops[rnd(4) as usize], vd, 7, rnd(7), o, rnd(2) == 0));
                }
                5 => body.extend(mubufe([mubuf::LOAD_DWORD, mubuf::LOAD_DWORDX2, mubuf::STORE_DWORD][rnd(3) as usize], rnd(5), 7, 8, z, o, true, false)),
                6 => body.extend(ds(ds::WRITE_B32, 0, 6, rnd(7), 4 * rnd(4))),
                7 => body.extend(ds(ds::READ_B32, vd, 6, 0, 4 * rnd(8))), // may read the next lane's words
                11 => body.extend(ds(0x20 /* add_rtn */, vd, 6, rnd(7), 4 * rnd(4))),
                8 => body.push(vop2e(vop2::V_ADD_NC_U32, vd as u16, v(rnd(7) as u16), rnd(7) as u16)),
                _ => body.push(vop2e(vop2::V_XOR_B32, vd as u16, int(rnd(60) as i32 - 16), rnd(7) as u16)),
            }
        }
        for size in [WaveSize::W32, WaveSize::W64] {
            let mov = if size == WaveSize::W64 { sop1::S_MOV_B64 } else { sop1::S_MOV_B32 };
            let mut code = vec![sop1e(mov, 12, EXEC), sop1e(mov, EXEC, int(-1))];
            code.extend(vop3e(vop3::V_MBCNT_LO_U32_B32, 7, [int(-1), int(0), 0]));
            code.extend(vop3e(vop3::V_MBCNT_HI_U32_B32, 7, [int(-1), v(7), 0]));
            code.push(vop2e(vop2::V_LSHLREV_B32, 6, int(4), 7)); // v6 = 16 lane: own LDS words
            code.push(vop2e(vop2::V_LSHLREV_B32, 7, int(6), 7)); // v7 = 64 lane
            code.extend([sop1e(sop1::S_MOV_B32, 2, LIT), GBASE as u32]);
            code.extend([sop1e(sop1::S_MOV_B32, 3, LIT), (GBASE >> 32) as u32]);
            code.extend([sop1e(sop1::S_MOV_B32, 8, LIT), GBASE as u32]);
            code.extend([sop1e(sop1::S_MOV_B32, 9, LIT), (GBASE >> 32) as u32]);
            code.extend([sop1e(sop1::S_MOV_B32, 10, LIT), 64 * 40]); // lanes >= 40 out of bounds
            code.extend([sop1e(sop1::S_MOV_B32, 11, LIT), 3 << 28]);
            code.push(sop1e(mov, EXEC, 12));
            code.extend(&body);
            code.push(soppe(sopp::S_WAITCNT, 0));
            code.push(ENDPGM);
            for sparse in [false, true] {
                let (sgprs, vgprs) = inputs(prog, size.lanes());
                let mut w = Wave::new(size, NV as usize);
                w.lds = vec![0; LDS_WORDS as usize];
                w.sgpr[..NS as usize].copy_from_slice(&sgprs);
                w.vgpr.clone_from(&vgprs);
                w.exec = exec_for(size, sparse);
                w.code_base = CODE_BASE;
                let mut memory = SparseMemory::new();
                memory.map(GBASE, 4 * table.len() as u64);
                assert!(memory.write(GBASE, &table));
                if let Err(e) = interp::run_with_memory(&code, &mut w, &mut memory, 10_000) {
                    if std::env::var("FUZZ_DEBUG").is_ok() {
                        eprintln!("refused: {e:?}");
                    }
                    skipped += 1;
                    continue;
                }
                ran += 1;
                check_mem(&code, size, prog, sparse, Some((GBASE, &table)));
            }
        }
    }
    assert!(ran > 3 * skipped, "ran {ran}, skipped {skipped}");
}

/// ds_swizzle_b32 in QUAD_PERM and BITMASK modes (SWAP 16, REVERSE 32,
/// BROADCAST 8 lane 3, "01pi0"), full exec (an inactive source lane is
/// refused); plus the alignbyte / msad formulas by hand.
#[test]
fn ds_swizzle_and_byte_ops() {
    use scarlett_gpu::interp::vop3_int;
    use scarlett_gpu::rdna::ds::swizzle_source;
    assert_eq!(vop3_int(vop3::V_ALIGNBYTE_B32, 0x1122_3344, 0x5566_7788, 1), Some(0x4455_6677));
    assert_eq!(vop3_int(vop3::V_MSAD_U8, 0x1000_0510, 0x0000_0805, 100), Some(100 + 3 + 11));
    assert_eq!(vop3_int(vop3::V_CVT_PK_U16_U32, 0x1_0000, 7, 0), Some(0x0007_FFFF));
    assert_eq!(vop3_int(vop3::V_CVT_PK_I16_I32, (-40000i32) as u32, 40000, 0), Some(0x7FFF_8000));
    // encodings as the LLVM assembler builds them
    let bitmask = |and: u32, or: u32, xor: u32| and | or << 5 | xor << 10;
    let swap16 = bitmask(31, 0, 16);
    let rev32 = bitmask(31, 0, 31);
    let bcast8_3 = bitmask(31 - 8 + 1, 3, 0);
    let pat = bitmask(0b00110, 0b01000, 0b00010); // "01pi0"
    let quad = 0x8000 | 3 | 2 << 2 | 1 << 4; // (3, 2, 1, 0)
    assert_eq!(swizzle_source(swap16, 3), Some(19));
    assert_eq!(swizzle_source(rev32, 33), Some(62));
    assert_eq!(swizzle_source(bcast8_3, 13), Some(11));
    assert_eq!(swizzle_source(pat, 0b10111), Some(0b01100));
    assert_eq!(swizzle_source(quad, 6), Some(5));
    assert_eq!(swizzle_source(0xE000, 0), None); // FFT
    let ds = |vdst: u32, addr: u32, off: u32| [0xD800_0000 | (53u32) << 18 | off, vdst << 24 | addr];
    let mut code = vec![];
    // every swizzle reads v0 or v1 (per-lane distinct inputs)
    for (d, off) in [(2u32, swap16), (3, rev32), (4, bcast8_3), (5, pat), (6, quad)] {
        code.extend(ds(d, d % 2, off));
    }
    code.push(ENDPGM);
    for size in [WaveSize::W32, WaveSize::W64] {
        for seed in 0..3 {
            check(&code, size, seed, false);
        }
    }
    // an inactive source lane: refused
    let mut w = Wave::new(WaveSize::W32, NV as usize);
    w.exec = 1;
    let mut c = ds(2, 1, swap16).to_vec();
    c.push(ENDPGM);
    assert!(interp::run(&c, &mut w, 10).is_err());
}

/// f64: per-lane doubles built from the random v0/v1 (exponents 1000..1127
/// in A = v[2:3], so |x| < 1, fractional and integral cases; B = v[4:5]
/// with four mantissa bits and exponents 1022..1025, so exact ties), the
/// four roundings into v[0:1] / v[6:7], and the 16 compares of A, B, a NaN
/// and ±0 packed bit by bit into v0 / v7 by v_add_co_ci_u32.
#[test]
fn f64_rounding_and_compares() {
    use scarlett_gpu::interp::{round_f64, Compare};
    let d = |x: f64| x.to_bits();
    let r = |op, x: f64| round_f64(op, d(x)).map(f64::from_bits);
    assert_eq!(r(vop1::V_TRUNC_F64, -2.5), Some(-2.0));
    assert_eq!(r(vop1::V_FLOOR_F64, -2.5), Some(-3.0));
    assert_eq!(r(vop1::V_CEIL_F64, -2.5), Some(-2.0));
    assert_eq!(r(vop1::V_RNDNE_F64, 2.5), Some(2.0));
    assert_eq!(r(vop1::V_RNDNE_F64, 3.5), Some(4.0));
    assert_eq!(r(vop1::V_CEIL_F64, 0.3), Some(1.0));
    assert_eq!(r(vop1::V_RNDNE_F64, 0.7), Some(1.0));
    assert_eq!(round_f64(vop1::V_RNDNE_F64, d(-0.5)), Some(d(-0.0)));
    assert_eq!(round_f64(vop1::V_CEIL_F64, d(-0.5)), Some(d(-0.0)));
    assert_eq!(r(vop1::V_RNDNE_F64, 4503599627370495.5), Some(4503599627370496.0));
    assert_eq!(r(vop1::V_FLOOR_F64, f64::NEG_INFINITY), Some(f64::NEG_INFINITY));
    assert_eq!(round_f64(vop1::V_FLOOR_F64, d(f64::NAN)), None);
    assert_eq!(round_f64(vop1::V_CEIL_F64, 1), None, "subnormal");
    let cmp = |op: u16, a: f64, b: f64| Compare::decode(op).unwrap().eval64(d(a), d(b));
    assert!(cmp(0x22, -0.0, 0.0) && !cmp(0x21, -0.0, 0.0), "eq/lt of -0, +0");
    assert!(cmp(0x21, -1e300, 1e-300) && cmp(0x24, -1e-300, -1e300));
    assert!(!cmp(0x22, f64::NAN, f64::NAN) && cmp(0x28, f64::NAN, 1.0) && cmp(0x2D, f64::NAN, f64::NAN));
    assert!(Compare::decode(0x30).unwrap().cmpx && Compare::decode(0x3F).is_some());
    assert_eq!(vopce(0x20, v(1), 2), 0x7C40_0501, "LLVM v_cmp_f_f64 vcc_lo, v[1:2], v[2:3]");
    assert_eq!(vop1e(vop1::V_TRUNC_F64, 5, v(1)), 0x7E0A_2F01);

    const L: u16 = 255;
    let mut setup = vec![
        vop1e(vop1::V_MOV_B32, 6, v(0)),
        vop2e(vop2::V_AND_B32, 3, L, 0), 0x07F0_0000,
        vop2e(vop2::V_ADD_NC_U32, 3, L, 3), 1000 << 20,
        vop2e(vop2::V_AND_B32, 7, L, 0), 0x800F_FFFF,
        vop2e(vop2::V_OR_B32, 3, v(3), 7),
        vop1e(vop1::V_MOV_B32, 2, v(1)),
        vop2e(vop2::V_AND_B32, 5, L, 0), 0x0030_0000,
        vop2e(vop2::V_ADD_NC_U32, 5, L, 5), 1022 << 20,
        vop2e(vop2::V_AND_B32, 7, L, 0), 0x800F_0000,
        vop2e(vop2::V_OR_B32, 5, v(5), 7),
        vop1e(vop1::V_MOV_B32, 4, int(0)),
    ];
    for op in [vop1::V_TRUNC_F64, vop1::V_CEIL_F64, vop1::V_RNDNE_F64, vop1::V_FLOOR_F64] {
        let mut code = setup.clone();
        code.push(vop1e(op, 0, v(2)));
        code.push(vop1e(op, 6, v(4)));
        code.push(vop1e(op, 4, 241)); // inline -0.5 as a double
        code.extend([sop1e(sop1::S_MOV_B32, 6, int(0)), sop1e(sop1::S_MOV_B32, 7, L), 0x8000_0000]);
        code.push(vop1e(op, 2, 6)); // -0.0
        code.push(ENDPGM);
        check_all(&code);
    }
    // s[4:5] = NaN, s[6:7] = -0.0
    setup.extend([
        sop1e(sop1::S_MOV_B32, 4, int(0)),
        sop1e(sop1::S_MOV_B32, 5, L), 0x7FF8_0000,
        sop1e(sop1::S_MOV_B32, 6, int(0)),
        sop1e(sop1::S_MOV_B32, 7, L), 0x8000_0000,
        vop1e(vop1::V_MOV_B32, 0, int(0)),
        vop1e(vop1::V_MOV_B32, 7, int(0)),
    ]);
    let acc = |code: &mut Vec<u32>, r: u16| code.push(vop2e(vop2::V_ADD_CO_CI_U32, r, v(r), r));
    let mut code = setup;
    for cond in 0..16u16 {
        code.push(vopce(0x20 + cond, v(2), 4));
        acc(&mut code, 0);
        code.push(vopce(0x20 + cond, v(4), 2));
        acc(&mut code, 7);
    }
    code.extend([vop2e(vop2::V_AND_B32, 5, L, 6), 0x8000_0000]);
    for cond in 0..16u16 {
        code.push(vopce(0x20 + cond, 4, 2)); // NaN vs A
        acc(&mut code, 7);
        code.push(vopce(0x20 + cond, int(0), 4)); // +0 vs ±0
        acc(&mut code, 0);
        code.push(vopce(0x20 + cond, 6, 4)); // -0 vs ±0
        acc(&mut code, 1);
    }
    // same high word, different low word (signs from r)
    code.extend([vop1e(vop1::V_MOV_B32, 4, v(6)), vop1e(vop1::V_MOV_B32, 5, v(3))]);
    for cond in 0..16u16 {
        code.push(vopce(0x20 + cond, v(2), 4));
        acc(&mut code, 1);
    }
    // +Inf in s[8:9] against A
    code.extend([
        sop1e(sop1::S_MOV_B32, 8, int(0)),
        sop1e(sop1::S_MOV_B32, 9, L), 0x7FF0_0000,
        vop1e(vop1::V_MOV_B32, 6, int(0)),
    ]);
    for cond in 0..16u16 {
        code.push(vopce(0x20 + cond, 8, 2));
        acc(&mut code, 6);
    }
    code.push(vopce(0x30 + 4, v(2), 4)); // v_cmpx_gt_f64
    code.push(vop1e(vop1::V_MOV_B32, 5, int(3)));
    code.push(ENDPGM);
    for (seed, sparse) in [(0, false), (1, true)] {
        check(&code, WaveSize::W32, seed, sparse);
        check(&code, WaveSize::W64, seed, sparse);
    }
}

/// v_add_f64 / v_mul_f64 / v_fma_f64 on the doubles of
/// f64_rounding_and_compares, with abs/neg, SGPR-pair and inline sources.
#[test]
fn f64_arithmetic() {
    let f64op = |op: u16, vdst: u16, s: [u16; 3], abs: u32, neg: u32| {
        let mut w = vop3e(op, vdst, s);
        w[0] |= abs << 8;
        w[1] |= neg << 29;
        w
    };
    assert_eq!(f64op(vop3::V_ADD_F64, 5, [v(1), v(2), 0], 0, 0), [0xD564_0005, 0x0002_0501]);
    assert_eq!(f64op(vop3::V_FMA_F64, 5, [v(1), v(2), v(3)], 0, 0), [0xD54C_0005, 0x040E_0501]);
    // hand checks through the interpreter: v[2:3] op v[4:5] (op v[6:7]) -> v[0:1]
    let run = |op: u16, x: [f64; 3]| {
        let mut w = Wave::new(WaveSize::W32, NV as usize);
        w.exec = 1;
        for (k, d) in x.iter().enumerate() {
            let b = d.to_bits();
            w.vgpr[2 + 2 * k][0] = b as u32;
            w.vgpr[3 + 2 * k][0] = (b >> 32) as u32;
        }
        let mut c = f64op(op, 0, [v(2), v(4), v(6)], 0, 0).to_vec();
        c.push(ENDPGM);
        interp::run(&c, &mut w, 10).map(|_| f64::from_bits(w.vgpr[0][0] as u64 | (w.vgpr[1][0] as u64) << 32))
    };
    let e = f64::EPSILON;
    assert_eq!(run(vop3::V_ADD_F64, [1.0, e / 2.0, 0.0]).unwrap(), 1.0, "tie to even");
    assert_eq!(run(vop3::V_ADD_F64, [1.0, e / 2.0 + e / 4.0, 0.0]).unwrap(), 1.0 + e);
    assert_eq!(run(vop3::V_FMA_F64, [1.0 + e, 1.0 - e, -1.0]).unwrap(), -e * e, "one rounding");
    assert_eq!(run(vop3::V_MUL_F64, [1e200, 1e200, 0.0]).unwrap(), f64::INFINITY);
    assert!(run(vop3::V_ADD_F64, [f64::INFINITY, f64::NEG_INFINITY, 0.0]).is_err(), "NaN result");
    assert!(run(vop3::V_MUL_F64, [f64::NAN, 1.0, 0.0]).is_err(), "NaN operand");
    assert!(run(vop3::V_MUL_F64, [1e-300, 1e-20, 0.0]).is_err(), "subnormal result");

    const L: u16 = 255;
    let setup = [
        vop2e(vop2::V_AND_B32, 3, L, 0), 0x07F0_0000,
        vop2e(vop2::V_ADD_NC_U32, 3, L, 3), 1000 << 20,
        vop2e(vop2::V_AND_B32, 7, L, 0), 0x800F_FFFF,
        vop2e(vop2::V_OR_B32, 3, v(3), 7),
        vop1e(vop1::V_MOV_B32, 2, v(1)),
        vop2e(vop2::V_AND_B32, 5, L, 0), 0x0030_0000,
        vop2e(vop2::V_ADD_NC_U32, 5, L, 5), 1022 << 20,
        vop2e(vop2::V_AND_B32, 7, L, 0), 0x800F_0000,
        vop2e(vop2::V_OR_B32, 5, v(5), 7),
        vop1e(vop1::V_MOV_B32, 4, v(0)),
        sop1e(sop1::S_MOV_B32, 6, L), 0x5555_5555,
        sop1e(sop1::S_MOV_B32, 7, L), 0xC009_2492, // about -3.14
    ];
    for op in [vop3::V_ADD_F64, vop3::V_MUL_F64, vop3::V_FMA_F64] {
        let mut code = setup.to_vec();
        // first result in v[0:1]; later ones in v[6:7], folded into v[0:1]
        code.extend(f64op(op, 0, [v(2), v(4), v(4)], 0, 0));
        let fold = [vop2e(vop2::V_XOR_B32, 0, v(6), 0), vop2e(vop2::V_XOR_B32, 1, v(7), 1)];
        code.extend(f64op(op, 6, [v(2), v(4), v(2)], 0b010, 0b101)); // -A, |B|, -A
        code.extend(fold);
        code.extend(f64op(op, 6, [v(2), v(2), v(2)], 0, 0b010)); // A + -A = +0 for add
        code.extend(fold);
        code.extend(f64op(op, 6, [6, v(4), 240], 0b001, 0b100)); // |s[6:7]|, B, -0.5
        code.extend(fold);
        code.extend(f64op(op, 4, [v(4), 240, 6], 0, 0)); // B, 0.5, s[6:7]
        code.push(ENDPGM);
        check_all(&code);
    }
}

/// v_xor3_b32, v_mad_u64_u32 (carry into an SGPR mask and VCC) and
/// v_min/max_f64 (quiet NaN, ±0, equal operands, modifiers).
#[test]
fn xor3_mad_u64_minmax_f64() {
    use scarlett_gpu::interp::{min_max_f64, vop3_int};
    assert_eq!(vop3_int(vop3::V_XOR3_B32, 0xF0F0, 0x0FF0, 0x1234), Some(0xF0F0 ^ 0x0FF0 ^ 0x1234));
    assert_eq!(vop3e(vop3::V_XOR3_B32, 5, [v(1), v(2), v(3)]), [0xD578_0005, 0x040E_0501]);
    let mut mad = vop3e(vop3::V_MAD_U64_U32, 5, [v(1), v(2), v(3)]);
    mad[0] |= 12 << 8;
    assert_eq!(mad, [0xD576_0C05, 0x040E_0501]);
    let d = |x: f64| x.to_bits();
    let qnan = 0x7FF8_0000_0000_0001u64;
    assert_eq!(min_max_f64(false, d(-0.0), d(0.0)), Some(d(-0.0)));
    assert_eq!(min_max_f64(true, d(-0.0), d(0.0)), Some(d(0.0)));
    assert_eq!(min_max_f64(false, qnan, d(2.0)), Some(d(2.0)));
    assert_eq!(min_max_f64(true, d(-3.0), qnan), Some(d(-3.0)));
    assert_eq!(min_max_f64(false, 0x7FF0_0000_0000_0001, d(1.0)), None, "signalling NaN");
    // mad: all ones * all ones + all ones = 2^65 - 2^33: low 0xFFFFFFFE_00000000, carry 1
    let mut w = Wave::new(WaveSize::W32, NV as usize);
    w.exec = 1;
    for r in 1..=4 {
        w.vgpr[r][0] = u32::MAX;
    }
    let mut c = mad.to_vec();
    c.push(ENDPGM);
    interp::run(&c, &mut w, 10).unwrap();
    assert_eq!((w.vgpr[5][0], w.vgpr[6][0], w.sgpr[12]), (0, 0xFFFF_FFFE, 1));

    let with_sdst = |mut x: [u32; 2], sdst: u32| {
        x[0] |= sdst << 8;
        x
    };
    let mods = |mut x: [u32; 2], abs: u32, neg: u32| {
        x[0] |= abs << 8;
        x[1] |= neg << 29;
        x
    };
    const L: u16 = 255;
    let mut code = vec![];
    code.extend(vop3e(vop3::V_XOR3_B32, 7, [v(0), v(1), v(2)]));
    code.extend(with_sdst(vop3e(vop3::V_MAD_U64_U32, 4, [v(0), v(1), v(2)]), 10));
    code.extend(with_sdst(vop3e(vop3::V_MAD_U64_U32, 2, [v(1), 3, v(4)]), VCC as u32));
    code.extend(with_sdst(vop3e(vop3::V_MAD_U64_U32, 0, [v(0), int(-1), int(-1)]), 12));
    // 1 * v0 + 2^64 - 1: the carry comes from the low-word carry alone
    code.extend(with_sdst(vop3e(vop3::V_MAD_U64_U32, 6, [int(1), v(0), int(-1)]), 14));
    code.push(ENDPGM);
    check_all(&code);

    let setup = [
        vop2e(vop2::V_AND_B32, 3, L, 0), 0x07F0_0000,
        vop2e(vop2::V_ADD_NC_U32, 3, L, 3), 1000 << 20,
        vop2e(vop2::V_AND_B32, 7, L, 0), 0x800F_FFFF,
        vop2e(vop2::V_OR_B32, 3, v(3), 7),
        vop1e(vop1::V_MOV_B32, 2, v(1)),
        vop2e(vop2::V_AND_B32, 5, L, 0), 0x8030_0000,
        vop2e(vop2::V_ADD_NC_U32, 5, L, 5), 1022 << 20,
        vop1e(vop1::V_MOV_B32, 4, int(0)),
        sop1e(sop1::S_MOV_B32, 4, int(1)),
        sop1e(sop1::S_MOV_B32, 5, L), 0x7FF8_0000, // quiet NaN
        sop1e(sop1::S_MOV_B32, 6, int(0)),
        sop1e(sop1::S_MOV_B32, 7, L), 0x8000_0000, // -0
    ];
    let fold = [vop2e(vop2::V_XOR_B32, 0, v(6), 0), vop2e(vop2::V_XOR_B32, 1, v(7), 1)];
    for op in [vop3::V_MIN_F64, vop3::V_MAX_F64] {
        let mut code = setup.to_vec();
        code.extend(vop3e(op, 0, [v(2), v(4), 0]));
        for (s0, s1, abs, neg) in [
            (v(4), v(2), 0, 0),
            (v(2), v(2), 0, 0b10), // A vs -A
            (v(2), v(2), 0b01, 0), // |A| vs A
            (4, v(2), 0, 0),       // NaN vs A
            (v(4), 4, 0, 0),       // B vs NaN
            (6, int(0), 0, 0),     // -0 vs +0
            (int(0), 6, 0, 0),
            (v(2), v(2), 0, 0),
        ] {
            code.extend(mods(vop3e(op, 6, [s0, s1, 0]), abs, neg));
            code.extend(fold);
        }
        code.push(ENDPGM);
        check_all(&code);
    }
}

/// v_ffbh_i32 / s_flbit_i32 (NIR ifind_msb_rev) and v_bfm_b32 (NIR bfm).
#[test]
fn ffbh_i32_and_bfm() {
    use scarlett_gpu::interp::{vop1_int, vop3_int};
    for (x, want) in [(0u32, u32::MAX), (u32::MAX, u32::MAX), (1, 31), (0x4000_0000, 1), (0xFFFF_FFFE, 31), (0x8000_0000, 1), (0xC000_0000, 2)] {
        assert_eq!(vop1_int(vop1::V_FFBH_I32, x), Some(want), "{x:#x}");
    }
    assert_eq!(vop1e(vop1::V_FFBH_I32, 5, v(1)), 0x7E0A_7701);
    assert_eq!(vop3e(vop3::V_BFM_B32, 5, [v(1), v(2), int(0)]), [0xD763_0005, 0x0202_0501]);
    assert_eq!(vop3_int(vop3::V_BFM_B32, 5, 3, 0), Some(0xF8));
    assert_eq!(vop3_int(vop3::V_BFM_B32, 31 + 32, 1, 0), Some(0xFFFF_FFFE));
    assert_eq!(vop3_int(vop3::V_BFM_B32, 0, 7, 0), Some(0));
    let mut code = vec![
        vop1e(vop1::V_FFBH_I32, 2, v(0)),
        vop1e(vop1::V_FFBH_I32, 3, int(-1)),
        vop1e(vop1::V_FFBH_I32, 4, int(0)),
        vop2e(vop2::V_LSHRREV_B32, 5, int(9), 1),
        vop1e(vop1::V_FFBH_I32, 5, v(5)),
        sop1e(sop1::S_FLBIT_I32, 6, 1),
        sop1e(sop1::S_FLBIT_I32, 7, int(-5)),
        sop1e(sop1::S_FLBIT_I32, 8, int(0)),
    ];
    code.extend(vop3e(vop3::V_BFM_B32, 6, [v(0), v(1), 0]));
    code.extend(vop3e(vop3::V_BFM_B32, 7, [2, v(1), 0]));
    code.push(ENDPGM);
    check_all(&code);
}

/// v_min3/max3/med3_f32 (quiet NaN, ±0, ties, modifiers) and
/// v_add_lshl_u32.
#[test]
fn min3_max3_med3_f32_and_add_lshl() {
    use scarlett_gpu::interp::{med3_f32, vop3_int};
    let f = |x: f32| x.to_bits();
    assert_eq!(med3_f32(f(3.0), f(1.0), f(2.0)), Some(f(2.0)));
    assert_eq!(med3_f32(f(2.0), f(2.0), f(1.0)), Some(f(2.0)));
    assert_eq!(med3_f32(f(-0.0), f(0.0), f(0.0)), Some(f(0.0)));
    assert_eq!(med3_f32(f(0.0), f(-0.0), f(-1.0)), Some(f(-0.0)));
    assert_eq!(med3_f32(f(f32::NAN), f(0.0), f(1.0)), None);
    assert_eq!(vop3_int(vop3::V_ADD_LSHL_U32, 0xFFFF_FFFF, 2, 33), Some(2));
    assert_eq!(vop3e(vop3::V_MIN3_F32, 5, [v(1), v(2), v(3)]), [0xD551_0005, 0x040E_0501]);
    assert_eq!(vop3e(vop3::V_ADD_LSHL_U32, 5, [v(1), v(2), v(3)]), [0xD747_0005, 0x040E_0501]);
    let mods = |mut x: [u32; 2], abs: u32, neg: u32| {
        x[0] |= abs << 8;
        x[1] |= neg << 29;
        x
    };
    const L: u16 = 255;
    // small floats from the random v0/v1: +-{0, 0.5, 1, ..} so ties and
    // signed zeros come up; s4 = quiet NaN, s5 = -0
    let mut base = vec![
        vop2e(vop2::V_AND_B32, 2, L, 0), 0x8000_0003,
        vop1e(vop1::V_CVT_F32_I32, 2, v(2)),
        vop2e(vop2::V_AND_B32, 3, L, 1), 0x8000_0003,
        vop1e(vop1::V_CVT_F32_I32, 3, v(3)),
        vop2e(vop2::V_LSHRREV_B32, 4, int(8), 0),
        vop2e(vop2::V_AND_B32, 4, L, 4), 0x8000_0001,
        vop1e(vop1::V_CVT_F32_I32, 4, v(4)),
        sop1e(sop1::S_MOV_B32, 4, L), 0x7FC0_0001,
        sop1e(sop1::S_MOV_B32, 5, L), 0x8000_0000,
    ];
    base.extend(vop3e(vop3::V_ADD_LSHL_U32, 7, [v(0), v(1), 2]));
    for op in [vop3::V_MIN3_F32, vop3::V_MAX3_F32, vop3::V_MED3_F32] {
        let mut code = base.clone();
        code.extend(vop3e(op, 0, [v(2), v(3), v(4)]));
        code.extend(mods(vop3e(op, 1, [v(3), v(2), v(4)]), 0b001, 0b100));
        code.extend(vop3e(op, 5, [5, int(0), v(2)]));
        code.extend(vop3e(op, 6, [v(4), 5, v(3)]));
        if op != vop3::V_MED3_F32 {
            code.extend(vop3e(op, 2, [4, v(3), v(4)])); // quiet NaN first
            code.extend(vop3e(op, 3, [v(2), v(3), 4])); // quiet NaN last
        }
        code.push(ENDPGM);
        check_all(&code);
    }
}

/// f64 <-> i32 / u32 / f32 conversions: saturation and NaN -> 0 for the
/// integer results, round to nearest even and overflow to Inf for f32.
#[test]
fn f64_conversions() {
    use scarlett_gpu::interp::cvt_from_f64;
    let i = |x: f64| cvt_from_f64(vop1::V_CVT_I32_F64, x).unwrap() as i32;
    let u = |x: f64| cvt_from_f64(vop1::V_CVT_U32_F64, x).unwrap();
    let f = |x: f64| cvt_from_f64(vop1::V_CVT_F32_F64, x).map(f32::from_bits);
    assert_eq!((i(3.99), i(-3.99), i(1e10), i(-1e10), i(f64::NAN)), (3, -3, i32::MAX, i32::MIN, 0));
    assert_eq!((u(-0.5), u(4294967295.9), u(1e300), u(f64::NAN)), (0, u32::MAX, u32::MAX, 0));
    assert_eq!(f(1.0 + f64::from(f32::EPSILON) / 2.0), Some(1.0), "tie to even");
    assert_eq!(f(1e39), Some(f32::INFINITY));
    assert_eq!(f(1e-39), None, "f32 subnormal");
    assert_eq!(f(f64::NAN), None);
    assert_eq!(vop1e(vop1::V_CVT_F32_F64, 5, v(1)), 0x7E0A_1F01);
    assert_eq!(vop1e(vop1::V_CVT_F64_U32, 5, v(1)), 0x7E0A_2D01);

    const L: u16 = 255;
    // A = v[2:3] with exponents 1000..1127, B = v[4:5] with 1100..1227
    let mut code = vec![
        vop2e(vop2::V_AND_B32, 3, L, 0), 0x07F0_0000,
        vop2e(vop2::V_ADD_NC_U32, 3, L, 3), 1000 << 20,
        vop2e(vop2::V_AND_B32, 7, L, 0), 0x800F_FFFF,
        vop2e(vop2::V_OR_B32, 3, v(3), 7),
        vop1e(vop1::V_MOV_B32, 2, v(1)),
        vop2e(vop2::V_ADD_NC_U32, 5, L, 3), 100 << 20,
        vop1e(vop1::V_MOV_B32, 4, v(0)),
        sop1e(sop1::S_MOV_B32, 4, int(0)),
        sop1e(sop1::S_MOV_B32, 5, L), 0xFFF8_0000, // NaN
    ];
    code.extend([
        vop1e(vop1::V_CVT_I32_F64, 0, v(2)),
        vop1e(vop1::V_CVT_U32_F64, 1, v(2)),
        vop1e(vop1::V_CVT_F32_F64, 6, v(2)),
        vop1e(vop1::V_CVT_I32_F64, 7, v(4)),
        vop1e(vop1::V_CVT_U32_F64, 2, v(4)),
        vop1e(vop1::V_CVT_F32_F64, 3, v(4)),
        vop1e(vop1::V_CVT_I32_F64, 4, 4),
        vop1e(vop1::V_CVT_U32_F64, 5, 241), // -0.5
        ENDPGM,
    ]);
    check_all(&code);
    let code = [
        vop1e(vop1::V_CVT_F64_I32, 2, v(0)),
        vop1e(vop1::V_CVT_F64_U32, 4, v(1)),
        vop1e(vop1::V_CVT_F32_I32, 6, v(0)),
        vop1e(vop1::V_CVT_F64_F32, 6, v(6)),
        vop1e(vop1::V_CVT_F64_I32, 0, int(-7)),
        ENDPGM,
    ];
    check_all(&code);    // exact range ends from SGPR pairs
    let mut code = vec![];
    for (k, d) in [2_147_483_648.0f64, -2_147_483_648.0, 4_294_967_296.0, 2_147_483_647.5].iter().enumerate() {
        let b = d.to_bits();
        code.extend([sop1e(sop1::S_MOV_B32, 4 + 2 * k as u16, L), b as u32, sop1e(sop1::S_MOV_B32, 5 + 2 * k as u16, L), (b >> 32) as u32]);
    }
    code.extend([
        vop1e(vop1::V_CVT_I32_F64, 0, 4),
        vop1e(vop1::V_CVT_I32_F64, 1, 6),
        vop1e(vop1::V_CVT_U32_F64, 2, 8),
        vop1e(vop1::V_CVT_I32_F64, 3, 10),
        vop1e(vop1::V_CVT_U32_F64, 4, 4),
        ENDPGM,
    ]);
    check_all(&code);
}

/// v_cvt_rpi/flr_i32_f32, v_frexp_exp/mant and v_ldexp_f32 on floats
/// with exponents 110..173 built from v0, ties, zeros and Inf.
#[test]
fn f32_rpi_flr_frexp_ldexp() {
    use scarlett_gpu::interp::{ldexp_f32, vop1_f32_special as sp};
    let f = |x: f32| x.to_bits();
    let i = |op, x: f32| sp(op, f(x)).map(|r| r as i32);
    assert_eq!(i(vop1::V_CVT_RPI_I32_F32, 2.5), Some(3));
    assert_eq!(i(vop1::V_CVT_RPI_I32_F32, -2.5), Some(-2));
    assert_eq!(i(vop1::V_CVT_RPI_I32_F32, 0.49999997), Some(1), "x + 0.5 rounds to 1.0");
    assert_eq!(i(vop1::V_CVT_FLR_I32_F32, -0.1), Some(-1));
    assert_eq!(i(vop1::V_CVT_FLR_I32_F32, 3e9), None, "out of range");
    assert_eq!(i(vop1::V_CVT_FLR_I32_F32, f32::NAN), None);
    assert_eq!(i(vop1::V_FREXP_EXP_I32_F32, 8.0), Some(4));
    assert_eq!(sp(vop1::V_FREXP_MANT_F32, f(-12.0)), Some(f(-0.75)));
    assert_eq!(i(vop1::V_FREXP_EXP_I32_F32, f32::INFINITY), Some(0));
    assert_eq!(i(vop1::V_FREXP_EXP_I32_F32, f32::NAN), Some(0));
    assert_eq!(sp(vop1::V_FREXP_MANT_F32, f(f32::NEG_INFINITY)), Some(f(f32::NEG_INFINITY)));
    assert_eq!(sp(vop1::V_FREXP_MANT_F32, f(f32::NAN)), None);
    assert_eq!(ldexp_f32(f(1.5), 3), Some(f(12.0)));
    assert_eq!(ldexp_f32(f(-1.0), 200), Some(f(f32::NEG_INFINITY)));
    assert_eq!(ldexp_f32(f(1.0), -127), None, "subnormal result");
    assert_eq!(ldexp_f32(f(-0.0), 5), Some(f(-0.0)));
    assert_eq!(vop1e(vop1::V_FREXP_MANT_F32, 5, v(1)), 0x7E0A_8101);
    assert_eq!(vop3e(vop3::V_LDEXP_F32, 5, [v(1), v(2), int(0)]), [0xD762_0005, 0x0202_0501]);

    const L: u16 = 255;
    let mut code = vec![
        // v2 = float with exponent 110 + (v0 >> 8 & 63)
        vop2e(vop2::V_LSHRREV_B32, 2, int(8), 0),
        vop2e(vop2::V_AND_B32, 2, int(63), 2),
        vop2e(vop2::V_ADD_NC_U32, 2, L, 2), 110,
        vop2e(vop2::V_LSHLREV_B32, 2, int(23), 2),
        vop2e(vop2::V_AND_B32, 3, L, 0), 0x807F_FFFF,
        vop2e(vop2::V_OR_B32, 2, v(2), 3),
        // v3 = n in -20..107
        vop2e(vop2::V_AND_B32, 3, L, 1), 0x7F,
        vop2e(vop2::V_SUBREV_NC_U32, 3, L, 3), 20,
        sop1e(sop1::S_MOV_B32, 4, L), 0x7F80_0000, // +Inf
        sop1e(sop1::S_MOV_B32, 5, L), 0x8000_0000, // -0
    ];
    code.extend(vop3e(vop3::V_LDEXP_F32, 4, [v(2), v(3), int(0)]));
    code.extend(vop3e(vop3::V_LDEXP_F32, 5, [4, v(3), int(0)]));
    code.extend(vop3e(vop3::V_LDEXP_F32, 6, [7, 6, int(0)])); // 1.5 * 2^128 = Inf
    code.splice(0..0, [sop1e(sop1::S_MOV_B32, 6, L), 128, sop1e(sop1::S_MOV_B32, 7, L), 0x3FC0_0000]);
    code.extend(vop3e(vop3::V_LDEXP_F32, 7, [v(2), L, int(0)]));
    code.push(0x7FFF_FFFF);
    code.extend([
        vop1e(vop1::V_FREXP_EXP_I32_F32, 0, v(2)),
        vop1e(vop1::V_FREXP_MANT_F32, 1, v(2)),
        vop1e(vop1::V_FREXP_MANT_F32, 3, 4), // Inf
        vop1e(vop1::V_FREXP_EXP_I32_F32, 5, 4),
        ENDPGM,
    ]);
    check_all(&code);
    // conversions on |x| < 2^23 (exponent 110 + 0..31) and ties
    let code = [
        vop2e(vop2::V_LSHRREV_B32, 2, int(8), 0),
        vop2e(vop2::V_AND_B32, 2, int(31), 2),
        vop2e(vop2::V_ADD_NC_U32, 2, L, 2), 110,
        vop2e(vop2::V_LSHLREV_B32, 2, int(23), 2),
        vop2e(vop2::V_AND_B32, 3, L, 0), 0x807F_FFFF,
        vop2e(vop2::V_OR_B32, 2, v(2), 3),
        vop2e(vop2::V_AND_B32, 4, L, 1), 0x8000_0007,
        vop1e(vop1::V_CVT_F32_I32, 4, v(4)),
        vop2e(vop2::V_ADD_F32, 4, 240, 4), // k + 0.5
        vop1e(vop1::V_CVT_RPI_I32_F32, 0, v(2)),
        vop1e(vop1::V_CVT_FLR_I32_F32, 1, v(2)),
        vop1e(vop1::V_CVT_RPI_I32_F32, 5, v(4)),
        vop1e(vop1::V_CVT_FLR_I32_F32, 6, v(4)),
        vop1e(vop1::V_CVT_RPI_I32_F32, 7, 241), // -0.5
        vop1e(vop1::V_CVT_FLR_I32_F32, 3, 241),
        ENDPGM,
    ];
    check_all(&code);
}

/// 64-bit scalar bit ops: s_brev_b64, s_bcnt1_i32_b64 (SCC),
/// s_ff1_i32_b64, s_flbit_i32_b64/i64, s_bitset0/1_b64; zero, high-only
/// and negative operands.
#[test]
fn scalar_bit_ops_b64() {
    use scarlett_gpu::rdna::sop1::bit64;
    assert_eq!(bit64(sop1::S_FF1_I32_B64, 0), Some(u32::MAX));
    assert_eq!(bit64(sop1::S_FF1_I32_B64, 1 << 40), Some(40));
    assert_eq!(bit64(sop1::S_FLBIT_I32_B64, 1 << 40), Some(23));
    assert_eq!(bit64(sop1::S_FLBIT_I32_B64, 0), Some(u32::MAX));
    assert_eq!(bit64(sop1::S_FLBIT_I32_I64, u64::MAX), Some(u32::MAX));
    assert_eq!(bit64(sop1::S_FLBIT_I32_I64, !(1u64 << 40)), Some(23));
    assert_eq!(bit64(sop1::S_BCNT1_I32_B64, u64::MAX), Some(64));
    let progs = [
        vec![
            sop1e(sop1::S_MOV_B32, 8, int(0)),
            sop1e(sop1::S_MOV_B32, 9, int(0)),
            sop1e(sop1::S_MOV_B32, 10, int(0)),
            sop1e(sop1::S_MOV_B32, 11, 1),
            sop1e(sop1::S_FF1_I32_B64, 12, 8),
            sop1e(sop1::S_FF1_I32_B64, 13, 10),
            sop1e(sop1::S_FF1_I32_B64, 14, 0),
            sop1e(sop1::S_FLBIT_I32_B64, 15, 10),
            sop1e(sop1::S_FLBIT_I32_B64, 7, 8),
            sop1e(sop1::S_FLBIT_I32_B64, 6, 2),
            sop1e(sop1::S_BCNT1_I32_B64, 5, 10), // low word 0, high word not
            sop1e(sop1::S_BREV_B64, 2, 0),
            sop1e(sop1::S_BITSET1_B64, 0, 4),
            sop1e(sop1::S_BITSET0_B64, 10, 3),
            ENDPGM,
        ],
        vec![
            sop1e(sop1::S_FLBIT_I32_I64, 8, 0),
            sop1e(sop1::S_FLBIT_I32_I64, 9, int(-1)),
            sop1e(sop1::S_FLBIT_I32_I64, 10, int(-3)),
            sop1e(sop1::S_MOV_B32, 12, int(-1)),
            sop1e(sop1::S_MOV_B32, 13, 2),
            sop1e(sop1::S_FLBIT_I32_I64, 11, 12),
            sop1e(sop1::S_MOV_B32, 4, int(0)),
            sop1e(sop1::S_MOV_B32, 5, int(0)),
            sop1e(sop1::S_BCNT1_I32_B64, 14, 4), // 0: SCC = 0
            sop1e(sop1::S_BITSET0_B64, 6, int(63)),
            sop1e(sop1::S_BITSET1_B64, 2, int(32)),
            ENDPGM,
        ],
    ];
    for code in &progs {
        check_all(code);
    }
}

/// v_mul_i32_i24, v_mul_hi_i32_i24, v_mul_hi_u32_u24: high 8 bits of the
/// operands ignored, bit 23 the sign for i24.
#[test]
fn mul24_lo_hi() {
    use scarlett_gpu::interp::vop2_int;
    let (mi, hi, hu) = (vop2::V_MUL_I32_I24, vop2::V_MUL_HI_I32_I24, vop2::V_MUL_HI_U32_U24);
    assert_eq!(vop2_int(mi, 0xFF80_0000, 2), Some((-(1i32 << 24)) as u32), "-2^23 * 2");
    assert_eq!(vop2_int(hi, 0x0080_0000, 0x0080_0000), Some(0x4000), "(-2^23)^2 = 2^46");
    assert_eq!(vop2_int(hi, 0x0080_0000, 1), Some(u32::MAX), "negative product: hi all ones");
    assert_eq!(vop2_int(hu, 0xFFFF_FFFF, 0xFFFF_FFFF), Some(0xFFFF), "(2^24-1)^2 >> 32");
    assert_eq!(vop2e(vop2::V_MUL_HI_U32_U24, 5, v(1), 2), 0x180A_0501);
    let mut code = vec![];
    for (k, op) in [mi, hi, hu].into_iter().enumerate() {
        code.push(vop2e(op, 2 + k as u16, v(0), 1));
        code.push(vop2e(op, 5 + k as u16, 3, 0));
    }
    code.extend(vop3e(vop3::FROM_VOP2 + hi, 1, [v(1), int(-7), int(0)]));
    code.push(ENDPGM);
    check_all(&code);
}

/// s_bitreplicate_b64_b32 into SGPRs, VCC and EXEC (a lane mask built
/// from its half, as ACO does).
#[test]
fn bitreplicate() {
    use scarlett_gpu::rdna::sop1::bitreplicate;
    assert_eq!(bitreplicate(0b101), 0b11_0011);
    assert_eq!(bitreplicate(0x8000_0001), 0xC000_0000_0000_0003);
    assert_eq!(bitreplicate(u32::MAX), u64::MAX);
    let code = [
        sop1e(sop1::S_BITREPLICATE_B64_B32, 4, 1),
        sop1e(sop1::S_BITREPLICATE_B64_B32, 106, 2),
        sop1e(sop1::S_BITREPLICATE_B64_B32, 6, 4),
        sop1e(sop1::S_BITREPLICATE_B64_B32, EXEC, 3),
        vop1e(vop1::V_MOV_B32, 0, int(7)),
        ENDPGM,
    ];
    check_all(&code);
}

/// ds_append / ds_consume (wave32): a uniform pre-op value in every active
/// lane, the counter moved by the active-lane count; LLVM gfx10_ds.txt:
/// `ds_append v5 offset:4` = 0xD8F80004 0x05000000.
#[test]
fn ds_append_consume() {
    assert_eq!(dse(ds::APPEND, 5, 0, 0, 4, 0), [0xD8F8_0004, 0x0500_0000]);
    let mut code = vec![vop1e(vop1::V_MOV_B32, 4, int(0))];
    code.extend(dse(ds::APPEND, 1, 0, 0, 8, 0)); // v1 = 0
    code.extend(dse(ds::APPEND, 2, 0, 0, 8, 0)); // v2 = n
    code.extend(dse(ds::CONSUME, 3, 0, 0, 8, 0)); // v3 = 2n
    code.extend(dse(ds::CONSUME, 6, 0, 0, 12, 0)); // v6 = 0, lds[3] = -n
    code.extend(dse(ds::READ_B32, 5, 4, 0, 8, 0)); // v5 = n
    code.extend(dse(ds::READ_B32, 7, 4, 0, 12, 0)); // v7 = -n
    code.push(soppe(sopp::S_WAITCNT, 0));
    code.push(ENDPGM);
    let mut w = Wave::new(WaveSize::W32, NV as usize);
    w.lds = vec![0; LDS_WORDS as usize];
    w.exec = 0b1011_0000;
    interp::run(&code, &mut w, 100).unwrap();
    for (r, want) in [(1, 0), (2, 3), (3, 6), (6, 0), (5, 3), (7, 3u32.wrapping_neg())] {
        assert_eq!(w.vgpr[r][4], want, "v{r}");
        assert_eq!(w.vgpr[r][0], 0, "v{r} inactive lane");
    }
    for seed in 0..3 {
        for sparse in [false, true] {
            check(&code, WaveSize::W32, seed, sparse);
        }
    }
    // refused: wave64, a non-zero M0
    let mut w = Wave::new(WaveSize::W64, NV as usize);
    w.lds = vec![0; LDS_WORDS as usize];
    w.exec = 1;
    assert!(interp::run(&code, &mut w, 100).is_err());
    let mut c = vec![sop1e(sop1::S_MOV_B32, 124, int(4))];
    c.extend(&code);
    let mut w = Wave::new(WaveSize::W32, NV as usize);
    w.lds = vec![0; LDS_WORDS as usize];
    w.exec = 1;
    assert!(interp::run(&c, &mut w, 100).is_err());
}

/// VOP3P: w0 = 0xCC000000 | op << 16 | clamp << 15 | op_sel_hi[2] << 14 |
/// vdst; w1 = op_sel_hi[1:0] << 27 | src2 << 18 | src1 << 9 | src0
/// (LLVM `gfx1011_dlops.s`: v_dot4_i32_i8 v0, v1, v2, v3 = 0xCC164000
/// 0x1C0E0501).
fn vop3pe(op: u16, vdst: u32, s: [u16; 3], clamp: bool) -> [u32; 2] {
    [
        0xCC00_0000 | (op as u32) << 16 | (clamp as u32) << 15 | 1 << 14 | vdst,
        3 << 27 | (s[2] as u32) << 18 | (s[1] as u32) << 9 | s[0] as u32,
    ]
}

/// Integer dot products, plain and saturating, on random and edge
/// inputs (programs the interpreter refuses are skipped, see vop3p::dot).
#[test]
fn dot_products() {
    use scarlett_gpu::rdna::vop3p::{self as p, dot};
    assert_eq!(vop3pe(p::V_DOT4_I32_I8, 0, [v(1), v(2), v(3)], false), [0xCC16_4000, 0x1C0E_0501]);
    assert_eq!(dot(p::V_DOT4_I32_I8, 0x80FF_0102, 0x7F01_0304, 10, false), Some((-128i32 * 127 - 1 + 3 + 8 + 10) as u32));
    assert_eq!(dot(p::V_DOT4_U32_U8, 0xFFFF_FFFF, 0xFFFF_FFFF, u32::MAX - 5, true), Some(u32::MAX));
    assert_eq!(dot(p::V_DOT2_I32_I16, 0x7FFF_7FFF, 0x7FFF_7FFF, 0x7FFF_FFFF, true), Some(i32::MAX as u32));
    assert_eq!(dot(p::V_DOT2_I32_I16, 0x8000_0001, 0x7FFF_FFFF, 0x8000_0000, true), Some(i32::MIN as u32));
    assert_eq!(dot(p::V_DOT2_U32_U16, 0xFFFF_FFFF, 0xFFFF_FFFF, 0, true), None); // NIR poison
    let mut code = vec![];
    for (k, op) in [p::V_DOT2_I32_I16, p::V_DOT2_U32_U16, p::V_DOT4_I32_I8, p::V_DOT4_U32_U8].into_iter().enumerate() {
        code.extend(vop3pe(op, 3 + k as u32, [v(0), v(1), v(2)], false));
        code.extend(vop3pe(op, 7, [v(1), 2, v(k as u16)], true)); // saturating
        code.push(vop2e(vop2::V_XOR_B32, 2, v(7), 2));
    }
    code.push(ENDPGM);
    let (mut ran, mut skipped) = (0, 0);
    for size in [WaveSize::W32, WaveSize::W64] {
        for seed in 0..12 {
            for sparse in [false, true] {
                let (sgprs, vgprs) = inputs(seed, size.lanes());
                let mut w = Wave::new(size, NV as usize);
                w.sgpr[..NS as usize].copy_from_slice(&sgprs);
                w.vgpr.clone_from(&vgprs);
                w.exec = exec_for(size, sparse);
                if interp::run(&code, &mut w, 1000).is_err() {
                    skipped += 1;
                    continue;
                }
                ran += 1;
                check(&code, size, seed, sparse);
            }
        }
    }
    assert!(ran >= 8, "ran {ran}, skipped {skipped}");
}
