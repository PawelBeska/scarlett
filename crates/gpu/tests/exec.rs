//! End to end: a PM4 stream programs a compute dispatch, the command
//! processor emits it, and `exec::run_dispatch` runs the shader from GPU
//! memory on the interpreter; the buffer it writes is checked.

mod common;
use common::*;

use scarlett_gpu::cp::{CommandProcessor, Event, Memory};
use scarlett_gpu::exec::{self, CS_W32_EN, COMPUTE_SHADER_EN};
use scarlett_gpu::interp::WaveSize;
use scarlett_gpu::pm4::{op, pkt3};
use scarlett_gpu::rdna::{global, vop2, vop3};
use scarlett_gpu::regs;
use scarlett_gpu::SparseMemory;

const CODE: u64 = 0x30_0000;
const OUT: u64 = 0x10_0000;

fn sh(reg: u32, values: &[u32]) -> Vec<u32> {
    let mut v = vec![pkt3(op::SET_SH_REG, 1 + values.len() as u32), reg - regs::SH_BASE];
    v.extend_from_slice(values);
    v
}

/// out[g * 64 + t] = 3 * (g * 64 + t) + 7, g = TGID.x (s2 after two user
/// SGPRs holding the buffer address), t = v0.
fn shader() -> Vec<u32> {
    let mut c = vec![];
    c.extend(vop3e(vop3::V_LSHL_ADD_U32, 1, [2, int(6), v(0)]));
    c.push(vop2e(vop2::V_LSHLREV_B32, 2, int(2), 1));
    c.extend(vop3e(vop3::V_MAD_U32_U24, 3, [v(1), int(3), int(7)]));
    // global_store_dword v2, v3, s[0:1]
    c.extend([0xDC00_0000 | (global::STORE_DWORD as u32) << 18 | 2 << 14, 3 << 8 | 2]);
    c.push(ENDPGM);
    c
}

fn run(wave32: bool) -> SparseMemory {
    let mut mem = SparseMemory::new();
    mem.map(CODE, 0x1000);
    assert!(mem.write(CODE, &shader()));
    mem.map(OUT, 0x1000);
    let mut s = Vec::new();
    s.extend(sh(regs::COMPUTE_PGM_LO, &[(CODE >> 8) as u32, 0]));
    // RSRC1 VGPRS = 0: 8 VGPRs in wave32, 4 in wave64; 1: 8 in wave64.
    s.extend(sh(regs::COMPUTE_PGM_RSRC1, &[if wave32 { 0 } else { 1 }, 2 << 1 | 1 << 7]));
    s.extend(sh(regs::COMPUTE_NUM_THREAD_X, &[64, 1, 1]));
    s.extend(sh(regs::COMPUTE_USER_DATA_0, &[OUT as u32, (OUT >> 32) as u32]));
    let init = COMPUTE_SHADER_EN | if wave32 { CS_W32_EN } else { 0 };
    s.extend([pkt3(op::DISPATCH_DIRECT, 4), 3, 1, 1, init]);
    let mut cp = CommandProcessor::new();
    let ev = cp.run(&mut mem, &s);
    let d = ev
        .iter()
        .find_map(|e| match e {
            Event::Dispatch(d) => Some(d.clone()),
            _ => None,
        })
        .expect("a dispatch");
    let (size, vgprs, lds) = exec::shape(&d);
    assert_eq!((size, vgprs, lds), (if wave32 { WaveSize::W32 } else { WaveSize::W64 }, 8, 0));
    exec::run_dispatch(&d, &mut mem, 10_000).expect("dispatch runs");
    mem
}

#[test]
fn pm4_dispatch_runs_the_shader() {
    for wave32 in [true, false] {
        let mem = run(wave32);
        let mut out = vec![0u32; 3 * 64];
        assert!(mem.read(OUT, &mut out));
        let want: Vec<u32> = (0..3 * 64).map(|i| 3 * i + 7).collect();
        assert_eq!(out, want, "wave32 {wave32}");
    }
}

#[test]
fn refuses_what_it_does_not_model() {
    let mut mem = SparseMemory::new();
    let mut s = Vec::new();
    s.extend(sh(regs::COMPUTE_PGM_RSRC1, &[0, 1])); // SCRATCH_EN
    s.extend([pkt3(op::DISPATCH_DIRECT, 4), 1, 1, 1, COMPUTE_SHADER_EN]);
    let mut cp = CommandProcessor::new();
    let ev = cp.run(&mut mem, &s);
    let Some(Event::Dispatch(d)) = ev.into_iter().find(|e| matches!(e, Event::Dispatch(_))) else {
        panic!("no dispatch")
    };
    assert!(exec::run_dispatch(&d, &mut mem, 100).is_err());
}

/// The same PM4 dispatch translated to SPIR-V and run on the evaluator
/// over a window holding the output buffer: memory must match the
/// interpreter's.
#[test]
fn pm4_dispatch_translated_matches_interpreter() {
    use scarlett_gpu::spirv_eval::{run_dispatch_subgroups, Buffers};
    use scarlett_gpu::translate::Window;
    for wave32 in [true, false] {
        let mut mem = SparseMemory::new();
        mem.map(CODE, 0x1000);
        assert!(mem.write(CODE, &shader()));
        mem.map(OUT, 0x1000);
        let mut s = Vec::new();
        s.extend(sh(regs::COMPUTE_PGM_LO, &[(CODE >> 8) as u32, 0]));
        s.extend(sh(regs::COMPUTE_PGM_RSRC1, &[if wave32 { 0 } else { 1 }, 2 << 1 | 1 << 7]));
        s.extend(sh(regs::COMPUTE_NUM_THREAD_X, &[64, 1, 1]));
        s.extend(sh(regs::COMPUTE_USER_DATA_0, &[OUT as u32, (OUT >> 32) as u32]));
        let init = COMPUTE_SHADER_EN | if wave32 { CS_W32_EN } else { 0 };
        s.extend([pkt3(op::DISPATCH_DIRECT, 4), 3, 1, 1, init]);
        let ev = CommandProcessor::new().run(&mut mem, &s);
        let Some(Event::Dispatch(d)) = ev.into_iter().find(|e| matches!(e, Event::Dispatch(_))) else {
            panic!("no dispatch")
        };
        let words = 3 * 64;
        let c = exec::translate_dispatch(&d, &mem, Window { base: OUT, words }).expect("translate");
        assert_eq!(exec::reachable_len(&exec::fetch_code(&mem, CODE)), Some(shader().len()));
        let mut buffers = Buffers::new();
        buffers.insert((0, 1), vec![0; words as usize + 1]);
        let lanes = c.options.lanes;
        run_dispatch_subgroups(&c.module, d.groups, d.threads, lanes, &mut buffers, &c.push, 1_000_000)
            .expect("evaluate");
        exec::run_dispatch(&d, &mut mem, 10_000).expect("interpreter");
        let mut want = vec![0u32; words as usize];
        assert!(mem.read(OUT, &mut want));
        assert_eq!(&buffers[&(0, 1)][..words as usize], &want[..], "wave32 {wave32}");
    }
}

#[test]
fn reachable_length_follows_branches() {
    use scarlett_gpu::rdna::{sopp, vop1};
    let code = [
        soppe(sopp::S_CBRANCH_SCC0, 1), // to dword 2
        ENDPGM,
        vop1e(vop1::V_MOV_B32, 0, int(1)),
        ENDPGM,
        0xFFFF_FFFF, // not reached
        0,
    ];
    assert_eq!(exec::reachable_len(&code), Some(4));
    assert_eq!(exec::reachable_len(&[soppe(sopp::S_BRANCH, 5), ENDPGM]), None); // leaves the code
}

/// A 128-thread work-group (4 waves of 32 or 2 of 64) through PM4 with
/// LDS from RSRC2.LDS_SIZE = 1 (512 bytes = 128 dwords): each thread
/// writes lds[t] = 5t, s_barrier, reads lds[127 - t] and stores it to
/// out[g * 128 + t]. Interpreter and SPIR-V must agree, and by hand.
#[test]
fn pm4_dispatch_with_lds_and_barrier() {
    use scarlett_gpu::rdna::{ds, sopp};
    use scarlett_gpu::spirv_eval::{run_dispatch_subgroups, Buffers};
    use scarlett_gpu::translate::Window;
    let dse = |op: u16, vdst: u32, addr: u32, data0: u32| [0xD800_0000 | (op as u32) << 18, vdst << 24 | data0 << 8 | addr];
    let mut c = vec![
        vop2e(vop2::V_LSHLREV_B32, 1, int(2), 0),
        vop2e(vop2::V_MUL_U32_U24, 2, int(5), 0),
    ];
    c.extend(dse(ds::WRITE_B32, 0, 1, 2));
    c.push(soppe(sopp::S_WAITCNT, 0));
    c.push(soppe(sopp::S_BARRIER, 0));
    c.extend([vop2e(vop2::V_SUB_NC_U32, 3, LIT, 0), 127]);
    c.push(vop2e(vop2::V_LSHLREV_B32, 3, int(2), 3));
    c.extend(dse(ds::READ_B32, 4, 3, 0));
    c.push(soppe(sopp::S_WAITCNT, 0));
    c.extend(vop3e(vop3::V_LSHL_ADD_U32, 5, [2, int(7), v(0)]));
    c.push(vop2e(vop2::V_LSHLREV_B32, 5, int(2), 5));
    c.extend([0xDC00_0000 | (global::STORE_DWORD as u32) << 18 | 2 << 14, 4 << 8 | 5]);
    c.push(ENDPGM);
    let groups = 2u32;
    for wave32 in [true, false] {
        let mut mem = SparseMemory::new();
        mem.map(CODE, 0x1000);
        assert!(mem.write(CODE, &c));
        mem.map(OUT, 0x1000);
        let mut s = Vec::new();
        s.extend(sh(regs::COMPUTE_PGM_LO, &[(CODE >> 8) as u32, 0]));
        // RSRC2: USER_SGPR 2, TGID_X_EN, LDS_SIZE 1 (bits 15..23)
        s.extend(sh(regs::COMPUTE_PGM_RSRC1, &[if wave32 { 0 } else { 1 }, 2 << 1 | 1 << 7 | 1 << 15]));
        s.extend(sh(regs::COMPUTE_NUM_THREAD_X, &[128, 1, 1]));
        s.extend(sh(regs::COMPUTE_USER_DATA_0, &[OUT as u32, (OUT >> 32) as u32]));
        let init = COMPUTE_SHADER_EN | if wave32 { CS_W32_EN } else { 0 };
        s.extend([pkt3(op::DISPATCH_DIRECT, 4), groups, 1, 1, init]);
        let ev = CommandProcessor::new().run(&mut mem, &s);
        let Some(Event::Dispatch(d)) = ev.into_iter().find(|e| matches!(e, Event::Dispatch(_))) else {
            panic!("no dispatch")
        };
        assert_eq!(exec::shape(&d).2, 128);
        let words = groups * 128;
        let comp = exec::translate_dispatch(&d, &mem, Window { base: OUT, words }).expect("translate");
        let mut buffers = Buffers::new();
        buffers.insert((0, 1), vec![0; words as usize + 1]);
        run_dispatch_subgroups(&comp.module, d.groups, d.threads, comp.options.lanes, &mut buffers, &comp.push, 10_000_000)
            .expect("evaluate");
        exec::run_dispatch(&d, &mut mem, 100_000).expect("interpreter");
        let mut want = vec![0u32; words as usize];
        assert!(mem.read(OUT, &mut want));
        let by_hand: Vec<u32> = (0..words).map(|i| 5 * (127 - i % 128)).collect();
        assert_eq!(want, by_hand, "interpreter, wave32 {wave32}");
        assert_eq!(&buffers[&(0, 1)][..words as usize], &want[..], "SPIR-V, wave32 {wave32}");
    }
}
