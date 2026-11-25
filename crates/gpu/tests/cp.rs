//! Command processor on synthetic PM4 streams.

use scarlett_gpu::cp::{CommandProcessor, DrawKind, Event, Memory};
use scarlett_gpu::pm4::{op, pkt3};
use scarlett_gpu::regs::{self, prim, Space};
use scarlett_gpu::SparseMemory;

fn set(space_base: u32, reg: u32, values: &[u32]) -> Vec<u32> {
    let op = match space_base {
        regs::CONTEXT_BASE => op::SET_CONTEXT_REG,
        regs::SH_BASE => op::SET_SH_REG,
        regs::UCONFIG_BASE => op::SET_UCONFIG_REG,
        _ => op::SET_CONFIG_REG,
    };
    let mut v = vec![pkt3(op, 1 + values.len() as u32), reg - space_base];
    v.extend_from_slice(values);
    v
}

fn draws(ev: &[Event]) -> Vec<&scarlett_gpu::cp::Draw> {
    ev.iter()
        .filter_map(|e| match e {
            Event::Draw(d) => Some(d),
            _ => None,
        })
        .collect()
}

#[test]
fn draws_capture_their_own_state() {
    let mut s = Vec::new();
    s.extend(set(
        regs::UCONFIG_BASE,
        regs::VGT_PRIMITIVE_TYPE,
        &[prim::TRILIST],
    ));
    s.extend(set(
        regs::SH_BASE,
        regs::SPI_SHADER_PGM_LO_VS,
        &[0x0010_0000, 0],
    ));
    s.extend(set(
        regs::SH_BASE,
        regs::SPI_SHADER_PGM_LO_PS,
        &[0x0020_0000, 0x1],
    ));
    s.extend(set(regs::CONTEXT_BASE, regs::CB_COLOR0_BASE, &[0xAAAA]));
    s.extend([pkt3(op::NUM_INSTANCES, 1), 3]);
    s.extend([pkt3(op::DRAW_INDEX_AUTO, 2), 36, 2]);
    // second draw: new render target, indexed 32-bit through INDEX_BASE + offset
    s.extend(set(regs::CONTEXT_BASE, regs::CB_COLOR0_BASE, &[0xBBBB]));
    s.extend([pkt3(op::INDEX_TYPE, 1), 1]);
    s.extend([pkt3(op::INDEX_BASE, 2), 0x4000_0000, 0x12]);
    s.extend([pkt3(op::DRAW_INDEX_OFFSET_2, 4), 1000, 10, 60, 0]);

    let mut cp = CommandProcessor::new();
    let ev = cp.run(&mut SparseMemory::new(), &s);
    let d = draws(&ev);
    assert_eq!(d.len(), 2, "{ev:?}");
    assert_eq!(d[0].kind, DrawKind::Auto { vertices: 36 });
    assert_eq!((d[0].instances, d[0].primitive), (3, prim::TRILIST));
    assert_eq!(d[0].vs, 0x0010_0000 << 8);
    assert_eq!(d[0].ps, (0x1_0020_0000u64) << 8);
    assert_eq!(
        d[0].regs.get(regs::CB_COLOR0_BASE),
        0xAAAA,
        "first draw keeps its target"
    );
    assert_eq!(d[1].regs.get(regs::CB_COLOR0_BASE), 0xBBBB);
    assert!(
        d[0].regs.shares(&d[1].regs, Space::Sh),
        "unchanged spaces are shared, not copied"
    );
    assert_eq!(
        d[1].kind,
        DrawKind::Indexed {
            index_va: 0x12_4000_0000 + 40,
            count: 60,
            max: 1000,
            index_size: 4
        }
    );
}

#[test]
fn indirect_buffers_fences_writes_and_waits() {
    let mut mem = SparseMemory::new();
    mem.map(0x1_0000, 0x3000);
    mem.clock = 0xDEAD_BEEF_0000;
    // IB at 0x11000: WRITE_DATA 2 dwords to 0x10000, then a 64-bit fence at 0x10010
    let ib = [
        pkt3(op::WRITE_DATA, 5),
        5 << 8, // dst_sel = memory
        0x1_0000,
        0,
        0x11,
        0x22,
        pkt3(op::RELEASE_MEM, 7),
        0,
        2 << 29, // data_sel = 64-bit value
        0x1_0010,
        0,
        0x0000_0007,
        0x0000_0001,
        0,
    ];
    assert!(mem.write(0x1_1000, &ib));
    let main = [
        pkt3(op::INDIRECT_BUFFER, 3),
        0x1_1000,
        0,
        ib.len() as u32,
        // timestamp via EVENT_WRITE_EOP (data_sel 3) at 0x10020
        pkt3(op::EVENT_WRITE_EOP, 5),
        0,
        0x1_0020,
        3 << 29,
        0,
        0,
        // wait until mem[0x10000] == 0x11 (satisfied), then until == 0x99 (not)
        pkt3(op::WAIT_REG_MEM, 6),
        3 | 1 << 4,
        0x1_0000,
        0,
        0x11,
        0xFFFF_FFFF,
        4,
        pkt3(op::WAIT_REG_MEM, 6),
        3 | 1 << 4,
        0x1_0000,
        0,
        0x99,
        0xFFFF_FFFF,
        4,
        // IB that is not mapped
        pkt3(op::INDIRECT_BUFFER, 3),
        0x9_0000,
        0,
        4,
        pkt3(0x7F, 1),
        0,
    ];
    let mut cp = CommandProcessor::new();
    let ev = cp.run(&mut mem, &main);
    let mut v = [0u32; 6];
    assert!(mem.read(0x1_0000, &mut v[..2]));
    assert_eq!(
        &v[..2],
        &[0x11, 0x22],
        "WRITE_DATA from the indirect buffer"
    );
    assert!(mem.read(0x1_0010, &mut v[..2]));
    assert_eq!(&v[..2], &[7, 1]);
    assert_eq!(
        ev[0],
        Event::Fence {
            addr: 0x1_0010,
            value: 0x1_0000_0007,
            bits: 64
        }
    );
    assert_eq!(
        ev[1],
        Event::Fence {
            addr: 0x1_0020,
            value: 0xDEAD_BEEF_0000,
            bits: 64
        }
    );
    assert_eq!(
        ev[2],
        Event::Stall {
            addr: 0x1_0000,
            reference: 0x99,
            mask: 0xFFFF_FFFF,
            function: 3,
            seen: 0x11
        }
    );
    assert!(matches!(&ev[3], Event::Error(e) if e.contains("not mapped")));
    assert_eq!(
        ev[4],
        Event::Unhandled {
            opcode: 0x7F,
            at: main.len() - 2
        }
    );
    assert_eq!(ev.len(), 5);
}

#[test]
fn dispatches_direct_and_indirect() {
    let mut mem = SparseMemory::new();
    mem.map(0x8000, 0x1000);
    assert!(mem.write(0x8010, &[4, 5, 6]));
    let mut s = Vec::new();
    s.extend(set(regs::SH_BASE, regs::COMPUTE_NUM_THREAD_X, &[64, 1, 1]));
    s.extend(set(regs::SH_BASE, regs::COMPUTE_PGM_LO, &[0x30_0000, 0]));
    s.extend([pkt3(op::DISPATCH_DIRECT, 4), 8, 8, 1, 1]);
    s.extend([pkt3(op::SET_BASE, 3), 1, 0x8000, 0]);
    s.extend([pkt3(op::DISPATCH_INDIRECT, 2), 0x10, 1]);
    // COPY_DATA immediate → register, then register → memory
    s.extend([
        pkt3(op::COPY_DATA, 5),
        5,
        0x77,
        0,
        regs::COMPUTE_USER_DATA_0,
        0,
    ]);
    s.extend([
        pkt3(op::COPY_DATA, 5),
        5 << 8,
        regs::COMPUTE_USER_DATA_0,
        0,
        0x8100,
        0,
    ]);
    let mut cp = CommandProcessor::new();
    let ev = cp.run(&mut mem, &s);
    let ds: Vec<_> = ev
        .iter()
        .filter_map(|e| match e {
            Event::Dispatch(d) => Some(d),
            _ => None,
        })
        .collect();
    assert_eq!(ds.len(), 2, "{ev:?}");
    assert_eq!(
        (ds[0].groups, ds[0].threads, ds[0].program),
        ([8, 8, 1], [64, 1, 1], 0x30_0000 << 8)
    );
    assert_eq!(ds[1].groups, [4, 5, 6]);
    let mut v = [0u32];
    assert!(mem.read(0x8100, &mut v));
    assert_eq!(v[0], 0x77);
    assert_eq!(cp.regs.get(regs::COMPUTE_USER_DATA_0), 0x77);
}
