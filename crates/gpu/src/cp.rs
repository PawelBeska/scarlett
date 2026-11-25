//! Command processor: executes PM4 streams against a register file and GPU
//! memory, and turns draws and dispatches into [`Event`]s that carry a
//! snapshot of the state they ran with (for a backend to translate).
//!
//! Memory side effects that the CPU side of a game waits on (fences from
//! `RELEASE_MEM`/`EVENT_WRITE_EOP`, `WRITE_DATA`, `COPY_DATA`) are applied
//! immediately through [`Memory`]. `WAIT_REG_MEM` is evaluated once: if
//! the condition does not hold yet it is reported as [`Event::Stall`]
//! (a threaded CP would block there).

use crate::pm4::{self, op, Packet};
use crate::regs::{self, Regs};

/// GPU-visible memory, by GPU virtual address.
pub trait Memory {
    /// Fill `out` from `addr`; false if any of it is unmapped.
    fn read(&self, addr: u64, out: &mut [u32]) -> bool;
    /// Store `data` at `addr`; false if any of it is unmapped.
    fn write(&mut self, addr: u64, data: &[u32]) -> bool;
    /// 64-bit GPU timestamp for `*_TIMESTAMP` data selects.
    fn gpu_clock(&self) -> u64 {
        0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DrawKind {
    /// Non-indexed: vertices 0..count.
    Auto { vertices: u32 },
    /// Indexed from `index_va` (`index_size` bytes per index).
    Indexed {
        index_va: u64,
        count: u32,
        max: u32,
        index_size: u8,
    },
    /// Arguments from memory (`DRAW_INDIRECT` / `DRAW_INDEX_INDIRECT`),
    /// already read: vertex/index count, instances, first vertex/index, first instance.
    Indirect {
        indexed: bool,
        args: [u32; 4],
        index_va: u64,
        index_size: u8,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Draw {
    pub kind: DrawKind,
    pub instances: u32,
    pub primitive: u32,
    /// Shader program addresses (vertex/NGG stage, pixel stage).
    pub vs: u64,
    pub ps: u64,
    pub regs: Regs,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dispatch {
    pub groups: [u32; 3],
    /// The packet's DISPATCH_INITIATOR dword (bit 15 CS_W32_EN, per Mesa's
    /// `gfx10.json`; RADV emits it last in DISPATCH_DIRECT/_INDIRECT).
    pub initiator: u32,
    pub threads: [u32; 3],
    pub program: u64,
    pub regs: Regs,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Draw(Draw),
    Dispatch(Dispatch),
    /// A fence/timestamp write the CPU may be waiting for.
    Fence {
        addr: u64,
        value: u64,
        bits: u8,
    },
    /// `WAIT_REG_MEM` whose condition did not hold.
    Stall {
        addr: u64,
        reference: u32,
        mask: u32,
        function: u8,
        seen: u32,
    },
    /// A packet the CP does not interpret (logged once per opcode).
    Unhandled {
        opcode: u8,
        at: usize,
    },
    /// Malformed stream or unreadable indirect buffer; execution of that
    /// buffer stops.
    Error(String),
}

/// Bytes per index for `INDEX_TYPE` / `VGT_INDEX_TYPE` values.
fn index_size(t: u32) -> u8 {
    match t & 3 {
        0 => 2,
        1 => 4,
        _ => 1,
    }
}

const MAX_IB_DEPTH: u32 = 4;

#[derive(Default)]
pub struct CommandProcessor {
    pub regs: Regs,
    index_base: u64,
    index_type: u32,
    index_buffer_size: u32,
    num_instances: u32,
    /// `SET_BASE` index 1: base address for indirect draw/dispatch args.
    indirect_base: u64,
    seen_unhandled: std::collections::HashSet<u8>,
}

impl CommandProcessor {
    pub fn new() -> Self {
        CommandProcessor {
            num_instances: 1,
            ..Default::default()
        }
    }

    /// Execute one command buffer (and the indirect buffers it calls).
    pub fn run(&mut self, mem: &mut dyn Memory, dw: &[u32]) -> Vec<Event> {
        let mut out = Vec::new();
        self.exec(mem, dw, 0, &mut out);
        out
    }

    /// Execute `size` dwords at GPU address `addr`.
    pub fn run_ib(&mut self, mem: &mut dyn Memory, addr: u64, size: u32) -> Vec<Event> {
        let mut out = Vec::new();
        self.exec_ib(mem, addr, size, 0, &mut out);
        out
    }

    fn exec_ib(
        &mut self,
        mem: &mut dyn Memory,
        addr: u64,
        size: u32,
        depth: u32,
        out: &mut Vec<Event>,
    ) {
        if depth >= MAX_IB_DEPTH {
            out.push(Event::Error(format!(
                "indirect buffers nested deeper than {MAX_IB_DEPTH}"
            )));
            return;
        }
        let mut buf = vec![0u32; size as usize];
        if !mem.read(addr, &mut buf) {
            out.push(Event::Error(format!(
                "indirect buffer {addr:#x}+{size} dwords not mapped"
            )));
            return;
        }
        self.exec(mem, &buf, depth, out);
    }

    fn exec(&mut self, mem: &mut dyn Memory, dw: &[u32], depth: u32, out: &mut Vec<Event>) {
        for p in pm4::packets(dw) {
            match p {
                Err(e) => {
                    out.push(Event::Error(e.to_string()));
                    return;
                }
                Ok((_, Packet::Type2)) => {}
                Ok((_, Packet::Type0 { reg, values })) => {
                    self.regs.set_range(reg as u32, values);
                }
                Ok((at, Packet::Type3 { opcode, body, .. })) => {
                    self.packet3(mem, at, opcode, body, depth, out)
                }
            }
        }
    }

    fn set_regs(&mut self, base: u32, body: &[u32]) {
        if let Some((&first, values)) = body.split_first() {
            self.regs.set_range(base + (first & 0xFFFF), values);
        }
    }

    fn draw(&self, kind: DrawKind, instances: u32) -> Event {
        Event::Draw(Draw {
            kind,
            instances,
            primitive: self.regs.get(regs::VGT_PRIMITIVE_TYPE) & 0x3F,
            vs: self.regs.get64(regs::SPI_SHADER_PGM_LO_VS) << 8,
            ps: self.regs.get64(regs::SPI_SHADER_PGM_LO_PS) << 8,
            regs: self.regs.clone(),
        })
    }

    fn dispatch(&self, groups: [u32; 3], initiator: u32) -> Event {
        let t = |r| (self.regs.get(r) & 0xFFFF).max(1);
        Event::Dispatch(Dispatch {
            groups,
            initiator,
            threads: [
                t(regs::COMPUTE_NUM_THREAD_X),
                t(regs::COMPUTE_NUM_THREAD_Y),
                t(regs::COMPUTE_NUM_THREAD_Z),
            ],
            program: self.regs.get64(regs::COMPUTE_PGM_LO) << 8,
            regs: self.regs.clone(),
        })
    }

    fn read_args<const N: usize>(&self, mem: &dyn Memory, addr: u64) -> Option<[u32; N]> {
        let mut a = [0u32; N];
        mem.read(addr, &mut a).then_some(a)
    }

    fn packet3(
        &mut self,
        mem: &mut dyn Memory,
        at: usize,
        opcode: u8,
        b: &[u32],
        depth: u32,
        out: &mut Vec<Event>,
    ) {
        let arg = |i: usize| b.get(i).copied().unwrap_or(0);
        let addr = |lo: usize| arg(lo) as u64 | ((arg(lo + 1) & 0xFFFF) as u64) << 32;
        match opcode {
            op::NOP
            | op::CONTEXT_CONTROL
            | op::PFP_SYNC_ME
            | op::ACQUIRE_MEM
            | op::SURFACE_SYNC
            | op::PREAMBLE_CNTL
            | op::EVENT_WRITE => {}
            op::CLEAR_STATE => self.regs.clear_context(),
            op::SET_CONFIG_REG => self.set_regs(regs::CONFIG_BASE, b),
            op::SET_CONTEXT_REG => self.set_regs(regs::CONTEXT_BASE, b),
            op::SET_SH_REG | op::SET_SH_REG_INDEX => self.set_regs(regs::SH_BASE, b),
            op::SET_UCONFIG_REG | op::SET_UCONFIG_REG_INDEX => self.set_regs(regs::UCONFIG_BASE, b),
            op::SET_BASE => {
                if arg(0) & 0xF == 1 {
                    self.indirect_base = addr(1);
                }
            }
            op::INDEX_BASE => self.index_base = addr(0),
            op::INDEX_BUFFER_SIZE => self.index_buffer_size = arg(0),
            op::INDEX_TYPE => {
                self.index_type = arg(0);
                self.regs.set(regs::VGT_INDEX_TYPE, arg(0));
            }
            op::NUM_INSTANCES => {
                self.num_instances = arg(0).max(1);
                self.regs.set(regs::VGT_NUM_INSTANCES, arg(0));
            }
            op::DRAW_INDEX_AUTO => {
                out.push(self.draw(DrawKind::Auto { vertices: arg(0) }, self.num_instances))
            }
            op::DRAW_INDEX_2 => out.push(self.draw(
                DrawKind::Indexed {
                    index_va: addr(1),
                    count: arg(3),
                    max: arg(0),
                    index_size: index_size(self.index_type),
                },
                self.num_instances,
            )),
            op::DRAW_INDEX_OFFSET_2 => {
                let size = index_size(self.index_type);
                out.push(self.draw(
                    DrawKind::Indexed {
                        index_va: self.index_base + arg(1) as u64 * size as u64,
                        count: arg(2),
                        max: arg(0),
                        index_size: size,
                    },
                    self.num_instances,
                ))
            }
            op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT => {
                let indexed = opcode == op::DRAW_INDEX_INDIRECT;
                // DRAW_INDIRECT: count, instances, first vertex, first instance.
                // DRAW_INDEX_INDIRECT: count, instances, first index, base vertex, first instance.
                let va = self.indirect_base + arg(0) as u64;
                let args = if indexed {
                    self.read_args::<5>(mem, va)
                        .map(|a| [a[0], a[1], a[2], a[4]])
                } else {
                    self.read_args::<4>(mem, va)
                };
                match args {
                    Some(args) => out.push(self.draw(
                        DrawKind::Indirect {
                            indexed,
                            args,
                            index_va: self.index_base,
                            index_size: index_size(self.index_type),
                        },
                        args[1],
                    )),
                    None => out.push(Event::Error(format!(
                        "indirect draw args at {va:#x} not mapped"
                    ))),
                }
            }
            op::DISPATCH_DIRECT => out.push(self.dispatch([arg(0), arg(1), arg(2)], arg(3))),
            op::DISPATCH_INDIRECT => {
                let va = self.indirect_base + arg(0) as u64;
                match self.read_args::<3>(mem, va) {
                    Some(g) => out.push(self.dispatch(g, arg(1))),
                    None => out.push(Event::Error(format!(
                        "indirect dispatch args at {va:#x} not mapped"
                    ))),
                }
            }
            op::INDIRECT_BUFFER | op::INDIRECT_BUFFER_CONST => {
                let va = arg(0) as u64 & !3 | ((arg(1) & 0xFFFF) as u64) << 32;
                self.exec_ib(mem, va, arg(2) & 0xF_FFFF, depth + 1, out);
            }
            op::WRITE_DATA => {
                // control: dst_sel [11:8] (0 = register, 1/2/5 = memory)
                let dst_sel = arg(0) >> 8 & 0xF;
                let data = b.get(3..).unwrap_or(&[]);
                if dst_sel == 0 {
                    self.regs.set_range(arg(1), data);
                } else if !mem.write(addr(1), data) {
                    out.push(Event::Error(format!(
                        "WRITE_DATA to unmapped {:#x}",
                        addr(1)
                    )));
                }
            }
            op::COPY_DATA => self.copy_data(mem, b, out),
            op::RELEASE_MEM => {
                // [1]: data_sel [31:29]; [2..3] address; [4..5] data
                self.fence_write(
                    mem,
                    arg(1) >> 29,
                    addr(2),
                    arg(4) as u64 | (arg(5) as u64) << 32,
                    out,
                )
            }
            op::EVENT_WRITE_EOP => {
                // [2]: addr hi [15:0], data_sel [31:29]
                self.fence_write(
                    mem,
                    arg(2) >> 29,
                    addr(1),
                    arg(3) as u64 | (arg(4) as u64) << 32,
                    out,
                )
            }
            op::WAIT_REG_MEM => {
                // [0]: function [2:0], mem_space [4]; [1..2] address/register; [3] ref; [4] mask
                let function = (arg(0) & 7) as u8;
                let (reference, mask) = (arg(3), arg(4));
                let (target, seen) = if arg(0) >> 4 & 1 == 1 {
                    let va = addr(1) & !3;
                    let mut v = [0u32];
                    if !mem.read(va, &mut v) {
                        out.push(Event::Error(format!("WAIT_REG_MEM on unmapped {va:#x}")));
                        return;
                    }
                    (va, v[0])
                } else {
                    (arg(1) as u64, self.regs.get(arg(1)))
                };
                let v = seen & mask;
                let ok = match function {
                    0 => true,
                    1 => v < reference,
                    2 => v <= reference,
                    3 => v == reference,
                    4 => v != reference,
                    5 => v >= reference,
                    6 => v > reference,
                    _ => false,
                };
                if !ok {
                    out.push(Event::Stall {
                        addr: target,
                        reference,
                        mask,
                        function,
                        seen,
                    });
                }
            }
            _ => {
                if self.seen_unhandled.insert(opcode) {
                    log::debug!(
                        "PM4: unhandled {} ({opcode:#04x}) at dword {at}",
                        op::name(opcode)
                    );
                }
                out.push(Event::Unhandled { opcode, at });
            }
        }
    }

    /// Data select: 0 none, 1 low 32 bits, 2 64 bits, 3 GPU timestamp.
    fn fence_write(
        &mut self,
        mem: &mut dyn Memory,
        sel: u32,
        va: u64,
        value: u64,
        out: &mut Vec<Event>,
    ) {
        let (value, bits) = match sel {
            0 => return,
            1 => (value & 0xFFFF_FFFF, 32),
            2 => (value, 64),
            3 => (mem.gpu_clock(), 64),
            _ => return,
        };
        let data = [value as u32, (value >> 32) as u32];
        if mem.write(va, &data[..bits as usize / 32]) {
            out.push(Event::Fence {
                addr: va,
                value,
                bits,
            });
        } else {
            out.push(Event::Error(format!("fence write to unmapped {va:#x}")));
        }
    }

    /// `COPY_DATA`: [0] control (src_sel [3:0], dst_sel [11:8], 64-bit [16]),
    /// [1..2] source (address, register or immediate), [3..4] destination.
    fn copy_data(&mut self, mem: &mut dyn Memory, b: &[u32], out: &mut Vec<Event>) {
        let arg = |i: usize| b.get(i).copied().unwrap_or(0);
        let (ctl, src_sel, dst_sel) = (arg(0), arg(0) & 0xF, arg(0) >> 8 & 0xF);
        let n = if ctl >> 16 & 1 == 1 { 2 } else { 1 };
        let src_addr = arg(1) as u64 | (arg(2) as u64) << 32;
        let mut v = [0u32; 2];
        match src_sel {
            0 => {
                v[0] = self.regs.get(arg(1));
                v[1] = self.regs.get(arg(1) + 1);
            }
            1 | 2 => {
                if !mem.read(src_addr, &mut v[..n]) {
                    out.push(Event::Error(format!(
                        "COPY_DATA from unmapped {src_addr:#x}"
                    )));
                    return;
                }
            }
            5 => v = [arg(1), arg(2)],
            9 => {
                let t = mem.gpu_clock();
                v = [t as u32, (t >> 32) as u32];
            }
            s => {
                out.push(Event::Error(format!(
                    "COPY_DATA source select {s} not supported"
                )));
                return;
            }
        }
        let dst = arg(3) as u64 | (arg(4) as u64) << 32;
        match dst_sel {
            0 => {
                self.regs.set_range(arg(3), &v[..n]);
            }
            1 | 2 | 5 => {
                if !mem.write(dst, &v[..n]) {
                    out.push(Event::Error(format!("COPY_DATA to unmapped {dst:#x}")));
                }
            }
            s => out.push(Event::Error(format!(
                "COPY_DATA destination select {s} not supported"
            ))),
        }
    }
}
