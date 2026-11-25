//! GPU front end for the Xbox Series (RDNA2-class) graphics engine.
//!
//! * [`pm4`]: command-stream packet decoding;
//! * [`regs`]: the register file the packets program, with copy-on-write
//!   snapshots;
//! * [`cp`]: the command processor that executes streams, applies memory
//!   side effects (fences, writes, copies) and emits draws/dispatches with
//!   their state for a backend;
//! * [`rdna`]: RDNA2 shader instruction decoding (formats, lengths,
//!   operands, branches, basic blocks, listing);
//! * [`interp`]: the reference RDNA2 interpreter;
//! * [`exec`]: runs command-processor dispatches on the interpreter;
//! * [`spirv`], [`translate`], [`spirv_eval`]: RDNA2 -> SPIR-V
//!   translation and an evaluator to check it against the interpreter.
//!
//! Not here yet: a rendering backend (no Vulkan device is driven). How titles hand command buffers to the GPU is not
//! known until real binaries are traced, so nothing in HLE calls this yet.

pub mod cp;
pub mod exec;
pub mod interp;
pub mod pm4;
pub mod rdna;
pub mod spirv;
pub mod spirv_eval;
pub mod translate;
pub mod regs;

/// Simple sparse GPU memory for tests and tools: 4 KiB pages on demand.
#[derive(Default)]
pub struct SparseMemory {
    pages: std::collections::HashMap<u64, Box<[u32; 1024]>>,
    pub clock: u64,
}

impl SparseMemory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Map (zeroed) pages covering `[addr, addr + bytes)`.
    pub fn map(&mut self, addr: u64, bytes: u64) {
        let mut p = addr & !0xFFF;
        while p < addr + bytes {
            self.pages.entry(p).or_insert_with(|| Box::new([0; 1024]));
            p += 0x1000;
        }
    }

    fn slot(&self, a: u64) -> Option<u32> {
        self.pages
            .get(&(a & !0xFFF))
            .map(|pg| pg[(a as usize & 0xFFF) / 4])
    }
}

impl cp::Memory for SparseMemory {
    fn read(&self, addr: u64, out: &mut [u32]) -> bool {
        for (i, o) in out.iter_mut().enumerate() {
            match self.slot(addr + 4 * i as u64) {
                Some(v) => *o = v,
                None => return false,
            }
        }
        true
    }

    fn write(&mut self, addr: u64, data: &[u32]) -> bool {
        if (0..data.len()).any(|i| self.slot(addr + 4 * i as u64).is_none()) {
            return false;
        }
        for (i, v) in data.iter().enumerate() {
            let a = addr + 4 * i as u64;
            self.pages.get_mut(&(a & !0xFFF)).unwrap()[(a as usize & 0xFFF) / 4] = *v;
        }
        true
    }

    fn gpu_clock(&self) -> u64 {
        self.clock
    }
}
