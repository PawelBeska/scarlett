//! High-level emulation (HLE) of guest imports.
//!
//! Guest code runs natively, so an import is "emulated" by pointing its IAT
//! slot at a tiny per-import thunk:
//!
//! ```text
//! mov r10, <thunk id>
//! mov r11, <hle_trampoline>
//! jmp r11
//! ```
//!
//! The trampoline (Win64 ABI, see [`trampoline`]) spills the four register
//! arguments, then calls [`dispatch`] with the id, a pointer to those
//! registers and a pointer to the caller's stack arguments. `dispatch` looks
//! the id up in the global thunk table and runs the Rust handler, or logs an
//! unimplemented call. r10/r11 are volatile in Win64, so clobbering them is
//! legal at a call boundary.
//!
//! Floating-point arguments (xmm0-3, then the stack) are read with
//! [`Call::arg_f64`] / [`Call::arg_f32`]; a handler returning a float or
//! vector sets it with [`Call::ret_f64`] / [`Call::ret_f32`] /
//! [`Call::ret_xmm`] and returns any integer (rax is then don't-care).

pub mod builtins;
#[cfg(target_arch = "x86_64")]
pub mod com;
#[cfg(target_arch = "x86_64")]
pub mod enter;
#[cfg(target_arch = "x86_64")]
pub mod file;
pub mod guest;
#[cfg(target_arch = "x86_64")]
pub mod memory;
#[cfg(target_arch = "x86_64")]
pub mod module;
#[cfg(target_arch = "x86_64")]
pub mod object;
pub mod policy;
#[cfg(target_arch = "x86_64")]
pub mod seh;
#[cfg(target_arch = "x86_64")]
pub mod sync;
#[cfg(target_arch = "x86_64")]
pub mod teb;
#[cfg(target_arch = "x86_64")]
pub mod threading;
#[cfg(target_arch = "x86_64")]
pub mod time;
#[cfg(target_arch = "x86_64")]
pub mod trampoline;
pub mod vfs;
#[cfg(target_arch = "x86_64")]
pub mod xgameruntime;

use scarlett_pe::ImportName;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

/// Arguments of one guest call, in Win64 order.
pub struct Call {
    regs: [u64; 4],
    xmm: [[u64; 2]; 4],
    stack: *const u64,
    xmm_ret: Cell<[u64; 2]>,
    /// Full trampoline frame when called from guest code; null in tests.
    #[cfg(target_arch = "x86_64")]
    pub(crate) frame: *const trampoline::Frame,
}

impl Call {
    /// # Safety
    /// `stack` must point at the caller's 5th argument slot, valid for as
    /// many reads as the handler performs via [`Call::arg`].
    pub unsafe fn new(regs: [u64; 4], stack: *const u64) -> Self {
        unsafe { Self::with_xmm(regs, [[0; 2]; 4], stack) }
    }

    /// Like [`Call::new`], with the caller's xmm0-3.
    ///
    /// # Safety
    /// As [`Call::new`].
    pub unsafe fn with_xmm(regs: [u64; 4], xmm: [[u64; 2]; 4], stack: *const u64) -> Self {
        Call {
            regs,
            xmm,
            stack,
            xmm_ret: Cell::new([0; 2]),
            #[cfg(target_arch = "x86_64")]
            frame: std::ptr::null(),
        }
    }

    /// Raw 64 bits of argument `i` as passed in its float position: the low
    /// half of xmm`i` for `i < 4`, else the stack slot.
    fn float_bits(&self, i: usize) -> u64 {
        if i < 4 {
            self.xmm[i][0]
        } else {
            self.arg(i)
        }
    }

    /// `double` argument `i`.
    pub fn arg_f64(&self, i: usize) -> f64 {
        f64::from_bits(self.float_bits(i))
    }

    /// `float` argument `i` (low 32 bits of its slot).
    pub fn arg_f32(&self, i: usize) -> f32 {
        f32::from_bits(self.float_bits(i) as u32)
    }

    /// Return a `double` in xmm0.
    pub fn ret_f64(&self, v: f64) {
        self.xmm_ret.set([v.to_bits(), 0]);
    }

    /// Return a `float` in xmm0.
    pub fn ret_f32(&self, v: f32) {
        self.xmm_ret.set([v.to_bits() as u64, 0]);
    }

    /// Return a full 128-bit vector (`__m128`, `XMVECTOR`) in xmm0.
    pub fn ret_xmm(&self, lo: u64, hi: u64) {
        self.xmm_ret.set([lo, hi]);
    }

    /// The caller's register state at the call (callee-saved registers,
    /// return address, stack pointer), when called from guest code.
    #[cfg(target_arch = "x86_64")]
    pub fn frame(&self) -> Option<&trampoline::Frame> {
        // SAFETY: set by the trampoline to its live frame for this call.
        unsafe { self.frame.as_ref() }
    }

    /// What the handler asked to leave in xmm0.
    pub fn xmm_ret(&self) -> [u64; 2] {
        self.xmm_ret.get()
    }

    /// Integer/pointer argument `i` (0-based).
    pub fn arg(&self, i: usize) -> u64 {
        if i < 4 {
            self.regs[i]
        } else {
            // SAFETY: guaranteed by the contract of `Call::new`: the guest
            // passed at least this many arguments if the handler asks for them.
            unsafe { self.stack.add(i - 4).read_unaligned() }
        }
    }

    pub fn arg_ptr<T>(&self, i: usize) -> *mut T {
        self.arg(i) as *mut T
    }
}

pub type Handler = fn(&Call) -> u64;

/// Name → handler table, filled before loading an image.
#[derive(Default, Clone)]
pub struct Registry {
    by_name: HashMap<(String, String), Handler>,
    by_ordinal: HashMap<(String, u16), Handler>,
}

/// DLL names are case-insensitive and the extension is optional.
pub fn normalize_dll(dll: &str) -> String {
    let d = dll.to_ascii_lowercase();
    d.strip_suffix(".dll").map(str::to_owned).unwrap_or(d)
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, dll: &str, symbol: &str, handler: Handler) -> &mut Self {
        self.by_name
            .insert((normalize_dll(dll), symbol.to_owned()), handler);
        self
    }

    pub fn register_ordinal(&mut self, dll: &str, ordinal: u16, handler: Handler) -> &mut Self {
        self.by_ordinal
            .insert((normalize_dll(dll), ordinal), handler);
        self
    }

    pub fn lookup(&self, dll: &str, name: &ImportName) -> Option<Handler> {
        let dll = normalize_dll(dll);
        match name {
            ImportName::Name { name, .. } => self.by_name.get(&(dll, name.clone())).copied(),
            ImportName::Ordinal(o) => self.by_ordinal.get(&(dll, *o)).copied(),
        }
    }

    /// Does any handler exist under this DLL name?
    pub fn has_dll(&self, dll: &str) -> bool {
        let dll = normalize_dll(dll);
        self.by_name.keys().any(|(d, _)| *d == dll)
            || self.by_ordinal.keys().any(|(d, _)| *d == dll)
    }

    pub fn len(&self) -> usize {
        self.by_name.len() + self.by_ordinal.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Register every HLE module under each of `dlls`. Games import the same
/// functions through many names (kernel32, kernelbase, kernelx, API sets
/// like `api-ms-win-core-synch-l1-2-0`), so callers usually pass the image's
/// whole import list.
#[cfg(target_arch = "x86_64")]
pub fn register_all(reg: &mut Registry, dlls: &[&str]) {
    builtins::register_basic(reg, dlls);
    threading::register_threading(reg, dlls);
    memory::register_memory(reg, dlls);
    sync::register_sync(reg, dlls);
    time::register_time(reg, dlls);
    object::register_objects(reg, dlls);
    file::register_files(reg, dlls);
    module::register_modules(reg, dlls);
    seh::register_seh(reg, dlls);
    xgameruntime::register_xgameruntime(reg);
}

/// One row of the global thunk table.
pub struct ThunkEntry {
    pub dll: String,
    pub name: String,
    pub handler: Option<Handler>,
    pub calls: AtomicU64,
}

static THUNKS: RwLock<Vec<ThunkEntry>> = RwLock::new(Vec::new());

/// Kept for callers of the old flag: `true` selects the `stop` policy.
pub fn set_strict(strict: bool) {
    policy::set_unimplemented_policy(if strict {
        policy::Unimplemented::Stop
    } else {
        policy::Unimplemented::Log
    });
}

pub use policy::{set_unimplemented_policy, unimplemented_policy, Unimplemented};

/// Add a thunk-table entry and return its id.
pub fn add_thunk(dll: &str, name: &ImportName, handler: Option<Handler>) -> u64 {
    let mut t = THUNKS.write().unwrap();
    t.push(ThunkEntry {
        dll: dll.to_owned(),
        name: name.to_string(),
        handler,
        calls: AtomicU64::new(0),
    });
    (t.len() - 1) as u64
}

/// Snapshot of (dll, name, implemented, call count) for reporting.
pub fn thunk_stats() -> Vec<(String, String, bool, u64)> {
    THUNKS
        .read()
        .unwrap()
        .iter()
        .map(|e| {
            (
                e.dll.clone(),
                e.name.clone(),
                e.handler.is_some(),
                e.calls.load(Ordering::Relaxed),
            )
        })
        .collect()
}

/// Entry point from the trampoline. Never unwinds into guest code.
pub fn dispatch(id: u64, call: &Call) -> u64 {
    let (handler, dll, name, first) = {
        let t = THUNKS.read().unwrap();
        let Some(e) = t.get(id as usize) else {
            log::error!("HLE: call through unknown thunk id {id}");
            std::process::abort();
        };
        let n = e.calls.fetch_add(1, Ordering::Relaxed);
        (e.handler, e.dll.clone(), e.name.clone(), n == 0)
    };
    match handler {
        Some(h) => match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| h(call))) {
            Ok(v) => v,
            Err(_) => {
                log::error!("HLE: handler {dll}!{name} panicked; aborting");
                std::process::abort();
            }
        },
        None => {
            let p = policy::unimplemented_policy();
            if p == Unimplemented::Stop {
                eprintln!(
                    "HLE: unimplemented {dll}!{name} called from {} \
                     (rcx={:#x}, rdx={:#x}, r8={:#x}, r9={:#x}) [stop]",
                    caller_location(call),
                    call.arg(0),
                    call.arg(1),
                    call.arg(2),
                    call.arg(3)
                );
                std::process::exit(policy::STOP_EXIT_CODE);
            }
            let ret = policy::unimplemented_return(p, &dll, &name);
            if first {
                log::warn!(
                    "HLE: unimplemented {dll}!{name}(rcx={:#x}, rdx={:#x}, r8={:#x}, r9={:#x}) -> {ret:#x}",
                    call.arg(0), call.arg(1), call.arg(2), call.arg(3)
                );
            }
            ret
        }
    }
}

/// The caller's return address as `module+0xRVA`, or a bare address.
fn caller_location(call: &Call) -> String {
    #[cfg(target_arch = "x86_64")]
    if let Some(f) = call.frame() {
        let rip = f.caller_rip;
        return match module::guest_containing(rip) {
            Some(m) => format!("{}+{:#x}", m.name, rip - m.base),
            None => format!("{rip:#x}"),
        };
    }
    let _ = call;
    "<unknown caller>".into()
}

/// Encode the 23-byte thunk for `id` that jumps to `target`.
pub fn encode_thunk(id: u64, target: u64) -> [u8; 23] {
    let mut b = [0u8; 23];
    b[0..2].copy_from_slice(&[0x49, 0xBA]); // mov r10, imm64
    b[2..10].copy_from_slice(&id.to_le_bytes());
    b[10..12].copy_from_slice(&[0x49, 0xBB]); // mov r11, imm64
    b[12..20].copy_from_slice(&target.to_le_bytes());
    b[20..23].copy_from_slice(&[0x41, 0xFF, 0xE3]); // jmp r11
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn add(c: &Call) -> u64 {
        c.arg(0) + c.arg(1)
    }

    #[test]
    fn registry_is_case_insensitive_and_extension_agnostic() {
        let mut r = Registry::new();
        r.register("KernelX.DLL", "Foo", add);
        let name = ImportName::Name {
            hint: 0,
            name: "Foo".into(),
        };
        assert!(r.lookup("kernelx", &name).is_some());
        assert!(r.lookup("KERNELX.dll", &name).is_some());
        assert!(r.lookup("other.dll", &name).is_none());
    }

    #[test]
    fn stack_args() {
        let stack = [50u64, 60];
        let c = unsafe { Call::new([1, 2, 3, 4], stack.as_ptr()) };
        assert_eq!((c.arg(0), c.arg(3), c.arg(4), c.arg(5)), (1, 4, 50, 60));
    }

    #[test]
    fn dispatch_counts_calls_and_stubs_unknown() {
        let name = ImportName::Name {
            hint: 0,
            name: "Missing".into(),
        };
        let id = add_thunk("x.dll", &name, None);
        let c = unsafe { Call::new([7, 0, 0, 0], std::ptr::null()) };
        assert_eq!(dispatch(id, &c), 0);
        assert_eq!(dispatch(id, &c), 0);
        let stats = thunk_stats();
        assert_eq!(stats[id as usize].3, 2);
    }

    #[test]
    fn float_args_by_position() {
        let stack = [1.5f64.to_bits(), 2.5f32.to_bits() as u64];
        let xmm = [
            [0; 2],
            [3.25f64.to_bits(), 0],
            [0; 2],
            [(-1.0f32).to_bits() as u64, 0],
        ];
        let c = unsafe { Call::with_xmm([7, 0, 9, 0], xmm, stack.as_ptr()) };
        assert_eq!(c.arg(0), 7);
        assert_eq!(c.arg_f64(1), 3.25);
        assert_eq!(c.arg(2), 9);
        assert_eq!(c.arg_f32(3), -1.0);
        assert_eq!(c.arg_f64(4), 1.5);
        assert_eq!(c.arg_f32(5), 2.5);
        c.ret_f32(0.5);
        assert_eq!(c.xmm_ret(), [0.5f32.to_bits() as u64, 0]);
    }

    #[test]
    fn thunk_encoding() {
        let t = encode_thunk(0x1122, 0xAABB);
        assert_eq!(&t[..4], &[0x49, 0xBA, 0x22, 0x11]);
        assert_eq!(&t[20..], &[0x41, 0xFF, 0xE3]);
    }
}
