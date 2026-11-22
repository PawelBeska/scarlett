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
//! Limitation (intentional for now): floating-point arguments passed in
//! xmm0-3 and float/vector return values are not forwarded.

pub mod builtins;
#[cfg(target_arch = "x86_64")]
pub mod trampoline;

use scarlett_pe::ImportName;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::RwLock;

/// Arguments of one guest call, in Win64 order.
pub struct Call {
    regs: [u64; 4],
    stack: *const u64,
}

impl Call {
    /// # Safety
    /// `stack` must point at the caller's 5th argument slot, valid for as
    /// many reads as the handler performs via [`Call::arg`].
    pub unsafe fn new(regs: [u64; 4], stack: *const u64) -> Self {
        Call { regs, stack }
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
#[derive(Default)]
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

    pub fn len(&self) -> usize {
        self.by_name.len() + self.by_ordinal.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One row of the global thunk table.
pub struct ThunkEntry {
    pub dll: String,
    pub name: String,
    pub handler: Option<Handler>,
    pub calls: AtomicU64,
}

static THUNKS: RwLock<Vec<ThunkEntry>> = RwLock::new(Vec::new());
static STRICT: AtomicBool = AtomicBool::new(false);

/// In strict mode an unimplemented import aborts the process instead of
/// returning 0. Useful once basic boot works, to find the next missing API.
pub fn set_strict(strict: bool) {
    STRICT.store(strict, Ordering::Relaxed);
}

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
            if STRICT.load(Ordering::Relaxed) {
                log::error!(
                    "HLE: unimplemented {dll}!{name}(rcx={:#x}, rdx={:#x}, r8={:#x}, r9={:#x}) [strict]",
                    call.arg(0), call.arg(1), call.arg(2), call.arg(3)
                );
                std::process::abort();
            }
            if first {
                log::warn!(
                    "HLE: unimplemented {dll}!{name}(rcx={:#x}, rdx={:#x}, r8={:#x}, r9={:#x}) -> 0",
                    call.arg(0), call.arg(1), call.arg(2), call.arg(3)
                );
            }
            0
        }
    }
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
    fn thunk_encoding() {
        let t = encode_thunk(0x1122, 0xAABB);
        assert_eq!(&t[..4], &[0x49, 0xBA, 0x22, 0x11]);
        assert_eq!(&t[20..], &[0x41, 0xFF, 0xE3]);
    }
}
