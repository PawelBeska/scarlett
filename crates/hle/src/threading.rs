//! Thread-environment APIs answered from the guest TEB: last error, thread
//! and process ids, pseudo handles, dynamic TLS (`TlsAlloc` & co.).
//!
//! On Windows the TEB is the host's real one, so slot allocation goes
//! through the host's `TlsAlloc` to avoid colliding with host libraries.
//! Elsewhere slots live in the fake TEB; freed indices are not reused, so a
//! fresh index always reads 0 on every thread without tracking all TEBs.

use crate::teb;
use crate::{Call, Registry};

pub const ERROR_INVALID_PARAMETER: u32 = 87;

fn get_last_error(_: &Call) -> u64 {
    teb::last_error() as u64
}

fn set_last_error(c: &Call) -> u64 {
    teb::set_last_error(c.arg(0) as u32);
    0
}

fn get_current_thread_id(_: &Call) -> u64 {
    teb::current_thread_id() as u64
}

fn get_current_process_id(_: &Call) -> u64 {
    teb::current_process_id() as u64
}

fn get_current_thread(_: &Call) -> u64 {
    (-2i64) as u64
}

fn get_current_process(_: &Call) -> u64 {
    (-1i64) as u64
}

#[cfg(not(windows))]
mod slots {
    use super::*;
    use std::sync::Mutex;

    const MAX: u32 = (teb::TLS_INLINE_SLOTS + teb::TLS_EXPANSION_SLOTS) as u32;

    /// (next never-used index, freed indices)
    static STATE: Mutex<(u32, Vec<u32>)> = Mutex::new((0, Vec::new()));

    fn live(i: u32) -> bool {
        let s = STATE.lock().unwrap();
        i < s.0 && !s.1.contains(&i)
    }

    pub fn alloc(_: &Call) -> u64 {
        let mut s = STATE.lock().unwrap();
        if s.0 >= MAX {
            teb::set_last_error(8); // ERROR_NOT_ENOUGH_MEMORY
            return teb::TLS_OUT_OF_INDEXES as u64;
        }
        s.0 += 1;
        (s.0 - 1) as u64
    }

    pub fn free(c: &Call) -> u64 {
        let i = c.arg(0) as u32;
        if !live(i) {
            teb::set_last_error(ERROR_INVALID_PARAMETER);
            return 0;
        }
        STATE.lock().unwrap().1.push(i);
        1
    }

    pub fn get(c: &Call) -> u64 {
        match teb::tls_slot(c.arg(0) as u32) {
            Some(p) => {
                teb::set_last_error(0);
                // SAFETY: tls_slot returns a valid slot of the current TEB.
                unsafe { p.read() }
            }
            None => {
                teb::set_last_error(ERROR_INVALID_PARAMETER);
                0
            }
        }
    }

    pub fn set(c: &Call) -> u64 {
        match teb::tls_slot(c.arg(0) as u32) {
            Some(p) => {
                // SAFETY: as in `get`.
                unsafe { p.write(c.arg(1)) };
                1
            }
            None => {
                teb::set_last_error(ERROR_INVALID_PARAMETER);
                0
            }
        }
    }
}

#[cfg(windows)]
mod slots {
    use super::*;
    use windows_sys::Win32::System::Threading::{TlsAlloc, TlsFree, TlsGetValue, TlsSetValue};

    // SAFETY (all four): plain kernel32 calls on the real TEB of this thread,
    // which is also the guest's TEB.
    pub fn alloc(_: &Call) -> u64 {
        unsafe { TlsAlloc() as u64 }
    }
    pub fn free(c: &Call) -> u64 {
        unsafe { TlsFree(c.arg(0) as u32) as u64 }
    }
    pub fn get(c: &Call) -> u64 {
        unsafe { TlsGetValue(c.arg(0) as u32) as u64 }
    }
    pub fn set(c: &Call) -> u64 {
        unsafe { TlsSetValue(c.arg(0) as u32, c.arg(1) as *const _) as u64 }
    }
}

/// Register the handlers under each of `dlls` (kernel32, kernelbase,
/// `api-ms-win-core-processthreads-l1-1-0`, …).
pub fn register_threading(reg: &mut Registry, dlls: &[&str]) {
    for dll in dlls {
        reg.register(dll, "GetLastError", get_last_error)
            .register(dll, "SetLastError", set_last_error)
            .register(dll, "GetCurrentThreadId", get_current_thread_id)
            .register(dll, "GetCurrentProcessId", get_current_process_id)
            .register(dll, "GetCurrentThread", get_current_thread)
            .register(dll, "GetCurrentProcess", get_current_process)
            .register(dll, "TlsAlloc", slots::alloc)
            .register(dll, "TlsFree", slots::free)
            .register(dll, "TlsGetValue", slots::get)
            .register(dll, "TlsSetValue", slots::set);
    }
}
