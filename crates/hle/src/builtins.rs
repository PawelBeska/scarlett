//! A few handlers every image tends to import, mainly so early boot produces
//! readable logs. Real API coverage lives in per-module files added later.

use crate::{Call, Registry};
use std::ffi::CStr;

fn output_debug_string_a(c: &Call) -> u64 {
    let p = c.arg_ptr::<std::ffi::c_char>(0);
    if !p.is_null() {
        // SAFETY: guest passes a NUL-terminated string per API contract.
        let s = unsafe { CStr::from_ptr(p) }.to_string_lossy();
        log::info!(target: "guest", "{}", s.trim_end());
    }
    0
}

fn output_debug_string_w(c: &Call) -> u64 {
    let mut p = c.arg_ptr::<u16>(0);
    if p.is_null() {
        return 0;
    }
    let mut buf = Vec::new();
    // SAFETY: guest passes a NUL-terminated UTF-16 string; cap the walk.
    unsafe {
        while buf.len() < 64 * 1024 && *p != 0 {
            buf.push(*p);
            p = p.add(1);
        }
    }
    log::info!(target: "guest", "{}", String::from_utf16_lossy(&buf).trim_end());
    0
}

fn exit_process(c: &Call) -> u64 {
    let code = c.arg(0) as u32;
    log::info!("guest called ExitProcess({code})");
    std::process::exit(code as i32)
}

/// Register the debug-output and process-exit helpers under the given DLL
/// names (e.g. `["kernel32", "kernelbase"]`, or whatever the image imports).
pub fn register_basic(reg: &mut Registry, dlls: &[&str]) {
    for dll in dlls {
        reg.register(dll, "OutputDebugStringA", output_debug_string_a)
            .register(dll, "OutputDebugStringW", output_debug_string_w)
            .register(dll, "ExitProcess", exit_process);
    }
}
