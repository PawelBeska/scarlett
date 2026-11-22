//! Windows-only pieces: host import forwarding and unwind registration.

use crate::mem::Region;
use scarlett_pe::{Dir, ImportName, PeFile};
use std::ffi::CString;
use windows_sys::Win32::System::Diagnostics::Debug::{
    RtlAddFunctionTable, RtlDeleteFunctionTable, IMAGE_RUNTIME_FUNCTION_ENTRY,
};
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};

/// Resolve `dll!name` against the host. `None` if either is missing.
pub fn host_proc(dll: &str, name: &ImportName) -> Option<u64> {
    let dll_c = CString::new(dll).ok()?;
    // SAFETY: plain Win32 calls with NUL-terminated strings.
    unsafe {
        let module = LoadLibraryA(dll_c.as_ptr().cast());
        if module.is_null() {
            return None;
        }
        let proc = match name {
            ImportName::Name { name, .. } => {
                let n = CString::new(name.as_str()).ok()?;
                GetProcAddress(module, n.as_ptr().cast())
            }
            // MAKEINTRESOURCEA: ordinal in the low word of the pointer.
            ImportName::Ordinal(o) => GetProcAddress(module, *o as usize as *const u8),
        };
        proc.map(|f| f as usize as u64)
    }
}

/// `.pdata` registration so SEH/C++ exceptions can unwind guest frames.
pub struct FunctionTable(*const IMAGE_RUNTIME_FUNCTION_ENTRY);

// SAFETY: the pointer refers into the image region, which outlives this.
unsafe impl Send for FunctionTable {}
unsafe impl Sync for FunctionTable {}

impl FunctionTable {
    pub fn register(image: &Region, pe: &PeFile) -> Option<FunctionTable> {
        let dir = pe.headers.dir(Dir::Exception);
        if !dir.is_present() {
            return None;
        }
        let count = dir.size / 12;
        // SAFETY: the directory lies inside the mapped image (validated by
        // the parser against SizeOfImage); entries are 12-byte RVAs.
        unsafe {
            let table = image.as_ptr().add(dir.rva as usize) as *const IMAGE_RUNTIME_FUNCTION_ENTRY;
            if RtlAddFunctionTable(table, count, image.base()) == 0 {
                log::warn!(
                    "RtlAddFunctionTable failed; exceptions will not unwind through guest code"
                );
                return None;
            }
            Some(FunctionTable(table))
        }
    }
}

impl Drop for FunctionTable {
    fn drop(&mut self) {
        // SAFETY: registered in `register` with this exact pointer.
        unsafe { RtlDeleteFunctionTable(self.0) };
    }
}
