//! Host virtual memory: reserve/commit, protect, release.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prot {
    Read,
    ReadWrite,
    ReadExec,
    ReadWriteExec,
}

#[derive(Debug)]
pub struct MemError(pub String);

impl fmt::Display for MemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

pub const PAGE: usize = 0x1000;

/// An owned, page-aligned block of host memory. Freed on drop.
pub struct Region {
    base: *mut u8,
    size: usize,
}

// SAFETY: Region is a plain owned allocation; synchronisation of guest
// execution against it is the caller's job.
unsafe impl Send for Region {}
unsafe impl Sync for Region {}

impl Region {
    /// Allocate `size` bytes (rounded up to pages), read-write, zeroed.
    /// With `at`, try that exact address first; the result may differ —
    /// check [`Region::base`].
    pub fn alloc(size: usize, at: Option<u64>) -> Result<Region, MemError> {
        let size = size.div_ceil(PAGE) * PAGE;
        let base = sys::alloc(size, at)?;
        Ok(Region { base, size })
    }

    pub fn base(&self) -> u64 {
        self.base as u64
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.base
    }

    fn check(&self, offset: usize, len: usize) -> Result<(), MemError> {
        match offset.checked_add(len) {
            Some(end) if end <= self.size => Ok(()),
            _ => Err(MemError(format!(
                "access {offset:#x}+{len:#x} outside region of {:#x}",
                self.size
            ))),
        }
    }

    /// Copy bytes in. Only valid while the target pages are writable.
    pub fn write(&self, offset: usize, bytes: &[u8]) -> Result<(), MemError> {
        self.check(offset, bytes.len())?;
        // SAFETY: bounds checked above; region is ours and currently RW.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.base.add(offset), bytes.len())
        };
        Ok(())
    }

    pub fn read_u64(&self, offset: usize) -> Result<u64, MemError> {
        self.check(offset, 8)?;
        // SAFETY: bounds checked; unaligned read is fine on x86.
        Ok(unsafe { self.base.add(offset).cast::<u64>().read_unaligned() })
    }

    pub fn write_u64(&self, offset: usize, v: u64) -> Result<(), MemError> {
        self.write(offset, &v.to_le_bytes())
    }

    pub fn read_u32(&self, offset: usize) -> Result<u32, MemError> {
        self.check(offset, 4)?;
        // SAFETY: as above.
        Ok(unsafe { self.base.add(offset).cast::<u32>().read_unaligned() })
    }

    pub fn write_u32(&self, offset: usize, v: u32) -> Result<(), MemError> {
        self.write(offset, &v.to_le_bytes())
    }

    /// Change protection of a page-aligned sub-range.
    pub fn protect(&self, offset: usize, len: usize, prot: Prot) -> Result<(), MemError> {
        if !offset.is_multiple_of(PAGE) {
            return Err(MemError(format!(
                "protect offset {offset:#x} not page aligned"
            )));
        }
        let len = len.div_ceil(PAGE) * PAGE;
        self.check(offset, len)?;
        // SAFETY: range is inside our allocation.
        unsafe { sys::protect(self.base.add(offset), len, prot) }
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        // SAFETY: base/size came from sys::alloc.
        unsafe { sys::free(self.base, self.size) }
    }
}

#[cfg(unix)]
mod sys {
    use super::{MemError, Prot};

    pub fn alloc(size: usize, at: Option<u64>) -> Result<*mut u8, MemError> {
        let hint = at.unwrap_or(0) as *mut libc::c_void;
        // SAFETY: anonymous private mapping; the hint is only a hint.
        let p = unsafe {
            libc::mmap(
                hint,
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(MemError(format!(
                "mmap({size:#x}) failed: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok(p.cast())
    }

    pub unsafe fn protect(p: *mut u8, len: usize, prot: Prot) -> Result<(), MemError> {
        let flags = match prot {
            Prot::Read => libc::PROT_READ,
            Prot::ReadWrite => libc::PROT_READ | libc::PROT_WRITE,
            Prot::ReadExec => libc::PROT_READ | libc::PROT_EXEC,
            Prot::ReadWriteExec => libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
        };
        if libc::mprotect(p.cast(), len, flags) != 0 {
            return Err(MemError(format!(
                "mprotect failed: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok(())
    }

    pub unsafe fn free(p: *mut u8, size: usize) {
        libc::munmap(p.cast(), size);
    }
}

#[cfg(windows)]
mod sys {
    use super::{MemError, Prot};
    use windows_sys::Win32::System::Memory::*;

    pub fn alloc(size: usize, at: Option<u64>) -> Result<*mut u8, MemError> {
        let flags = MEM_RESERVE | MEM_COMMIT;
        // SAFETY: plain VirtualAlloc calls; a null result is handled.
        unsafe {
            if let Some(addr) = at {
                let p = VirtualAlloc(addr as *const _, size, flags, PAGE_READWRITE);
                if !p.is_null() {
                    return Ok(p.cast());
                }
            }
            let p = VirtualAlloc(std::ptr::null(), size, flags, PAGE_READWRITE);
            if p.is_null() {
                return Err(MemError(format!(
                    "VirtualAlloc({size:#x}) failed: {}",
                    std::io::Error::last_os_error()
                )));
            }
            Ok(p.cast())
        }
    }

    pub unsafe fn protect(p: *mut u8, len: usize, prot: Prot) -> Result<(), MemError> {
        let flags = match prot {
            Prot::Read => PAGE_READONLY,
            Prot::ReadWrite => PAGE_READWRITE,
            Prot::ReadExec => PAGE_EXECUTE_READ,
            Prot::ReadWriteExec => PAGE_EXECUTE_READWRITE,
        };
        let mut old = 0;
        if VirtualProtect(p as *const _, len, flags, &mut old) == 0 {
            return Err(MemError(format!(
                "VirtualProtect failed: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok(())
    }

    pub unsafe fn free(p: *mut u8, _size: usize) {
        VirtualFree(p.cast(), 0, MEM_RELEASE);
    }
}
