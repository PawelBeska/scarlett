//! Memory APIs: `Virtual*` (page-level) and `Heap*` (allocator).
//!
//! On Windows `Virtual*` go straight to the host's own functions. Elsewhere
//! they are rebuilt on `mmap`/`mprotect` with a reservation table, so
//! reserve/commit/decommit/release and `VirtualQuery` behave as on Windows:
//! reservations are 64 KiB aligned, committed pages read as zero, a
//! release must name the reservation base with size 0.
//!
//! `Heap*` is a portable allocator on top of the Rust global allocator
//! (16-byte aligned blocks, like the x64 NT heap). All heaps share it; the
//! handles only exist so guest code has something non-null to pass around.

use crate::teb::set_last_error;
use crate::{Call, Registry};

pub const MEM_COMMIT: u32 = 0x1000;
pub const MEM_RESERVE: u32 = 0x2000;
pub const MEM_DECOMMIT: u32 = 0x4000;
pub const MEM_RELEASE: u32 = 0x8000;
pub const MEM_FREE: u32 = 0x1_0000;
pub const MEM_PRIVATE: u32 = 0x2_0000;

pub const PAGE_NOACCESS: u32 = 0x01;
pub const PAGE_READONLY: u32 = 0x02;
pub const PAGE_READWRITE: u32 = 0x04;
pub const PAGE_WRITECOPY: u32 = 0x08;
pub const PAGE_EXECUTE: u32 = 0x10;
pub const PAGE_EXECUTE_READ: u32 = 0x20;
pub const PAGE_EXECUTE_READWRITE: u32 = 0x40;
pub const PAGE_EXECUTE_WRITECOPY: u32 = 0x80;

pub const ERROR_NOT_ENOUGH_MEMORY: u32 = 8;
pub const ERROR_INVALID_PARAMETER: u32 = 87;
pub const ERROR_INVALID_ADDRESS: u32 = 487;

pub const PAGE: u64 = 0x1000;
pub const ALLOCATION_GRANULARITY: u64 = 0x1_0000;

/// Size of `MEMORY_BASIC_INFORMATION` on x64.
pub const MBI_SIZE: u64 = 48;

// ---------------------------------------------------------------- Virtual*

#[cfg(unix)]
mod virt {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    struct Reservation {
        size: u64,
        alloc_prot: u32,
        /// Protection per page; 0 = reserved, not committed.
        pages: Vec<u32>,
    }

    static TABLE: Mutex<BTreeMap<u64, Reservation>> = Mutex::new(BTreeMap::new());

    fn host_prot(p: u32) -> Option<libc::c_int> {
        Some(match p & 0xFF {
            PAGE_NOACCESS => libc::PROT_NONE,
            PAGE_READONLY => libc::PROT_READ,
            PAGE_READWRITE | PAGE_WRITECOPY => libc::PROT_READ | libc::PROT_WRITE,
            PAGE_EXECUTE => libc::PROT_EXEC,
            PAGE_EXECUTE_READ => libc::PROT_READ | libc::PROT_EXEC,
            PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY => {
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC
            }
            _ => return None,
        })
    }

    fn map(at: u64, len: u64, fixed: bool) -> Option<u64> {
        let flags =
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | if fixed { libc::MAP_FIXED } else { 0 };
        // SAFETY: anonymous mapping; MAP_FIXED only over ranges we own.
        let p = unsafe { libc::mmap(at as *mut _, len as usize, libc::PROT_NONE, flags, -1, 0) };
        (p != libc::MAP_FAILED).then_some(p as u64)
    }

    fn unmap(at: u64, len: u64) {
        // SAFETY: only called on ranges we mapped.
        unsafe { libc::munmap(at as *mut _, len as usize) };
    }

    /// Reserve `size` bytes at a 64 KiB aligned address (`at`, or anywhere).
    fn reserve(at: u64, size: u64) -> Option<u64> {
        let g = ALLOCATION_GRANULARITY;
        if at != 0 {
            let base = at & !(g - 1);
            let p = map(base, size, false)?;
            if p != base {
                unmap(p, size);
                return None;
            }
            return Some(p);
        }
        // Over-allocate and trim to get 64 KiB alignment.
        let p = map(0, size + g, false)?;
        let base = (p + g - 1) & !(g - 1);
        if base > p {
            unmap(p, base - p);
        }
        let tail = p + size + g - (base + size);
        if tail > 0 {
            unmap(base + size, tail);
        }
        Some(base)
    }

    /// Reservation containing `[addr, addr+len)`.
    fn find(
        t: &mut BTreeMap<u64, Reservation>,
        addr: u64,
        len: u64,
    ) -> Option<(u64, &mut Reservation)> {
        let (&base, r) = t.range_mut(..=addr).next_back()?;
        (addr + len <= base + r.size).then_some((base, r))
    }

    fn page_range(addr: u64, size: u64) -> (u64, u64) {
        let lo = addr & !(PAGE - 1);
        let hi = (addr + size).div_ceil(PAGE) * PAGE;
        (lo, hi - lo)
    }

    pub fn alloc(c: &Call) -> u64 {
        let (addr, size, kind, prot) = (c.arg(0), c.arg(1), c.arg(2) as u32, c.arg(3) as u32);
        let Some(hp) = host_prot(prot) else {
            set_last_error(ERROR_INVALID_PARAMETER);
            return 0;
        };
        if size == 0 || kind & (MEM_COMMIT | MEM_RESERVE) == 0 {
            set_last_error(ERROR_INVALID_PARAMETER);
            return 0;
        }
        let mut t = TABLE.lock().unwrap();
        // MEM_COMMIT with no address reserves too, as on Windows.
        let reserving = kind & MEM_RESERVE != 0 || addr == 0;
        let (lo, len) = if reserving {
            let len = size.div_ceil(PAGE) * PAGE;
            let Some(base) = reserve(addr, len) else {
                set_last_error(if addr != 0 {
                    ERROR_INVALID_ADDRESS
                } else {
                    ERROR_NOT_ENOUGH_MEMORY
                });
                return 0;
            };
            t.insert(
                base,
                Reservation {
                    size: len,
                    alloc_prot: prot,
                    pages: vec![0; (len / PAGE) as usize],
                },
            );
            if kind & MEM_COMMIT == 0 {
                return base;
            }
            (base, len)
        } else {
            page_range(addr, size)
        };
        let Some((base, r)) = find(&mut t, lo, len) else {
            set_last_error(ERROR_INVALID_ADDRESS);
            return 0;
        };
        // SAFETY: range lies inside a reservation we mapped.
        if unsafe { libc::mprotect(lo as *mut _, len as usize, hp) } != 0 {
            set_last_error(ERROR_NOT_ENOUGH_MEMORY);
            return 0;
        }
        let first = ((lo - base) / PAGE) as usize;
        r.pages[first..first + (len / PAGE) as usize].fill(prot);
        lo
    }

    pub fn free(c: &Call) -> u64 {
        let (addr, size, kind) = (c.arg(0), c.arg(1), c.arg(2) as u32);
        let mut t = TABLE.lock().unwrap();
        match kind {
            MEM_RELEASE => {
                if size != 0 || !t.contains_key(&addr) {
                    set_last_error(ERROR_INVALID_PARAMETER);
                    return 0;
                }
                let r = t.remove(&addr).unwrap();
                unmap(addr, r.size);
                1
            }
            MEM_DECOMMIT => {
                // size 0 = from addr to the end of the reservation.
                let Some((base, r)) = find(&mut t, addr, 1) else {
                    set_last_error(ERROR_INVALID_ADDRESS);
                    return 0;
                };
                let (lo, len) = if size == 0 {
                    page_range(addr, base + r.size - addr)
                } else {
                    page_range(addr, size)
                };
                if lo + len > base + r.size {
                    set_last_error(ERROR_INVALID_PARAMETER);
                    return 0;
                }
                // Fresh PROT_NONE mapping drops the contents (zero on recommit).
                if map(lo, len, true) != Some(lo) {
                    set_last_error(ERROR_NOT_ENOUGH_MEMORY);
                    return 0;
                }
                let first = ((lo - base) / PAGE) as usize;
                r.pages[first..first + (len / PAGE) as usize].fill(0);
                1
            }
            _ => {
                set_last_error(ERROR_INVALID_PARAMETER);
                0
            }
        }
    }

    pub fn protect(c: &Call) -> u64 {
        let (addr, size, prot, old) = (c.arg(0), c.arg(1), c.arg(2) as u32, c.arg_ptr::<u32>(3));
        let Some(hp) = host_prot(prot) else {
            set_last_error(ERROR_INVALID_PARAMETER);
            return 0;
        };
        let (lo, len) = page_range(addr, size.max(1));
        let mut t = TABLE.lock().unwrap();
        let Some((base, r)) = find(&mut t, lo, len) else {
            set_last_error(ERROR_INVALID_ADDRESS);
            return 0;
        };
        let first = ((lo - base) / PAGE) as usize;
        let pages = &mut r.pages[first..first + (len / PAGE) as usize];
        if pages.contains(&0) {
            set_last_error(ERROR_INVALID_ADDRESS); // not committed
            return 0;
        }
        // SAFETY: committed range inside our reservation.
        if unsafe { libc::mprotect(lo as *mut _, len as usize, hp) } != 0 {
            set_last_error(ERROR_INVALID_PARAMETER);
            return 0;
        }
        if old.is_null() {
            // Windows requires lpflOldProtect.
            set_last_error(ERROR_INVALID_PARAMETER);
            return 0;
        }
        // SAFETY: guest-provided out pointer, checked non-null.
        unsafe { old.write_unaligned(pages[0]) };
        pages.fill(prot);
        1
    }

    pub fn query(c: &Call) -> u64 {
        let (addr, out, len) = (c.arg(0), c.arg_ptr::<u8>(1), c.arg(2));
        if out.is_null() || len < MBI_SIZE {
            set_last_error(ERROR_INVALID_PARAMETER);
            return 0;
        }
        let page = addr & !(PAGE - 1);
        let mut t = TABLE.lock().unwrap();
        let mut mbi = [0u64; 6];
        let put32 = |mbi: &mut [u64; 6], off: usize, v: u32| {
            let w = &mut mbi[off / 8];
            let shift = (off % 8) * 8;
            *w = (*w & !(0xFFFF_FFFFu64 << shift)) | ((v as u64) << shift);
        };
        match find(&mut t, page, PAGE) {
            Some((base, r)) => {
                let i = ((page - base) / PAGE) as usize;
                let prot = r.pages[i];
                let run = r.pages[i..].iter().take_while(|&&p| p == prot).count() as u64;
                mbi[0] = page;
                mbi[1] = base;
                put32(&mut mbi, 16, r.alloc_prot);
                mbi[3] = run * PAGE;
                put32(
                    &mut mbi,
                    32,
                    if prot == 0 { MEM_RESERVE } else { MEM_COMMIT },
                );
                put32(&mut mbi, 36, prot);
                put32(&mut mbi, 40, MEM_PRIVATE);
            }
            None => {
                let next = t.range(page..).next().map_or(page + PAGE, |(&b, _)| b);
                mbi[0] = page;
                mbi[3] = next - page;
                put32(&mut mbi, 32, MEM_FREE);
                put32(&mut mbi, 36, PAGE_NOACCESS);
            }
        }
        // SAFETY: guest buffer of at least MBI_SIZE bytes.
        unsafe { std::ptr::copy_nonoverlapping(mbi.as_ptr().cast::<u8>(), out, MBI_SIZE as usize) };
        MBI_SIZE
    }
}

#[cfg(windows)]
mod virt {
    use super::*;
    use windows_sys::Win32::System::Memory::*;

    // SAFETY (all): the guest passes the same arguments the host API takes.
    pub fn alloc(c: &Call) -> u64 {
        unsafe {
            VirtualAlloc(
                c.arg(0) as _,
                c.arg(1) as usize,
                c.arg(2) as u32,
                c.arg(3) as u32,
            ) as u64
        }
    }
    pub fn free(c: &Call) -> u64 {
        unsafe { VirtualFree(c.arg(0) as _, c.arg(1) as usize, c.arg(2) as u32) as u64 }
    }
    pub fn protect(c: &Call) -> u64 {
        unsafe {
            VirtualProtect(
                c.arg(0) as _,
                c.arg(1) as usize,
                c.arg(2) as u32,
                c.arg_ptr(3),
            ) as u64
        }
    }
    pub fn query(c: &Call) -> u64 {
        unsafe { VirtualQuery(c.arg(0) as _, c.arg_ptr(1), c.arg(2) as usize) as u64 }
    }
}

// ---------------------------------------------------------------- Heap*

pub mod heap {
    use super::*;
    use std::alloc::{alloc, alloc_zeroed, dealloc, realloc, Layout};
    use std::sync::atomic::{AtomicU64, Ordering};

    pub const HEAP_ZERO_MEMORY: u32 = 0x08;
    pub const HEAP_REALLOC_IN_PLACE_ONLY: u32 = 0x10;
    /// Handle returned by `GetProcessHeap`.
    pub const PROCESS_HEAP: u64 = 0x0000_0001_0000_0000 | 0x5CA1;

    /// Header right before the user pointer: `[user size, MAGIC | capacity]`.
    /// Capacity (what was allocated) can exceed the size after an in-place shrink.
    const HDR: usize = 16;
    const MAGIC: u64 = 0x5CA7 << 48;
    const CAP_MASK: u64 = (1 << 48) - 1;

    static NEXT_HEAP: AtomicU64 = AtomicU64::new(PROCESS_HEAP + 0x10);

    fn layout(size: usize) -> Option<Layout> {
        Layout::from_size_align(size.checked_add(HDR)?, 16).ok()
    }

    /// Header of a block from [`allocate`]; `None` if the magic is missing.
    unsafe fn header(p: *mut u8) -> Option<*mut u64> {
        let h = unsafe { p.sub(HDR) }.cast::<u64>();
        (unsafe { h.add(1).read() } & !CAP_MASK == MAGIC).then_some(h)
    }

    unsafe fn capacity(h: *mut u64) -> usize {
        (unsafe { h.add(1).read() } & CAP_MASK) as usize
    }

    unsafe fn write_header(raw: *mut u8, size: usize, cap: usize) {
        unsafe {
            raw.cast::<u64>().write(size as u64);
            raw.cast::<u64>().add(1).write(MAGIC | cap as u64);
        }
    }

    pub fn allocate(size: usize, zero: bool) -> *mut u8 {
        let Some(l) = layout(size).filter(|_| (size as u64) <= CAP_MASK) else {
            return std::ptr::null_mut();
        };
        // SAFETY: non-zero layout (HDR > 0).
        let raw = unsafe {
            if zero {
                alloc_zeroed(l)
            } else {
                alloc(l)
            }
        };
        if raw.is_null() {
            return raw;
        }
        // SAFETY: raw has room for the header.
        unsafe {
            write_header(raw, size, size);
            raw.add(HDR)
        }
    }

    /// Size of a block from [`allocate`], or `None` for a foreign pointer.
    fn block_size(p: *mut u8) -> Option<usize> {
        if p.is_null() || !(p as usize).is_multiple_of(16) {
            return None;
        }
        // SAFETY: p came from `allocate` if the magic matches.
        unsafe { header(p).map(|h| h.read() as usize) }
    }

    fn release(p: *mut u8) -> bool {
        if block_size(p).is_none() {
            return false;
        }
        // SAFETY: block from `allocate`; clear the magic against double free.
        unsafe {
            let h = header(p).unwrap();
            let cap = capacity(h);
            h.add(1).write(0);
            dealloc(p.sub(HDR), layout(cap).unwrap());
        }
        true
    }

    fn get_process_heap(_: &Call) -> u64 {
        PROCESS_HEAP
    }

    fn heap_create(_: &Call) -> u64 {
        NEXT_HEAP.fetch_add(0x10, Ordering::Relaxed)
    }

    fn heap_destroy(_: &Call) -> u64 {
        1 // blocks of a destroyed heap are leaked
    }

    fn heap_alloc(c: &Call) -> u64 {
        let p = allocate(c.arg(2) as usize, c.arg(1) as u32 & HEAP_ZERO_MEMORY != 0);
        if p.is_null() {
            set_last_error(ERROR_NOT_ENOUGH_MEMORY);
        }
        p as u64
    }

    fn heap_free(c: &Call) -> u64 {
        let p = c.arg_ptr::<u8>(2);
        if p.is_null() || release(p) {
            1
        } else {
            set_last_error(ERROR_INVALID_PARAMETER);
            0
        }
    }

    fn heap_size(c: &Call) -> u64 {
        block_size(c.arg_ptr(2)).map_or(u64::MAX, |s| s as u64)
    }

    fn heap_realloc(c: &Call) -> u64 {
        let (flags, p, new) = (c.arg(1) as u32, c.arg_ptr::<u8>(2), c.arg(3) as usize);
        let Some(old) = block_size(p) else {
            set_last_error(ERROR_INVALID_PARAMETER);
            return 0;
        };
        if flags & HEAP_REALLOC_IN_PLACE_ONLY != 0 {
            if new > old {
                set_last_error(ERROR_NOT_ENOUGH_MEMORY);
                return 0;
            }
            // Shrink in place: only the recorded size changes.
            // SAFETY: valid header (block_size succeeded).
            unsafe { header(p).unwrap().write(new as u64) };
            return p as u64;
        }
        let Some(nl) = layout(new).filter(|_| (new as u64) <= CAP_MASK) else {
            set_last_error(ERROR_NOT_ENOUGH_MEMORY);
            return 0;
        };
        // SAFETY: block from `allocate` with layout(capacity).
        let raw = unsafe {
            let cap = capacity(header(p).unwrap());
            realloc(p.sub(HDR), layout(cap).unwrap(), nl.size())
        };
        if raw.is_null() {
            set_last_error(ERROR_NOT_ENOUGH_MEMORY);
            return 0;
        }
        // SAFETY: raw is a live block of nl.size() bytes.
        unsafe {
            write_header(raw, new, new);
            if flags & HEAP_ZERO_MEMORY != 0 && new > old {
                std::ptr::write_bytes(raw.add(HDR + old), 0, new - old);
            }
            raw.add(HDR) as u64
        }
    }

    pub(super) fn register(reg: &mut Registry, dll: &str) {
        reg.register(dll, "GetProcessHeap", get_process_heap)
            .register(dll, "HeapCreate", heap_create)
            .register(dll, "HeapDestroy", heap_destroy)
            .register(dll, "HeapAlloc", heap_alloc)
            .register(dll, "HeapFree", heap_free)
            .register(dll, "HeapSize", heap_size)
            .register(dll, "HeapReAlloc", heap_realloc);
    }
}

/// Register `Virtual*` and `Heap*` under each of `dlls`.
pub fn register_memory(reg: &mut Registry, dlls: &[&str]) {
    for dll in dlls {
        reg.register(dll, "VirtualAlloc", virt::alloc)
            .register(dll, "VirtualFree", virt::free)
            .register(dll, "VirtualProtect", virt::protect)
            .register(dll, "VirtualQuery", virt::query);
        heap::register(reg, dll);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(f: fn(&Call) -> u64, args: [u64; 4]) -> u64 {
        // SAFETY: no handler here reads stack arguments.
        f(&unsafe { Call::new(args, std::ptr::null()) })
    }

    #[test]
    fn reserve_commit_query_protect_release() {
        let size = 0x30000;
        let base = call(
            virt::alloc,
            [0, size, MEM_RESERVE as u64, PAGE_NOACCESS as u64],
        );
        assert_ne!(base, 0);
        assert_eq!(base % ALLOCATION_GRANULARITY, 0);

        let mbi = [0u64; 6];
        let q = |addr: u64| {
            call(virt::query, [addr, mbi.as_ptr() as u64, MBI_SIZE, 0]);
            let state = (mbi[4] & 0xFFFF_FFFF) as u32;
            let prot = (mbi[4] >> 32) as u32;
            (mbi[0], mbi[1], mbi[3], state, prot)
        };
        assert_eq!(q(base + 5), (base, base, size, MEM_RESERVE, 0));

        // Commit the middle page.
        let mid = base + 0x10000;
        assert_eq!(
            call(
                virt::alloc,
                [mid + 8, 0x10, MEM_COMMIT as u64, PAGE_READWRITE as u64]
            ),
            mid
        );
        unsafe {
            assert_eq!((mid as *const u64).read(), 0, "fresh commit is zeroed");
            (mid as *mut u64).write(0x1234);
        }
        assert_eq!(q(mid), (mid, base, PAGE, MEM_COMMIT, PAGE_READWRITE));
        assert_eq!(q(base).2, 0x10000, "reserved run before the commit");

        let mut old = 0u32;
        assert_eq!(
            call(
                virt::protect,
                [mid, 8, PAGE_READONLY as u64, &mut old as *mut u32 as u64]
            ),
            1
        );
        assert_eq!(old, PAGE_READWRITE);
        assert_eq!(unsafe { (mid as *const u64).read() }, 0x1234);
        // Protecting uncommitted pages fails.
        assert_eq!(
            call(
                virt::protect,
                [base, 8, PAGE_READONLY as u64, &mut old as *mut u32 as u64]
            ),
            0
        );

        // Decommit + recommit drops the contents.
        assert_eq!(call(virt::free, [mid, PAGE, MEM_DECOMMIT as u64, 0]), 1);
        assert_eq!(q(mid).3, MEM_RESERVE);
        call(
            virt::alloc,
            [mid, PAGE, MEM_COMMIT as u64, PAGE_READWRITE as u64],
        );
        assert_eq!(unsafe { (mid as *const u64).read() }, 0);

        assert_eq!(
            call(virt::free, [mid, 0, MEM_RELEASE as u64, 0]),
            0,
            "not the base"
        );
        assert_eq!(call(virt::free, [base, 0, MEM_RELEASE as u64, 0]), 1);
        assert_eq!(q(base).3, MEM_FREE);
    }

    #[test]
    fn heap_alloc_realloc_free() {
        let h = heap::PROCESS_HEAP;
        let reg = {
            let mut r = Registry::new();
            register_memory(&mut r, &["k"]);
            r
        };
        let f = |n: &str| {
            reg.lookup(
                "k",
                &scarlett_pe::ImportName::Name {
                    hint: 0,
                    name: n.into(),
                },
            )
            .unwrap()
        };
        let p = call(f("HeapAlloc"), [h, heap::HEAP_ZERO_MEMORY as u64, 100, 0]);
        assert_eq!(p % 16, 0);
        assert_eq!(call(f("HeapSize"), [h, 0, p, 0]), 100);
        unsafe { std::ptr::write_bytes(p as *mut u8, 0xAB, 100) };
        let q = call(f("HeapReAlloc"), [h, heap::HEAP_ZERO_MEMORY as u64, p, 300]);
        assert_ne!(q, 0);
        unsafe {
            assert_eq!(*(q as *const u8).add(99), 0xAB);
            assert_eq!(*(q as *const u8).add(100), 0);
            assert_eq!(*(q as *const u8).add(299), 0);
        }
        assert_eq!(call(f("HeapSize"), [h, 0, q, 0]), 300);
        let flags = heap::HEAP_REALLOC_IN_PLACE_ONLY as u64;
        assert_eq!(
            call(f("HeapReAlloc"), [h, flags, q, 40]),
            q,
            "in-place shrink"
        );
        assert_eq!(call(f("HeapSize"), [h, 0, q, 0]), 40);
        assert_eq!(
            call(f("HeapReAlloc"), [h, flags, q, 400]),
            0,
            "cannot grow in place"
        );
        assert_eq!(call(f("HeapFree"), [h, 0, q, 0]), 1);
        assert_eq!(
            call(f("HeapFree"), [h, 0, 0, 0]),
            1,
            "freeing NULL succeeds"
        );
    }
}
