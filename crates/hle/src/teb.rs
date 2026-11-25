//! Thread and process environment blocks as guest code sees them via `gs`.
//!
//! Guest code reads its TEB directly: `gs:[0x30]` (self), `gs:[0x58]`
//! (static TLS array), `gs:[0x10]` (stack limit, used by `__chkstk`),
//! `gs:[0x68]` (last error), `gs:[0x1480]` (dynamic TLS slots). How `gs`
//! gets there depends on the host:
//!
//! | host          | TEB                     | `gs` handling                                  |
//! |---------------|-------------------------|------------------------------------------------|
//! | Windows       | the real one            | untouched; only `ThreadLocalStoragePointer` is swapped for an extended array |
//! | Linux         | fake, allocated here    | `arch_prctl(ARCH_SET_GS)` once per thread; glibc never uses `gs` |
//! | macOS/Rosetta | fake, allocated here    | libpthread owns `gs`, so it is switched on every guest↔host transition |
//!
//! Static TLS (`__declspec(thread)`): every loaded module with a TLS
//! directory registers a [`TlsTemplate`] and gets an index, which the loader
//! writes to the module's `_tls_index`. Each [`GuestThread`] allocates one
//! block per module, copied from the template, and points the TEB's TLS
//! array at them — the same thing `ntdll` does for a native thread.

use crate::enter::{self, EnterCtx};
use std::alloc::Layout;
use std::cell::Cell;
use std::sync::{OnceLock, RwLock};

/// TEB field offsets (x64).
pub mod off {
    pub const EXCEPTION_LIST: usize = 0x00;
    pub const STACK_BASE: usize = 0x08;
    pub const STACK_LIMIT: usize = 0x10;
    pub const SELF: usize = 0x30;
    pub const CLIENT_PID: usize = 0x40;
    pub const CLIENT_TID: usize = 0x48;
    pub const TLS_POINTER: usize = 0x58;
    pub const PEB: usize = 0x60;
    pub const LAST_ERROR: usize = 0x68;
    pub const TLS_SLOTS: usize = 0x1480;
    pub const TLS_EXPANSION: usize = 0x1780;
    /// Not a Windows field: past the end of the real TEB (0x1838). Fake
    /// TEBs keep the host's `gs` base here so the trampoline can restore it.
    pub const HOST_GS: usize = 0x1F00;
}

/// PEB field offsets (x64).
pub mod peb_off {
    pub const BEING_DEBUGGED: usize = 0x02;
    pub const IMAGE_BASE: usize = 0x10;
}

pub const TEB_SIZE: usize = 0x2000;
pub const PEB_SIZE: usize = 0x800;
/// `TLS_MINIMUM_AVAILABLE`: slots inline in the TEB.
pub const TLS_INLINE_SLOTS: usize = 64;
/// `TLS_EXPANSION_SLOTS`: slots in the side array.
pub const TLS_EXPANSION_SLOTS: usize = 1024;
pub const TLS_OUT_OF_INDEXES: u32 = 0xFFFF_FFFF;

/// One module's static TLS image: `len` bytes copied from `data` (a host
/// address inside the mapped module), followed by `zero_fill` zero bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TlsTemplate {
    pub data: u64,
    pub len: usize,
    pub zero_fill: usize,
    pub align: usize,
}

impl TlsTemplate {
    fn layout(&self) -> Layout {
        Layout::from_size_align((self.len + self.zero_fill).max(1), self.align.max(16))
            .expect("TLS template layout")
    }
}

static MODULES: RwLock<Vec<Option<TlsTemplate>>> = RwLock::new(Vec::new());

/// First guest TLS index. On Windows the host's own modules occupy the
/// low indices of the array, so guest modules go after them.
fn index_base() -> u32 {
    static BASE: OnceLock<u32> = OnceLock::new();
    *BASE.get_or_init(sys::host_tls_entries)
}

/// Register a module's TLS template and return the value for its `_tls_index`.
///
/// Threads that already exist do not get a block for it (same as Windows
/// before 10 for `LoadLibrary`ed modules); load images before running them.
pub fn register_static_tls(t: TlsTemplate) -> u32 {
    let mut m = MODULES.write().unwrap();
    m.push(Some(t));
    index_base() + (m.len() - 1) as u32
}

/// Forget a template (its module is being unmapped). The index is not reused.
pub fn unregister_static_tls(index: u32) {
    let mut m = MODULES.write().unwrap();
    if let Some(slot) = index
        .checked_sub(index_base())
        .and_then(|i| m.get_mut(i as usize))
    {
        *slot = None;
    }
}

/// TLS callbacks of loaded modules, run with `DLL_THREAD_ATTACH`/`DETACH`
/// around every thread created by the guest. `(module base, callback VAs)`.
/// `(module base, callbacks)`; `None` once the module is unloaded.
type CallbackSlot = Option<(u64, Vec<u64>)>;
static TLS_CALLBACKS: RwLock<Vec<CallbackSlot>> = RwLock::new(Vec::new());

pub const DLL_PROCESS_ATTACH: u64 = 1;
pub const DLL_THREAD_ATTACH: u64 = 2;
pub const DLL_THREAD_DETACH: u64 = 3;

/// Register a module's TLS callbacks for thread attach/detach; returns a
/// token for [`unregister_tls_callbacks`].
pub fn register_tls_callbacks(base: u64, callbacks: Vec<u64>) -> usize {
    let mut t = TLS_CALLBACKS.write().unwrap();
    t.push(Some((base, callbacks)));
    t.len() - 1
}

pub fn unregister_tls_callbacks(token: usize) {
    if let Some(slot) = TLS_CALLBACKS.write().unwrap().get_mut(token) {
        *slot = None;
    }
}

/// Snapshot of `(module base, callback)` pairs in load order.
pub fn tls_callbacks() -> Vec<(u64, u64)> {
    TLS_CALLBACKS
        .read()
        .unwrap()
        .iter()
        .flatten()
        .flat_map(|(b, cbs)| cbs.iter().map(move |&cb| (*b, cb)))
        .collect()
}

// ---------------------------------------------------------------- process

/// Process-wide fake PEB (non-Windows hosts; Windows has a real one).
fn peb() -> *mut u8 {
    static PEB: OnceLock<usize> = OnceLock::new();
    *PEB.get_or_init(|| {
        Box::leak(vec![0u64; PEB_SIZE / 8].into_boxed_slice()).as_mut_ptr() as usize
    }) as *mut u8
}

/// Record the main image's base in the PEB (`GetModuleHandle(NULL)`).
/// No-op on Windows, where the real PEB describes the host executable.
pub fn set_image_base(base: u64) {
    if !cfg!(windows) {
        // SAFETY: the fake PEB is a leaked, process-lifetime allocation.
        unsafe { peb().add(peb_off::IMAGE_BASE).cast::<u64>().write(base) };
    }
}

pub fn image_base() -> u64 {
    if cfg!(windows) {
        return 0;
    }
    // SAFETY: as above.
    unsafe { peb().add(peb_off::IMAGE_BASE).cast::<u64>().read() }
}

/// macOS `thread_fast_set_cthread_self64`: machine-dependent trap that sets
/// the calling thread's `gs` base. Rosetta 2 implements it.
#[cfg(target_os = "macos")]
pub const SET_GS_TRAP: u64 = 0x300_0003;

/// Fake thread ids, Windows-style multiples of 4.
#[cfg(not(windows))]
static NEXT_TID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0x1004);

// ---------------------------------------------------------------- threads

thread_local! {
    static CURRENT: Cell<*mut u8> = const { Cell::new(std::ptr::null_mut()) };
    static CURRENT_CTX: Cell<*mut EnterCtx> = const { Cell::new(std::ptr::null_mut()) };
}

/// TEB of the guest thread attached to the calling OS thread, if any.
/// HLE handlers use this; `gs` is the host's own while they run.
pub fn current_teb() -> Option<*mut u8> {
    let t = CURRENT.with(Cell::get);
    (!t.is_null()).then_some(t)
}

fn teb_u64(off: usize) -> Option<u64> {
    // SAFETY: current_teb points at a live TEB of at least TEB_SIZE bytes.
    current_teb().map(|t| unsafe { t.add(off).cast::<u64>().read() })
}

/// Call back into guest code (`f(a0, a1, a2)`, Win64) from an HLE handler
/// running on a guest thread, e.g. an `InitOnce` or thread-pool callback.
///
/// # Safety
/// `f` must be guest code that is safe to run natively.
///
/// # Panics
/// If the calling thread has no [`GuestThread`].
pub unsafe fn call_guest(f: u64, a0: u64, a1: u64, a2: u64) -> u64 {
    unsafe { call_guest4(f, [a0, a1, a2, 0]) }
}

/// [`call_guest`] with four register arguments.
///
/// # Safety
/// As [`call_guest`].
pub unsafe fn call_guest4(f: u64, args: [u64; 4]) -> u64 {
    let ctx = CURRENT_CTX.with(Cell::get);
    assert!(!ctx.is_null(), "call_guest outside a guest thread");
    unsafe { enter::enter(f, args, ctx) }
}

/// `ExitThread` from an HLE handler: abandon every guest frame on this
/// thread and make the outermost [`GuestThread::call`] return `value`.
///
/// # Safety
/// Must be called from a handler on a guest thread that is inside
/// [`GuestThread::call`], holding no locks or resources that need dropping.
pub unsafe fn exit_guest(value: u64) -> ! {
    let ctx = CURRENT_CTX.with(Cell::get);
    assert!(
        !ctx.is_null() && unsafe { (*ctx).exit_rsp } != 0,
        "exit_guest outside guest code"
    );
    unsafe { enter::exit(ctx, value) }
}

pub fn last_error() -> u32 {
    // SAFETY: see teb_u64.
    current_teb().map_or(0, |t| unsafe {
        t.add(off::LAST_ERROR).cast::<u32>().read()
    })
}

pub fn set_last_error(code: u32) {
    if let Some(t) = current_teb() {
        // SAFETY: see teb_u64.
        unsafe { t.add(off::LAST_ERROR).cast::<u32>().write(code) };
    }
}

pub fn current_thread_id() -> u32 {
    teb_u64(off::CLIENT_TID).unwrap_or(0) as u32
}

pub fn current_process_id() -> u32 {
    teb_u64(off::CLIENT_PID).unwrap_or(0) as u32
}

/// Address of dynamic TLS slot `index` in the current TEB, if the index
/// is in range (allocating the expansion array is the caller's job on
/// Windows; fake TEBs always have one).
pub fn tls_slot(index: u32) -> Option<*mut u64> {
    let teb = current_teb()?;
    let i = index as usize;
    // SAFETY: offsets inside TEB_SIZE; expansion array has TLS_EXPANSION_SLOTS entries.
    unsafe {
        if i < TLS_INLINE_SLOTS {
            Some(teb.add(off::TLS_SLOTS + i * 8).cast())
        } else if i < TLS_INLINE_SLOTS + TLS_EXPANSION_SLOTS {
            let exp = teb.add(off::TLS_EXPANSION).cast::<*mut u64>().read();
            (!exp.is_null()).then(|| exp.add(i - TLS_INLINE_SLOTS))
        } else {
            None
        }
    }
}

/// Guest execution context of one OS thread: TEB (fake or real) plus its
/// static TLS blocks. Created on the thread that will run guest code and
/// detached on drop; not `Send`.
pub struct GuestThread {
    teb: *mut u8,
    ctx: Box<EnterCtx>,
    /// Indexed like the module registry; `None` for unregistered modules.
    blocks: Vec<Option<(*mut u8, Layout)>>,
    _tls_array: Box<[u64]>,
    #[cfg(not(windows))]
    _teb_mem: Box<[u64]>,
    #[cfg(not(windows))]
    _expansion: Box<[u64]>,
    #[cfg(windows)]
    prev_tls_pointer: u64,
}

impl GuestThread {
    /// Set up guest state for the calling thread. `stack_base` is the high
    /// end of the usable stack and `stack_limit` the low end.
    ///
    /// # Panics
    /// If the thread already has a `GuestThread`.
    pub fn attach(stack_base: u64, stack_limit: u64) -> GuestThread {
        assert!(current_teb().is_none(), "thread already has a guest TEB");
        #[cfg(unix)]
        crate::seh::install_fault_handlers();

        let modules = MODULES.read().unwrap();
        let mut blocks = Vec::with_capacity(modules.len());
        let mut ptrs = Vec::with_capacity(modules.len());
        for t in modules.iter() {
            match t {
                Some(t) => {
                    let layout = t.layout();
                    // SAFETY: non-zero size; template lies in a mapped image
                    // that stays registered (and mapped) while we hold the lock.
                    let p = unsafe {
                        let p = std::alloc::alloc_zeroed(layout);
                        assert!(!p.is_null(), "TLS block allocation failed");
                        std::ptr::copy_nonoverlapping(t.data as *const u8, p, t.len);
                        p
                    };
                    blocks.push(Some((p, layout)));
                    ptrs.push(p as u64);
                }
                None => {
                    blocks.push(None);
                    ptrs.push(0);
                }
            }
        }
        drop(modules);

        let mut me = sys::attach(stack_base, stack_limit, ptrs, blocks);
        me.ctx.teb = me.teb as u64;
        CURRENT.with(|c| c.set(me.teb));
        CURRENT_CTX.with(|c| c.set(&mut *me.ctx));
        sys::install(&mut me);
        me
    }

    /// [`GuestThread::attach`] with stack bounds estimated for a thread
    /// spawned with `stack_size` bytes of stack: the top is just above the
    /// caller's frame, the limit 64 KiB above the far end (guard page).
    #[inline(never)]
    pub fn attach_current(stack_size: usize) -> GuestThread {
        let here = &stack_size as *const usize as u64;
        let page = 0x1000u64;
        let top = (here + page) & !(page - 1);
        let limit = top.saturating_sub(stack_size as u64) + 0x10000;
        Self::attach(top, limit)
    }

    pub fn teb(&self) -> *mut u8 {
        self.teb
    }

    /// Static TLS block of module `index`, as the guest would find it.
    pub fn tls_block(&self, index: u32) -> Option<*mut u8> {
        let i = index.checked_sub(index_base())? as usize;
        self.blocks.get(i).copied().flatten().map(|b| b.0)
    }

    /// Call guest function `f(a0, a1, a2)` (Win64 ABI) on this thread with
    /// the guest `gs` in place.
    ///
    /// # Safety
    /// `f` must be guest code that is safe to run natively.
    pub unsafe fn call(&self, f: u64, a0: u64, a1: u64, a2: u64) -> u64 {
        let ctx = CURRENT_CTX.with(Cell::get);
        debug_assert!(std::ptr::eq(ctx, &*self.ctx));
        unsafe { enter::enter(f, [a0, a1, a2, 0], ctx) }
    }
}

impl Drop for GuestThread {
    fn drop(&mut self) {
        sys::uninstall(self);
        CURRENT.with(|c| c.set(std::ptr::null_mut()));
        CURRENT_CTX.with(|c| c.set(std::ptr::null_mut()));
        for (p, layout) in self.blocks.drain(..).flatten() {
            // SAFETY: allocated in `attach` with this layout.
            unsafe { std::alloc::dealloc(p, layout) };
        }
    }
}

/// Build a fake TEB (non-Windows hosts).
#[cfg(not(windows))]
fn fake_thread(
    stack_base: u64,
    stack_limit: u64,
    ptrs: Vec<u64>,
    blocks: Vec<Option<(*mut u8, Layout)>>,
) -> GuestThread {
    let mut teb_mem = vec![0u64; TEB_SIZE / 8].into_boxed_slice();
    let mut expansion = vec![0u64; TLS_EXPANSION_SLOTS].into_boxed_slice();
    let tls_array = ptrs.into_boxed_slice();
    let teb = teb_mem.as_mut_ptr().cast::<u8>();
    let put = |o: usize, v: u64| teb_mem_write(teb, o, v);
    put(off::EXCEPTION_LIST, u64::MAX);
    put(off::STACK_BASE, stack_base);
    put(off::STACK_LIMIT, stack_limit);
    put(off::SELF, teb as u64);
    put(off::CLIENT_PID, std::process::id() as u64);
    put(
        off::CLIENT_TID,
        NEXT_TID.fetch_add(4, std::sync::atomic::Ordering::Relaxed) as u64,
    );
    put(off::TLS_POINTER, tls_array.as_ptr() as u64);
    put(off::PEB, peb() as u64);
    put(off::TLS_EXPANSION, expansion.as_mut_ptr() as u64);
    GuestThread {
        teb,
        ctx: Box::new(EnterCtx {
            teb: 0,
            exit_rsp: 0,
        }),
        blocks,
        _tls_array: tls_array,
        _teb_mem: teb_mem,
        _expansion: expansion,
    }
}

#[cfg(not(windows))]
fn teb_mem_write(teb: *mut u8, o: usize, v: u64) {
    debug_assert!(o + 8 <= TEB_SIZE);
    // SAFETY: `teb` is a TEB_SIZE allocation owned by the caller.
    unsafe { teb.add(o).cast::<u64>().write(v) }
}

// ---------------------------------------------------------------- hosts

#[cfg(target_os = "linux")]
mod sys {
    use super::*;

    pub fn host_tls_entries() -> u32 {
        0
    }

    pub fn attach(sb: u64, sl: u64, p: Vec<u64>, b: Vec<Option<(*mut u8, Layout)>>) -> GuestThread {
        fake_thread(sb, sl, p, b)
    }

    fn set_gs(base: u64) {
        const ARCH_SET_GS: u64 = 0x1001;
        let ret: i64;
        // SAFETY: arch_prctl(ARCH_SET_GS) only changes this thread's gs base,
        // which glibc and Rust std do not use on x86-64 Linux.
        unsafe {
            core::arch::asm!("syscall",
                inlateout("rax") 158i64 => ret,
                in("rdi") ARCH_SET_GS, in("rsi") base,
                lateout("rcx") _, lateout("r11") _, options(nostack));
        }
        assert_eq!(ret, 0, "arch_prctl(ARCH_SET_GS) failed");
    }

    pub fn install(t: &mut GuestThread) {
        set_gs(t.teb as u64);
    }

    pub fn uninstall(_: &mut GuestThread) {
        set_gs(0);
    }
}

#[cfg(target_os = "macos")]
mod sys {
    use super::*;

    pub fn host_tls_entries() -> u32 {
        0
    }

    /// libpthread's TSD array is at `gs:0`: slot 0 holds `pthread_self()`
    /// and slot `k` the value of pthread key `k`. Find the array inside the
    /// pthread struct by matching both, so a stray self-pointer elsewhere in
    /// the struct cannot fool us. Done once; the offset is the same for all
    /// threads.
    fn host_gs_base() -> u64 {
        type Key = usize; // pthread_key_t is unsigned long on macOS
        extern "C" {
            fn pthread_self() -> *const u64;
            fn pthread_key_create(key: *mut Key, dtor: Option<extern "C" fn(*mut u8)>) -> i32;
            fn pthread_setspecific(key: Key, value: *const u8) -> i32;
            fn pthread_getspecific(key: Key) -> *const u8;
        }
        static TSD_OFFSET: OnceLock<usize> = OnceLock::new();
        let off = *TSD_OFFSET.get_or_init(|| unsafe {
            // SAFETY: plain libpthread calls; the scan reads the first 4 KiB
            // of this thread's pthread struct, which is always mapped.
            let mut key: Key = 0;
            assert_eq!(pthread_key_create(&mut key, None), 0, "pthread_key_create");
            let marker = &key as *const Key as *const u8;
            let old = pthread_getspecific(key);
            pthread_setspecific(key, marker);
            let me = pthread_self();
            let found = (0..512 - key.min(511))
                .find(|&i| me.add(i).read() == me as u64 && me.add(i + key).read() == marker as u64)
                .expect("cannot locate pthread TSD array");
            pthread_setspecific(key, old);
            found * 8
        });
        // SAFETY: pthread_self is always valid.
        unsafe { pthread_self() as u64 + off as u64 }
    }

    pub fn attach(sb: u64, sl: u64, p: Vec<u64>, b: Vec<Option<(*mut u8, Layout)>>) -> GuestThread {
        let t = fake_thread(sb, sl, p, b);
        teb_mem_write(t.teb, off::HOST_GS, host_gs_base());
        t
    }

    // gs is switched per call in `crate::enter` and the trampoline.
    pub fn install(_: &mut GuestThread) {}
    pub fn uninstall(_: &mut GuestThread) {}
}

#[cfg(windows)]
mod sys {
    use super::*;
    use windows_sys::Win32::System::Memory::{GetProcessHeap, HeapSize};

    fn read_gs(o: usize) -> u64 {
        let v: u64;
        // SAFETY: reads a field of this thread's TEB.
        unsafe {
            core::arch::asm!("mov {}, gs:[{}]", out(reg) v, in(reg) o, options(nostack, readonly))
        };
        v
    }

    fn write_gs(o: usize, v: u64) {
        // SAFETY: writes a field of this thread's TEB (only TLS_POINTER).
        unsafe { core::arch::asm!("mov gs:[{}], {}", in(reg) o, in(reg) v, options(nostack)) };
    }

    /// Number of entries ntdll allocated in this thread's TLS array (one per
    /// host module with static TLS). The array comes from the process heap.
    pub fn host_tls_entries() -> u32 {
        let arr = read_gs(off::TLS_POINTER);
        if arr == 0 {
            return 0;
        }
        // SAFETY: ntdll allocates ThreadLocalStoragePointer with RtlAllocateHeap
        // on the process heap.
        let size = unsafe { HeapSize(GetProcessHeap(), 0, arr as *const _) };
        assert_ne!(
            size,
            usize::MAX,
            "cannot size host ThreadLocalStoragePointer"
        );
        (size / 8) as u32
    }

    pub fn attach(
        _sb: u64,
        _sl: u64,
        ptrs: Vec<u64>,
        blocks: Vec<Option<(*mut u8, Layout)>>,
    ) -> GuestThread {
        let host_n = index_base() as usize;
        let old = read_gs(off::TLS_POINTER) as *const u64;
        let mut array = vec![0u64; host_n + ptrs.len()];
        for (i, slot) in array.iter_mut().take(host_n).enumerate() {
            // SAFETY: the host array has at least host_n entries (HeapSize).
            *slot = unsafe { old.add(i).read() };
        }
        array[host_n..].copy_from_slice(&ptrs);
        GuestThread {
            teb: read_gs(off::SELF) as *mut u8,
            ctx: Box::new(EnterCtx {
                teb: 0,
                exit_rsp: 0,
            }),
            blocks,
            _tls_array: array.into_boxed_slice(),
            prev_tls_pointer: old as u64,
        }
    }

    pub fn install(t: &mut GuestThread) {
        write_gs(off::TLS_POINTER, t._tls_array.as_ptr() as u64);
    }

    pub fn uninstall(t: &mut GuestThread) {
        // ntdll frees the original array at thread exit; give it back.
        write_gs(off::TLS_POINTER, t.prev_tls_pointer);
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
compile_error!("guest TEB support exists for Windows, Linux and macOS hosts only");
