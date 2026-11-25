//! Module registry and the module APIs: `GetModuleHandle*`,
//! `GetModuleFileName*`, `GetProcAddress`, `LoadLibrary*`, `FreeLibrary`.
//!
//! Two kinds of module exist for the guest:
//!
//! * **guest modules**: PE images the loader mapped (the game and its
//!   DLLs). The loader registers each with its exports, so imports between
//!   guest modules bind directly and `GetProcAddress` finds them.
//! * **virtual modules**: DLLs that only exist as HLE handlers (kernel32,
//!   kernelx, API sets...). `GetModuleHandle("kernel32")` returns a small
//!   fake image (just an `MZ` page), and `GetProcAddress` on it mints a
//!   thunk to the handler at runtime, the same thunk the loader would have
//!   put in an IAT. Unknown names return NULL (as Windows does) and are
//!   recorded in [`proc_misses`] for the end-of-run report.
//!
//! Loading new guest DLLs needs the loader crate, which depends on this
//! one, so it is injected as a hook ([`install`]).

use crate::guest::{cstr, write_wstr, wstr};
use crate::teb::set_last_error;
use crate::{normalize_dll, Call, Handler, Registry};
use scarlett_pe::ImportName;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex, OnceLock, RwLock};

pub const ERROR_MOD_NOT_FOUND: u32 = 126;
pub const ERROR_PROC_NOT_FOUND: u32 = 127;
pub const ERROR_INSUFFICIENT_BUFFER: u32 = 122;
pub const ERROR_INVALID_PARAMETER: u32 = 87;

pub const GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT: u32 = 0x2;
pub const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: u32 = 0x4;

/// A PE image mapped by the loader.
#[derive(Debug, Clone)]
pub struct GuestModule {
    /// File name as the guest would name it (`game.exe`, `engine.dll`).
    pub name: String,
    /// Full guest path (`G:\engine.dll`).
    pub guest_path: String,
    pub base: u64,
    pub size: u64,
    pub is_dll: bool,
    pub exports: HashMap<String, u64>,
    pub ordinals: HashMap<u32, u64>,
    /// Name or `#ordinal` → `"dll.Symbol"` / `"dll.#7"`.
    pub forwarders: HashMap<String, String>,
    /// Exception directory (`RUNTIME_FUNCTION` table): RVA and entry count.
    pub pdata_rva: u32,
    pub pdata_count: u32,
    /// DllMain (absolute), 0 for EXEs and DLLs without an entry point.
    pub entry: u64,
}

/// Loads a guest DLL by guest path and returns its base. Errors are Win32
/// codes. Installed by the loader crate.
pub type LoadHook = fn(&str) -> Result<u64, u32>;

struct Virtual {
    name: String,
    base: u64,
}

#[derive(Default)]
struct State {
    guests: Vec<GuestModule>,
    /// Bases of DLLs that called `DisableThreadLibraryCalls`.
    no_thread_calls: Vec<u64>,
    virtuals: Vec<Virtual>,
    thunks: HashMap<(String, String), u64>,
    misses: Vec<(String, String)>,
}

static STATE: LazyLock<RwLock<State>> = LazyLock::new(Default::default);
static REGISTRY: OnceLock<Registry> = OnceLock::new();
static HOOK: OnceLock<LoadHook> = OnceLock::new();

/// Make `registry` available for runtime `GetProcAddress` on HLE DLLs, and
/// set the hook that maps new guest DLLs. First call wins.
pub fn install(registry: Registry, hook: Option<LoadHook>) {
    let _ = REGISTRY.set(registry);
    if let Some(h) = hook {
        let _ = HOOK.set(h);
    }
}

pub fn register_guest(m: GuestModule) {
    STATE.write().unwrap().guests.push(m);
}

pub fn unregister_guest(base: u64) {
    STATE.write().unwrap().guests.retain(|m| m.base != base);
}

/// `GetProcAddress` names that resolved to nothing, in call order.
pub fn proc_misses() -> Vec<(String, String)> {
    STATE.read().unwrap().misses.clone()
}

/// File-name part of a guest path, for module matching.
fn file_part(name: &str) -> &str {
    name.rsplit(['\\', '/']).next().unwrap_or(name)
}

/// Guest module whose name matches (case-insensitive, `.dll` optional).
pub fn find_guest(name: &str) -> Option<GuestModule> {
    let key = normalize_dll(file_part(name));
    STATE
        .read()
        .unwrap()
        .guests
        .iter()
        .find(|m| normalize_dll(&m.name) == key)
        .cloned()
}

pub fn guest_by_base(base: u64) -> Option<GuestModule> {
    STATE
        .read()
        .unwrap()
        .guests
        .iter()
        .find(|m| m.base == base)
        .cloned()
}

/// Guest module whose image contains `addr`.
pub fn guest_containing(addr: u64) -> Option<GuestModule> {
    STATE
        .read()
        .unwrap()
        .guests
        .iter()
        .find(|m| (m.base..m.base + m.size).contains(&addr))
        .cloned()
}

/// [`guest_containing`] for signal handlers: `None` if the table is locked
/// (never blocks), else whether `addr` is in a guest image.
pub fn guest_containing_try(addr: u64) -> Option<bool> {
    let st = STATE.try_read().ok()?;
    Some(
        st.guests
            .iter()
            .any(|m| (m.base..m.base + m.size).contains(&addr)),
    )
}

/// `(base, DllMain)` of every guest DLL that wants thread notifications,
/// in load order.
pub fn thread_notify_entries() -> Vec<(u64, u64)> {
    let st = STATE.read().unwrap();
    st.guests
        .iter()
        .filter(|m| m.is_dll && m.entry != 0 && !st.no_thread_calls.contains(&m.base))
        .map(|m| (m.base, m.entry))
        .collect()
}

/// Main executable: the first registered non-DLL image.
pub fn main_module() -> Option<GuestModule> {
    STATE
        .read()
        .unwrap()
        .guests
        .iter()
        .find(|m| !m.is_dll)
        .cloned()
}

/// Is `dll` implemented by HLE (any handler registered under its name)?
pub fn is_hle_dll(dll: &str) -> bool {
    REGISTRY.get().is_some_and(|r| r.has_dll(file_part(dll)))
}

/// Export `sym` of guest module `m`, following forwarders.
fn guest_export(m: &GuestModule, sym: &ImportName, depth: u32) -> Option<u64> {
    let key = match sym {
        ImportName::Name { name, .. } => name.clone(),
        ImportName::Ordinal(o) => format!("#{o}"),
    };
    if let Some(fwd) = m.forwarders.get(&key) {
        if depth > 8 {
            return None;
        }
        let (dll, target) = fwd.rsplit_once('.')?;
        let target = match target.strip_prefix('#') {
            Some(o) => ImportName::Ordinal(o.parse().ok()?),
            None => ImportName::Name {
                hint: 0,
                name: target.to_string(),
            },
        };
        return resolve_depth(dll, &target, depth + 1);
    }
    match sym {
        ImportName::Name { name, .. } => m.exports.get(name).copied(),
        ImportName::Ordinal(o) => m.ordinals.get(&(*o as u32)).copied(),
    }
}

/// Address for `dll!sym` from guest modules or HLE (minting a thunk).
pub fn resolve(dll: &str, sym: &ImportName) -> Option<u64> {
    resolve_depth(dll, sym, 0)
}

fn resolve_depth(dll: &str, sym: &ImportName, depth: u32) -> Option<u64> {
    if let Some(m) = find_guest(dll) {
        return guest_export(&m, sym, depth);
    }
    let handler = REGISTRY.get()?.lookup(file_part(dll), sym)?;
    Some(thunk_for(dll, sym, handler))
}

/// Export of an already loaded guest module only (no HLE), for binding
/// imports between guest images at load time.
pub fn guest_import(dll: &str, sym: &ImportName) -> Option<u64> {
    find_guest(dll).and_then(|m| guest_export(&m, sym, 0))
}

// ---------------------------------------------------------------- runtime thunks

/// Executable pool for thunks created after load. Pages are RWX because a
/// page may gain thunks while other threads execute earlier ones.
struct Pool {
    page: u64,
    used: usize,
}

static POOL: Mutex<Pool> = Mutex::new(Pool { page: 0, used: 0 });
const POOL_PAGE: usize = 0x1000;
const SLOT: usize = 32;

#[cfg(unix)]
fn exec_page() -> u64 {
    // SAFETY: anonymous private mapping.
    let p = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            POOL_PAGE,
            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    assert_ne!(p, libc::MAP_FAILED, "cannot map thunk page");
    p as u64
}

#[cfg(windows)]
fn exec_page() -> u64 {
    use windows_sys::Win32::System::Memory::*;
    // SAFETY: plain VirtualAlloc.
    let p = unsafe {
        VirtualAlloc(
            std::ptr::null(),
            POOL_PAGE,
            MEM_RESERVE | MEM_COMMIT,
            PAGE_EXECUTE_READWRITE,
        )
    };
    assert!(!p.is_null(), "cannot map thunk page");
    p as u64
}

pub(crate) fn thunk_for(dll: &str, sym: &ImportName, handler: Handler) -> u64 {
    let key = (normalize_dll(file_part(dll)), sym.to_string());
    if let Some(&t) = STATE.read().unwrap().thunks.get(&key) {
        return t;
    }
    let mut st = STATE.write().unwrap();
    if let Some(&t) = st.thunks.get(&key) {
        return t;
    }
    let id = crate::add_thunk(dll, sym, Some(handler));
    let bytes = crate::encode_thunk(id, crate::trampoline::trampoline_address());
    let mut pool = POOL.lock().unwrap();
    if pool.page == 0 || pool.used + SLOT > POOL_PAGE {
        pool.page = exec_page();
        pool.used = 0;
    }
    let at = pool.page + pool.used as u64;
    // SAFETY: RWX pool page owned by us; slot not yet handed out.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), at as *mut u8, bytes.len());
        std::ptr::write_bytes((at as *mut u8).add(bytes.len()), 0xCC, SLOT - bytes.len());
    }
    pool.used += SLOT;
    st.thunks.insert(key, at);
    at
}

/// Fake image base for an HLE DLL: a zeroed, 64 KiB aligned page starting
/// with `MZ`, so code that peeks at its module header does not fault.
fn virtual_handle(dll: &str) -> u64 {
    let name = normalize_dll(file_part(dll));
    if let Some(v) = STATE
        .read()
        .unwrap()
        .virtuals
        .iter()
        .find(|v| v.name == name)
    {
        return v.base;
    }
    let mut st = STATE.write().unwrap();
    if let Some(v) = st.virtuals.iter().find(|v| v.name == name) {
        return v.base;
    }
    let layout = std::alloc::Layout::from_size_align(0x1000, 0x10000).unwrap();
    // SAFETY: non-zero layout; leaked for the process lifetime.
    let p = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!p.is_null());
    // SAFETY: fresh page.
    unsafe { p.cast::<[u8; 2]>().write(*b"MZ") };
    st.virtuals.push(Virtual {
        name,
        base: p as u64,
    });
    p as u64
}

fn virtual_name(base: u64) -> Option<String> {
    STATE
        .read()
        .unwrap()
        .virtuals
        .iter()
        .find(|v| v.base == base)
        .map(|v| v.name.clone())
}

/// Handle for an already present module (guest or HLE), without loading.
pub fn module_handle(name: &str) -> Option<u64> {
    if let Some(m) = find_guest(name) {
        return Some(m.base);
    }
    is_hle_dll(name).then(|| virtual_handle(name))
}

/// Find or load `name` (`LoadLibrary` semantics).
pub fn load_library(name: &str) -> Result<u64, u32> {
    if let Some(h) = module_handle(name) {
        return Ok(h);
    }
    let hook = HOOK.get().ok_or(ERROR_MOD_NOT_FOUND)?;
    for path in search_paths(name) {
        if crate::vfs::resolve(&path).is_ok_and(|p| p.is_file()) {
            // Absolute guest path, so GetModuleFileName does not depend on
            // the current directory at load time.
            let abs = match crate::vfs::normalize(&path) {
                Some((dev, parts)) => format!("{}:\\{}", dev.to_uppercase(), parts.join("\\")),
                None => path,
            };
            return hook(&abs);
        }
    }
    Err(ERROR_MOD_NOT_FOUND)
}

/// Candidate guest paths for a DLL name: as given if it has a directory,
/// else next to the main image, then in the current directory.
fn search_paths(name: &str) -> Vec<String> {
    let mut name = name.to_string();
    if !file_part(&name).contains('.') {
        name.push_str(".dll");
    }
    if name.contains(['\\', '/', ':']) {
        return vec![name];
    }
    let mut out = Vec::new();
    if let Some(m) = main_module() {
        if let Some((dir, _)) = m.guest_path.rsplit_once('\\') {
            out.push(format!("{dir}\\{name}"));
        }
    }
    out.push(name);
    out
}

// ---------------------------------------------------------------- handlers

fn fail(code: u32) -> u64 {
    set_last_error(code);
    0
}

fn wide(c: &Call, i: usize) -> Option<String> {
    // SAFETY: argument `i` is a wide string or null per API contract.
    unsafe { wstr(c.arg_ptr(i)) }
}

fn ansi(c: &Call, i: usize) -> Option<String> {
    // SAFETY: as above, ANSI.
    unsafe { cstr(c.arg_ptr(i)) }
}

fn main_base() -> u64 {
    main_module().map_or_else(crate::teb::image_base, |m| m.base)
}

fn get_module_handle(name: Option<String>) -> u64 {
    match name {
        None => main_base(),
        Some(n) => module_handle(&n).unwrap_or_else(|| fail(ERROR_MOD_NOT_FOUND)),
    }
}

fn get_module_handle_w(c: &Call) -> u64 {
    get_module_handle(wide(c, 0))
}

fn get_module_handle_a(c: &Call) -> u64 {
    get_module_handle(ansi(c, 0))
}

/// `GetModuleHandleExW(flags, name_or_address, *module)`.
fn get_module_handle_ex(c: &Call, ansi_name: bool) -> u64 {
    let (flags, arg, out) = (c.arg(0) as u32, c.arg(1), c.arg_ptr::<u64>(2));
    if out.is_null() {
        return fail(ERROR_INVALID_PARAMETER);
    }
    let h = if flags & GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS != 0 {
        STATE
            .read()
            .unwrap()
            .guests
            .iter()
            .find(|m| (m.base..m.base + m.size).contains(&arg))
            .map(|m| m.base)
    } else if arg == 0 {
        Some(main_base())
    } else {
        let n = if ansi_name { ansi(c, 1) } else { wide(c, 1) };
        n.and_then(|n| module_handle(&n))
    };
    // SAFETY: guest out pointer.
    unsafe { out.write_unaligned(h.unwrap_or(0)) };
    match h {
        Some(_) => 1,
        None => fail(ERROR_MOD_NOT_FOUND),
    }
}

fn get_module_handle_ex_w(c: &Call) -> u64 {
    get_module_handle_ex(c, false)
}

fn get_module_handle_ex_a(c: &Call) -> u64 {
    get_module_handle_ex(c, true)
}

fn module_path(h: u64) -> Option<String> {
    let h = if h == 0 { main_base() } else { h };
    if let Some(m) = guest_by_base(h) {
        return Some(m.guest_path);
    }
    virtual_name(h).map(|n| format!("C:\\Windows\\System32\\{n}.dll"))
}

/// `GetModuleFileNameW(h, buf, size)`: chars copied (without NUL); on
/// truncation returns `size` with ERROR_INSUFFICIENT_BUFFER, as Windows does.
fn get_module_file_name_w(c: &Call) -> u64 {
    let (h, buf, size) = (c.arg(0), c.arg_ptr::<u16>(1), c.arg(2) as u32 as usize);
    let Some(path) = module_path(h) else {
        return fail(ERROR_MOD_NOT_FOUND);
    };
    // SAFETY: guest buffer of `size` u16s.
    let n = unsafe { write_wstr(buf, size, &path) };
    if n < path.encode_utf16().count() {
        set_last_error(ERROR_INSUFFICIENT_BUFFER);
        return size as u64;
    }
    set_last_error(0);
    n as u64
}

fn get_module_file_name_a(c: &Call) -> u64 {
    let (h, buf, size) = (c.arg(0), c.arg_ptr::<u8>(1), c.arg(2) as u32 as usize);
    let Some(path) = module_path(h) else {
        return fail(ERROR_MOD_NOT_FOUND);
    };
    if buf.is_null() || size == 0 {
        set_last_error(ERROR_INSUFFICIENT_BUFFER);
        return 0;
    }
    let b = path.as_bytes();
    let n = b.len().min(size - 1);
    // SAFETY: guest buffer of `size` bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(b.as_ptr(), buf, n);
        buf.add(n).write(0);
    }
    if n < b.len() {
        set_last_error(ERROR_INSUFFICIENT_BUFFER);
        return size as u64;
    }
    set_last_error(0);
    n as u64
}

/// `GetProcAddress(module, name_or_ordinal)`.
fn get_proc_address(c: &Call) -> u64 {
    let (h, p) = (c.arg(0), c.arg(1));
    let sym = if p < 0x10000 {
        ImportName::Ordinal(p as u16)
    } else {
        // SAFETY: a non-ordinal argument is a NUL-terminated name.
        match unsafe { cstr(p as *const u8) } {
            Some(name) => ImportName::Name { hint: 0, name },
            None => return fail(ERROR_INVALID_PARAMETER),
        }
    };
    let h = if h == 0 { main_base() } else { h };
    let dll = match (guest_by_base(h), virtual_name(h)) {
        (Some(m), _) => m.name,
        (None, Some(v)) => v,
        _ => return fail(ERROR_MOD_NOT_FOUND),
    };
    match resolve(&dll, &sym) {
        Some(addr) => addr,
        None => {
            log::warn!("GetProcAddress({dll}, {sym}): not found");
            STATE.write().unwrap().misses.push((dll, sym.to_string()));
            fail(ERROR_PROC_NOT_FOUND)
        }
    }
}

fn load_library_common(name: Option<String>) -> u64 {
    let Some(name) = name else {
        return fail(ERROR_INVALID_PARAMETER);
    };
    match load_library(&name) {
        Ok(h) => h,
        Err(e) => {
            log::warn!("LoadLibrary({name}): error {e}");
            fail(e)
        }
    }
}

fn load_library_w(c: &Call) -> u64 {
    load_library_common(wide(c, 0))
}

fn load_library_a(c: &Call) -> u64 {
    load_library_common(ansi(c, 0))
}

fn free_library(_: &Call) -> u64 {
    1 // modules stay mapped for the process lifetime
}

fn disable_thread_library_calls(c: &Call) -> u64 {
    STATE.write().unwrap().no_thread_calls.push(c.arg(0));
    1
}

pub fn register_modules(reg: &mut Registry, dlls: &[&str]) {
    for dll in dlls {
        reg.register(dll, "GetModuleHandleW", get_module_handle_w)
            .register(dll, "GetModuleHandleA", get_module_handle_a)
            .register(dll, "GetModuleHandleExW", get_module_handle_ex_w)
            .register(dll, "GetModuleHandleExA", get_module_handle_ex_a)
            .register(dll, "GetModuleFileNameW", get_module_file_name_w)
            .register(dll, "GetModuleFileNameA", get_module_file_name_a)
            .register(dll, "GetProcAddress", get_proc_address)
            .register(dll, "LoadLibraryW", load_library_w)
            .register(dll, "LoadLibraryA", load_library_a)
            .register(dll, "LoadLibraryExW", load_library_w)
            .register(dll, "LoadLibraryExA", load_library_a)
            .register(dll, "LoadPackagedLibrary", load_library_w)
            .register(dll, "FreeLibrary", free_library)
            .register(
                dll,
                "DisableThreadLibraryCalls",
                disable_thread_library_calls,
            );
    }
}
