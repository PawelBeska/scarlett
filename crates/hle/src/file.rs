//! File APIs over the [`crate::vfs`] mount table: `CreateFile*`,
//! `ReadFile`/`WriteFile` (including `OVERLAPPED` offsets, completed
//! synchronously), seeking and sizes, attributes, `FindFirstFile*`,
//! directory and delete/move operations, standard handles.
//!
//! File handles are I/O objects in the [`crate::object`] table, so
//! `CloseHandle`, `DuplicateHandle` and waits work on them unchanged.

use crate::guest::{cstr, write_wstr, wstr};
use crate::object::{self, insert_io, io};
use crate::teb::set_last_error;
use crate::time::FILETIME_UNIX_EPOCH;
use crate::vfs;
use crate::{Call, Registry};
use std::collections::VecDeque;
use std::fs::{File, Metadata, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

pub const INVALID_HANDLE_VALUE: u64 = u64::MAX;
pub const INVALID_FILE_ATTRIBUTES: u32 = 0xFFFF_FFFF;
pub const INVALID_SET_FILE_POINTER: u32 = 0xFFFF_FFFF;

pub const GENERIC_READ: u32 = 0x8000_0000;
pub const GENERIC_WRITE: u32 = 0x4000_0000;
pub const GENERIC_ALL: u32 = 0x1000_0000;
pub const FILE_READ_DATA: u32 = 0x1;
pub const FILE_WRITE_DATA: u32 = 0x2;
pub const FILE_APPEND_DATA: u32 = 0x4;

pub const CREATE_NEW: u32 = 1;
pub const CREATE_ALWAYS: u32 = 2;
pub const OPEN_EXISTING: u32 = 3;
pub const OPEN_ALWAYS: u32 = 4;
pub const TRUNCATE_EXISTING: u32 = 5;

pub const FILE_ATTRIBUTE_READONLY: u32 = 0x01;
pub const FILE_ATTRIBUTE_DIRECTORY: u32 = 0x10;
pub const FILE_ATTRIBUTE_ARCHIVE: u32 = 0x20;

pub const FILE_TYPE_DISK: u32 = 1;
pub const FILE_TYPE_CHAR: u32 = 2;

pub const ERROR_FILE_NOT_FOUND: u32 = 2;
pub const ERROR_PATH_NOT_FOUND: u32 = 3;
pub const ERROR_ACCESS_DENIED: u32 = 5;
pub const ERROR_INVALID_HANDLE: u32 = 6;
pub const ERROR_NO_MORE_FILES: u32 = 18;
pub const ERROR_GEN_FAILURE: u32 = 31;
pub const ERROR_HANDLE_EOF: u32 = 38;
pub const ERROR_FILE_EXISTS: u32 = 80;
pub const ERROR_INVALID_PARAMETER: u32 = 87;
pub const ERROR_DISK_FULL: u32 = 112;
pub const ERROR_DIR_NOT_EMPTY: u32 = 145;
pub const ERROR_ALREADY_EXISTS: u32 = 183;

pub const MOVEFILE_REPLACE_EXISTING: u32 = 1;

/// `STD_INPUT_HANDLE` etc. as the 32-bit values the guest passes.
const STD_INPUT: u32 = -10i32 as u32;
const STD_OUTPUT: u32 = -11i32 as u32;
const STD_ERROR: u32 = -12i32 as u32;

#[derive(Debug)]
pub enum Target {
    Disk(Mutex<File>),
    Stdin,
    Stdout,
    Stderr,
}

#[derive(Debug)]
pub struct FileObj {
    pub target: Target,
    pub path: PathBuf,
}

#[derive(Debug)]
struct FindObj {
    rest: Mutex<VecDeque<(String, Metadata)>>,
}

fn win32_error(e: &std::io::Error) -> u32 {
    use std::io::ErrorKind::*;
    match e.kind() {
        NotFound => ERROR_FILE_NOT_FOUND,
        PermissionDenied => ERROR_ACCESS_DENIED,
        AlreadyExists => ERROR_FILE_EXISTS,
        NotADirectory => ERROR_PATH_NOT_FOUND,
        IsADirectory => ERROR_ACCESS_DENIED,
        DirectoryNotEmpty => ERROR_DIR_NOT_EMPTY,
        StorageFull => ERROR_DISK_FULL,
        _ => ERROR_GEN_FAILURE,
    }
}

/// `NotFound` for `a\b\c` where `a\b` is missing is a *path* error.
fn not_found_code(host: &Path) -> u32 {
    match host.parent() {
        Some(p) if !p.exists() => ERROR_PATH_NOT_FOUND,
        _ => ERROR_FILE_NOT_FOUND,
    }
}

fn io_error(host: &Path, e: &std::io::Error) -> u32 {
    if e.kind() == std::io::ErrorKind::NotFound {
        not_found_code(host)
    } else {
        win32_error(e)
    }
}

fn fail(code: u32) -> u64 {
    set_last_error(code);
    0
}

fn fail_handle(code: u32) -> u64 {
    set_last_error(code);
    INVALID_HANDLE_VALUE
}

fn wide(c: &Call, i: usize) -> Option<String> {
    // SAFETY: the API contract says argument `i` is a wide string or null.
    unsafe { wstr(c.arg_ptr(i)) }
}

fn ansi(c: &Call, i: usize) -> Option<String> {
    // SAFETY: as above, ANSI.
    unsafe { cstr(c.arg_ptr(i)) }
}

fn resolve(path: Option<String>) -> Result<PathBuf, u32> {
    vfs::resolve(&path.ok_or(ERROR_INVALID_PARAMETER)?)
}

unsafe fn put<T>(p: *mut u8, off: usize, v: T) {
    unsafe { p.add(off).cast::<T>().write_unaligned(v) }
}

unsafe fn get<T: Copy>(p: *const u8, off: usize) -> T {
    unsafe { p.add(off).cast::<T>().read_unaligned() }
}

// ---------------------------------------------------------------- open

fn open(path: Option<String>, access: u32, disposition: u32) -> u64 {
    let host = match resolve(path) {
        Ok(h) => h,
        Err(e) => return fail_handle(e),
    };
    if host.is_dir() {
        // Directory handles (FILE_FLAG_BACKUP_SEMANTICS) are not supported yet.
        return fail_handle(ERROR_ACCESS_DENIED);
    }
    let read = access & (GENERIC_READ | GENERIC_ALL | FILE_READ_DATA) != 0;
    let write = access & (GENERIC_WRITE | GENERIC_ALL | FILE_WRITE_DATA | FILE_APPEND_DATA) != 0;
    let existed = host.exists();
    let mut o = OpenOptions::new();
    // Attribute-only opens (access 0) still need a readable host handle.
    o.read(read || !write).write(write);
    match disposition {
        CREATE_NEW => o.write(true).create_new(true),
        CREATE_ALWAYS => o.write(true).create(true).truncate(true),
        OPEN_EXISTING => &mut o,
        OPEN_ALWAYS => o.write(write).create(true),
        TRUNCATE_EXISTING => o.write(true).truncate(true),
        _ => return fail_handle(ERROR_INVALID_PARAMETER),
    };
    if disposition == OPEN_ALWAYS && !write {
        // create(true) needs write access on the host; create the file
        // first, then open it the way the guest asked.
        if !existed {
            if let Err(e) = File::create(&host) {
                return fail_handle(io_error(&host, &e));
            }
        }
        o = OpenOptions::new();
        o.read(true);
    }
    match o.open(&host) {
        Ok(f) => {
            let reopened = existed && matches!(disposition, CREATE_ALWAYS | OPEN_ALWAYS);
            set_last_error(if reopened { ERROR_ALREADY_EXISTS } else { 0 });
            insert_io(Arc::new(FileObj {
                target: Target::Disk(Mutex::new(f)),
                path: host,
            }))
        }
        Err(e) => fail_handle(io_error(&host, &e)),
    }
}

/// `CreateFileW(name, access, share, sa, disposition, flags, template)`.
fn create_file_w(c: &Call) -> u64 {
    open(wide(c, 0), c.arg(1) as u32, c.arg(4) as u32)
}

fn create_file_a(c: &Call) -> u64 {
    open(ansi(c, 0), c.arg(1) as u32, c.arg(4) as u32)
}

/// `CreateFile2(name, access, share, disposition, params)`.
fn create_file2(c: &Call) -> u64 {
    open(wide(c, 0), c.arg(1) as u32, c.arg(3) as u32)
}

fn std_handle(which: u32) -> u64 {
    static HANDLES: OnceLock<[u64; 3]> = OnceLock::new();
    let h = HANDLES.get_or_init(|| {
        [Target::Stdin, Target::Stdout, Target::Stderr].map(|target| {
            insert_io(Arc::new(FileObj {
                target,
                path: PathBuf::new(),
            }))
        })
    });
    match which {
        STD_INPUT => h[0],
        STD_OUTPUT => h[1],
        STD_ERROR => h[2],
        _ => fail_handle(ERROR_INVALID_PARAMETER),
    }
}

fn get_std_handle(c: &Call) -> u64 {
    std_handle(c.arg(0) as u32)
}

// ---------------------------------------------------------------- read / write

fn file(h: u64) -> Option<Arc<FileObj>> {
    io::<FileObj>(h)
}

/// `OVERLAPPED`: Internal @0, InternalHigh @8, Offset @16, OffsetHigh @20,
/// hEvent @24.
fn overlapped_offset(ov: *const u8) -> u64 {
    // SAFETY: guest OVERLAPPED.
    unsafe { get::<u32>(ov, 16) as u64 | (get::<u32>(ov, 20) as u64) << 32 }
}

fn complete_overlapped(ov: *mut u8, bytes: usize) {
    // SAFETY: guest OVERLAPPED.
    unsafe {
        put::<u64>(ov, 0, 0); // STATUS_SUCCESS
        put::<u64>(ov, 8, bytes as u64);
        let ev = get::<u64>(ov, 24) & !1; // low bit = "no completion port"
        if ev != 0 {
            object::signal_event(ev);
        }
    }
}

fn read_full(f: &mut File, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut done = 0;
    while done < buf.len() {
        match f.read(&mut buf[done..]) {
            Ok(0) => break,
            Ok(n) => done += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(done)
}

/// `ReadFile(h, buf, n, *read, overlapped)`.
fn read_file(c: &Call) -> u64 {
    let (h, buf, n, out, ov) = (
        c.arg(0),
        c.arg_ptr::<u8>(1),
        c.arg(2) as u32 as usize,
        c.arg_ptr::<u32>(3),
        c.arg_ptr::<u8>(4),
    );
    let Some(f) = file(h) else {
        return fail(ERROR_INVALID_HANDLE);
    };
    if buf.is_null() && n > 0 {
        return fail(ERROR_INVALID_PARAMETER);
    }
    // SAFETY: guest buffer of `n` bytes.
    let dst = if n == 0 {
        &mut [][..]
    } else {
        unsafe { std::slice::from_raw_parts_mut(buf, n) }
    };
    let got = match &f.target {
        Target::Disk(m) => {
            let mut file = m.lock().unwrap();
            let r = if ov.is_null() {
                read_full(&mut file, dst)
            } else {
                file.seek(SeekFrom::Start(overlapped_offset(ov)))
                    .and_then(|_| read_full(&mut file, dst))
            };
            match r {
                Ok(k) => k,
                Err(e) => return fail(win32_error(&e)),
            }
        }
        Target::Stdin => 0,
        Target::Stdout | Target::Stderr => return fail(ERROR_ACCESS_DENIED),
    };
    if !out.is_null() {
        // SAFETY: guest out pointer.
        unsafe { out.write_unaligned(got as u32) };
    }
    if !ov.is_null() {
        complete_overlapped(ov, got);
        if got == 0 && n > 0 {
            return fail(ERROR_HANDLE_EOF);
        }
    }
    1
}

/// `WriteFile(h, buf, n, *written, overlapped)`.
fn write_file(c: &Call) -> u64 {
    let (h, buf, n, out, ov) = (
        c.arg(0),
        c.arg_ptr::<u8>(1),
        c.arg(2) as u32 as usize,
        c.arg_ptr::<u32>(3),
        c.arg_ptr::<u8>(4),
    );
    let Some(f) = file(h) else {
        return fail(ERROR_INVALID_HANDLE);
    };
    if buf.is_null() && n > 0 {
        return fail(ERROR_INVALID_PARAMETER);
    }
    // SAFETY: guest buffer of `n` bytes.
    let src = if n == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(buf, n) }
    };
    let r = match &f.target {
        Target::Disk(m) => {
            let mut file = m.lock().unwrap();
            if !ov.is_null() {
                if let Err(e) = file.seek(SeekFrom::Start(overlapped_offset(ov))) {
                    return fail(win32_error(&e));
                }
            }
            file.write_all(src)
        }
        Target::Stdout => std::io::stdout().write_all(src),
        Target::Stderr => std::io::stderr().write_all(src),
        Target::Stdin => return fail(ERROR_ACCESS_DENIED),
    };
    if let Err(e) = r {
        return fail(win32_error(&e));
    }
    if !out.is_null() {
        // SAFETY: guest out pointer.
        unsafe { out.write_unaligned(n as u32) };
    }
    if !ov.is_null() {
        complete_overlapped(ov, n);
    }
    1
}

/// `GetOverlappedResult(h, ov, *bytes, wait)`: everything completes
/// synchronously, so just report the stored count.
fn get_overlapped_result(c: &Call) -> u64 {
    let (ov, out) = (c.arg_ptr::<u8>(1), c.arg_ptr::<u32>(2));
    if ov.is_null() {
        return fail(ERROR_INVALID_PARAMETER);
    }
    if !out.is_null() {
        // SAFETY: guest pointers.
        unsafe { out.write_unaligned(get::<u64>(ov, 8) as u32) };
    }
    1
}

// ---------------------------------------------------------------- position / size

fn with_disk<R>(h: u64, f: impl FnOnce(&mut File) -> std::io::Result<R>) -> Result<R, u32> {
    let obj = file(h).ok_or(ERROR_INVALID_HANDLE)?;
    match &obj.target {
        Target::Disk(m) => f(&mut m.lock().unwrap()).map_err(|e| win32_error(&e)),
        _ => Err(ERROR_INVALID_HANDLE),
    }
}

fn seek(h: u64, dist: i64, method: u32) -> Result<u64, u32> {
    with_disk(h, |file| {
        let from = match method {
            0 => SeekFrom::Start(dist.max(0) as u64),
            1 => SeekFrom::Current(dist),
            2 => SeekFrom::End(dist),
            _ => return Err(std::io::ErrorKind::InvalidInput.into()),
        };
        if method != 0 && dist < 0 {
            // Seeking before the start is ERROR_NEGATIVE_SEEK on Windows.
            let here = file.stream_position()?;
            let end = file.metadata()?.len();
            let base = if method == 1 { here } else { end };
            if (base as i64) + dist < 0 {
                return Err(std::io::ErrorKind::InvalidInput.into());
            }
        }
        file.seek(from)
    })
    .map_err(|e| {
        if e == ERROR_GEN_FAILURE {
            ERROR_INVALID_PARAMETER
        } else {
            e
        }
    })
}

/// `SetFilePointer(h, lo, *hi, method)`.
fn set_file_pointer(c: &Call) -> u64 {
    let (h, lo, hi, method) = (
        c.arg(0),
        c.arg(1) as u32,
        c.arg_ptr::<i32>(2),
        c.arg(3) as u32,
    );
    let dist = if hi.is_null() {
        lo as i32 as i64
    } else {
        // SAFETY: guest in/out pointer.
        ((unsafe { hi.read_unaligned() } as i64) << 32) | lo as i64
    };
    match seek(h, dist, method) {
        Ok(pos) => {
            if !hi.is_null() {
                // SAFETY: as above.
                unsafe { hi.write_unaligned((pos >> 32) as i32) };
            }
            set_last_error(0);
            pos as u32 as u64
        }
        Err(e) => {
            set_last_error(e);
            INVALID_SET_FILE_POINTER as u64
        }
    }
}

/// `SetFilePointerEx(h, dist, *new_pos, method)`.
fn set_file_pointer_ex(c: &Call) -> u64 {
    let (h, dist, out, method) = (
        c.arg(0),
        c.arg(1) as i64,
        c.arg_ptr::<u64>(2),
        c.arg(3) as u32,
    );
    match seek(h, dist, method) {
        Ok(pos) => {
            if !out.is_null() {
                // SAFETY: guest out pointer.
                unsafe { out.write_unaligned(pos) };
            }
            1
        }
        Err(e) => fail(e),
    }
}

fn file_len(h: u64) -> Result<u64, u32> {
    with_disk(h, |f| f.metadata().map(|m| m.len()))
}

fn get_file_size(c: &Call) -> u64 {
    let hi = c.arg_ptr::<u32>(1);
    match file_len(c.arg(0)) {
        Ok(n) => {
            if !hi.is_null() {
                // SAFETY: guest out pointer.
                unsafe { hi.write_unaligned((n >> 32) as u32) };
            }
            set_last_error(0);
            n as u32 as u64
        }
        Err(e) => {
            set_last_error(e);
            INVALID_FILE_ATTRIBUTES as u64 // INVALID_FILE_SIZE, same value
        }
    }
}

fn get_file_size_ex(c: &Call) -> u64 {
    let out = c.arg_ptr::<u64>(1);
    match file_len(c.arg(0)) {
        Ok(n) if !out.is_null() => {
            // SAFETY: guest out pointer.
            unsafe { out.write_unaligned(n) };
            1
        }
        Ok(_) => fail(ERROR_INVALID_PARAMETER),
        Err(e) => fail(e),
    }
}

fn get_file_type(c: &Call) -> u64 {
    match file(c.arg(0)).map(|f| matches!(f.target, Target::Disk(_))) {
        Some(true) => FILE_TYPE_DISK as u64,
        Some(false) => FILE_TYPE_CHAR as u64,
        None => fail(ERROR_INVALID_HANDLE),
    }
}

fn flush_file_buffers(c: &Call) -> u64 {
    match file(c.arg(0)) {
        Some(f) => {
            let r = match &f.target {
                Target::Disk(m) => m.lock().unwrap().flush(),
                Target::Stdout => std::io::stdout().flush(),
                Target::Stderr => std::io::stderr().flush(),
                Target::Stdin => Ok(()),
            };
            r.map_or_else(|e| fail(win32_error(&e)), |_| 1)
        }
        None => fail(ERROR_INVALID_HANDLE),
    }
}

fn set_end_of_file(c: &Call) -> u64 {
    match with_disk(c.arg(0), |f| {
        let pos = f.stream_position()?;
        f.set_len(pos)
    }) {
        Ok(()) => 1,
        Err(e) => fail(e),
    }
}

// ---------------------------------------------------------------- attributes

fn filetime(t: std::io::Result<SystemTime>) -> u64 {
    t.ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| FILETIME_UNIX_EPOCH + (d.as_nanos() / 100) as u64)
}

fn attributes(m: &Metadata) -> u32 {
    if m.is_dir() {
        FILE_ATTRIBUTE_DIRECTORY
    } else if m.permissions().readonly() {
        FILE_ATTRIBUTE_READONLY | FILE_ATTRIBUTE_ARCHIVE
    } else {
        FILE_ATTRIBUTE_ARCHIVE
    }
}

/// Fill the common prefix of `WIN32_FILE_ATTRIBUTE_DATA` / `WIN32_FIND_DATA`:
/// attrs @0, creation @4, access @12, write @20, size high @28, low @32.
unsafe fn put_attr_data(p: *mut u8, m: &Metadata) {
    let mtime = filetime(m.modified());
    let ctime = match filetime(m.created()) {
        0 => mtime, // creation time is not available on every host fs
        t => t,
    };
    unsafe {
        put::<u32>(p, 0, attributes(m));
        put::<u64>(p, 4, ctime);
        put::<u64>(p, 12, filetime(m.accessed()));
        put::<u64>(p, 20, mtime);
        put::<u32>(p, 28, (m.len() >> 32) as u32);
        put::<u32>(p, 32, m.len() as u32);
    }
}

fn metadata(path: Option<String>) -> Result<Metadata, u32> {
    let host = resolve(path)?;
    std::fs::metadata(&host).map_err(|e| io_error(&host, &e))
}

fn get_attrs(path: Option<String>) -> u64 {
    match metadata(path) {
        Ok(m) => attributes(&m) as u64,
        Err(e) => {
            set_last_error(e);
            INVALID_FILE_ATTRIBUTES as u64
        }
    }
}

fn get_file_attributes_w(c: &Call) -> u64 {
    get_attrs(wide(c, 0))
}

fn get_file_attributes_a(c: &Call) -> u64 {
    get_attrs(ansi(c, 0))
}

fn get_attrs_ex(path: Option<String>, out: *mut u8) -> u64 {
    if out.is_null() {
        return fail(ERROR_INVALID_PARAMETER);
    }
    match metadata(path) {
        // SAFETY: guest WIN32_FILE_ATTRIBUTE_DATA (36 bytes).
        Ok(m) => unsafe {
            put_attr_data(out, &m);
            1
        },
        Err(e) => fail(e),
    }
}

/// `GetFileAttributesExW(path, GetFileExInfoStandard, *data)`.
fn get_file_attributes_ex_w(c: &Call) -> u64 {
    get_attrs_ex(wide(c, 0), c.arg_ptr(2))
}

fn get_file_attributes_ex_a(c: &Call) -> u64 {
    get_attrs_ex(ansi(c, 0), c.arg_ptr(2))
}

// ---------------------------------------------------------------- find

/// Size of `WIN32_FIND_DATAW` / `WIN32_FIND_DATAA`.
pub const FIND_DATA_W: usize = 592;
pub const FIND_DATA_A: usize = 320;

unsafe fn put_find_data(p: *mut u8, wide: bool, name: &str, m: &Metadata) {
    unsafe {
        std::ptr::write_bytes(p, 0, if wide { FIND_DATA_W } else { FIND_DATA_A });
        put_attr_data(p, m);
        if wide {
            write_wstr(p.add(44).cast(), 260, name);
        } else {
            let b = name.as_bytes();
            let n = b.len().min(259);
            std::ptr::copy_nonoverlapping(b.as_ptr(), p.add(44), n);
        }
    }
}

fn find_first(pattern: Option<String>, out: *mut u8, wide: bool) -> u64 {
    let Some(pattern) = pattern else {
        return fail_handle(ERROR_INVALID_PARAMETER);
    };
    if out.is_null() {
        return fail_handle(ERROR_INVALID_PARAMETER);
    }
    let p = pattern.replace('/', "\\");
    let (dir, mask) = match p.rfind('\\') {
        Some(i) => (p[..=i].to_string(), p[i + 1..].to_string()),
        None => (String::new(), p.clone()),
    };
    let host_dir = match vfs::resolve(if dir.is_empty() { "." } else { &dir }) {
        Ok(d) => d,
        Err(e) => return fail_handle(e),
    };
    let rd = match std::fs::read_dir(&host_dir) {
        Ok(rd) => rd,
        Err(e) => {
            return fail_handle(if e.kind() == std::io::ErrorKind::NotFound {
                ERROR_PATH_NOT_FOUND
            } else {
                win32_error(&e)
            })
        }
    };
    let mut hits: Vec<(String, Metadata)> = rd
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            (vfs::wildcard(&mask, &name)).then(|| e.metadata().ok().map(|m| (name, m)))?
        })
        .collect();
    hits.sort_by_key(|h| h.0.to_lowercase());
    let mut hits = VecDeque::from(hits);
    let Some((name, m)) = hits.pop_front() else {
        return fail_handle(ERROR_FILE_NOT_FOUND);
    };
    // SAFETY: guest WIN32_FIND_DATA buffer.
    unsafe { put_find_data(out, wide, &name, &m) };
    insert_io(Arc::new(FindObj {
        rest: Mutex::new(hits),
    }))
}

fn find_next(h: u64, out: *mut u8, wide: bool) -> u64 {
    let Some(f) = io::<FindObj>(h) else {
        return fail(ERROR_INVALID_HANDLE);
    };
    if out.is_null() {
        return fail(ERROR_INVALID_PARAMETER);
    }
    let next = f.rest.lock().unwrap().pop_front();
    match next {
        Some((name, m)) => {
            // SAFETY: guest WIN32_FIND_DATA buffer.
            unsafe { put_find_data(out, wide, &name, &m) };
            1
        }
        None => fail(ERROR_NO_MORE_FILES),
    }
}

fn find_first_file_w(c: &Call) -> u64 {
    find_first(wide(c, 0), c.arg_ptr(1), true)
}

fn find_first_file_a(c: &Call) -> u64 {
    find_first(ansi(c, 0), c.arg_ptr(1), false)
}

/// `FindFirstFileExW(pattern, level, *data, search_op, filter, flags)`.
fn find_first_file_ex_w(c: &Call) -> u64 {
    find_first(wide(c, 0), c.arg_ptr(2), true)
}

fn find_next_file_w(c: &Call) -> u64 {
    find_next(c.arg(0), c.arg_ptr(1), true)
}

fn find_next_file_a(c: &Call) -> u64 {
    find_next(c.arg(0), c.arg_ptr(1), false)
}

fn find_close(c: &Call) -> u64 {
    let h = c.arg(0);
    if io::<FindObj>(h).is_some() && object::close(h) {
        1
    } else {
        fail(ERROR_INVALID_HANDLE)
    }
}

// ---------------------------------------------------------------- namespace

fn path_op(path: Option<String>, op: impl FnOnce(&Path) -> std::io::Result<()>) -> u64 {
    match resolve(path) {
        Ok(host) => match op(&host) {
            Ok(()) => 1,
            Err(e) => fail(io_error(&host, &e)),
        },
        Err(e) => fail(e),
    }
}

fn delete_file_w(c: &Call) -> u64 {
    path_op(wide(c, 0), |p| std::fs::remove_file(p))
}

fn delete_file_a(c: &Call) -> u64 {
    path_op(ansi(c, 0), |p| std::fs::remove_file(p))
}

fn create_dir(path: Option<String>) -> u64 {
    match resolve(path) {
        Ok(host) if host.exists() => fail(ERROR_ALREADY_EXISTS),
        Ok(host) => match std::fs::create_dir(&host) {
            Ok(()) => 1,
            Err(e) => fail(if e.kind() == std::io::ErrorKind::NotFound {
                ERROR_PATH_NOT_FOUND
            } else {
                win32_error(&e)
            }),
        },
        Err(e) => fail(e),
    }
}

fn create_directory_w(c: &Call) -> u64 {
    create_dir(wide(c, 0))
}

fn create_directory_a(c: &Call) -> u64 {
    create_dir(ansi(c, 0))
}

fn remove_directory_w(c: &Call) -> u64 {
    path_op(wide(c, 0), |p| std::fs::remove_dir(p))
}

/// `MoveFileExW(src, dst, flags)`; `MoveFileW(src, dst)` is flags = 0.
fn move_file(src: Option<String>, dst: Option<String>, flags: u32) -> u64 {
    let (src, dst) = match (resolve(src), resolve(dst)) {
        (Ok(s), Ok(d)) => (s, d),
        (Err(e), _) | (_, Err(e)) => return fail(e),
    };
    if dst.exists() && flags & MOVEFILE_REPLACE_EXISTING == 0 {
        return fail(ERROR_ALREADY_EXISTS);
    }
    match std::fs::rename(&src, &dst) {
        Ok(()) => 1,
        Err(e) => fail(io_error(&src, &e)),
    }
}

fn move_file_ex_w(c: &Call) -> u64 {
    move_file(wide(c, 0), wide(c, 1), c.arg(2) as u32)
}

fn move_file_w(c: &Call) -> u64 {
    move_file(wide(c, 0), wide(c, 1), 0)
}

/// `GetCurrentDirectoryW(len, buf)`: chars written, or the size needed
/// (including NUL) if the buffer is too small.
fn get_current_directory_w(c: &Call) -> u64 {
    let (len, buf) = (c.arg(0) as u32 as usize, c.arg_ptr::<u16>(1));
    let cwd = vfs::current_dir();
    let need = cwd.encode_utf16().count();
    if buf.is_null() || len <= need {
        return (need + 1) as u64;
    }
    // SAFETY: guest buffer of `len` u16s.
    unsafe { write_wstr(buf, len, &cwd) as u64 }
}

fn set_current_directory_w(c: &Call) -> u64 {
    let Some(path) = wide(c, 0) else {
        return fail(ERROR_INVALID_PARAMETER);
    };
    let Some((dev, parts)) = vfs::normalize(&path) else {
        return fail(ERROR_PATH_NOT_FOUND);
    };
    match vfs::resolve(&path) {
        Ok(h) if h.is_dir() => {
            vfs::set_current_dir(&format!("{}:\\{}", dev.to_uppercase(), parts.join("\\")));
            1
        }
        Ok(_) => fail(ERROR_PATH_NOT_FOUND),
        Err(e) => fail(e),
    }
}

pub fn register_files(reg: &mut Registry, dlls: &[&str]) {
    for dll in dlls {
        reg.register(dll, "CreateFileW", create_file_w)
            .register(dll, "CreateFileA", create_file_a)
            .register(dll, "CreateFile2", create_file2)
            .register(dll, "GetStdHandle", get_std_handle)
            .register(dll, "ReadFile", read_file)
            .register(dll, "WriteFile", write_file)
            .register(dll, "GetOverlappedResult", get_overlapped_result)
            .register(dll, "SetFilePointer", set_file_pointer)
            .register(dll, "SetFilePointerEx", set_file_pointer_ex)
            .register(dll, "GetFileSize", get_file_size)
            .register(dll, "GetFileSizeEx", get_file_size_ex)
            .register(dll, "GetFileType", get_file_type)
            .register(dll, "FlushFileBuffers", flush_file_buffers)
            .register(dll, "SetEndOfFile", set_end_of_file)
            .register(dll, "GetFileAttributesW", get_file_attributes_w)
            .register(dll, "GetFileAttributesA", get_file_attributes_a)
            .register(dll, "GetFileAttributesExW", get_file_attributes_ex_w)
            .register(dll, "GetFileAttributesExA", get_file_attributes_ex_a)
            .register(dll, "FindFirstFileW", find_first_file_w)
            .register(dll, "FindFirstFileA", find_first_file_a)
            .register(dll, "FindFirstFileExW", find_first_file_ex_w)
            .register(dll, "FindNextFileW", find_next_file_w)
            .register(dll, "FindNextFileA", find_next_file_a)
            .register(dll, "FindClose", find_close)
            .register(dll, "DeleteFileW", delete_file_w)
            .register(dll, "DeleteFileA", delete_file_a)
            .register(dll, "CreateDirectoryW", create_directory_w)
            .register(dll, "CreateDirectoryA", create_directory_a)
            .register(dll, "RemoveDirectoryW", remove_directory_w)
            .register(dll, "MoveFileExW", move_file_ex_w)
            .register(dll, "MoveFileW", move_file_w)
            .register(dll, "GetCurrentDirectoryW", get_current_directory_w)
            .register(dll, "SetCurrentDirectoryW", set_current_directory_w);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guest::to_wide;
    use crate::teb::{last_error, GuestThread};

    /// Call with up to 8 args (the rest go on the "stack").
    fn call(f: fn(&Call) -> u64, a: &[u64]) -> u64 {
        let mut regs = [0u64; 4];
        let mut stack = [0u64; 4];
        for (i, v) in a.iter().enumerate() {
            if i < 4 {
                regs[i] = *v
            } else {
                stack[i - 4] = *v
            }
        }
        // SAFETY: stack holds 4 slots, enough for every handler here.
        f(&unsafe { Call::new(regs, stack.as_ptr()) })
    }

    fn w(s: &str) -> Vec<u16> {
        to_wide(s)
    }

    fn p(v: &[u16]) -> u64 {
        v.as_ptr() as u64
    }

    #[test]
    fn create_write_seek_read_find_delete() {
        let _g = GuestThread::attach(0, 0);
        let root = std::env::temp_dir().join(format!("scarlett-vfs-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("Saves")).unwrap();
        vfs::mount("Q", &root);

        let name = w("Q:\\SAVES\\Slot1.DAT");
        let h = call(
            create_file_w,
            &[
                p(&name),
                (GENERIC_READ | GENERIC_WRITE) as u64,
                0,
                0,
                CREATE_ALWAYS as u64,
                0,
                0,
            ],
        );
        assert_ne!(h, INVALID_HANDLE_VALUE, "err {}", last_error());
        assert!(
            root.join("Saves/Slot1.DAT").exists(),
            "created inside the existing dir, guest spelling for the new file"
        );

        let data = b"hello, scarlett";
        let mut n = 0u32;
        assert_eq!(
            call(
                write_file,
                &[
                    h,
                    data.as_ptr() as u64,
                    data.len() as u64,
                    &mut n as *mut u32 as u64,
                    0
                ]
            ),
            1
        );
        assert_eq!(n as usize, data.len());
        let mut size = 0u64;
        assert_eq!(
            call(get_file_size_ex, &[h, &mut size as *mut u64 as u64]),
            1
        );
        assert_eq!(size, data.len() as u64);

        assert_eq!(call(set_file_pointer, &[h, 7, 0, 0]), 7);
        let mut buf = [0u8; 32];
        assert_eq!(
            call(
                read_file,
                &[h, buf.as_mut_ptr() as u64, 32, &mut n as *mut u32 as u64, 0]
            ),
            1
        );
        assert_eq!(&buf[..n as usize], b"scarlett");

        // OVERLAPPED read at offset 0, past EOF fails with ERROR_HANDLE_EOF.
        let mut ov = [0u8; 32];
        assert_eq!(
            call(
                read_file,
                &[h, buf.as_mut_ptr() as u64, 5, 0, ov.as_mut_ptr() as u64]
            ),
            1
        );
        assert_eq!(&buf[..5], b"hello");
        assert_eq!(u64::from_le_bytes(ov[8..16].try_into().unwrap()), 5);
        ov[16..20].copy_from_slice(&1000u32.to_le_bytes());
        assert_eq!(
            call(
                read_file,
                &[h, buf.as_mut_ptr() as u64, 5, 0, ov.as_mut_ptr() as u64]
            ),
            0
        );
        assert_eq!(last_error(), ERROR_HANDLE_EOF);
        assert_eq!(
            object::wait(&[h], false, 0),
            object::WAIT_OBJECT_0,
            "file handles are waitable"
        );

        // Reopen: OPEN_ALWAYS on an existing file reports ERROR_ALREADY_EXISTS.
        let h2 = call(
            create_file_w,
            &[
                p(&w("q:/saves/slot1.dat")),
                GENERIC_READ as u64,
                0,
                0,
                OPEN_ALWAYS as u64,
                0,
                0,
            ],
        );
        assert_ne!(h2, INVALID_HANDLE_VALUE);
        assert_eq!(last_error(), ERROR_ALREADY_EXISTS);

        // Attributes and find.
        assert_eq!(
            call(get_file_attributes_w, &[p(&w("Q:\\saves"))]),
            FILE_ATTRIBUTE_DIRECTORY as u64
        );
        let mut fd = [0u8; FIND_DATA_W];
        let f = call(
            find_first_file_w,
            &[p(&w("Q:\\Saves\\*.dat")), fd.as_mut_ptr() as u64],
        );
        assert_ne!(f, INVALID_HANDLE_VALUE);
        let fname: Vec<u16> = fd[44..44 + 18]
            .chunks(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(String::from_utf16_lossy(&fname), "Slot1.DAT");
        assert_eq!(
            u32::from_le_bytes(fd[32..36].try_into().unwrap()),
            data.len() as u32
        );
        assert_eq!(call(find_next_file_w, &[f, fd.as_mut_ptr() as u64]), 0);
        assert_eq!(last_error(), ERROR_NO_MORE_FILES);
        assert_eq!(call(find_close, &[f]), 1);

        // Missing paths.
        let missing = call(
            create_file_w,
            &[
                p(&w("Q:\\nope\\x.bin")),
                GENERIC_READ as u64,
                0,
                0,
                OPEN_EXISTING as u64,
                0,
                0,
            ],
        );
        assert_eq!(missing, INVALID_HANDLE_VALUE);
        assert_eq!(last_error(), ERROR_PATH_NOT_FOUND);
        call(
            create_file_w,
            &[
                p(&w("Q:\\saves\\none.bin")),
                GENERIC_READ as u64,
                0,
                0,
                OPEN_EXISTING as u64,
                0,
                0,
            ],
        );
        assert_eq!(last_error(), ERROR_FILE_NOT_FOUND);
        assert_eq!(
            call(get_file_attributes_w, &[p(&w("Z:\\unmounted"))]),
            INVALID_FILE_ATTRIBUTES as u64
        );

        for h in [h, h2] {
            assert!(object::close(h));
        }
        assert_eq!(call(delete_file_w, &[p(&w("Q:\\saves\\SLOT1.dat"))]), 1);
        assert!(!root.join("Saves/Slot1.DAT").exists());
        std::fs::remove_dir_all(&root).unwrap();
    }
}
