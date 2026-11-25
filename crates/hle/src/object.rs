//! Kernel objects behind handles: events, semaphores, mutexes and threads,
//! plus `WaitFor*Object*`, `CloseHandle`, `DuplicateHandle` and the thread
//! APIs (`CreateThread`, `ExitThread`, `ResumeThread`, ...).
//!
//! One process-wide table guarded by a mutex, and one condition variable
//! that every state change notifies. Waiters re-check their objects on each
//! wake-up. That is O(waiters) per signal, which is fine at the object
//! counts games use and keeps wait-any/wait-all trivially correct.
//!
//! Object names are accepted and ignored (no `Open*` by name yet).

use crate::teb::{self, set_last_error, GuestThread};
use crate::{Call, Registry};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard};
use std::time::{Duration, Instant};

pub const WAIT_OBJECT_0: u32 = 0;
pub const WAIT_TIMEOUT: u32 = 0x102;
pub const WAIT_IO_COMPLETION: u32 = 0xC0;
pub const WAIT_FAILED: u32 = 0xFFFF_FFFF;
pub const STILL_ACTIVE: u32 = 259;
pub const MAXIMUM_WAIT_OBJECTS: usize = 64;
pub const INFINITE: u32 = crate::time::INFINITE;

pub const CREATE_SUSPENDED: u32 = 0x4;
pub const CREATE_EVENT_MANUAL_RESET: u32 = 0x1;
pub const CREATE_EVENT_INITIAL_SET: u32 = 0x2;
pub const CREATE_MUTEX_INITIAL_OWNER: u32 = 0x1;
pub const DUPLICATE_CLOSE_SOURCE: u32 = 0x1;

pub const ERROR_FILE_NOT_FOUND: u32 = 2;
pub const ERROR_INVALID_HANDLE: u32 = 6;
pub const ERROR_ALREADY_EXISTS: u32 = 183;
pub const ERROR_NOT_ENOUGH_MEMORY: u32 = 8;
pub const ERROR_INVALID_PARAMETER: u32 = 87;
pub const ERROR_NOT_OWNER: u32 = 288;
pub const ERROR_TOO_MANY_POSTS: u32 = 298;

/// `GetCurrentProcess()` / `GetCurrentThread()` pseudo handles.
pub const CURRENT_PROCESS: u64 = u64::MAX;
pub const CURRENT_THREAD: u64 = u64::MAX - 1;

#[derive(Debug)]
enum Kind {
    /// Files, consoles and find handles. I/O is synchronous, so they are
    /// always signaled. The object sits behind an `Arc` so I/O runs without
    /// holding the table lock.
    Io(Arc<dyn std::any::Any + Send + Sync>),
    Event {
        manual: bool,
        set: bool,
    },
    Semaphore {
        count: i64,
        max: i64,
    },
    Mutex {
        owner: u32,
        depth: u32,
    },
    Thread {
        tid: u32,
        done: bool,
        code: u32,
        suspend: u32,
    },
}

struct Entry {
    kind: Kind,
    /// Open handles plus internal references (a running thread holds one
    /// on its own object).
    refs: u32,
}

#[derive(Default)]
struct Table {
    objects: HashMap<u64, Entry>,
    handles: HashMap<u64, u64>,
    next_obj: u64,
    next_handle: u64,
    /// Named objects: lower-case name (without `Local\`/`Global\`) → id.
    names: HashMap<String, u64>,
    /// Pending user APCs per thread id: (function, argument).
    apcs: HashMap<u32, VecDeque<(u64, u64)>>,
}

impl Table {
    fn new_handle(&mut self, obj: u64) -> u64 {
        // Windows-looking values: small multiples of 4, never 0.
        self.next_handle = self.next_handle.max(0x100 - 4) + 4;
        self.handles.insert(self.next_handle, obj);
        self.objects.get_mut(&obj).unwrap().refs += 1;
        self.next_handle
    }

    fn create(&mut self, kind: Kind) -> u64 {
        self.next_obj += 1;
        let id = self.next_obj;
        self.objects.insert(id, Entry { kind, refs: 0 });
        self.new_handle(id)
    }

    fn id(&self, h: u64) -> Option<u64> {
        self.handles.get(&h).copied()
    }

    fn kind(&mut self, h: u64) -> Option<&mut Kind> {
        let id = self.id(h)?;
        self.objects.get_mut(&id).map(|e| &mut e.kind)
    }

    fn unref(&mut self, id: u64) {
        if let Some(e) = self.objects.get_mut(&id) {
            e.refs -= 1;
            if e.refs == 0 {
                self.objects.remove(&id);
                self.names.retain(|_, v| *v != id);
            }
        }
    }

    fn close(&mut self, h: u64) -> bool {
        match self.handles.remove(&h) {
            Some(id) => {
                self.unref(id);
                true
            }
            None => false,
        }
    }
}

static STATE: LazyLock<(Mutex<Table>, Condvar)> = LazyLock::new(Default::default);

fn table() -> MutexGuard<'static, Table> {
    STATE.0.lock().unwrap()
}

fn notify() {
    STATE.1.notify_all();
}

fn fail(code: u32) -> u64 {
    set_last_error(code);
    0
}

// ---------------------------------------------------------------- waiting

fn signaled(k: &Kind, me: u32) -> bool {
    match *k {
        Kind::Event { set, .. } => set,
        Kind::Semaphore { count, .. } => count > 0,
        Kind::Mutex { owner, .. } => owner == 0 || owner == me,
        Kind::Thread { done, .. } => done,
        Kind::Io(_) => true,
    }
}

/// Side effect of a satisfied wait.
fn acquire(k: &mut Kind, me: u32) {
    match k {
        Kind::Event { manual: false, set } => *set = false,
        Kind::Semaphore { count, .. } => *count -= 1,
        Kind::Mutex { owner, depth } => {
            *owner = me;
            *depth += 1;
        }
        _ => {}
    }
}

/// Wait for one (`all = false`) or all of `handles`. Returns
/// `WAIT_OBJECT_0 + i`, `WAIT_TIMEOUT` or `WAIT_FAILED`.
pub fn wait(handles: &[u64], all: bool, ms: u32) -> u32 {
    wait_ex(handles, all, ms, false)
}

/// [`wait`], optionally alertable: queued user APCs of this thread run
/// (in FIFO order) and the wait returns `WAIT_IO_COMPLETION`. With no
/// handles it is an alertable sleep.
pub fn wait_ex(handles: &[u64], all: bool, ms: u32, alertable: bool) -> u32 {
    if handles.len() > MAXIMUM_WAIT_OBJECTS || (handles.is_empty() && !alertable) {
        set_last_error(ERROR_INVALID_PARAMETER);
        return WAIT_FAILED;
    }
    let me = teb::current_thread_id();
    let deadline = (ms != INFINITE).then(|| Instant::now() + Duration::from_millis(ms as u64));
    let mut t = table();
    loop {
        if alertable {
            if let Some(q) = t.apcs.get_mut(&me).filter(|q| !q.is_empty()) {
                let pending: Vec<(u64, u64)> = q.drain(..).collect();
                drop(t);
                for (f, arg) in pending {
                    // SAFETY: APC routine queued by the guest: VOID f(ULONG_PTR).
                    unsafe { teb::call_guest(f, arg, 0, 0) };
                }
                return WAIT_IO_COMPLETION;
            }
        }
        let mut ids = Vec::with_capacity(handles.len());
        for &h in handles {
            match t.id(h) {
                Some(id) => ids.push(id),
                None => {
                    set_last_error(ERROR_INVALID_HANDLE);
                    return WAIT_FAILED;
                }
            }
        }
        let ready = |t: &Table, id: &u64| signaled(&t.objects[id].kind, me);
        let hit = if ids.is_empty() {
            None
        } else if all {
            ids.iter().all(|id| ready(&t, id)).then_some(0)
        } else {
            ids.iter().position(|id| ready(&t, id))
        };
        if let Some(i) = hit {
            let chosen: Vec<u64> = if all { ids } else { vec![ids[i]] };
            for id in chosen {
                acquire(&mut t.objects.get_mut(&id).unwrap().kind, me);
            }
            drop(t);
            notify();
            return WAIT_OBJECT_0 + i as u32;
        }
        t = match deadline {
            None => STATE.1.wait(t).unwrap(),
            Some(d) => {
                let now = Instant::now();
                if now >= d {
                    return WAIT_TIMEOUT;
                }
                STATE.1.wait_timeout(t, d - now).unwrap().0
            }
        };
    }
}

fn wait_for_single_object(c: &Call) -> u64 {
    wait(&[c.arg(0)], false, c.arg(1) as u32) as u64
}

fn wait_for_single_object_ex(c: &Call) -> u64 {
    wait_ex(&[c.arg(0)], false, c.arg(1) as u32, c.arg(2) as u32 != 0) as u64
}

/// `SleepEx(ms, alertable)`: 0 on timeout, `WAIT_IO_COMPLETION` if APCs ran.
fn sleep_ex(c: &Call) -> u64 {
    let (ms, alertable) = (c.arg(0) as u32, c.arg(1) as u32 != 0);
    if !alertable {
        crate::time::sleep_ms(ms);
        return 0;
    }
    match wait_ex(&[], false, ms, true) {
        WAIT_IO_COMPLETION => WAIT_IO_COMPLETION as u64,
        _ => 0,
    }
}

/// `QueueUserAPC(fn, thread, data)`.
fn queue_user_apc(c: &Call) -> u64 {
    let (f, h, data) = (c.arg(0), c.arg(1), c.arg(2));
    let mut t = table();
    let tid = if h == CURRENT_THREAD {
        teb::current_thread_id()
    } else {
        match t.kind(h) {
            Some(Kind::Thread {
                tid, done: false, ..
            }) => *tid,
            _ => {
                drop(t);
                return fail(ERROR_INVALID_HANDLE);
            }
        }
    };
    t.apcs.entry(tid).or_default().push_back((f, data));
    drop(t);
    notify();
    1
}

fn wait_for_multiple_objects(c: &Call) -> u64 {
    let n = c.arg(0) as usize;
    let arr = c.arg_ptr::<u64>(1);
    if arr.is_null() || n == 0 || n > MAXIMUM_WAIT_OBJECTS {
        set_last_error(ERROR_INVALID_PARAMETER);
        return WAIT_FAILED as u64;
    }
    // SAFETY: guest array of `n` handles.
    let handles = unsafe { std::slice::from_raw_parts(arr, n) }.to_vec();
    wait(&handles, c.arg(2) as u32 != 0, c.arg(3) as u32) as u64
}

/// `WaitForMultipleObjectsEx(n, handles, all, ms, alertable)`.
fn wait_for_multiple_objects_ex(c: &Call) -> u64 {
    let n = c.arg(0) as usize;
    let arr = c.arg_ptr::<u64>(1);
    if arr.is_null() || n == 0 || n > MAXIMUM_WAIT_OBJECTS {
        set_last_error(ERROR_INVALID_PARAMETER);
        return WAIT_FAILED as u64;
    }
    // SAFETY: guest array of `n` handles.
    let handles = unsafe { std::slice::from_raw_parts(arr, n) }.to_vec();
    wait_ex(
        &handles,
        c.arg(2) as u32 != 0,
        c.arg(3) as u32,
        c.arg(4) as u32 != 0,
    ) as u64
}

// ---------------------------------------------------------------- handles

fn close_handle(c: &Call) -> u64 {
    let h = c.arg(0);
    if h == CURRENT_PROCESS || h == CURRENT_THREAD {
        return 1;
    }
    if table().close(h) {
        1
    } else {
        fail(ERROR_INVALID_HANDLE)
    }
}

/// `DuplicateHandle(src_proc, src, dst_proc, *dst, access, inherit, options)`
/// within the current process.
fn duplicate_handle(c: &Call) -> u64 {
    let (src, out, options) = (c.arg(1), c.arg_ptr::<u64>(3), c.arg(6) as u32);
    let mut t = table();
    let Some(id) = t.id(src) else {
        drop(t);
        return fail(ERROR_INVALID_HANDLE);
    };
    let dup = t.new_handle(id);
    if options & DUPLICATE_CLOSE_SOURCE != 0 {
        t.close(src);
    }
    if !out.is_null() {
        // SAFETY: guest out pointer.
        unsafe { out.write_unaligned(dup) };
    }
    1
}

fn object_name(c: &Call, i: usize, wide: bool) -> Option<String> {
    let p = c.arg(i);
    // SAFETY: the API takes a NUL-terminated name or NULL here.
    let n = unsafe {
        if wide {
            crate::guest::wstr(p as *const u16)
        } else {
            crate::guest::cstr(p as *const u8)
        }
    }?;
    let n = n.to_lowercase();
    let n = n
        .strip_prefix("local\\")
        .or_else(|| n.strip_prefix("global\\"))
        .unwrap_or(&n)
        .to_string();
    (!n.is_empty()).then_some(n)
}

/// Create `kind`, or open the existing object of the same kind if `name`
/// is taken (last error ERROR_ALREADY_EXISTS, as on Windows).
fn create_named(name: Option<String>, kind: Kind) -> u64 {
    let mut t = table();
    if let Some(n) = &name {
        if let Some(&id) = t.names.get(n) {
            let same =
                std::mem::discriminant(&t.objects[&id].kind) == std::mem::discriminant(&kind);
            drop(t);
            if !same {
                return fail(ERROR_INVALID_HANDLE);
            }
            let h = table().new_handle(id);
            set_last_error(ERROR_ALREADY_EXISTS);
            return h;
        }
    }
    let h = t.create(kind);
    if let Some(n) = name {
        let id = t.id(h).unwrap();
        t.names.insert(n, id);
    }
    drop(t);
    set_last_error(0);
    h
}

/// `Open{Event,Mutex,Semaphore}{A,W}(access, inherit, name)`.
fn open_named(c: &Call, wide: bool, probe: Kind) -> u64 {
    let Some(name) = object_name(c, 2, wide) else {
        return fail(ERROR_INVALID_PARAMETER);
    };
    let mut t = table();
    match t.names.get(&name).copied() {
        Some(id)
            if std::mem::discriminant(&t.objects[&id].kind) == std::mem::discriminant(&probe) =>
        {
            t.new_handle(id)
        }
        Some(_) => {
            drop(t);
            fail(ERROR_INVALID_HANDLE)
        }
        None => {
            drop(t);
            fail(ERROR_FILE_NOT_FOUND)
        }
    }
}

const EVENT_PROBE: Kind = Kind::Event {
    manual: false,
    set: false,
};
const MUTEX_PROBE: Kind = Kind::Mutex { owner: 0, depth: 0 };
const SEMAPHORE_PROBE: Kind = Kind::Semaphore { count: 0, max: 1 };

fn open_event_w(c: &Call) -> u64 {
    open_named(c, true, EVENT_PROBE)
}
fn open_event_a(c: &Call) -> u64 {
    open_named(c, false, EVENT_PROBE)
}
fn open_mutex_w(c: &Call) -> u64 {
    open_named(c, true, MUTEX_PROBE)
}
fn open_mutex_a(c: &Call) -> u64 {
    open_named(c, false, MUTEX_PROBE)
}
fn open_semaphore_w(c: &Call) -> u64 {
    open_named(c, true, SEMAPHORE_PROBE)
}
fn open_semaphore_a(c: &Call) -> u64 {
    open_named(c, false, SEMAPHORE_PROBE)
}

// ---------------------------------------------------------------- I/O objects

/// Put an I/O object (file, find handle, ...) in the handle table.
pub fn insert_io<T: std::any::Any + Send + Sync>(obj: Arc<T>) -> u64 {
    table().create(Kind::Io(obj))
}

/// The I/O object behind `h`, if it is one of type `T`.
pub fn io<T: std::any::Any + Send + Sync>(h: u64) -> Option<Arc<T>> {
    match table().kind(h)? {
        Kind::Io(o) => o.clone().downcast::<T>().ok(),
        _ => None,
    }
}

/// Close `h` (any kind). Used by `FindClose`.
pub fn close(h: u64) -> bool {
    table().close(h)
}

// ---------------------------------------------------------------- events

#[cfg(test)]
fn new_event(manual: bool, set: bool) -> u64 {
    create_named(None, Kind::Event { manual, set })
}

/// `CreateEvent{A,W}(sa, manual, initial, name)`.
fn create_event(c: &Call, wide: bool) -> u64 {
    let kind = Kind::Event {
        manual: c.arg(1) as u32 != 0,
        set: c.arg(2) as u32 != 0,
    };
    create_named(object_name(c, 3, wide), kind)
}
fn create_event_w(c: &Call) -> u64 {
    create_event(c, true)
}
fn create_event_a(c: &Call) -> u64 {
    create_event(c, false)
}

/// `CreateEventEx{A,W}(sa, name, flags, access)`.
fn create_event_ex(c: &Call, wide: bool) -> u64 {
    let f = c.arg(2) as u32;
    let kind = Kind::Event {
        manual: f & CREATE_EVENT_MANUAL_RESET != 0,
        set: f & CREATE_EVENT_INITIAL_SET != 0,
    };
    create_named(object_name(c, 1, wide), kind)
}
fn create_event_ex_w(c: &Call) -> u64 {
    create_event_ex(c, true)
}
fn create_event_ex_a(c: &Call) -> u64 {
    create_event_ex(c, false)
}

fn set_event_state(h: u64, value: bool) -> u64 {
    let mut t = table();
    match t.kind(h) {
        Some(Kind::Event { set, .. }) => *set = value,
        _ => {
            drop(t);
            return fail(ERROR_INVALID_HANDLE);
        }
    }
    drop(t);
    notify();
    1
}

/// Signal an event by handle (e.g. `OVERLAPPED::hEvent` on completion).
pub fn signal_event(h: u64) -> bool {
    set_event_state(h, true) != 0
}

fn set_event(c: &Call) -> u64 {
    set_event_state(c.arg(0), true)
}

fn reset_event(c: &Call) -> u64 {
    set_event_state(c.arg(0), false)
}

// ---------------------------------------------------------------- semaphores

fn new_semaphore_named(name: Option<String>, initial: i64, max: i64) -> u64 {
    if max <= 0 || initial < 0 || initial > max {
        return fail(ERROR_INVALID_PARAMETER);
    }
    create_named(
        name,
        Kind::Semaphore {
            count: initial,
            max,
        },
    )
}

#[cfg(test)]
fn new_semaphore(initial: i64, max: i64) -> u64 {
    new_semaphore_named(None, initial, max)
}

/// `CreateSemaphore{,Ex}{A,W}(sa, initial, max, name, ..)`.
fn create_semaphore(c: &Call, wide: bool) -> u64 {
    new_semaphore_named(
        object_name(c, 3, wide),
        c.arg(1) as i32 as i64,
        c.arg(2) as i32 as i64,
    )
}
fn create_semaphore_w(c: &Call) -> u64 {
    create_semaphore(c, true)
}
fn create_semaphore_a(c: &Call) -> u64 {
    create_semaphore(c, false)
}

fn release_semaphore(c: &Call) -> u64 {
    let (h, n, prev) = (c.arg(0), c.arg(1) as i32 as i64, c.arg_ptr::<i32>(2));
    let mut t = table();
    let old = match t.kind(h) {
        Some(Kind::Semaphore { count, max }) => {
            if n <= 0 || *count + n > *max {
                drop(t);
                return fail(ERROR_TOO_MANY_POSTS);
            }
            let old = *count;
            *count += n;
            old
        }
        _ => {
            drop(t);
            return fail(ERROR_INVALID_HANDLE);
        }
    };
    drop(t);
    notify();
    if !prev.is_null() {
        // SAFETY: guest out pointer.
        unsafe { prev.write_unaligned(old as i32) };
    }
    1
}

// ---------------------------------------------------------------- mutexes

/// A named mutex that already exists is opened, ignoring `owned`.
fn new_mutex_named(name: Option<String>, owned: bool) -> u64 {
    let me = teb::current_thread_id();
    create_named(
        name,
        Kind::Mutex {
            owner: if owned { me } else { 0 },
            depth: owned as u32,
        },
    )
}

#[cfg(test)]
fn new_mutex(owned: bool) -> u64 {
    new_mutex_named(None, owned)
}

/// `CreateMutex{A,W}(sa, initial_owner, name)`.
fn create_mutex_w(c: &Call) -> u64 {
    new_mutex_named(object_name(c, 2, true), c.arg(1) as u32 != 0)
}
fn create_mutex_a(c: &Call) -> u64 {
    new_mutex_named(object_name(c, 2, false), c.arg(1) as u32 != 0)
}

/// `CreateMutexEx{A,W}(sa, name, flags, access)`.
fn create_mutex_ex_w(c: &Call) -> u64 {
    new_mutex_named(
        object_name(c, 1, true),
        c.arg(2) as u32 & CREATE_MUTEX_INITIAL_OWNER != 0,
    )
}
fn create_mutex_ex_a(c: &Call) -> u64 {
    new_mutex_named(
        object_name(c, 1, false),
        c.arg(2) as u32 & CREATE_MUTEX_INITIAL_OWNER != 0,
    )
}

fn release_mutex(c: &Call) -> u64 {
    let me = teb::current_thread_id();
    let mut t = table();
    match t.kind(c.arg(0)) {
        Some(Kind::Mutex { owner, depth }) if *owner == me && *depth > 0 => {
            *depth -= 1;
            if *depth == 0 {
                *owner = 0;
            }
        }
        Some(Kind::Mutex { .. }) => {
            drop(t);
            return fail(ERROR_NOT_OWNER);
        }
        _ => {
            drop(t);
            return fail(ERROR_INVALID_HANDLE);
        }
    }
    drop(t);
    notify();
    1
}

// ---------------------------------------------------------------- threads

static DEFAULT_STACK: AtomicU64 = AtomicU64::new(1 << 20);

/// Stack size for `CreateThread(.., 0, ..)`: the main image's
/// `SizeOfStackReserve`.
pub fn set_default_stack_size(bytes: u64) {
    if bytes > 0 {
        DEFAULT_STACK.store(bytes, Ordering::Relaxed);
    }
}

fn thread_main(
    obj: u64,
    start: u64,
    param: u64,
    stack: usize,
    ready: std::sync::mpsc::Sender<u32>,
) {
    let thread = GuestThread::attach_current(stack);
    let tid = teb::current_thread_id();
    let mut t = table();
    if let Some(Entry {
        kind: Kind::Thread { tid: slot, .. },
        ..
    }) = t.objects.get_mut(&obj)
    {
        *slot = tid;
    }
    let _ = ready.send(tid);
    // CREATE_SUSPENDED gate.
    while matches!(
        t.objects.get(&obj),
        Some(Entry {
            kind: Kind::Thread { suspend: 1.., .. },
            ..
        })
    ) {
        t = STATE.1.wait(t).unwrap();
    }
    drop(t);

    // TLS callbacks, then DllMain of each DLL (unless it called
    // DisableThreadLibraryCalls), as the Windows loader does.
    let mut notify_list = teb::tls_callbacks();
    notify_list.extend(crate::module::thread_notify_entries());
    // SAFETY (all calls): guest code the guest asked to run on this thread.
    for &(base, cb) in &notify_list {
        unsafe { thread.call(cb, base, teb::DLL_THREAD_ATTACH, 0) };
    }
    let code = unsafe { thread.call(start, param, 0, 0) } as u32;
    for &(base, cb) in notify_list.iter().rev() {
        unsafe { thread.call(cb, base, teb::DLL_THREAD_DETACH, 0) };
    }
    table().apcs.remove(&tid);
    drop(thread);

    let mut t = table();
    if let Some(Entry {
        kind: Kind::Thread { done, code: c, .. },
        ..
    }) = t.objects.get_mut(&obj)
    {
        *done = true;
        *c = code;
    }
    // Mutexes this thread still owns become free (Windows: abandoned).
    for e in t.objects.values_mut() {
        if let Kind::Mutex { owner, depth } = &mut e.kind {
            if *owner == tid {
                *owner = 0;
                *depth = 0;
            }
        }
    }
    t.unref(obj);
    drop(t);
    notify();
}

/// `CreateThread(sa, stack, start, param, flags, *tid)`.
fn create_thread(c: &Call) -> u64 {
    let (stack, start, param, flags, tid_out) = (
        c.arg(1),
        c.arg(2),
        c.arg(3),
        c.arg(4) as u32,
        c.arg_ptr::<u32>(5),
    );
    if start == 0 {
        return fail(ERROR_INVALID_PARAMETER);
    }
    let stack = match stack {
        0 => DEFAULT_STACK.load(Ordering::Relaxed),
        s => s.max(0x10000),
    } as usize;
    let (handle, obj) = {
        let mut t = table();
        let h = t.create(Kind::Thread {
            tid: 0,
            done: false,
            code: STILL_ACTIVE,
            suspend: (flags & CREATE_SUSPENDED != 0) as u32,
        });
        let obj = t.id(h).unwrap();
        t.objects.get_mut(&obj).unwrap().refs += 1; // the running thread's own ref
        (h, obj)
    };
    let (tx, rx) = std::sync::mpsc::channel();
    let spawned = std::thread::Builder::new()
        .name(format!("guest-{handle:#x}"))
        .stack_size(stack)
        .spawn(move || thread_main(obj, start, param, stack, tx));
    if spawned.is_err() {
        let mut t = table();
        t.unref(obj);
        t.close(handle);
        drop(t);
        return fail(ERROR_NOT_ENOUGH_MEMORY);
    }
    let tid = rx.recv().unwrap_or(0);
    if !tid_out.is_null() {
        // SAFETY: guest out pointer.
        unsafe { tid_out.write_unaligned(tid) };
    }
    handle
}

fn exit_thread(c: &Call) -> u64 {
    let code = c.arg(0) as u32 as u64;
    // SAFETY: handlers hold no locks here; dispatch released the thunk
    // table before calling us.
    unsafe { teb::exit_guest(code) }
}

fn resume_thread(c: &Call) -> u64 {
    let mut t = table();
    let prev = match t.kind(c.arg(0)) {
        Some(Kind::Thread { suspend, .. }) => {
            let p = *suspend;
            *suspend = suspend.saturating_sub(1);
            p
        }
        _ => {
            drop(t);
            set_last_error(ERROR_INVALID_HANDLE);
            return u32::MAX as u64;
        }
    };
    drop(t);
    notify();
    prev as u64
}

fn get_exit_code_thread(c: &Call) -> u64 {
    let out = c.arg_ptr::<u32>(1);
    let mut t = table();
    let code = match t.kind(c.arg(0)) {
        Some(Kind::Thread { code, .. }) => *code,
        _ => {
            drop(t);
            return fail(ERROR_INVALID_HANDLE);
        }
    };
    drop(t);
    if out.is_null() {
        return fail(ERROR_INVALID_PARAMETER);
    }
    // SAFETY: guest out pointer.
    unsafe { out.write_unaligned(code) };
    1
}

fn get_thread_id(c: &Call) -> u64 {
    let h = c.arg(0);
    if h == CURRENT_THREAD {
        return teb::current_thread_id() as u64;
    }
    match table().kind(h) {
        Some(Kind::Thread { tid, .. }) => *tid as u64,
        _ => fail(ERROR_INVALID_HANDLE),
    }
}

fn succeed(_: &Call) -> u64 {
    1
}

fn set_thread_affinity_mask(c: &Call) -> u64 {
    // Report "all cores" as the previous mask; affinity is not applied.
    if c.arg(1) == 0 {
        return fail(ERROR_INVALID_PARAMETER);
    }
    u64::MAX >> (64 - std::thread::available_parallelism().map_or(8, |n| n.get().min(64)))
}

fn set_thread_description(_: &Call) -> u64 {
    0 // S_OK
}

pub fn register_objects(reg: &mut Registry, dlls: &[&str]) {
    for dll in dlls {
        reg.register(dll, "CloseHandle", close_handle)
            .register(dll, "DuplicateHandle", duplicate_handle)
            .register(dll, "WaitForSingleObject", wait_for_single_object)
            .register(dll, "WaitForSingleObjectEx", wait_for_single_object_ex)
            .register(dll, "SleepEx", sleep_ex)
            .register(dll, "QueueUserAPC", queue_user_apc)
            .register(dll, "WaitForMultipleObjects", wait_for_multiple_objects)
            .register(
                dll,
                "WaitForMultipleObjectsEx",
                wait_for_multiple_objects_ex,
            )
            .register(dll, "CreateEventA", create_event_a)
            .register(dll, "CreateEventW", create_event_w)
            .register(dll, "CreateEventExA", create_event_ex_a)
            .register(dll, "CreateEventExW", create_event_ex_w)
            .register(dll, "OpenEventA", open_event_a)
            .register(dll, "OpenEventW", open_event_w)
            .register(dll, "OpenMutexA", open_mutex_a)
            .register(dll, "OpenMutexW", open_mutex_w)
            .register(dll, "OpenSemaphoreA", open_semaphore_a)
            .register(dll, "OpenSemaphoreW", open_semaphore_w)
            .register(dll, "SetEvent", set_event)
            .register(dll, "ResetEvent", reset_event)
            .register(dll, "CreateSemaphoreA", create_semaphore_a)
            .register(dll, "CreateSemaphoreW", create_semaphore_w)
            .register(dll, "CreateSemaphoreExA", create_semaphore_a)
            .register(dll, "CreateSemaphoreExW", create_semaphore_w)
            .register(dll, "ReleaseSemaphore", release_semaphore)
            .register(dll, "CreateMutexA", create_mutex_a)
            .register(dll, "CreateMutexW", create_mutex_w)
            .register(dll, "CreateMutexExA", create_mutex_ex_a)
            .register(dll, "CreateMutexExW", create_mutex_ex_w)
            .register(dll, "ReleaseMutex", release_mutex)
            .register(dll, "CreateThread", create_thread)
            .register(dll, "ExitThread", exit_thread)
            .register(dll, "ResumeThread", resume_thread)
            .register(dll, "GetExitCodeThread", get_exit_code_thread)
            .register(dll, "GetThreadId", get_thread_id)
            .register(dll, "SetThreadPriority", succeed)
            .register(dll, "SetThreadPriorityBoost", succeed)
            .register(dll, "SetThreadAffinityMask", set_thread_affinity_mask)
            .register(dll, "SetThreadDescription", set_thread_description);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(f: fn(&Call) -> u64, args: [u64; 4]) -> u64 {
        // SAFETY: none of the tested handlers reads stack arguments.
        f(&unsafe { Call::new(args, std::ptr::null()) })
    }

    #[test]
    fn auto_and_manual_events() {
        let auto = new_event(false, true);
        assert_eq!(wait(&[auto], false, 0), WAIT_OBJECT_0);
        assert_eq!(wait(&[auto], false, 0), WAIT_TIMEOUT, "auto-reset consumed");
        let manual = new_event(true, false);
        call(set_event, [manual, 0, 0, 0]);
        assert_eq!(wait(&[manual], false, 0), WAIT_OBJECT_0);
        assert_eq!(wait(&[manual], false, 0), WAIT_OBJECT_0, "manual stays set");
        call(reset_event, [manual, 0, 0, 0]);
        let t0 = Instant::now();
        assert_eq!(wait(&[manual], false, 20), WAIT_TIMEOUT);
        assert!(t0.elapsed() >= Duration::from_millis(20));
        assert_eq!(call(close_handle, [auto, 0, 0, 0]), 1);
        assert_eq!(call(close_handle, [auto, 0, 0, 0]), 0);
        assert_eq!(wait(&[auto], false, 0), WAIT_FAILED);
    }

    #[test]
    fn semaphore_counts_and_limits() {
        let s = new_semaphore(1, 2);
        let mut prev = -1i32;
        assert_eq!(
            call(release_semaphore, [s, 1, &mut prev as *mut i32 as u64, 0]),
            1
        );
        assert_eq!(prev, 1);
        assert_eq!(call(release_semaphore, [s, 1, 0, 0]), 0, "over max");
        assert_eq!(wait(&[s], false, 0), WAIT_OBJECT_0);
        assert_eq!(wait(&[s], false, 0), WAIT_OBJECT_0);
        assert_eq!(wait(&[s], false, 0), WAIT_TIMEOUT);
    }

    #[test]
    fn wait_any_and_all() {
        let a = new_event(true, false);
        let b = new_event(true, true);
        assert_eq!(wait(&[a, b], false, 0), WAIT_OBJECT_0 + 1);
        assert_eq!(wait(&[a, b], true, 0), WAIT_TIMEOUT);
        let setter = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(5));
            set_event_state(a, true);
        });
        assert_eq!(wait(&[a, b], true, 5000), WAIT_OBJECT_0);
        setter.join().unwrap();
    }

    #[test]
    fn named_objects_are_shared_and_typed() {
        let _g = GuestThread::attach(0, 0);
        let name = crate::guest::to_wide("Local\\ScarlettTestEvent");
        let p = name.as_ptr() as u64;
        let a = call(create_event_w, [0, 1, 0, p]);
        assert_eq!(teb::last_error(), 0);
        let b = call(create_event_w, [0, 1, 0, p]);
        assert_eq!(teb::last_error(), ERROR_ALREADY_EXISTS);
        assert_ne!(a, b);
        call(set_event, [a, 0, 0, 0]);
        assert_eq!(
            wait(&[b], false, 0),
            WAIT_OBJECT_0,
            "same object behind both handles"
        );
        // OpenEventW(access, inherit, name) without the namespace prefix
        let bare = crate::guest::to_wide("scarletttestevent");
        let o = call(open_event_w, [0, 0, bare.as_ptr() as u64, 0]);
        assert_ne!(o, 0);
        assert_eq!(
            call(open_mutex_w, [0, 0, bare.as_ptr() as u64, 0]),
            0,
            "wrong kind"
        );
        assert_eq!(teb::last_error(), ERROR_INVALID_HANDLE);
        for h in [a, b, o] {
            assert_eq!(call(close_handle, [h, 0, 0, 0]), 1);
        }
        assert_eq!(
            call(open_event_w, [0, 0, bare.as_ptr() as u64, 0]),
            0,
            "gone with its last handle"
        );
        assert_eq!(teb::last_error(), ERROR_FILE_NOT_FOUND);
    }

    static APC_SUM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    /// Stands in for guest code: Win64 ABI, no thread-locals (on macOS it
    /// runs with the guest `gs`).
    extern "win64" fn apc(arg: u64) {
        APC_SUM.fetch_add(arg, std::sync::atomic::Ordering::Relaxed);
    }

    #[test]
    fn apcs_run_in_alertable_waits_only() {
        let _g = GuestThread::attach(0, 0);
        let f = apc as *const () as usize as u64;
        assert_eq!(call(queue_user_apc, [f, CURRENT_THREAD, 5, 0]), 1);
        assert_eq!(call(queue_user_apc, [f, CURRENT_THREAD, 7, 0]), 1);
        let ev = new_event(true, false);
        assert_eq!(
            wait(&[ev], false, 0),
            WAIT_TIMEOUT,
            "non-alertable wait ignores APCs"
        );
        assert_eq!(APC_SUM.load(std::sync::atomic::Ordering::Relaxed), 0);
        let t0 = Instant::now();
        assert_eq!(call(sleep_ex, [10_000, 1, 0, 0]), WAIT_IO_COMPLETION as u64);
        assert!(t0.elapsed() < Duration::from_secs(5));
        assert_eq!(APC_SUM.load(std::sync::atomic::Ordering::Relaxed), 12);
        assert_eq!(
            call(sleep_ex, [1, 1, 0, 0]),
            0,
            "queue drained: plain timeout"
        );
        assert_eq!(wait_ex(&[ev], false, 0, true), WAIT_TIMEOUT);
    }

    #[test]
    fn mutex_is_recursive_and_owned() {
        let _g = GuestThread::attach(0, 0);
        let m = new_mutex(true);
        assert_eq!(wait(&[m], false, 0), WAIT_OBJECT_0, "owner re-enters");
        let other = std::thread::spawn(move || {
            let _g = GuestThread::attach(0, 0);
            (wait(&[m], false, 0), call(release_mutex, [m, 0, 0, 0]))
        })
        .join()
        .unwrap();
        assert_eq!(other, (WAIT_TIMEOUT, 0));
        assert_eq!(call(release_mutex, [m, 0, 0, 0]), 1);
        assert_eq!(call(release_mutex, [m, 0, 0, 0]), 1);
        assert_eq!(call(release_mutex, [m, 0, 0, 0]), 0, "no longer owned");
    }
}
