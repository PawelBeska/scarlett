//! User-mode synchronisation: `CRITICAL_SECTION`, `SRWLOCK`,
//! `CONDITION_VARIABLE`, `INIT_ONCE`. All state lives in guest memory, in
//! the same sizes as on Windows, and is driven with host atomics.
//!
//! Waiting is spin → yield → short sleep (no kernel wait queue yet); that
//! keeps the code simple and correct at the cost of wake-up latency under
//! contention. Condition variables are a generation counter, so a wake may
//! wake more waiters than asked (the API allows spurious wake-ups).

use crate::teb::{self, set_last_error};
use crate::{Call, Registry};
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering::*};
use std::time::{Duration, Instant};

pub const ERROR_TIMEOUT: u32 = 1460;

/// Escalating wait: spin, then yield, then sleep 50 µs.
pub struct Backoff(u32);

impl Backoff {
    pub fn new() -> Self {
        Backoff(0)
    }

    pub fn snooze(&mut self) {
        match self.0 {
            0..64 => std::hint::spin_loop(),
            64..128 => std::thread::yield_now(),
            _ => std::thread::sleep(Duration::from_micros(50)),
        }
        self.0 = self.0.saturating_add(1);
    }
}

impl Default for Backoff {
    fn default() -> Self {
        Self::new()
    }
}

fn atomic_u64<'a>(p: u64) -> &'a AtomicU64 {
    assert!(
        p != 0 && p.is_multiple_of(8),
        "unaligned or null sync object {p:#x}"
    );
    // SAFETY: guest sync objects are 8-byte aligned, pointer-sized fields
    // that the guest only touches through these APIs.
    unsafe { AtomicU64::from_ptr(p as *mut u64) }
}

fn atomic_i32<'a>(p: u64) -> &'a AtomicI32 {
    // SAFETY: as above, 4-byte field.
    unsafe { AtomicI32::from_ptr(p as *mut i32) }
}

// ---------------------------------------------------------------- CRITICAL_SECTION
//
// x64 layout: DebugInfo @0, LockCount @8 (i32), RecursionCount @12 (i32),
// OwningThread @16, LockSemaphore @24, SpinCount @32. 40 bytes.

mod cs {
    pub const LOCK_COUNT: u64 = 8;
    pub const RECURSION: u64 = 12;
    pub const OWNER: u64 = 16;
    pub const SPIN: u64 = 32;
    pub const SIZE: usize = 40;
}

fn me() -> u64 {
    teb::current_thread_id() as u64
}

pub fn cs_init(p: u64, spin: u64) {
    // SAFETY: guest CRITICAL_SECTION of cs::SIZE bytes.
    unsafe { std::ptr::write_bytes(p as *mut u8, 0, cs::SIZE) };
    atomic_i32(p + cs::LOCK_COUNT).store(-1, Relaxed);
    atomic_u64(p + cs::SPIN).store(spin & 0x00FF_FFFF, Relaxed);
}

pub fn cs_try_enter(p: u64) -> bool {
    let owner = atomic_u64(p + cs::OWNER);
    let rec = atomic_i32(p + cs::RECURSION);
    let me = me();
    if owner.load(Relaxed) == me {
        rec.fetch_add(1, Relaxed);
        return true;
    }
    if owner.compare_exchange(0, me, Acquire, Relaxed).is_ok() {
        rec.store(1, Relaxed);
        atomic_i32(p + cs::LOCK_COUNT).store(-2, Relaxed);
        return true;
    }
    false
}

pub fn cs_enter(p: u64) {
    let mut b = Backoff::new();
    while !cs_try_enter(p) {
        b.snooze();
    }
}

pub fn cs_leave(p: u64) {
    let owner = atomic_u64(p + cs::OWNER);
    if owner.load(Relaxed) != me() {
        log::warn!("LeaveCriticalSection({p:#x}) by a thread that does not own it");
        return;
    }
    if atomic_i32(p + cs::RECURSION).fetch_sub(1, Relaxed) == 1 {
        atomic_i32(p + cs::LOCK_COUNT).store(-1, Relaxed);
        owner.store(0, Release);
    }
}

fn initialize_critical_section(c: &Call) -> u64 {
    cs_init(c.arg(0), 0);
    0
}

fn initialize_critical_section_and_spin_count(c: &Call) -> u64 {
    cs_init(c.arg(0), c.arg(1));
    1
}

fn initialize_critical_section_ex(c: &Call) -> u64 {
    cs_init(c.arg(0), c.arg(1));
    1
}

fn enter_critical_section(c: &Call) -> u64 {
    cs_enter(c.arg(0));
    0
}

fn try_enter_critical_section(c: &Call) -> u64 {
    cs_try_enter(c.arg(0)) as u64
}

fn leave_critical_section(c: &Call) -> u64 {
    cs_leave(c.arg(0));
    0
}

fn delete_critical_section(_: &Call) -> u64 {
    0
}

fn set_critical_section_spin_count(c: &Call) -> u64 {
    atomic_u64(c.arg(0) + cs::SPIN).swap(c.arg(1) & 0x00FF_FFFF, Relaxed)
}

// ---------------------------------------------------------------- SRWLOCK
//
// One pointer: bit 0 = held exclusive, bits 1.. = shared holders.

const EXCL: u64 = 1;
const SHARED: u64 = 2;

pub fn srw_try_exclusive(p: u64) -> bool {
    atomic_u64(p)
        .compare_exchange(0, EXCL, Acquire, Relaxed)
        .is_ok()
}

pub fn srw_try_shared(p: u64) -> bool {
    let a = atomic_u64(p);
    let s = a.load(Relaxed);
    s & EXCL == 0 && a.compare_exchange(s, s + SHARED, Acquire, Relaxed).is_ok()
}

pub fn srw_exclusive(p: u64) {
    let mut b = Backoff::new();
    while !srw_try_exclusive(p) {
        b.snooze();
    }
}

pub fn srw_shared(p: u64) {
    let mut b = Backoff::new();
    while !srw_try_shared(p) {
        b.snooze();
    }
}

pub fn srw_release_exclusive(p: u64) {
    atomic_u64(p).store(0, Release);
}

pub fn srw_release_shared(p: u64) {
    atomic_u64(p).fetch_sub(SHARED, Release);
}

fn initialize_srw_lock(c: &Call) -> u64 {
    atomic_u64(c.arg(0)).store(0, Relaxed);
    0
}

fn acquire_srw_lock_exclusive(c: &Call) -> u64 {
    srw_exclusive(c.arg(0));
    0
}

fn acquire_srw_lock_shared(c: &Call) -> u64 {
    srw_shared(c.arg(0));
    0
}

fn release_srw_lock_exclusive(c: &Call) -> u64 {
    srw_release_exclusive(c.arg(0));
    0
}

fn release_srw_lock_shared(c: &Call) -> u64 {
    srw_release_shared(c.arg(0));
    0
}

fn try_acquire_srw_lock_exclusive(c: &Call) -> u64 {
    srw_try_exclusive(c.arg(0)) as u64
}

fn try_acquire_srw_lock_shared(c: &Call) -> u64 {
    srw_try_shared(c.arg(0)) as u64
}

// ---------------------------------------------------------------- CONDITION_VARIABLE
//
// One pointer used as a wake generation counter.

pub const CONDITION_VARIABLE_LOCKMODE_SHARED: u32 = 1;

/// Wait until the generation at `cv` moves past `seen` or `ms` expires.
fn cv_wait(cv: u64, seen: u64, ms: u32) -> bool {
    let deadline =
        (ms != crate::time::INFINITE).then(|| Instant::now() + Duration::from_millis(ms as u64));
    let mut b = Backoff::new();
    while atomic_u64(cv).load(Acquire) == seen {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return false;
        }
        b.snooze();
    }
    true
}

fn initialize_condition_variable(c: &Call) -> u64 {
    atomic_u64(c.arg(0)).store(0, Relaxed);
    0
}

fn wake_condition_variable(c: &Call) -> u64 {
    atomic_u64(c.arg(0)).fetch_add(1, Release);
    0
}

fn sleep_condition_variable_cs(c: &Call) -> u64 {
    let (cv, lock, ms) = (c.arg(0), c.arg(1), c.arg(2) as u32);
    let seen = atomic_u64(cv).load(Acquire);
    // Release fully, as Windows does, and restore the recursion depth after.
    let depth = atomic_i32(lock + cs::RECURSION).load(Relaxed);
    for _ in 0..depth {
        cs_leave(lock);
    }
    let woke = cv_wait(cv, seen, ms);
    for _ in 0..depth {
        cs_enter(lock);
    }
    if !woke {
        set_last_error(ERROR_TIMEOUT);
    }
    woke as u64
}

fn sleep_condition_variable_srw(c: &Call) -> u64 {
    let (cv, lock, ms, flags) = (c.arg(0), c.arg(1), c.arg(2) as u32, c.arg(3) as u32);
    let shared = flags & CONDITION_VARIABLE_LOCKMODE_SHARED != 0;
    let seen = atomic_u64(cv).load(Acquire);
    if shared {
        srw_release_shared(lock);
    } else {
        srw_release_exclusive(lock);
    }
    let woke = cv_wait(cv, seen, ms);
    if shared {
        srw_shared(lock);
    } else {
        srw_exclusive(lock);
    }
    if !woke {
        set_last_error(ERROR_TIMEOUT);
    }
    woke as u64
}

// ---------------------------------------------------------------- INIT_ONCE
//
// One pointer: 0 = not run, 1 = running, otherwise done with
// (context & !3) | 2.

const INIT_RUNNING: u64 = 1;
const INIT_DONE: u64 = 2;

/// `InitOnceExecuteOnce(once, fn, param, context)`: runs the guest callback
/// `BOOL fn(PINIT_ONCE, PVOID param, PVOID *context)` at most once.
fn init_once_execute_once(c: &Call) -> u64 {
    let (once, f, param, ctx_out) = (c.arg(0), c.arg(1), c.arg(2), c.arg_ptr::<u64>(3));
    let a = atomic_u64(once);
    let mut b = Backoff::new();
    loop {
        match a.compare_exchange(0, INIT_RUNNING, Acquire, Acquire) {
            Ok(_) => {
                let mut ctx = 0u64;
                // SAFETY: guest callback with the documented signature,
                // called back into on this guest thread.
                let ok =
                    unsafe { teb::call_guest(f, once, param, &mut ctx as *mut u64 as u64) } as u32;
                if ok == 0 {
                    a.store(0, Release);
                    return 0;
                }
                let done = (ctx & !3) | INIT_DONE;
                a.store(done, Release);
                if !ctx_out.is_null() {
                    // SAFETY: guest out pointer.
                    unsafe { ctx_out.write_unaligned(done & !3) };
                }
                return 1;
            }
            Err(INIT_RUNNING) => b.snooze(),
            Err(done) => {
                if !ctx_out.is_null() {
                    // SAFETY: guest out pointer.
                    unsafe { ctx_out.write_unaligned(done & !3) };
                }
                return 1;
            }
        }
    }
}

fn init_once_initialize(c: &Call) -> u64 {
    atomic_u64(c.arg(0)).store(0, Relaxed);
    0
}

pub fn register_sync(reg: &mut Registry, dlls: &[&str]) {
    for dll in dlls {
        reg.register(
            dll,
            "InitializeCriticalSection",
            initialize_critical_section,
        )
        .register(
            dll,
            "InitializeCriticalSectionAndSpinCount",
            initialize_critical_section_and_spin_count,
        )
        .register(
            dll,
            "InitializeCriticalSectionEx",
            initialize_critical_section_ex,
        )
        .register(dll, "EnterCriticalSection", enter_critical_section)
        .register(dll, "TryEnterCriticalSection", try_enter_critical_section)
        .register(dll, "LeaveCriticalSection", leave_critical_section)
        .register(dll, "DeleteCriticalSection", delete_critical_section)
        .register(
            dll,
            "SetCriticalSectionSpinCount",
            set_critical_section_spin_count,
        )
        .register(dll, "InitializeSRWLock", initialize_srw_lock)
        .register(dll, "AcquireSRWLockExclusive", acquire_srw_lock_exclusive)
        .register(dll, "AcquireSRWLockShared", acquire_srw_lock_shared)
        .register(dll, "ReleaseSRWLockExclusive", release_srw_lock_exclusive)
        .register(dll, "ReleaseSRWLockShared", release_srw_lock_shared)
        .register(
            dll,
            "TryAcquireSRWLockExclusive",
            try_acquire_srw_lock_exclusive,
        )
        .register(dll, "TryAcquireSRWLockShared", try_acquire_srw_lock_shared)
        .register(
            dll,
            "InitializeConditionVariable",
            initialize_condition_variable,
        )
        .register(dll, "WakeConditionVariable", wake_condition_variable)
        .register(dll, "WakeAllConditionVariable", wake_condition_variable)
        .register(dll, "SleepConditionVariableCS", sleep_condition_variable_cs)
        .register(
            dll,
            "SleepConditionVariableSRW",
            sleep_condition_variable_srw,
        )
        .register(dll, "InitOnceInitialize", init_once_initialize)
        .register(dll, "InitOnceExecuteOnce", init_once_execute_once);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::teb::GuestThread;
    use std::sync::Arc;

    #[test]
    fn critical_section_is_recursive_and_exclusive() {
        let cs = Arc::new([0u64; 5]);
        let p = cs.as_ptr() as u64;
        cs_init(p, 4000);
        let counter = Arc::new(AtomicU64::new(0));
        let threads: Vec<_> = (0..4)
            .map(|_| {
                let (cs, counter) = (cs.clone(), counter.clone());
                std::thread::spawn(move || {
                    let _g = GuestThread::attach(0, 0);
                    let p = cs.as_ptr() as u64;
                    for _ in 0..2000 {
                        cs_enter(p);
                        cs_enter(p); // recursion
                                     // Non-atomic read-modify-write: only safe under the lock.
                        let v = counter.load(Relaxed);
                        std::hint::spin_loop();
                        counter.store(v + 1, Relaxed);
                        cs_leave(p);
                        cs_leave(p);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(counter.load(Relaxed), 8000);
        assert_eq!(atomic_u64(p + cs::OWNER).load(Relaxed), 0);
        assert_eq!(atomic_i32(p + cs::LOCK_COUNT).load(Relaxed), -1);
    }

    #[test]
    fn srw_shared_excludes_exclusive() {
        let lock = Box::new(0u64);
        let p = &*lock as *const u64 as u64;
        assert!(srw_try_shared(p));
        assert!(srw_try_shared(p));
        assert!(!srw_try_exclusive(p));
        srw_release_shared(p);
        srw_release_shared(p);
        assert!(srw_try_exclusive(p));
        assert!(!srw_try_shared(p));
        srw_release_exclusive(p);
        assert_eq!(*lock, 0);
    }

    #[test]
    fn condition_variable_times_out_and_wakes() {
        let _g = GuestThread::attach(0, 0);
        let objs = Arc::new([0u64; 2]); // [cv, srw]
        let (cv, lock) = (objs.as_ptr() as u64, objs.as_ptr() as u64 + 8);
        let call = |a: [u64; 4]| unsafe { Call::new(a, std::ptr::null()) };
        srw_exclusive(lock);
        assert_eq!(sleep_condition_variable_srw(&call([cv, lock, 5, 0])), 0);
        assert_eq!(teb::last_error(), ERROR_TIMEOUT);
        let waker = {
            let objs = objs.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(5));
                wake_condition_variable(&unsafe {
                    Call::new([objs.as_ptr() as u64, 0, 0, 0], std::ptr::null())
                });
            })
        };
        assert_eq!(sleep_condition_variable_srw(&call([cv, lock, 5000, 0])), 1);
        assert_eq!(atomic_u64(lock).load(Relaxed), EXCL, "lock re-acquired");
        srw_release_exclusive(lock);
        waker.join().unwrap();
    }
}
