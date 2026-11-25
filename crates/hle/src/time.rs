//! Clocks and sleeping: `QueryPerformance*`, `GetTickCount*`,
//! `GetSystemTime*AsFileTime`, `Sleep`, `SwitchToThread` (`SleepEx` lives in `object`, since it is
//! alertable).
//!
//! The performance counter runs at 10 MHz (what Windows 10+ reports on
//! invariant-TSC machines) from a monotonic host clock.

use crate::{Call, Registry};
use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const QPC_FREQUENCY: u64 = 10_000_000;
/// 100 ns intervals between 1601-01-01 and 1970-01-01.
pub const FILETIME_UNIX_EPOCH: u64 = 116_444_736_000_000_000;
pub const INFINITE: u32 = 0xFFFF_FFFF;

fn start() -> Instant {
    static START: OnceLock<Instant> = OnceLock::new();
    *START.get_or_init(Instant::now)
}

/// Monotonic time since emulator start. Starts at a nonzero offset so a
/// counter value of 0 never appears (some code treats 0 as "unset").
fn uptime() -> Duration {
    start().elapsed() + Duration::from_secs(1)
}

pub fn qpc() -> u64 {
    (uptime().as_nanos() / (1_000_000_000 / QPC_FREQUENCY as u128)) as u64
}

pub fn filetime_now() -> u64 {
    let since = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    FILETIME_UNIX_EPOCH + (since.as_nanos() / 100) as u64
}

fn write_u64(p: *mut u64, v: u64) -> u64 {
    if p.is_null() {
        crate::teb::set_last_error(crate::memory::ERROR_INVALID_PARAMETER);
        return 0;
    }
    // SAFETY: guest-provided out pointer, checked non-null.
    unsafe { p.write_unaligned(v) };
    1
}

fn query_performance_counter(c: &Call) -> u64 {
    write_u64(c.arg_ptr(0), qpc())
}

fn query_performance_frequency(c: &Call) -> u64 {
    write_u64(c.arg_ptr(0), QPC_FREQUENCY)
}

fn get_tick_count(_: &Call) -> u64 {
    uptime().as_millis() as u32 as u64
}

fn get_tick_count64(_: &Call) -> u64 {
    uptime().as_millis() as u64
}

fn get_system_time_as_file_time(c: &Call) -> u64 {
    write_u64(c.arg_ptr(0), filetime_now());
    0 // void
}

pub fn sleep_ms(ms: u32) {
    match ms {
        0 => std::thread::yield_now(),
        INFINITE => loop {
            std::thread::park();
        },
        ms => std::thread::sleep(Duration::from_millis(ms as u64)),
    }
}

fn sleep(c: &Call) -> u64 {
    sleep_ms(c.arg(0) as u32);
    0
}

fn switch_to_thread(_: &Call) -> u64 {
    std::thread::yield_now();
    1
}

pub fn register_time(reg: &mut Registry, dlls: &[&str]) {
    for dll in dlls {
        reg.register(dll, "QueryPerformanceCounter", query_performance_counter)
            .register(
                dll,
                "QueryPerformanceFrequency",
                query_performance_frequency,
            )
            .register(dll, "GetTickCount", get_tick_count)
            .register(dll, "GetTickCount64", get_tick_count64)
            .register(dll, "GetSystemTimeAsFileTime", get_system_time_as_file_time)
            .register(
                dll,
                "GetSystemTimePreciseAsFileTime",
                get_system_time_as_file_time,
            )
            .register(dll, "Sleep", sleep)
            .register(dll, "SwitchToThread", switch_to_thread);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_are_monotonic_and_filetime_is_sane() {
        let mut a = 0u64;
        let mut b = 0u64;
        let call =
            |p: &mut u64| unsafe { Call::new([p as *mut u64 as u64, 0, 0, 0], std::ptr::null()) };
        assert_eq!(query_performance_counter(&call(&mut a)), 1);
        std::thread::sleep(Duration::from_millis(2));
        query_performance_counter(&call(&mut b));
        assert!(b > a && b - a >= QPC_FREQUENCY / 1000, "{a} {b}");
        let mut f = 0;
        query_performance_frequency(&call(&mut f));
        assert_eq!(f, QPC_FREQUENCY);
        // 2020-01-01 .. 2100-01-01 in FILETIME units.
        let ft = filetime_now();
        assert!((132_223_104_000_000_000..157_469_184_000_000_000).contains(&ft));
    }
}
