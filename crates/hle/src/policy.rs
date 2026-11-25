//! What a call to an import with no handler does.
//!
//! Chosen by `SCARLETT_UNIMPLEMENTED` (read on first use) or
//! [`set_unimplemented_policy`]:
//!
//! - `log` (default): warn once per import, return 0 in rax and xmm0.
//! - `fail`: as `log`, but imports [`classify`] knows to return an
//!   `HRESULT` get `E_NOTIMPL`, so callers that check `FAILED(hr)` take
//!   their error path instead of using an object pointer that was never
//!   written.
//! - `stop`: on the first unimplemented call, print the import, the caller
//!   as `module+RVA` and rcx..r9, then end the process with exit code 3.
//!
//! Out parameters are never written in any mode.

use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unimplemented {
    Log,
    Fail,
    Stop,
}

/// What an import returns, as far as [`classify`] can tell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReturnClass {
    HResult,
    Other,
}

pub const E_NOTIMPL: u32 = 0x8000_4001;

/// Exit code of the `stop` policy.
pub const STOP_EXIT_CODE: i32 = 3;

const UNSET: u8 = 0xFF;
static POLICY: AtomicU8 = AtomicU8::new(UNSET);

/// `"log"`, `"fail"` or `"stop"`, any case. `None` for anything else.
pub fn parse(text: &str) -> Option<Unimplemented> {
    match text.trim().to_ascii_lowercase().as_str() {
        "log" => Some(Unimplemented::Log),
        "fail" => Some(Unimplemented::Fail),
        "stop" => Some(Unimplemented::Stop),
        _ => None,
    }
}

fn encode(p: Unimplemented) -> u8 {
    p as u8
}

fn decode(v: u8) -> Unimplemented {
    match v {
        1 => Unimplemented::Fail,
        2 => Unimplemented::Stop,
        _ => Unimplemented::Log,
    }
}

/// Override the environment (CLI flags, tests).
pub fn set_unimplemented_policy(p: Unimplemented) {
    POLICY.store(encode(p), Ordering::Relaxed);
}

pub fn unimplemented_policy() -> Unimplemented {
    let v = POLICY.load(Ordering::Relaxed);
    if v != UNSET {
        return decode(v);
    }
    let p = match std::env::var("SCARLETT_UNIMPLEMENTED") {
        Ok(s) if !s.is_empty() => parse(&s).unwrap_or_else(|| {
            log::warn!("SCARLETT_UNIMPLEMENTED={s:?} is not log/fail/stop; using log");
            Unimplemented::Log
        }),
        _ => Unimplemented::Log,
    };
    // A concurrent first call may race here; both compute the same value.
    let _ = POLICY.compare_exchange(UNSET, encode(p), Ordering::Relaxed, Ordering::Relaxed);
    decode(POLICY.load(Ordering::Relaxed))
}

/// DLLs whose every export returns an `HRESULT` (COM-style or GDK flat API).
/// Public: the GDK and D3D12 headers declare these functions `HRESULT`.
const HRESULT_DLLS: &[&str] = &[
    "xgameruntime",
    "gameinput",
    "xgameplatform",
    "gameplatformservices",
];

/// Function-name prefixes that return `HRESULT` regardless of DLL.
/// Public (Windows SDK / GDK headers): `D3D12CreateDevice`,
/// `D3D12SerializeRootSignature`, `CreateDXGIFactory*`, `CoCreateInstance`,
/// `CoInitializeEx`, `DllGetClassObject`, `XGameRuntimeInitialize`,
/// `XUser*`, `XGameSave*`, `XStore*`, `XAsync*`, `XTaskQueue*`.
///
/// ASSUMPTION: every `XUser`/`XGameSave`/`XStore`/`XAsync`/`XTaskQueue`
/// export returns HRESULT. A few return `bool` or `void` (e.g.
/// `XTaskQueueCloseHandle`); for them E_NOTIMPL is a harmless non-zero.
const HRESULT_PREFIXES: &[&str] = &[
    "D3D12",
    "CreateDXGIFactory",
    "DXGIGetDebugInterface",
    "CoCreate",
    "CoInitialize",
    "CoGetClassObject",
    "DllGetClassObject",
    "DllCanUnloadNow",
    "XGameRuntime",
    "XUser",
    "XGameSave",
    "XStore",
    "XAsync",
    "XTaskQueue",
    "XGameInvite",
    "XGameUi",
    "XGameStreaming",
    "XPackage",
    "XNetworking",
    "XSystem",
    "GameInputCreate",
];

/// Whether `dll!name` is known to return an `HRESULT`. Judged by name only.
pub fn classify(dll: &str, name: &str) -> ReturnClass {
    let dll = crate::normalize_dll(dll);
    if HRESULT_DLLS.contains(&dll.as_str()) || dll.starts_with("d3d12") || dll.starts_with("dxgi")
    {
        return ReturnClass::HResult;
    }
    if HRESULT_PREFIXES.iter().any(|p| name.starts_with(p)) {
        return ReturnClass::HResult;
    }
    ReturnClass::Other
}

/// Return value of an unimplemented `dll!name` under policy `p`
/// (`Stop` never returns; the caller handles it).
pub fn unimplemented_return(p: Unimplemented, dll: &str, name: &str) -> u64 {
    match (p, classify(dll, name)) {
        (Unimplemented::Fail, ReturnClass::HResult) => E_NOTIMPL as u64,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_policies() {
        assert_eq!(parse("LOG"), Some(Unimplemented::Log));
        assert_eq!(parse(" fail "), Some(Unimplemented::Fail));
        assert_eq!(parse("Stop"), Some(Unimplemented::Stop));
        assert_eq!(parse("abort"), None);
    }

    #[test]
    fn classifies_hresult_imports() {
        assert_eq!(classify("d3d12_xs.dll", "anything"), ReturnClass::HResult);
        assert_eq!(classify("XGameRuntime", "QueryApiImpl"), ReturnClass::HResult);
        assert_eq!(
            classify("api-ms-win-core-com-l1-1-0", "CoCreateInstance"),
            ReturnClass::HResult
        );
        assert_eq!(classify("kernel32", "XUserAddAsync"), ReturnClass::HResult);
        assert_eq!(classify("kernel32", "CreateFileW"), ReturnClass::Other);
        assert_eq!(classify("xmem", "XMemAlloc"), ReturnClass::Other);
    }

    #[test]
    fn fail_returns_e_notimpl_only_for_hresult() {
        let f = Unimplemented::Fail;
        assert_eq!(unimplemented_return(f, "dxgi", "CreateDXGIFactory2"), 0x8000_4001);
        assert_eq!(unimplemented_return(f, "kernel32", "GetTickCount"), 0);
        assert_eq!(
            unimplemented_return(Unimplemented::Log, "dxgi", "CreateDXGIFactory2"),
            0
        );
    }
}
