//! The GDK runtime, `xgameruntime.dll`.
//!
//! Titles link `xgameruntime.lib`, whose functions are thunks: they load
//! `xgameruntime.dll` at run time, `GetProcAddress` its initializers and
//! `QueryApiImpl`, ask `QueryApiImpl` for a per-feature class, call one
//! vtable slot and `Release` it. So an executable usually imports nothing
//! GDK-related by name, and the binding report says nothing about it; this
//! module makes the DLL exist for `LoadLibrary`/`GetProcAddress` (as a
//! virtual HLE module) and reports every class a title asks for.
//!
//! The flat exports declared in the public GDK headers
//! (`XGameRuntimeInitialize`, `XGameRuntimeUninitialize`) are served too.
//!
//! ASSUMPTION: `QueryApiImpl(const GUID* clsid, const GUID* iid, void** out)`
//! returning `HRESULT`. To be confirmed against the first real caller; the
//! log line prints all four argument registers for that reason.
//!
//! No class layouts are built in: they are not public. A class is added
//! with [`register_runtime_class`] once its slots are measured from a
//! title's thunks; until then `QueryApiImpl` refuses it with
//! `HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED)` and the class id lands in
//! [`requested_classes`] for the end-of-run report.

use crate::com::{self, Guid, State};
use crate::{Call, Registry};
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex, RwLock};

/// `HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED)`.
pub const E_NOT_SUPPORTED: u32 = 0x8007_0032;

/// DLL names this module answers to.
pub const DLLS: &[&str] = &["xgameruntime"];

struct RuntimeClass {
    class: u32,
    make: fn() -> State,
}

static CLASSES: LazyLock<RwLock<HashMap<Guid, RuntimeClass>>> = LazyLock::new(Default::default);
/// (class id, interface id, served) in request order, first request only.
static REQUESTS: Mutex<Vec<(Guid, Guid, bool)>> = Mutex::new(Vec::new());

/// Serve class id `clsid` with COM class `class` (from
/// [`com::register_class`]); `make` builds each new object's state.
/// `QueryApiImpl` accepts the class id itself as an interface id, plus
/// every interface id the COM class declares.
pub fn register_runtime_class(clsid: Guid, class: u32, make: fn() -> State) {
    CLASSES
        .write()
        .unwrap()
        .insert(clsid, RuntimeClass { class, make });
}

/// Every (class id, interface id) a title asked `QueryApiImpl` for, and
/// whether it was served.
pub fn requested_classes() -> Vec<(Guid, Guid, bool)> {
    REQUESTS.lock().unwrap().clone()
}

fn note(clsid: Guid, iid: Guid, served: bool) -> bool {
    let mut r = REQUESTS.lock().unwrap();
    if r.iter().any(|(c, i, _)| *c == clsid && *i == iid) {
        return false;
    }
    r.push((clsid, iid, served));
    true
}

fn query_api_impl(c: &Call) -> u64 {
    let (pclsid, piid, out) = (c.arg(0), c.arg(1), c.arg(2));
    if out == 0 {
        return com::E_POINTER as u64;
    }
    // SAFETY: REFCLSID / REFIID from the guest.
    let (Some(clsid), Some(iid)) = (unsafe { Guid::read(pclsid) }, unsafe { Guid::read(piid) })
    else {
        com::write_out(out, 0);
        return com::E_POINTER as u64;
    };
    let classes = CLASSES.read().unwrap();
    let served = classes.get(&clsid).filter(|rc| {
        iid == clsid || iid == com::IID_IUNKNOWN || com::class_iids(rc.class).contains(&iid)
    });
    match served {
        Some(rc) => {
            if note(clsid, iid, true) {
                log::info!("QueryApiImpl({clsid}, {iid}) -> {}", com::class_name(rc.class));
            }
            com::write_out(out, com::new_object(rc.class, (rc.make)()));
            com::S_OK
        }
        None => {
            if note(clsid, iid, false) {
                log::warn!(
                    "QueryApiImpl({clsid}, {iid}, out={out:#x}, r9={:#x}): class not served",
                    c.arg(3)
                );
            }
            com::write_out(out, 0);
            E_NOT_SUPPORTED as u64
        }
    }
}

/// `InitializeApiImpl*`: nothing to start offline. Arguments are logged
/// once (they carry the GDK version the title was built with).
fn initialize_api_impl(c: &Call) -> u64 {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        log::info!(
            "xgameruntime: InitializeApiImpl({:#x}, {:#x}, {:#x}, {:#x})",
            c.arg(0),
            c.arg(1),
            c.arg(2),
            c.arg(3)
        )
    });
    com::S_OK
}

fn s_ok(_: &Call) -> u64 {
    com::S_OK
}

pub fn register_xgameruntime(reg: &mut Registry) {
    for dll in DLLS {
        reg.register(dll, "InitializeApiImpl", initialize_api_impl)
            .register(dll, "InitializeApiImplEx", initialize_api_impl)
            .register(dll, "InitializeApiImplEx2", initialize_api_impl)
            .register(dll, "UninitializeApiImpl", s_ok)
            .register(dll, "QueryApiImpl", query_api_impl)
            .register(dll, "XGameRuntimeInitialize", initialize_api_impl)
            .register(dll, "XGameRuntimeInitializeWithOptions", initialize_api_impl)
            .register(dll, "XGameRuntimeUninitialize", s_ok);
    }
}
