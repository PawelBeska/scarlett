//! Loading guest DLLs after the main image: import dependencies at load
//! time, and `LoadLibrary` at run time (via the HLE module hook).
//!
//! Mapped DLLs stay mapped for the rest of the process. `DllMain` (and the
//! DLL's TLS callbacks) run with `DLL_PROCESS_ATTACH`:
//! * immediately, if the load happens on a guest thread (`LoadLibrary`);
//! * otherwise (dependencies of the main image, loaded on the host thread
//!   before it starts) queued and run first thing by [`crate::LoadedImage::run`].

use crate::{load, LoadOptions, LoadedImage};
use scarlett_hle::{module, teb, vfs, Registry};
use std::sync::{Mutex, OnceLock};

pub const ERROR_MOD_NOT_FOUND: u32 = 126;
pub const ERROR_BAD_EXE_FORMAT: u32 = 193;
pub const ERROR_DLL_INIT_FAILED: u32 = 1114;

static CONFIG: OnceLock<(Registry, LoadOptions)> = OnceLock::new();
static LOADED: Mutex<Vec<LoadedImage>> = Mutex::new(Vec::new());
static PENDING: Mutex<Vec<DllInit>> = Mutex::new(Vec::new());

/// What it takes to initialise a freshly mapped DLL.
pub struct DllInit {
    base: u64,
    callbacks: Vec<u64>,
    entry: Option<u64>,
    name: String,
}

impl DllInit {
    /// Run TLS callbacks and DllMain through `call` (a guest-entry
    /// function). Returns DllMain's BOOL (1 if there is none).
    ///
    /// # Safety
    /// Runs guest code.
    pub unsafe fn run(&self, call: impl Fn(u64, u64, u64, u64) -> u64) -> u64 {
        for &cb in &self.callbacks {
            call(cb, self.base, teb::DLL_PROCESS_ATTACH, 0);
        }
        let ok = self.entry.map_or(1, |e| {
            call(e, self.base, teb::DLL_PROCESS_ATTACH, 0) as u32 as u64
        });
        if ok == 0 {
            log::warn!("DllMain of {} returned FALSE", self.name);
        }
        ok
    }
}

/// Enable dependency loading and `LoadLibrary` of guest DLLs, using
/// `registry` for their imports and `opts` as the base load options. Also
/// installs the registry for runtime `GetProcAddress`. First call wins.
pub fn install_dynamic_loading(registry: Registry, opts: LoadOptions) {
    module::install(registry.clone(), Some(hook));
    let _ = CONFIG.set((registry, opts));
}

pub(crate) fn installed() -> bool {
    CONFIG.get().is_some()
}

pub(crate) fn take_pending() -> Vec<DllInit> {
    std::mem::take(&mut *PENDING.lock().unwrap())
}

fn hook(guest_path: &str) -> Result<u64, u32> {
    let (registry, base_opts) = CONFIG.get().ok_or(ERROR_MOD_NOT_FOUND)?;
    let host = vfs::resolve(guest_path)?;
    let bytes = std::fs::read(&host).map_err(|_| ERROR_MOD_NOT_FOUND)?;
    let name = guest_path
        .rsplit(['\\', '/'])
        .next()
        .unwrap_or(guest_path)
        .to_string();
    let opts = LoadOptions {
        name: Some(name.clone()),
        guest_path: Some(guest_path.to_string()),
        prefer_image_base: base_opts.prefer_image_base,
        forward_to_host: base_opts.forward_to_host,
    };
    // Not holding any lock: `load` recurses into this hook for dependencies.
    let img = load(&bytes, registry, &opts).map_err(|e| {
        log::warn!("loading {guest_path}: {e}");
        ERROR_BAD_EXE_FORMAT
    })?;
    let base = img.base();
    log::info!("loaded {name} at {base:#x}");
    let init = DllInit {
        base,
        callbacks: img.tls_callbacks.clone(),
        entry: (img.is_dll && img.entry_rva != 0).then(|| img.entry()),
        name,
    };
    LOADED.lock().unwrap().push(img);
    if teb::current_teb().is_some() {
        // SAFETY: we are on a guest thread inside a LoadLibrary call.
        let ok = unsafe { init.run(|f, a, b, c| teb::call_guest(f, a, b, c)) };
        if ok == 0 {
            return Err(ERROR_DLL_INIT_FAILED);
        }
    } else {
        PENDING.lock().unwrap().push(init);
    }
    Ok(base)
}
