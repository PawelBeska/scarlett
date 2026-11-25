//! Maps a PE32+ image into host memory for native execution.
//!
//! Pipeline: parse → reserve `SizeOfImage` (preferred base first) → copy
//! headers and sections → apply base relocations → bind imports (HLE
//! handler, optional host forward, or logging stub) → set per-section page
//! protections → register `.pdata` unwind info (Windows) → ready to run.
//!
//! Input is a plain, already-extracted PE file. Unpacking or decrypting
//! console packages is out of scope.

mod dynamic;
pub mod mem;
#[cfg(windows)]
mod win;

pub use dynamic::install_dynamic_loading;
use mem::{MemError, Prot, Region, PAGE};
use scarlett_hle::teb::{self, GuestThread, TlsTemplate};
use scarlett_hle::Registry;
use scarlett_pe::{PeFile, REL_BASED_DIR64, REL_BASED_HIGHLOW};
use std::fmt;

#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Try to map at the image's preferred base (avoids relocation).
    pub prefer_image_base: bool,
    /// Windows only: resolve imports not covered by HLE against same-named
    /// host DLLs via LoadLibrary/GetProcAddress.
    pub forward_to_host: bool,
    /// Module name the guest sees (`game.exe`). Default: from the export
    /// directory, else `image.exe` / `image.dll`.
    pub name: Option<String>,
    /// Full guest path (`G:\game.exe`) for `GetModuleFileName`.
    /// Default: `G:\<name>`.
    pub guest_path: Option<String>,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions {
            prefer_image_base: true,
            forward_to_host: false,
            name: None,
            guest_path: None,
        }
    }
}

#[derive(Debug)]
pub enum LoadError {
    Pe(scarlett_pe::Error),
    Mem(MemError),
    BaseUnavailable { preferred: u64 },
    UnsupportedReloc { rva: u32, kind: u8 },
    BadTls(&'static str),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Pe(e) => write!(f, "PE: {e}"),
            LoadError::Mem(e) => write!(f, "memory: {e}"),
            LoadError::BaseUnavailable { preferred } => {
                write!(
                    f,
                    "preferred base {preferred:#x} unavailable and image has no relocations"
                )
            }
            LoadError::UnsupportedReloc { rva, kind } => {
                write!(f, "unsupported relocation type {kind} at RVA {rva:#x}")
            }
            LoadError::BadTls(why) => write!(f, "TLS directory: {why}"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<scarlett_pe::Error> for LoadError {
    fn from(e: scarlett_pe::Error) -> Self {
        LoadError::Pe(e)
    }
}

impl From<MemError> for LoadError {
    fn from(e: MemError) -> Self {
        LoadError::Mem(e)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Binding {
    /// Export of another guest module.
    Guest,
    Hle,
    Host,
    Stub,
}

#[derive(Debug, Clone)]
pub struct BoundImport {
    pub dll: String,
    pub name: String,
    pub binding: Binding,
    pub iat_rva: u32,
}

/// An image mapped and bound, ready to execute.
pub struct LoadedImage {
    // Field order is drop order: unregister unwind info before unmapping.
    #[cfg(windows)]
    _unwind: Option<win::FunctionTable>,
    // Before `image`: these point into it.
    _module: ModuleRegistration,
    _tls: Option<TlsRegistration>,
    _tls_callbacks: Option<CallbackRegistration>,
    image: Region,
    _thunks: Option<Region>,
    pub preferred_base: u64,
    pub entry_rva: u32,
    pub is_dll: bool,
    pub stack_reserve: u64,
    pub tls_callbacks: Vec<u64>,
    /// Value written to the image's `_tls_index`, if it has static TLS.
    pub tls_index: Option<u32>,
    pub imports: Vec<BoundImport>,
    pub relocations_applied: usize,
}

impl LoadedImage {
    pub fn base(&self) -> u64 {
        self.image.base()
    }

    pub fn region(&self) -> &Region {
        &self.image
    }

    pub fn entry(&self) -> u64 {
        self.base() + self.entry_rva as u64
    }

    /// Call TLS callbacks (DLL_PROCESS_ATTACH) and then the entry point on a
    /// fresh thread sized to the image's stack reserve. Returns the entry's
    /// return value (exit code for an EXE, BOOL for a DLL).
    ///
    /// # Safety
    /// Executes arbitrary guest machine code natively. Only run images you
    /// trust as much as any native program.
    pub unsafe fn run(&self) -> std::thread::Result<u64> {
        let base = self.base();
        let entry = self.entry();
        let is_dll = self.is_dll;
        let callbacks = self.tls_callbacks.clone();
        let stack = (self.stack_reserve as usize).max(1 << 20);
        let worker = std::thread::Builder::new()
            .name("guest-main".into())
            .stack_size(stack);
        let handle = worker
            .spawn(move || {
                let thread = GuestThread::attach_current(stack);
                // SAFETY (all calls): caller of `run` accepted executing guest code.
                // DLLs loaded as dependencies initialise first, in load order.
                for init in dynamic::take_pending() {
                    unsafe { init.run(|f, a, b, c| thread.call(f, a, b, c)) };
                }
                for cb in callbacks {
                    unsafe { thread.call(cb, base, teb::DLL_PROCESS_ATTACH, 0) };
                }
                if is_dll {
                    unsafe { thread.call(entry, base, teb::DLL_PROCESS_ATTACH, 0) }
                } else {
                    unsafe { thread.call(entry, 0, 0, 0) }
                }
            })
            .expect("spawn guest thread");
        handle.join()
    }
}

/// Keeps the module visible to `GetModuleHandle`/`GetProcAddress` and to
/// import binding of later images while it is mapped.
struct ModuleRegistration(u64);

impl Drop for ModuleRegistration {
    fn drop(&mut self) {
        scarlett_hle::module::unregister_guest(self.0);
    }
}

/// Keeps a module's TLS template registered while its image is mapped.
struct TlsRegistration(u32);

impl Drop for TlsRegistration {
    fn drop(&mut self) {
        teb::unregister_static_tls(self.0);
    }
}

/// Keeps a module's TLS callbacks registered for thread attach/detach.
struct CallbackRegistration(usize);

impl Drop for CallbackRegistration {
    fn drop(&mut self) {
        teb::unregister_tls_callbacks(self.0);
    }
}

pub fn load(
    bytes: &[u8],
    registry: &Registry,
    opts: &LoadOptions,
) -> Result<LoadedImage, LoadError> {
    let pe = PeFile::parse(bytes)?;
    let h = &pe.headers;
    let preferred = h.image_base;

    // 1. reserve
    let image = Region::alloc(
        h.size_of_image as usize,
        opts.prefer_image_base.then_some(preferred),
    )?;
    let base = image.base();
    // Same rule as the Windows loader: only an explicit RELOCS_STRIPPED
    // flag pins an image; otherwise an empty reloc table means "nothing to fix".
    if base != preferred && h.relocs_stripped() {
        return Err(LoadError::BaseUnavailable { preferred });
    }
    log::debug!(
        "mapped image at {base:#x} (preferred {preferred:#x}, size {:#x})",
        image.size()
    );

    // 2. copy
    let hdr_len = (h.size_of_headers as usize).min(bytes.len());
    image.write(0, &bytes[..hdr_len])?;
    for s in &pe.sections {
        let n = s.raw_size.min(s.mapped_size()) as usize;
        if n > 0 {
            let off = s.raw_offset as usize;
            image.write(s.virtual_address as usize, &bytes[off..off + n])?;
        }
    }

    // 3. relocate
    let delta = base.wrapping_sub(preferred);
    let mut relocations_applied = 0;
    if delta != 0 {
        for r in pe.relocations()? {
            let at = r.rva as usize;
            match r.kind {
                REL_BASED_DIR64 => image.write_u64(at, image.read_u64(at)?.wrapping_add(delta))?,
                REL_BASED_HIGHLOW => {
                    image.write_u32(at, image.read_u32(at)?.wrapping_add(delta as u32))?
                }
                kind => return Err(LoadError::UnsupportedReloc { rva: r.rva, kind }),
            }
            relocations_applied += 1;
        }
        // Keep the in-memory header consistent for code that reads its own ImageBase.
        let nt = image.read_u32(0x3C)? as usize;
        image.write_u64(nt + 4 + 20 + 24, base)?;
    }

    // 3b. make it findable by name before binding imports, so dependency
    // search can look next to this image
    let exports = pe.exports()?;
    let name = opts
        .name
        .clone()
        .or_else(|| pe.export_dll_name().ok().flatten())
        .unwrap_or_else(|| if h.is_dll() { "image.dll" } else { "image.exe" }.into());
    let mut module = scarlett_hle::module::GuestModule {
        guest_path: opts
            .guest_path
            .clone()
            .unwrap_or_else(|| format!("G:\\{name}")),
        name,
        base,
        size: image.size() as u64,
        is_dll: h.is_dll(),
        exports: Default::default(),
        ordinals: Default::default(),
        forwarders: Default::default(),
        pdata_rva: h.dir(scarlett_pe::Dir::Exception).rva,
        pdata_count: h.dir(scarlett_pe::Dir::Exception).size / 12,
        entry: if h.is_dll() && h.entry_point != 0 {
            base + h.entry_point as u64
        } else {
            0
        },
    };
    for e in exports {
        let va = base + e.rva as u64;
        match &e.forwarder {
            Some(f) => {
                module
                    .forwarders
                    .insert(format!("#{}", e.ordinal), f.clone());
                if let Some(n) = &e.name {
                    module.forwarders.insert(n.clone(), f.clone());
                }
            }
            None => {
                module.ordinals.insert(e.ordinal, va);
                if let Some(n) = e.name {
                    module.exports.insert(n, va);
                }
            }
        }
    }
    scarlett_hle::module::register_guest(module);
    let module_reg = ModuleRegistration(base);

    // 4. bind imports
    let modules = pe.imports()?;
    let mut imports = Vec::new();
    let mut pending_thunks = Vec::new(); // (iat_rva, thunk id)
                                         // Guest DLLs this image needs that HLE does not provide: map them first
                                         // so their exports can be bound below.
    if dynamic::installed() {
        for m in &modules {
            if !registry.has_dll(&m.dll) && scarlett_hle::module::find_guest(&m.dll).is_none() {
                if let Err(e) = scarlett_hle::module::load_library(&m.dll) {
                    log::debug!(
                        "dependency {} not loaded (error {e}); its imports will be stubs",
                        m.dll
                    );
                }
            }
        }
    }
    for m in &modules {
        for sym in &m.symbols {
            if let Some(addr) = scarlett_hle::module::guest_import(&m.dll, &sym.name) {
                image.write_u64(sym.iat_rva as usize, addr)?;
                imports.push(BoundImport {
                    dll: m.dll.clone(),
                    name: sym.name.to_string(),
                    binding: Binding::Guest,
                    iat_rva: sym.iat_rva,
                });
                continue;
            }
            let handler = registry.lookup(&m.dll, &sym.name);
            #[cfg_attr(not(windows), allow(unused_mut))]
            let mut binding = if handler.is_some() {
                Binding::Hle
            } else {
                Binding::Stub
            };
            if handler.is_none() && opts.forward_to_host {
                #[cfg(windows)]
                if let Some(addr) = win::host_proc(&m.dll, &sym.name) {
                    image.write_u64(sym.iat_rva as usize, addr)?;
                    binding = Binding::Host;
                }
                #[cfg(not(windows))]
                log::warn!("forward_to_host requested but only supported on Windows");
            }
            if binding != Binding::Host {
                pending_thunks.push((
                    sym.iat_rva,
                    scarlett_hle::add_thunk(&m.dll, &sym.name, handler),
                ));
            }
            imports.push(BoundImport {
                dll: m.dll.clone(),
                name: sym.name.to_string(),
                binding,
                iat_rva: sym.iat_rva,
            });
        }
    }
    let thunks = if pending_thunks.is_empty() {
        None
    } else {
        const SLOT: usize = 32;
        let region = Region::alloc(pending_thunks.len() * SLOT, None)?;
        let target = trampoline_address();
        for (i, (iat_rva, id)) in pending_thunks.iter().enumerate() {
            let off = i * SLOT;
            region.write(off, &scarlett_hle::encode_thunk(*id, target))?;
            region.write(off + 23, &[0xCC; SLOT - 23])?; // int3 padding
            image.write_u64(*iat_rva as usize, region.base() + off as u64)?;
        }
        region.protect(0, region.size(), Prot::ReadExec)?;
        Some(region)
    };

    // 5. TLS
    let tls = pe.tls()?;
    let tls_callbacks: Vec<u64> = tls
        .as_ref()
        .map(|t| {
            t.callbacks
                .iter()
                .map(|cb| cb.wrapping_add(delta))
                .collect()
        })
        .unwrap_or_default();
    let mut tls_reg = None;
    if let Some(t) = &tls {
        // VAs in the directory are preferred-base values; the copy in
        // memory was relocated, so shift them the same way.
        let rel = |va: u64| -> Result<usize, LoadError> {
            va.wrapping_add(delta)
                .checked_sub(base)
                .filter(|&o| o < image.size() as u64)
                .map(|o| o as usize)
                .ok_or(LoadError::BadTls("address outside image"))
        };
        let len = t.template_len() as usize;
        let data = if len > 0 {
            rel(t.raw_data_start_va)?
        } else {
            0
        };
        if len > 0 && data + len > image.size() {
            return Err(LoadError::BadTls("template runs past image end"));
        }
        if t.index_va != 0 {
            let index_at = rel(t.index_va)?;
            let index = teb::register_static_tls(TlsTemplate {
                data: base + data as u64,
                len,
                zero_fill: t.zero_fill as usize,
                align: t.alignment(),
            });
            tls_reg = Some(TlsRegistration(index));
            image.write_u32(index_at, index)?;
        } else if len > 0 || t.zero_fill > 0 {
            return Err(LoadError::BadTls("template without AddressOfIndex"));
        }
    }

    // 6. protections
    let align = h.section_alignment as usize;
    if align.is_multiple_of(PAGE) {
        image.protect(0, (h.size_of_headers as usize).max(1), Prot::Read)?;
        for s in &pe.sections {
            let prot = match (s.is_exec(), s.is_write()) {
                (true, true) => Prot::ReadWriteExec,
                (true, false) => Prot::ReadExec,
                (false, true) => Prot::ReadWrite,
                (false, false) => Prot::Read,
            };
            image.protect(
                s.virtual_address as usize,
                (s.mapped_size() as usize).max(1),
                prot,
            )?;
        }
    } else {
        log::warn!("section alignment {align:#x} below page size; mapping whole image RWX");
        image.protect(0, image.size(), Prot::ReadWriteExec)?;
    }

    // 7. unwind info
    #[cfg(windows)]
    let unwind = win::FunctionTable::register(&image, &pe);

    if !h.is_dll() {
        teb::set_image_base(base);
        scarlett_hle::object::set_default_stack_size(h.stack_reserve);
    }
    let callback_reg = (!tls_callbacks.is_empty())
        .then(|| CallbackRegistration(teb::register_tls_callbacks(base, tls_callbacks.clone())));

    Ok(LoadedImage {
        _module: module_reg,
        image,
        _thunks: thunks,
        preferred_base: preferred,
        entry_rva: h.entry_point,
        is_dll: h.is_dll(),
        stack_reserve: h.stack_reserve,
        tls_callbacks,
        tls_index: tls_reg.as_ref().map(|r| r.0),
        _tls: tls_reg,
        _tls_callbacks: callback_reg,
        imports,
        relocations_applied,
        #[cfg(windows)]
        _unwind: unwind,
    })
}

#[cfg(target_arch = "x86_64")]
fn trampoline_address() -> u64 {
    scarlett_hle::trampoline::trampoline_address()
}

#[cfg(not(target_arch = "x86_64"))]
compile_error!("scarlett runs guest code natively and requires an x86-64 host");
