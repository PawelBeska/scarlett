//! Maps a PE32+ image into host memory for native execution.
//!
//! Pipeline: parse → reserve `SizeOfImage` (preferred base first) → copy
//! headers and sections → apply base relocations → bind imports (HLE
//! handler, optional host forward, or logging stub) → set per-section page
//! protections → register `.pdata` unwind info (Windows) → ready to run.
//!
//! Input is a plain, already-extracted PE file. Unpacking or decrypting
//! console packages is out of scope.

pub mod mem;
#[cfg(windows)]
mod win;

use mem::{MemError, Prot, Region, PAGE};
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
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions {
            prefer_image_base: true,
            forward_to_host: false,
        }
    }
}

#[derive(Debug)]
pub enum LoadError {
    Pe(scarlett_pe::Error),
    Mem(MemError),
    BaseUnavailable { preferred: u64 },
    UnsupportedReloc { rva: u32, kind: u8 },
    UnsupportedTls,
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
            LoadError::UnsupportedTls => {
                f.write_str("image uses static TLS, which is not implemented yet")
            }
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
    image: Region,
    _thunks: Option<Region>,
    pub preferred_base: u64,
    pub entry_rva: u32,
    pub is_dll: bool,
    pub stack_reserve: u64,
    pub tls_callbacks: Vec<u64>,
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
        type Exe = unsafe extern "win64" fn() -> u64;
        type Dll = unsafe extern "win64" fn(u64, u32, u64) -> u64;
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
                for cb in callbacks {
                    // SAFETY: caller of `run` accepted executing guest code.
                    let f: Dll = unsafe { std::mem::transmute(cb as usize) };
                    unsafe { f(base, 1, 0) };
                }
                if is_dll {
                    let f: Dll = unsafe { std::mem::transmute(entry as usize) };
                    unsafe { f(base, 1, 0) }
                } else {
                    let f: Exe = unsafe { std::mem::transmute(entry as usize) };
                    unsafe { f() }
                }
            })
            .expect("spawn guest thread");
        handle.join()
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

    // 4. bind imports
    let modules = pe.imports()?;
    let mut imports = Vec::new();
    let mut pending_thunks = Vec::new(); // (iat_rva, thunk id)
    for m in &modules {
        for sym in &m.symbols {
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
    if let Some(t) = &tls {
        if t.raw_data_end_va > t.raw_data_start_va || t.zero_fill > 0 {
            return Err(LoadError::UnsupportedTls);
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

    Ok(LoadedImage {
        image,
        _thunks: thunks,
        preferred_base: preferred,
        entry_rva: h.entry_point,
        is_dll: h.is_dll(),
        stack_reserve: h.stack_reserve,
        tls_callbacks,
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
