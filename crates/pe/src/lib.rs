//! PE32+ (x86-64) image parser.
//!
//! Parses an image straight from its on-disk bytes: headers, sections,
//! imports (regular and delay-load), exports, base relocations, TLS and the
//! exception (`.pdata`) directory. No allocation-heavy abstractions, no
//! unsafe code: every read is bounds-checked.

#[cfg(feature = "builder")]
pub mod builder;
mod read;

use read::Reader;
use std::fmt;

pub const MACHINE_AMD64: u16 = 0x8664;
pub const OPTIONAL_MAGIC_PE32_PLUS: u16 = 0x20B;

pub const FILE_RELOCS_STRIPPED: u16 = 0x0001;
pub const FILE_DLL: u16 = 0x2000;

pub const SCN_MEM_EXECUTE: u32 = 0x2000_0000;
pub const SCN_MEM_READ: u32 = 0x4000_0000;
pub const SCN_MEM_WRITE: u32 = 0x8000_0000;
pub const SCN_CNT_UNINITIALIZED_DATA: u32 = 0x0000_0080;

pub const REL_BASED_ABSOLUTE: u8 = 0;
pub const REL_BASED_HIGHLOW: u8 = 3;
pub const REL_BASED_DIR64: u8 = 10;

/// Indices into the data directory table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Dir {
    Export = 0,
    Import = 1,
    Resource = 2,
    Exception = 3,
    Security = 4,
    BaseReloc = 5,
    Debug = 6,
    Tls = 9,
    LoadConfig = 10,
    Iat = 12,
    DelayImport = 13,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Truncated { what: &'static str, offset: usize },
    BadMagic(&'static str),
    UnsupportedMachine(u16),
    NotPe32Plus(u16),
    RvaNotMapped(u32),
    Malformed(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Truncated { what, offset } => {
                write!(f, "truncated {what} at offset {offset:#x}")
            }
            Error::BadMagic(w) => write!(f, "bad {w} signature"),
            Error::UnsupportedMachine(m) => write!(f, "unsupported machine {m:#06x} (only AMD64)"),
            Error::NotPe32Plus(m) => write!(f, "optional header magic {m:#06x} is not PE32+"),
            Error::RvaNotMapped(r) => write!(f, "RVA {r:#x} is not backed by file data"),
            Error::Malformed(w) => write!(f, "malformed {w}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DataDir {
    pub rva: u32,
    pub size: u32,
}

impl DataDir {
    pub fn is_present(&self) -> bool {
        self.rva != 0 && self.size != 0
    }
}

#[derive(Debug, Clone)]
pub struct Section {
    pub name: String,
    pub virtual_size: u32,
    pub virtual_address: u32,
    pub raw_size: u32,
    pub raw_offset: u32,
    pub characteristics: u32,
}

impl Section {
    /// Size the section occupies once mapped.
    pub fn mapped_size(&self) -> u32 {
        self.virtual_size.max(self.raw_size)
    }
    pub fn contains_rva(&self, rva: u32) -> bool {
        rva >= self.virtual_address && rva - self.virtual_address < self.mapped_size()
    }
    pub fn is_exec(&self) -> bool {
        self.characteristics & SCN_MEM_EXECUTE != 0
    }
    pub fn is_write(&self) -> bool {
        self.characteristics & SCN_MEM_WRITE != 0
    }
    pub fn is_read(&self) -> bool {
        self.characteristics & SCN_MEM_READ != 0
    }
}

#[derive(Debug, Clone)]
pub struct Headers {
    pub machine: u16,
    pub characteristics: u16,
    pub timestamp: u32,
    pub entry_point: u32,
    pub image_base: u64,
    pub section_alignment: u32,
    pub file_alignment: u32,
    pub size_of_image: u32,
    pub size_of_headers: u32,
    pub subsystem: u16,
    pub dll_characteristics: u16,
    pub stack_reserve: u64,
    pub stack_commit: u64,
    pub data_dirs: Vec<DataDir>,
}

impl Headers {
    pub fn dir(&self, d: Dir) -> DataDir {
        self.data_dirs.get(d as usize).copied().unwrap_or_default()
    }
    pub fn is_dll(&self) -> bool {
        self.characteristics & FILE_DLL != 0
    }
    pub fn relocs_stripped(&self) -> bool {
        self.characteristics & FILE_RELOCS_STRIPPED != 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImportName {
    Name { hint: u16, name: String },
    Ordinal(u16),
}

impl fmt::Display for ImportName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ImportName::Name { name, .. } => f.write_str(name),
            ImportName::Ordinal(o) => write!(f, "#{o}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ImportSymbol {
    pub name: ImportName,
    /// RVA of the IAT slot the loader must fill with the resolved address.
    pub iat_rva: u32,
}

#[derive(Debug, Clone)]
pub struct ImportModule {
    pub dll: String,
    pub delay_load: bool,
    pub symbols: Vec<ImportSymbol>,
}

#[derive(Debug, Clone)]
pub struct Export {
    pub ordinal: u32,
    pub name: Option<String>,
    pub rva: u32,
    /// Set when the export forwards to another DLL ("dll.Symbol").
    pub forwarder: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reloc {
    pub rva: u32,
    pub kind: u8,
}

#[derive(Debug, Clone, Default)]
pub struct Tls {
    pub raw_data_start_va: u64,
    pub raw_data_end_va: u64,
    pub index_va: u64,
    pub callbacks_va: u64,
    pub zero_fill: u32,
    /// Callback VAs (relative to the preferred image base), if any.
    pub callbacks: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeFunction {
    pub begin_rva: u32,
    pub end_rva: u32,
    pub unwind_rva: u32,
}

/// A parsed image borrowing its file bytes.
pub struct PeFile<'a> {
    data: &'a [u8],
    pub headers: Headers,
    pub sections: Vec<Section>,
}

impl<'a> PeFile<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        let r = Reader::new(data);
        if r.u16(0, "DOS header")? != 0x5A4D {
            return Err(Error::BadMagic("MZ"));
        }
        let nt = r.u32(0x3C, "e_lfanew")? as usize;
        if r.u32(nt, "NT signature")? != 0x0000_4550 {
            return Err(Error::BadMagic("PE"));
        }
        let coff = nt + 4;
        let machine = r.u16(coff, "COFF header")?;
        if machine != MACHINE_AMD64 {
            return Err(Error::UnsupportedMachine(machine));
        }
        let nsections = r.u16(coff + 2, "COFF header")? as usize;
        let timestamp = r.u32(coff + 4, "COFF header")?;
        let opt_size = r.u16(coff + 16, "COFF header")? as usize;
        let characteristics = r.u16(coff + 18, "COFF header")?;

        let opt = coff + 20;
        let magic = r.u16(opt, "optional header")?;
        if magic != OPTIONAL_MAGIC_PE32_PLUS {
            return Err(Error::NotPe32Plus(magic));
        }
        if opt_size < 112 {
            return Err(Error::Malformed("optional header size"));
        }
        let ndirs = (r.u32(opt + 108, "optional header")? as usize).min(16);
        if 112 + ndirs * 8 > opt_size {
            return Err(Error::Malformed("data directory count"));
        }
        let data_dirs = (0..ndirs)
            .map(|i| {
                let o = opt + 112 + i * 8;
                Ok(DataDir {
                    rva: r.u32(o, "data directory")?,
                    size: r.u32(o + 4, "data directory")?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let headers = Headers {
            machine,
            characteristics,
            timestamp,
            entry_point: r.u32(opt + 16, "optional header")?,
            image_base: r.u64(opt + 24, "optional header")?,
            section_alignment: r.u32(opt + 32, "optional header")?,
            file_alignment: r.u32(opt + 36, "optional header")?,
            size_of_image: r.u32(opt + 56, "optional header")?,
            size_of_headers: r.u32(opt + 60, "optional header")?,
            subsystem: r.u16(opt + 68, "optional header")?,
            dll_characteristics: r.u16(opt + 70, "optional header")?,
            stack_reserve: r.u64(opt + 72, "optional header")?,
            stack_commit: r.u64(opt + 80, "optional header")?,
            data_dirs,
        };
        if headers.size_of_image == 0 || headers.section_alignment == 0 {
            return Err(Error::Malformed("image size/alignment"));
        }

        let sec_table = opt + opt_size;
        let mut sections = Vec::with_capacity(nsections);
        for i in 0..nsections {
            let s = sec_table + i * 40;
            let raw_name = r.bytes(s, 8, "section header")?;
            let end = raw_name.iter().position(|&b| b == 0).unwrap_or(8);
            let sec = Section {
                name: String::from_utf8_lossy(&raw_name[..end]).into_owned(),
                virtual_size: r.u32(s + 8, "section header")?,
                virtual_address: r.u32(s + 12, "section header")?,
                raw_size: r.u32(s + 16, "section header")?,
                raw_offset: r.u32(s + 20, "section header")?,
                characteristics: r.u32(s + 36, "section header")?,
            };
            let end = sec.virtual_address as u64 + sec.mapped_size() as u64;
            if end > headers.size_of_image as u64 {
                return Err(Error::Malformed("section outside SizeOfImage"));
            }
            if sec.raw_size > 0 {
                r.bytes(
                    sec.raw_offset as usize,
                    sec.raw_size as usize,
                    "section data",
                )?;
            }
            sections.push(sec);
        }

        Ok(PeFile {
            data,
            headers,
            sections,
        })
    }

    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Translate an RVA to a file offset. Bytes inside a section's virtual
    /// tail (beyond raw data) are not file-backed and yield an error.
    pub fn rva_to_offset(&self, rva: u32) -> Result<usize> {
        if rva < self.headers.size_of_headers {
            return Ok(rva as usize);
        }
        for s in &self.sections {
            if s.contains_rva(rva) {
                let delta = rva - s.virtual_address;
                if delta < s.raw_size {
                    return Ok((s.raw_offset + delta) as usize);
                }
                return Err(Error::RvaNotMapped(rva));
            }
        }
        Err(Error::RvaNotMapped(rva))
    }

    fn at(&self, rva: u32) -> Result<(Reader<'a>, usize)> {
        Ok((Reader::new(self.data), self.rva_to_offset(rva)?))
    }

    fn cstr_at(&self, rva: u32) -> Result<String> {
        let (r, off) = self.at(rva)?;
        r.cstr(off, "string")
    }

    fn import_name(&self, thunk: u64) -> Result<ImportName> {
        if thunk & (1 << 63) != 0 {
            Ok(ImportName::Ordinal((thunk & 0xFFFF) as u16))
        } else {
            let rva = (thunk & 0x7FFF_FFFF) as u32;
            let (r, off) = self.at(rva)?;
            Ok(ImportName::Name {
                hint: r.u16(off, "hint")?,
                name: r.cstr(off + 2, "import name")?,
            })
        }
    }

    fn thunk_list(&self, ilt_rva: u32, iat_rva: u32) -> Result<Vec<ImportSymbol>> {
        let mut out = Vec::new();
        for i in 0u32.. {
            if i > 1 << 16 {
                return Err(Error::Malformed("import thunk table (unterminated)"));
            }
            let (r, off) = self.at(ilt_rva + i * 8)?;
            let thunk = r.u64(off, "import thunk")?;
            if thunk == 0 {
                break;
            }
            out.push(ImportSymbol {
                name: self.import_name(thunk)?,
                iat_rva: iat_rva + i * 8,
            });
        }
        Ok(out)
    }

    pub fn imports(&self) -> Result<Vec<ImportModule>> {
        let mut mods = Vec::new();
        let dir = self.headers.dir(Dir::Import);
        if dir.is_present() {
            for i in 0u32.. {
                let (r, off) = self.at(dir.rva + i * 20)?;
                let ilt = r.u32(off, "import descriptor")?;
                let name = r.u32(off + 12, "import descriptor")?;
                let iat = r.u32(off + 16, "import descriptor")?;
                if ilt == 0 && name == 0 && iat == 0 {
                    break;
                }
                let lookup = if ilt != 0 { ilt } else { iat };
                mods.push(ImportModule {
                    dll: self.cstr_at(name)?,
                    delay_load: false,
                    symbols: self.thunk_list(lookup, iat)?,
                });
            }
        }
        let dir = self.headers.dir(Dir::DelayImport);
        if dir.is_present() {
            for i in 0u32.. {
                let (r, off) = self.at(dir.rva + i * 32)?;
                let attrs = r.u32(off, "delay descriptor")?;
                let name = r.u32(off + 4, "delay descriptor")?;
                let iat = r.u32(off + 12, "delay descriptor")?;
                let int = r.u32(off + 16, "delay descriptor")?;
                if name == 0 {
                    break;
                }
                if attrs & 1 == 0 {
                    // Pre-VC7 delay descriptors hold VAs, not RVAs. Never seen in x64 output.
                    return Err(Error::Malformed("VA-based delay import descriptor"));
                }
                mods.push(ImportModule {
                    dll: self.cstr_at(name)?,
                    delay_load: true,
                    symbols: self.thunk_list(int, iat)?,
                });
            }
        }
        Ok(mods)
    }

    pub fn exports(&self) -> Result<Vec<Export>> {
        let dir = self.headers.dir(Dir::Export);
        if !dir.is_present() {
            return Ok(Vec::new());
        }
        let (r, off) = self.at(dir.rva)?;
        let base = r.u32(off + 16, "export directory")?;
        let nfuncs = r.u32(off + 20, "export directory")?;
        let nnames = r.u32(off + 24, "export directory")?;
        let funcs = r.u32(off + 28, "export directory")?;
        let names = r.u32(off + 32, "export directory")?;
        let ords = r.u32(off + 36, "export directory")?;
        if nfuncs > 1 << 16 || nnames > nfuncs {
            return Err(Error::Malformed("export directory counts"));
        }
        let mut name_of = vec![None; nfuncs as usize];
        for i in 0..nnames {
            let (r, o) = self.at(names + i * 4)?;
            let name_rva = r.u32(o, "export name")?;
            let (r, o) = self.at(ords + i * 2)?;
            let idx = r.u16(o, "export ordinal")? as usize;
            if idx < name_of.len() {
                name_of[idx] = Some(self.cstr_at(name_rva)?);
            }
        }
        let mut out = Vec::new();
        for (i, name) in name_of.into_iter().enumerate() {
            let (r, o) = self.at(funcs + i as u32 * 4)?;
            let rva = r.u32(o, "export address")?;
            if rva == 0 {
                continue;
            }
            let in_dir = rva >= dir.rva && rva - dir.rva < dir.size;
            let forwarder = if in_dir {
                Some(self.cstr_at(rva)?)
            } else {
                None
            };
            out.push(Export {
                ordinal: base + i as u32,
                name,
                rva,
                forwarder,
            });
        }
        Ok(out)
    }

    pub fn relocations(&self) -> Result<Vec<Reloc>> {
        let dir = self.headers.dir(Dir::BaseReloc);
        if !dir.is_present() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        let mut pos = 0u32;
        while pos + 8 <= dir.size {
            let (r, off) = self.at(dir.rva + pos)?;
            let page = r.u32(off, "reloc block")?;
            let size = r.u32(off + 4, "reloc block")?;
            if size < 8 || pos + size > dir.size {
                return Err(Error::Malformed("reloc block size"));
            }
            for e in 0..(size - 8) / 2 {
                let v = r.u16(off + 8 + e as usize * 2, "reloc entry")?;
                let kind = (v >> 12) as u8;
                if kind == REL_BASED_ABSOLUTE {
                    continue;
                }
                out.push(Reloc {
                    rva: page + (v & 0xFFF) as u32,
                    kind,
                });
            }
            pos += size;
        }
        Ok(out)
    }

    fn va_to_rva(&self, va: u64) -> Result<u32> {
        va.checked_sub(self.headers.image_base)
            .filter(|&d| d < self.headers.size_of_image as u64)
            .map(|d| d as u32)
            .ok_or(Error::Malformed("VA outside image"))
    }

    pub fn tls(&self) -> Result<Option<Tls>> {
        let dir = self.headers.dir(Dir::Tls);
        if !dir.is_present() {
            return Ok(None);
        }
        let (r, off) = self.at(dir.rva)?;
        let mut tls = Tls {
            raw_data_start_va: r.u64(off, "TLS directory")?,
            raw_data_end_va: r.u64(off + 8, "TLS directory")?,
            index_va: r.u64(off + 16, "TLS directory")?,
            callbacks_va: r.u64(off + 24, "TLS directory")?,
            zero_fill: r.u32(off + 32, "TLS directory")?,
            callbacks: Vec::new(),
        };
        if tls.callbacks_va != 0 {
            let base = self.va_to_rva(tls.callbacks_va)?;
            for i in 0..256u32 {
                let (r, o) = self.at(base + i * 8)?;
                let cb = r.u64(o, "TLS callback")?;
                if cb == 0 {
                    break;
                }
                tls.callbacks.push(cb);
            }
        }
        Ok(Some(tls))
    }

    pub fn runtime_functions(&self) -> Result<Vec<RuntimeFunction>> {
        let dir = self.headers.dir(Dir::Exception);
        if !dir.is_present() {
            return Ok(Vec::new());
        }
        let (r, off) = self.at(dir.rva)?;
        (0..dir.size as usize / 12)
            .map(|i| {
                let o = off + i * 12;
                Ok(RuntimeFunction {
                    begin_rva: r.u32(o, "RUNTIME_FUNCTION")?,
                    end_rva: r.u32(o + 4, "RUNTIME_FUNCTION")?,
                    unwind_rva: r.u32(o + 8, "RUNTIME_FUNCTION")?,
                })
            })
            .collect()
    }
}
