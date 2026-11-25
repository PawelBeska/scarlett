//! Minimal PE32+ writer for tests. Produces real, well-formed images with a
//! fixed layout so test code can hand-assemble machine code against known
//! RVAs:
//!
//! | section | RVA      | contents                                  |
//! |---------|----------|-------------------------------------------|
//! | .text   | 0x1000   | code from the caller                      |
//! | .rdata  | 0x2000   | import + export tables                    |
//! | .data   | 0x3000   | caller data, optionally with DIR64 relocs |
//! | .reloc  | 0x4000   | base relocations                          |

use std::collections::HashMap;

pub const TEXT_RVA: u32 = 0x1000;
pub const RDATA_RVA: u32 = 0x2000;
pub const DATA_RVA: u32 = 0x3000;
pub const RELOC_RVA: u32 = 0x4000;
const SECTION_SPAN: u32 = 0x1000;
const FILE_ALIGN: u32 = 0x200;
const HEADERS_SIZE: u32 = 0x400;

pub struct Layout {
    iat: HashMap<(String, String), u32>,
    pub image_base: u64,
}

impl Layout {
    /// RVA of the IAT slot for `dll!sym`.
    pub fn iat(&self, dll: &str, sym: &str) -> u32 {
        *self
            .iat
            .get(&(dll.to_string(), sym.to_string()))
            .unwrap_or_else(|| panic!("no import {dll}!{sym}"))
    }
}

pub struct PeBuilder {
    image_base: u64,
    imports: Vec<(String, Vec<String>)>,
    exports: Vec<(String, u32)>,
    data: Vec<u8>,
    data_relocs: Vec<u32>,
    dll: bool,
    with_relocs: bool,
    entry_offset: u32,
    tls_dir: Option<u32>,
    extra_dirs: Vec<(usize, u32, u32)>,
}

impl PeBuilder {
    pub fn new(image_base: u64) -> Self {
        PeBuilder {
            image_base,
            imports: Vec::new(),
            exports: Vec::new(),
            data: Vec::new(),
            data_relocs: Vec::new(),
            dll: false,
            with_relocs: true,
            entry_offset: 0,
            tls_dir: None,
            extra_dirs: Vec::new(),
        }
    }

    pub fn import(mut self, dll: &str, symbols: &[&str]) -> Self {
        self.imports
            .push((dll.into(), symbols.iter().map(|s| s.to_string()).collect()));
        self
    }

    /// Export a symbol at `offset` within .text.
    pub fn export(mut self, name: &str, text_offset: u32) -> Self {
        self.exports.push((name.into(), text_offset));
        self
    }

    pub fn dll(mut self) -> Self {
        self.dll = true;
        self
    }

    pub fn strip_relocs(mut self) -> Self {
        self.with_relocs = false;
        self
    }

    pub fn entry_offset(mut self, off: u32) -> Self {
        self.entry_offset = off;
        self
    }

    pub fn data(mut self, bytes: &[u8]) -> Self {
        self.data = bytes.to_vec();
        self
    }

    /// Write the absolute VA of `target_rva` (at the preferred base) into
    /// .data at `data_offset` and emit a DIR64 relocation for it.
    pub fn data_pointer(mut self, data_offset: u32, target_rva: u32) -> Self {
        let end = data_offset as usize + 8;
        if self.data.len() < end {
            self.data.resize(end, 0);
        }
        let va = self.image_base + target_rva as u64;
        self.data[data_offset as usize..end].copy_from_slice(&va.to_le_bytes());
        self.data_relocs.push(data_offset);
        self
    }

    /// Point the TLS data directory at a 40-byte `IMAGE_TLS_DIRECTORY64`
    /// the caller laid out in .data at `data_offset` (use [`Self::data_pointer`]
    /// for its VA fields so they get relocated).
    pub fn tls_directory(mut self, data_offset: u32) -> Self {
        self.tls_dir = Some(data_offset);
        self
    }

    /// Set data directory `index` to (`rva`, `size`), e.g. the exception
    /// directory (3) over a `RUNTIME_FUNCTION` table placed in .data.
    pub fn directory(mut self, index: usize, rva: u32, size: u32) -> Self {
        self.extra_dirs.push((index, rva, size));
        self
    }

    pub fn build(self, code: impl FnOnce(&Layout) -> Vec<u8>) -> Vec<u8> {
        // ---- .rdata: imports, then exports ----
        let mut rdata = Vec::<u8>::new();
        let mut iat_map = HashMap::new();
        let ndesc = self.imports.len();
        let desc_size = if ndesc > 0 { (ndesc + 1) * 20 } else { 0 };
        rdata.resize(desc_size, 0);
        let mut descs = Vec::new();
        for (dll, syms) in &self.imports {
            let ilt = rdata.len();
            rdata.resize(ilt + (syms.len() + 1) * 8, 0);
            let iat = rdata.len();
            rdata.resize(iat + (syms.len() + 1) * 8, 0);
            for (i, s) in syms.iter().enumerate() {
                align(&mut rdata, 2);
                let hn = RDATA_RVA as u64 + rdata.len() as u64;
                rdata.extend_from_slice(&0u16.to_le_bytes());
                rdata.extend_from_slice(s.as_bytes());
                rdata.push(0);
                for table in [ilt, iat] {
                    let o = table + i * 8;
                    rdata[o..o + 8].copy_from_slice(&hn.to_le_bytes());
                }
                iat_map.insert((dll.clone(), s.clone()), RDATA_RVA + (iat + i * 8) as u32);
            }
            let name = rdata.len();
            rdata.extend_from_slice(dll.as_bytes());
            rdata.push(0);
            descs.push((ilt, name, iat));
        }
        for (i, (ilt, name, iat)) in descs.iter().enumerate() {
            let o = i * 20;
            put32(&mut rdata, o, RDATA_RVA + *ilt as u32);
            put32(&mut rdata, o + 12, RDATA_RVA + *name as u32);
            put32(&mut rdata, o + 16, RDATA_RVA + *iat as u32);
        }
        let import_dir = (if ndesc > 0 { RDATA_RVA } else { 0 }, desc_size as u32);

        let mut export_dir = (0, 0);
        if !self.exports.is_empty() {
            align(&mut rdata, 4);
            let start = rdata.len();
            let n = self.exports.len();
            rdata.resize(start + 40, 0);
            let funcs = rdata.len();
            rdata.resize(funcs + n * 4, 0);
            let names = rdata.len();
            rdata.resize(names + n * 4, 0);
            let ords = rdata.len();
            rdata.resize(ords + n * 2, 0);
            let mut sorted: Vec<_> = self.exports.iter().enumerate().collect();
            sorted.sort_by(|a, b| a.1 .0.cmp(&b.1 .0));
            for (i, (_, off)) in self.exports.iter().enumerate() {
                put32(&mut rdata, funcs + i * 4, TEXT_RVA + off);
            }
            for (slot, (idx, (name, _))) in sorted.into_iter().enumerate() {
                let s = rdata.len();
                rdata.extend_from_slice(name.as_bytes());
                rdata.push(0);
                put32(&mut rdata, names + slot * 4, RDATA_RVA + s as u32);
                rdata[ords + slot * 2..ords + slot * 2 + 2]
                    .copy_from_slice(&(idx as u16).to_le_bytes());
            }
            let dll_name = rdata.len();
            rdata.extend_from_slice(b"test.dll\0");
            put32(&mut rdata, start + 12, RDATA_RVA + dll_name as u32);
            put32(&mut rdata, start + 16, 1);
            put32(&mut rdata, start + 20, n as u32);
            put32(&mut rdata, start + 24, n as u32);
            put32(&mut rdata, start + 28, RDATA_RVA + funcs as u32);
            put32(&mut rdata, start + 32, RDATA_RVA + names as u32);
            put32(&mut rdata, start + 36, RDATA_RVA + ords as u32);
            export_dir = (RDATA_RVA + start as u32, (rdata.len() - start) as u32);
        }

        // ---- .reloc ----
        let mut reloc = Vec::<u8>::new();
        if self.with_relocs && !self.data_relocs.is_empty() {
            let mut pages: std::collections::BTreeMap<u32, Vec<u16>> = Default::default();
            for off in &self.data_relocs {
                let rva = DATA_RVA + off;
                pages
                    .entry(rva & !0xFFF)
                    .or_default()
                    .push(((crate::REL_BASED_DIR64 as u16) << 12) | (rva & 0xFFF) as u16);
            }
            for (page, mut entries) in pages {
                if entries.len() % 2 == 1 {
                    entries.push(0); // ABSOLUTE padding
                }
                let start = reloc.len();
                reloc.resize(start + 8, 0);
                put32(&mut reloc, start, page);
                put32(&mut reloc, start + 4, 8 + entries.len() as u32 * 2);
                for e in entries {
                    reloc.extend_from_slice(&e.to_le_bytes());
                }
            }
        }

        let layout = Layout {
            iat: iat_map,
            image_base: self.image_base,
        };
        let text = code(&layout);

        // ---- assemble ----
        let mut sections: Vec<(&str, u32, Vec<u8>, u32)> = vec![
            (
                ".text",
                TEXT_RVA,
                text,
                crate::SCN_MEM_EXECUTE | crate::SCN_MEM_READ | 0x20,
            ),
            (".rdata", RDATA_RVA, rdata, crate::SCN_MEM_READ | 0x40),
            (
                ".data",
                DATA_RVA,
                self.data,
                crate::SCN_MEM_READ | crate::SCN_MEM_WRITE | 0x40,
            ),
        ];
        let reloc_dir = (
            if reloc.is_empty() { 0 } else { RELOC_RVA },
            reloc.len() as u32,
        );
        if !reloc.is_empty() {
            sections.push((
                ".reloc",
                RELOC_RVA,
                reloc,
                crate::SCN_MEM_READ | 0x0200_0040,
            ));
        }
        for (name, _, bytes, _) in &sections {
            assert!(
                bytes.len() as u32 <= SECTION_SPAN,
                "{name} exceeds builder section span"
            );
        }

        let mut file = vec![0u8; HEADERS_SIZE as usize];
        file[0] = b'M';
        file[1] = b'Z';
        put32(&mut file, 0x3C, 0x80);
        let nt = 0x80;
        put32(&mut file, nt, 0x0000_4550);
        let coff = nt + 4;
        put16(&mut file, coff, crate::MACHINE_AMD64);
        put16(&mut file, coff + 2, sections.len() as u16);
        put16(&mut file, coff + 16, 240);
        let mut chars = 0x0022u16; // EXECUTABLE_IMAGE | LARGE_ADDRESS_AWARE
        if self.dll {
            chars |= crate::FILE_DLL;
        }
        if !self.with_relocs {
            chars |= crate::FILE_RELOCS_STRIPPED;
        }
        put16(&mut file, coff + 18, chars);
        let opt = coff + 20;
        put16(&mut file, opt, crate::OPTIONAL_MAGIC_PE32_PLUS);
        put32(&mut file, opt + 16, TEXT_RVA + self.entry_offset);
        put32(&mut file, opt + 20, TEXT_RVA);
        file[opt + 24..opt + 32].copy_from_slice(&self.image_base.to_le_bytes());
        put32(&mut file, opt + 32, SECTION_SPAN);
        put32(&mut file, opt + 36, FILE_ALIGN);
        put16(&mut file, opt + 40, 10);
        put16(&mut file, opt + 48, 10);
        let last = sections.last().unwrap().1;
        put32(&mut file, opt + 56, last + SECTION_SPAN);
        put32(&mut file, opt + 60, HEADERS_SIZE);
        put16(&mut file, opt + 68, 3); // console
        put16(
            &mut file,
            opt + 70,
            if self.with_relocs { 0x0160 } else { 0x0100 },
        );
        file[opt + 72..opt + 80].copy_from_slice(&0x10_0000u64.to_le_bytes());
        file[opt + 80..opt + 88].copy_from_slice(&0x1000u64.to_le_bytes());
        put32(&mut file, opt + 108, 16);
        let dd = |i: usize| opt + 112 + i * 8;
        let tls_dir = self.tls_dir.map_or((0, 0), |o| (DATA_RVA + o, 40));
        for (i, (rva, size)) in [
            (0, export_dir),
            (1, import_dir),
            (5, reloc_dir),
            (9, tls_dir),
        ] {
            put32(&mut file, dd(i), rva);
            put32(&mut file, dd(i) + 4, size);
        }
        for &(i, rva, size) in &self.extra_dirs {
            put32(&mut file, dd(i), rva);
            put32(&mut file, dd(i) + 4, size);
        }

        let mut raw_off = HEADERS_SIZE;
        let sec_table = opt + 240;
        let mut bodies = Vec::new();
        for (i, (name, rva, bytes, chars)) in sections.into_iter().enumerate() {
            let s = sec_table + i * 40;
            file[s..s + name.len()].copy_from_slice(name.as_bytes());
            let raw_size = (bytes.len() as u32).div_ceil(FILE_ALIGN) * FILE_ALIGN;
            put32(&mut file, s + 8, bytes.len().max(1) as u32);
            put32(&mut file, s + 12, rva);
            put32(&mut file, s + 16, raw_size);
            put32(&mut file, s + 20, if raw_size > 0 { raw_off } else { 0 });
            put32(&mut file, s + 36, chars);
            if raw_size > 0 {
                let mut body = bytes;
                body.resize(raw_size as usize, 0);
                bodies.push(body);
                raw_off += raw_size;
            }
        }
        for b in bodies {
            file.extend_from_slice(&b);
        }
        file
    }
}

fn align(v: &mut Vec<u8>, a: usize) {
    while !v.len().is_multiple_of(a) {
        v.push(0);
    }
}

fn put16(v: &mut [u8], o: usize, x: u16) {
    v[o..o + 2].copy_from_slice(&x.to_le_bytes());
}

fn put32(v: &mut [u8], o: usize, x: u32) {
    v[o..o + 4].copy_from_slice(&x.to_le_bytes());
}
