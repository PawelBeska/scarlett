//! `scarlett inventory <dir>`: every PE image under a game folder and, for
//! each of its imports, who provides it: another image of the package (by
//! export name, ordinal or forwarder), one of our HLE handlers, or nobody.
//! Host forwarding (Windows DLLs at run time) is not evaluated here: those
//! imports count as missing, which is what they are on a host without them.

use scarlett_hle::{normalize_dll, Registry};
use scarlett_pe::{ImportName, PeFile};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Who provides one import.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provider {
    Guest,
    Hle,
    Missing,
}

/// One image of the package.
pub struct Image {
    /// Path relative to the inventoried folder.
    pub path: String,
    /// (target module, symbol, provider) for every import.
    pub imports: Vec<(String, String, Provider)>,
}

pub struct Inventory {
    pub images: Vec<Image>,
    /// Files with a PE extension that did not parse, with the reason.
    pub unreadable: Vec<(String, String)>,
}

impl Inventory {
    pub fn count(&self, p: Provider) -> usize {
        self.images.iter().flat_map(|i| &i.imports).filter(|x| x.2 == p).count()
    }

    pub fn total(&self) -> usize {
        self.images.iter().map(|i| i.imports.len()).sum()
    }

    /// Missing imports per target module: (module, distinct symbols, import
    /// sites), most sites first.
    pub fn missing_by_module(&self) -> Vec<(String, usize, usize)> {
        let mut m: BTreeMap<String, (HashSet<String>, usize)> = BTreeMap::new();
        for (dll, sym, p) in self.images.iter().flat_map(|i| &i.imports) {
            if *p == Provider::Missing {
                let e = m.entry(normalize_dll(dll)).or_default();
                e.0.insert(sym.clone());
                e.1 += 1;
            }
        }
        let mut v: Vec<_> = m.into_iter().map(|(k, (s, n))| (k, s.len(), n)).collect();
        v.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));
        v
    }

    /// Missing (module, symbol) with the number of images importing it, most
    /// importers first.
    pub fn missing_symbols(&self) -> Vec<(String, String, usize)> {
        let mut m: BTreeMap<(String, String), usize> = BTreeMap::new();
        for img in &self.images {
            let mut seen = HashSet::new();
            for (dll, sym, p) in &img.imports {
                if *p == Provider::Missing && seen.insert((normalize_dll(dll), sym.clone())) {
                    *m.entry((normalize_dll(dll), sym.clone())).or_default() += 1;
                }
            }
        }
        let mut v: Vec<_> = m.into_iter().map(|((d, s), n)| (d, s, n)).collect();
        v.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
        v
    }
}

fn is_pe_name(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("exe") || e.eq_ignore_ascii_case("dll"))
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.collect::<Result<_, _>>()?;
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let p = e.path();
        if e.file_type()?.is_dir() {
            walk(&p, out)?;
        } else if is_pe_name(&p) {
            out.push(p);
        }
    }
    Ok(())
}

/// Exports of one package module: names and ordinals.
#[derive(Default)]
struct Exports {
    names: HashSet<String>,
    ordinals: HashSet<u32>,
}

pub fn inventory(dir: &Path) -> std::io::Result<Inventory> {
    let mut files = Vec::new();
    walk(dir, &mut files)?;
    let mut parsed = Vec::new();
    let mut unreadable = Vec::new();
    for f in &files {
        let rel = f.strip_prefix(dir).unwrap_or(f).display().to_string();
        let bytes = std::fs::read(f)?;
        match PeFile::parse(&bytes) {
            Ok(_) => parsed.push((rel, bytes)),
            Err(e) => unreadable.push((rel, e.to_string())),
        }
    }
    // package modules by file name (normalised as the HLE registry does)
    let mut exports: HashMap<String, Exports> = HashMap::new();
    let mut imports_of = Vec::new();
    for (rel, bytes) in &parsed {
        let pe = PeFile::parse(bytes).expect("parsed above");
        let name = Path::new(rel).file_name().and_then(|n| n.to_str()).unwrap_or(rel);
        let ex = exports.entry(normalize_dll(name)).or_default();
        match pe.exports() {
            Ok(list) => {
                for e in list {
                    ex.ordinals.insert(e.ordinal);
                    if let Some(n) = e.name {
                        ex.names.insert(n);
                    }
                }
            }
            Err(e) => unreadable.push((rel.clone(), format!("exports: {e}"))),
        }
        match pe.imports() {
            Ok(m) => imports_of.push((rel.clone(), m)),
            Err(e) => unreadable.push((rel.clone(), format!("imports: {e}"))),
        }
    }
    let all_dlls: Vec<String> = imports_of.iter().flat_map(|(_, m)| m.iter().map(|x| x.dll.clone())).collect();
    let refs: Vec<&str> = all_dlls.iter().map(String::as_str).collect();
    let mut reg = Registry::new();
    scarlett_hle::register_all(&mut reg, &refs);
    let mut images = Vec::new();
    for (rel, modules) in imports_of {
        let mut imports = Vec::new();
        for m in &modules {
            let ex = exports.get(&normalize_dll(&m.dll));
            for s in &m.symbols {
                let in_package = ex.is_some_and(|ex| match &s.name {
                    ImportName::Name { name, .. } => ex.names.contains(name),
                    ImportName::Ordinal(o) => ex.ordinals.contains(&(*o as u32)),
                });
                let p = if in_package {
                    Provider::Guest
                } else if reg.lookup(&m.dll, &s.name).is_some() {
                    Provider::Hle
                } else {
                    Provider::Missing
                };
                imports.push((m.dll.clone(), s.name.to_string(), p));
            }
        }
        images.push(Image { path: rel, imports });
    }
    Ok(Inventory { images, unreadable })
}

fn pct(n: usize, of: usize) -> String {
    if of == 0 {
        "-".into()
    } else {
        format!("{:.1}%", 100.0 * n as f64 / of as f64)
    }
}

/// The report as Markdown (also printed to stdout as is).
pub fn markdown(inv: &Inventory, top: usize) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let (g, h, m, t) = (
        inv.count(Provider::Guest),
        inv.count(Provider::Hle),
        inv.count(Provider::Missing),
        inv.total(),
    );
    let _ = writeln!(s, "# Import inventory\n");
    let _ = writeln!(
        s,
        "{} images, {t} imports: {g} from the package, {h} HLE, {m} missing; provided: {}\n",
        inv.images.len(),
        pct(g + h, t)
    );
    let _ = writeln!(s, "Host forwarding is not evaluated: imports a Windows DLL would serve count as missing.\n");
    let _ = writeln!(s, "## Images\n\n| image | imports | package | HLE | missing | provided |\n|---|---:|---:|---:|---:|---:|");
    for img in &inv.images {
        let c = |p| img.imports.iter().filter(|x| x.2 == p).count();
        let (g, h, m) = (c(Provider::Guest), c(Provider::Hle), c(Provider::Missing));
        let _ = writeln!(s, "| {} | {} | {g} | {h} | {m} | {} |", img.path, img.imports.len(), pct(g + h, img.imports.len()));
    }
    let _ = writeln!(s, "\n## Missing by module\n\n| module | symbols | import sites |\n|---|---:|---:|");
    for (d, n, sites) in inv.missing_by_module() {
        let _ = writeln!(s, "| {d} | {n} | {sites} |");
    }
    let miss = inv.missing_symbols();
    let _ = writeln!(s, "\n## Most imported missing symbols (top {})\n\n| module | symbol | images |\n|---|---|---:|", top.min(miss.len()));
    for (d, sym, n) in miss.iter().take(top) {
        let _ = writeln!(s, "| {d} | {sym} | {n} |");
    }
    if !inv.unreadable.is_empty() {
        let _ = writeln!(s, "\n## Not parsed\n");
        for (p, e) in &inv.unreadable {
            let _ = writeln!(s, "- {p}: {e}");
        }
    }
    s
}
