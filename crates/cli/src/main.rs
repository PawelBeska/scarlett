mod inventory;

use scarlett_hle::{vfs, Registry};
use scarlett_loader::{load, Binding, LoadOptions};
use scarlett_pe::{Dir, PeFile};
use std::process::ExitCode;

const USAGE: &str = "\
usage:
  scarlett inspect <image>
  scarlett inventory <game dir> [--md <report.md>] [--top N]
                       every PE image under the folder; each import marked as
                       provided by the package, by HLE, or missing (host
                       forwarding not evaluated); per image, per module and
                       the most imported missing symbols (default top 50)
  scarlett disasm <shader.bin> [byte offset]   RDNA2 listing of a raw shader dump
  scarlett translate <shader.bin> -o <out.spv> [--offset N] [--wave32]
                       [--emulate-cross-lane]   (wave64 on 32-wide subgroups)
                       [--lds-bytes N] [--rsrc2 COMPUTE_PGM_RSRC2 --workgroup XxYxZ]
                       (with --rsrc2: the hardware compute contract, user
                       SGPRs from push constants, results via memory)
                       [--sgprs N] [--vgprs N]
                       RDNA2 -> SPIR-V (register-dump I/O model; see
                       scarlett_gpu::translate), structurally validated
  scarlett pm4 <stream.bin> [--load ADDR=<file>]... [--alloc ADDR:BYTES]...
                       [--dump ADDR:BYTES=<file>]... [--spirv <dir> [--window ADDR:BYTES]]
                       run a PM4 command stream on GPU memory built from
                       files; dispatches execute on the reference
                       interpreter (--spirv also writes each dispatch's
                       translated compute shader there, its memory binding
                       being the --window range)
  scarlett run <image> [--unimplemented log|fail|stop] [--dry-run]
                       [--no-preferred-base] [--host-forward]
                       [--mount X=<host dir>]...

  inspect              headers, sections with entropy (near 8.0 = encrypted or
                       compressed), imports marked HLE / missing, exports, TLS
  --unimplemented P    what a call to an import with no handler does:
                         log   warn once, return 0 (default)
                         fail  as log, but E_NOTIMPL for HRESULT functions
                         stop  print import + caller module+RVA, exit code 3
                       (also SCARLETT_UNIMPLEMENTED=P; --strict = stop)
  --dry-run            map the image and its DLLs and bind every import, then
                       report what is missing without running guest code
  --no-preferred-base  always relocate (exercises the relocation path)
  --host-forward       Windows: bind non-HLE imports to same-named host DLLs
  --mount X=<dir>      expose a host directory as guest drive X: (or a device
                       name like game:). The image's directory is always G:,
                       which is also the current directory.

  RUST_LOG=debug for verbose loader output.";

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("inspect") if args.len() == 2 => inspect(&args[1]),
        Some("inventory") if args.len() >= 2 => inventory_cmd(&args[1], &args[2..]),
        Some("run") if args.len() >= 2 => run(&args[1], &args[2..]),
        Some("disasm") if (2..=3).contains(&args.len()) => disasm(&args[1], args.get(2)),
        Some("translate") if args.len() >= 2 => translate(&args[1], &args[2..]),
        Some("pm4") if args.len() >= 2 => pm4(&args[1], &args[2..]),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

type Res = Result<ExitCode, Box<dyn std::error::Error>>;

fn inventory_cmd(dir: &str, flags: &[String]) -> Res {
    let (mut md, mut top) = (None, 50);
    let mut it = flags.iter();
    while let Some(f) = it.next() {
        match f.as_str() {
            "--md" => md = Some(it.next().ok_or("--md needs a path")?.clone()),
            "--top" => top = parse_num(it.next().ok_or("--top needs a number")?)?,
            other => return Err(format!("unknown flag {other}").into()),
        }
    }
    let inv = inventory::inventory(std::path::Path::new(dir))?;
    let report = inventory::markdown(&inv, top);
    print!("{report}");
    if let Some(p) = md {
        std::fs::write(&p, &report)?;
        eprintln!("report written to {p}");
    }
    Ok(ExitCode::SUCCESS)
}

fn inspect(path: &str) -> Res {
    let bytes = std::fs::read(path)?;
    let pe = PeFile::parse(&bytes)?;
    let h = &pe.headers;
    println!("image base      {:#x}", h.image_base);
    println!("entry point     RVA {:#x}", h.entry_point);
    println!("size of image   {:#x}", h.size_of_image);
    println!("type            {}", if h.is_dll() { "DLL" } else { "EXE" });
    println!("subsystem       {}", h.subsystem);
    println!("stack reserve   {:#x}", h.stack_reserve);
    println!("relocatable     {}", !h.relocs_stripped());
    println!("\nsections:");
    for s in &pe.sections {
        let p = format!(
            "{}{}{}",
            if s.is_read() { 'r' } else { '-' },
            if s.is_write() { 'w' } else { '-' },
            if s.is_exec() { 'x' } else { '-' }
        );
        let raw = bytes
            .get(s.raw_offset as usize..)
            .map(|b| &b[..b.len().min(s.raw_size as usize)])
            .unwrap_or_default();
        let e = entropy(raw);
        println!(
            "  {:<8} rva {:#010x} vsize {:#010x} raw {:#010x} {p} entropy {e:.2}{}",
            s.name,
            s.virtual_address,
            s.virtual_size,
            s.raw_size,
            if e > 7.9 { "  <- encrypted or compressed?" } else { "" }
        );
    }
    let imports = pe.imports()?;
    let dlls: Vec<&str> = imports.iter().map(|m| m.dll.as_str()).collect();
    let mut reg = Registry::new();
    scarlett_hle::register_all(&mut reg, &dlls);
    let (mut have, mut total) = (0, 0);
    println!("\nimports ({} modules; + = HLE handler, - = missing):", imports.len());
    for m in &imports {
        let n = m
            .symbols
            .iter()
            .filter(|s| reg.lookup(&m.dll, &s.name).is_some())
            .count();
        have += n;
        total += m.symbols.len();
        println!(
            "  {}{} ({}/{} implemented)",
            m.dll,
            if m.delay_load { " [delay]" } else { "" },
            n,
            m.symbols.len()
        );
        for s in &m.symbols {
            let mark = if reg.lookup(&m.dll, &s.name).is_some() { '+' } else { '-' };
            println!("    {mark} {}", s.name);
        }
    }
    println!("HLE coverage    {have}/{total} imports (guest DLLs resolved at load time)");
    let exports = pe.exports()?;
    if !exports.is_empty() {
        println!("\nexports ({}):", exports.len());
        for e in &exports {
            let name = e.name.as_deref().unwrap_or("<ordinal only>");
            match &e.forwarder {
                Some(f) => println!("  #{:<5} {name} -> {f}", e.ordinal),
                None => println!("  #{:<5} {name} @ {:#x}", e.ordinal, e.rva),
            }
        }
    }
    println!("\nrelocations     {}", pe.relocations()?.len());
    println!("unwind entries  {}", pe.runtime_functions()?.len());
    match pe.tls()? {
        Some(t) => println!(
            "TLS             {} callbacks, data {:#x}..{:#x} +{:#x} zero, align {}",
            t.callbacks.len(),
            t.raw_data_start_va,
            t.raw_data_end_va,
            t.zero_fill,
            t.alignment()
        ),
        None => println!("TLS             none"),
    }
    if h.dir(Dir::LoadConfig).is_present() {
        println!("load config     present");
    }
    Ok(ExitCode::SUCCESS)
}

fn parse_num(s: &str) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(match s.strip_prefix("0x") {
        Some(h) => usize::from_str_radix(h, 16)?,
        None => s.parse()?,
    })
}

fn translate(path: &str, flags: &[String]) -> Res {
    use scarlett_gpu::translate::{translate, ComputeIo, CrossLane, Io, Options};
    let (mut out, mut offset, mut lanes, mut sgprs, mut vgprs) = (None, 0, 64, 106, 64);
    let mut emulate = false;
    let mut lds = 0u32;
    let mut rsrc2: Option<u32> = None;
    let mut workgroup = [64u32, 1, 1];
    let mut it = flags.iter();
    while let Some(f) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{f} needs a value"));
        match f.as_str() {
            "-o" => out = Some(val()?.clone()),
            "--offset" => offset = parse_num(val()?)?,
            "--sgprs" => sgprs = parse_num(val()?)? as u32,
            "--vgprs" => vgprs = parse_num(val()?)? as u32,
            "--wave32" => lanes = 32,
            "--emulate-cross-lane" => emulate = true,
            "--lds-bytes" => lds = (parse_num(val()?)? / 4) as u32,
            "--rsrc2" => rsrc2 = Some(parse_num(val()?)? as u32),
            "--workgroup" => {
                let v = val()?;
                let dims: Vec<u32> = v
                    .split(['x', ','])
                    .map(|d| parse_num(d).map(|n| n as u32))
                    .collect::<Result<_, _>>()?;
                workgroup = [dims[0], *dims.get(1).unwrap_or(&1), *dims.get(2).unwrap_or(&1)];
            }
            other => return Err(format!("unknown flag {other}\n\n{USAGE}").into()),
        }
    }
    let out = out.ok_or("translate needs -o <out.spv>")?;
    let bytes = std::fs::read(path)?;
    let body = bytes.get(offset..).ok_or("offset past end of file")?;
    let words: Vec<u32> = body
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    let io = match rsrc2 {
        Some(r) => Io::Compute(ComputeIo::from_rsrc2(r, workgroup)),
        None => Io::RegisterDump,
    };
    let opts = Options {
        io,
        num_sgprs: sgprs,
        num_vgprs: vgprs,
        lanes,
        memory: None,
        cross_lane: if emulate { CrossLane::Workgroup } else { CrossLane::Subgroup },
        lds_words: lds,
        code_base: 0,
    };
    let module = translate(&words, &opts).map_err(|e| format!("{e:?}"))?;
    scarlett_gpu::spirv::validate(&module).map_err(|e| format!("invalid SPIR-V: {e:?}"))?;
    let bytes: Vec<u8> = module.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(&out, &bytes)?;
    println!("{} words -> {out} ({} bytes, wave{lanes})", words.len(), bytes.len());
    Ok(ExitCode::SUCCESS)
}

fn words_of(bytes: &[u8]) -> Vec<u32> {
    bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect()
}

/// `ADDR:BYTES` (or `ADDR:BYTES=<file>` with the file split off).
fn range(s: &str) -> Result<(u64, u64), Box<dyn std::error::Error>> {
    let (a, n) = s.split_once(':').ok_or_else(|| format!("{s}: expected ADDR:BYTES"))?;
    Ok((parse_num(a)? as u64, parse_num(n)? as u64))
}

fn pm4(path: &str, flags: &[String]) -> Res {
    use scarlett_gpu::cp::{CommandProcessor, Event, Memory};
    use scarlett_gpu::{exec, translate::Window, SparseMemory};
    let mut mem = SparseMemory::new();
    let mut dumps = Vec::new();
    let mut spirv_dir = None;
    let mut window = Window { base: 0, words: 0 };
    let mut it = flags.iter();
    while let Some(f) = it.next() {
        let v = it.next().ok_or_else(|| format!("{f} needs a value"))?;
        match f.as_str() {
            "--load" => {
                let (a, file) = v.split_once('=').ok_or("--load ADDR=<file>")?;
                let addr = parse_num(a)? as u64;
                let data = std::fs::read(file)?;
                if data.len() % 4 != 0 {
                    return Err(format!("{file}: size not a multiple of 4").into());
                }
                mem.map(addr, data.len() as u64);
                if !mem.write(addr, &words_of(&data)) {
                    return Err(format!("{file}: cannot write at {addr:#x}").into());
                }
            }
            "--alloc" => {
                let (a, n) = range(v)?;
                mem.map(a, n);
            }
            "--dump" => {
                let (r, file) = v.split_once('=').ok_or("--dump ADDR:BYTES=<file>")?;
                dumps.push((range(r)?, file.to_string()));
            }
            "--spirv" => spirv_dir = Some(std::path::PathBuf::from(v)),
            "--window" => {
                let (a, n) = range(v)?;
                window = Window { base: a, words: (n / 4) as u32 };
            }
            other => return Err(format!("unknown flag {other}\n\n{USAGE}").into()),
        }
    }
    let stream = words_of(&std::fs::read(path)?);
    let events = CommandProcessor::new().run(&mut mem, &stream);
    let mut failed = false;
    for (n, ev) in events.iter().enumerate() {
        match ev {
            Event::Dispatch(d) => {
                let (size, vgprs, lds) = exec::shape(d);
                print!(
                    "dispatch {n}: groups {:?} threads {:?} program {:#x} {size:?} {vgprs} VGPRs {} LDS bytes",
                    d.groups, d.threads, d.program, lds * 4
                );
                if let Some(dir) = &spirv_dir {
                    match exec::translate_dispatch(d, &mem, window) {
                        Ok(c) => {
                            let out = dir.join(format!("dispatch{n}.spv"));
                            let bytes: Vec<u8> = c.module.iter().flat_map(|w| w.to_le_bytes()).collect();
                            std::fs::write(&out, bytes)?;
                            print!(" -> {}", out.display());
                        }
                        Err(e) => print!(" (not translated: {e:?})"),
                    }
                }
                match exec::run_dispatch(d, &mut mem, 10_000_000) {
                    Ok(()) => println!(": ran"),
                    Err(e) => {
                        println!(": {e:?}");
                        failed = true;
                    }
                }
            }
            Event::Draw(d) => println!("draw {n}: {:?} (not executed)", d.kind),
            Event::Error(e) => {
                println!("error {n}: {e}");
                failed = true;
            }
            other => println!("{n}: {other:?}"),
        }
    }
    for ((addr, bytes), file) in dumps {
        let mut out = vec![0u32; bytes.div_ceil(4) as usize];
        if !mem.read(addr, &mut out) {
            return Err(format!("dump {addr:#x}:{bytes}: not mapped").into());
        }
        let raw: Vec<u8> = out.iter().flat_map(|w| w.to_le_bytes()).take(bytes as usize).collect();
        std::fs::write(&file, raw)?;
    }
    Ok(if failed { ExitCode::FAILURE } else { ExitCode::SUCCESS })
}

fn disasm(path: &str, offset: Option<&String>) -> Res {
    let bytes = std::fs::read(path)?;
    let off = match offset {
        Some(o) => match o.strip_prefix("0x") {
            Some(h) => usize::from_str_radix(h, 16)?,
            None => o.parse()?,
        },
        None => 0,
    };
    let body = bytes.get(off..).ok_or("offset past end of file")?;
    let words: Vec<u32> = body
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    print!("{}", scarlett_gpu::rdna::disassemble(&words));
    Ok(ExitCode::SUCCESS)
}

fn run(path: &str, flags: &[String]) -> Res {
    let mut opts = LoadOptions::default();
    let image_dir = std::path::Path::new(path)
        .canonicalize()?
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_default();
    vfs::mount("G", &image_dir);
    vfs::set_current_dir("G:\\");
    let mut dry_run = false;
    let mut flags = flags.iter();
    while let Some(f) = flags.next() {
        match f.as_str() {
            "--strict" => scarlett_hle::set_strict(true),
            "--dry-run" => dry_run = true,
            "--unimplemented" => {
                let v = flags.next().ok_or("--unimplemented needs log|fail|stop")?;
                let p = scarlett_hle::policy::parse(v)
                    .ok_or("--unimplemented needs log|fail|stop")?;
                scarlett_hle::set_unimplemented_policy(p);
            }
            "--no-preferred-base" => opts.prefer_image_base = false,
            "--host-forward" => opts.forward_to_host = true,
            "--mount" => {
                let spec = flags.next().ok_or("--mount needs X=<dir>")?;
                let (dev, dir) = spec.split_once('=').ok_or("--mount needs X=<dir>")?;
                let dir = std::path::Path::new(dir).canonicalize()?;
                log::info!("mount {dev}: -> {}", dir.display());
                vfs::mount(dev, dir);
            }
            other => return Err(format!("unknown flag {other}\n\n{USAGE}").into()),
        }
    }
    let bytes = std::fs::read(path)?;
    let pe = PeFile::parse(&bytes)?;
    let dlls: Vec<String> = pe.imports()?.into_iter().map(|m| m.dll).collect();
    let dlls: Vec<&str> = dlls.iter().map(String::as_str).collect();
    let mut reg = Registry::new();
    scarlett_hle::register_all(&mut reg, &dlls);
    let file_name = std::path::Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "image.exe".into());
    opts.guest_path = Some(format!("G:\\{file_name}"));
    opts.name = Some(file_name);
    scarlett_loader::install_dynamic_loading(reg.clone(), opts.clone());

    let img = load(&bytes, &reg, &opts)?;
    let count = |b| img.imports.iter().filter(|i| i.binding == b).count();
    log::info!(
        "loaded at {:#x} ({} relocs applied); imports: {} guest, {} HLE, {} host, {} stubbed",
        img.base(),
        img.relocations_applied,
        count(Binding::Guest),
        count(Binding::Hle),
        count(Binding::Host),
        count(Binding::Stub)
    );

    if dry_run {
        let missing: Vec<_> = img
            .imports
            .iter()
            .filter(|i| i.binding == Binding::Stub)
            .collect();
        println!(
            "dry run: {} imports, {} guest, {} HLE, {} host, {} missing",
            img.imports.len(),
            count(Binding::Guest),
            count(Binding::Hle),
            count(Binding::Host),
            missing.len()
        );
        print_missing(missing.iter().map(|i| (i.dll.as_str(), i.name.as_str())));
        return Ok(ExitCode::SUCCESS);
    }

    // SAFETY: the user asked to run this image natively.
    let outcome = unsafe { img.run() };
    report();
    match outcome {
        Ok(ret) => {
            log::info!("entry returned {ret:#x}");
            Ok(ExitCode::from(ret as u8))
        }
        Err(_) => Err("guest thread panicked".into()),
    }
}

/// Shannon entropy in bits per byte.
fn entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut counts = [0u64; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let n = data.len() as f64;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.log2()
        })
        .sum()
}

/// Missing imports grouped by DLL, HRESULT functions marked.
fn print_missing<'a>(missing: impl Iterator<Item = (&'a str, &'a str)>) {
    let mut by_dll: std::collections::BTreeMap<&str, Vec<&str>> = Default::default();
    for (dll, name) in missing {
        by_dll.entry(dll).or_default().push(name);
    }
    for (dll, names) in by_dll {
        println!("  {dll} ({} missing)", names.len());
        for n in names {
            let hr = scarlett_hle::policy::classify(dll, n) == scarlett_hle::policy::ReturnClass::HResult;
            println!("    {n}{}", if hr { "  [HRESULT]" } else { "" });
        }
    }
}

fn report() {
    let classes = scarlett_hle::xgameruntime::requested_classes();
    if classes.iter().any(|c| !c.2) {
        eprintln!("\nGDK runtime classes requested through QueryApiImpl and not served:");
        for (clsid, iid, _) in classes.iter().filter(|c| !c.2) {
            eprintln!("  class {clsid}  interface {iid}");
        }
    }
    let proc_misses = scarlett_hle::module::proc_misses();
    if !proc_misses.is_empty() {
        eprintln!("\nGetProcAddress lookups that found nothing:");
        for (dll, name) in proc_misses.iter().take(25) {
            eprintln!("  {dll}!{name}");
        }
    }
    let mut missing: Vec<_> = scarlett_hle::thunk_stats()
        .into_iter()
        .filter(|s| !s.2 && s.3 > 0)
        .collect();
    if missing.is_empty() {
        return;
    }
    missing.sort_by_key(|s| std::cmp::Reverse(s.3));
    eprintln!("\nunimplemented imports that were called (implement these next):");
    for (dll, name, _, calls) in missing.iter().take(25) {
        eprintln!("  {calls:>8}x  {dll}!{name}");
    }
}
