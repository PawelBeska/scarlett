use scarlett_hle::{builtins, Registry};
use scarlett_loader::{load, Binding, LoadOptions};
use scarlett_pe::{Dir, PeFile};
use std::process::ExitCode;

const USAGE: &str = "\
usage:
  scarlett inspect <image>
  scarlett run <image> [--strict] [--no-preferred-base] [--host-forward]

  --strict             abort on the first call to an unimplemented import
  --no-preferred-base  always relocate (exercises the relocation path)
  --host-forward       Windows: bind non-HLE imports to same-named host DLLs

  RUST_LOG=debug for verbose loader output.";

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("inspect") if args.len() == 2 => inspect(&args[1]),
        Some("run") if args.len() >= 2 => run(&args[1], &args[2..]),
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
        println!(
            "  {:<8} rva {:#010x} vsize {:#010x} raw {:#010x} {p}",
            s.name, s.virtual_address, s.virtual_size, s.raw_size
        );
    }
    let imports = pe.imports()?;
    println!("\nimports ({} modules):", imports.len());
    for m in &imports {
        println!(
            "  {}{} ({} symbols)",
            m.dll,
            if m.delay_load { " [delay]" } else { "" },
            m.symbols.len()
        );
        for s in &m.symbols {
            println!("    {}", s.name);
        }
    }
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
            "TLS             {} callbacks, data {:#x}..{:#x}",
            t.callbacks.len(),
            t.raw_data_start_va,
            t.raw_data_end_va
        ),
        None => println!("TLS             none"),
    }
    if h.dir(Dir::LoadConfig).is_present() {
        println!("load config     present");
    }
    Ok(ExitCode::SUCCESS)
}

fn run(path: &str, flags: &[String]) -> Res {
    let mut opts = LoadOptions::default();
    for f in flags {
        match f.as_str() {
            "--strict" => scarlett_hle::set_strict(true),
            "--no-preferred-base" => opts.prefer_image_base = false,
            "--host-forward" => opts.forward_to_host = true,
            other => return Err(format!("unknown flag {other}\n\n{USAGE}").into()),
        }
    }
    let bytes = std::fs::read(path)?;
    let pe = PeFile::parse(&bytes)?;
    let dlls: Vec<String> = pe.imports()?.into_iter().map(|m| m.dll).collect();
    let mut reg = Registry::new();
    builtins::register_basic(
        &mut reg,
        &dlls.iter().map(String::as_str).collect::<Vec<_>>(),
    );

    let img = load(&bytes, &reg, &opts)?;
    let count = |b| img.imports.iter().filter(|i| i.binding == b).count();
    log::info!(
        "loaded at {:#x} ({} relocs applied); imports: {} HLE, {} host, {} stubbed",
        img.base(),
        img.relocations_applied,
        count(Binding::Hle),
        count(Binding::Host),
        count(Binding::Stub)
    );

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

fn report() {
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
