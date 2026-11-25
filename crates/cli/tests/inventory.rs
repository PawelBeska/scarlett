//! `scarlett inventory`: a synthetic package (an exe and two DLLs, one in a
//! subfolder, plus a file named .dll that is not a PE) and the report's
//! classification of each import.

use scarlett_pe::builder::PeBuilder;
use std::process::Command;

fn ret(_: &scarlett_pe::builder::Layout) -> Vec<u8> {
    vec![0xC3]
}

#[test]
fn inventory_classifies_imports() {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("inventory_pkg");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    let game = PeBuilder::new(0x1_4000_0000)
        .import("kernel32.dll", &["GetTickCount", "Sleep", "XNotAFunction"])
        .import("engine.dll", &["EngineInit", "EngineMissing"])
        .import("missing.dll", &["Anything"])
        .build(ret);
    let engine = PeBuilder::new(0x1_8000_0000)
        .dll()
        .export("EngineInit", 0)
        .import("Helper.DLL", &["HelperRun"])
        .build(ret);
    let helper = PeBuilder::new(0x1_9000_0000).dll().export("HelperRun", 0).build(ret);
    std::fs::write(dir.join("game.exe"), game).unwrap();
    std::fs::write(dir.join("engine.dll"), engine).unwrap();
    std::fs::write(dir.join("sub").join("helper.dll"), helper).unwrap();
    std::fs::write(dir.join("notes.dll"), b"not a PE").unwrap();
    let md = dir.join("report.md");
    let r = Command::new(env!("CARGO_BIN_EXE_scarlett"))
        .arg("inventory")
        .arg(&dir)
        .arg("--md")
        .arg(&md)
        .output()
        .unwrap();
    let out = String::from_utf8_lossy(&r.stdout);
    assert!(r.status.success(), "{out}\n{}", String::from_utf8_lossy(&r.stderr));
    // 3 images, 7 imports: EngineInit and HelperRun (case-insensitive module
    // name, other folder) from the package, GetTickCount and Sleep by HLE,
    // XNotAFunction, EngineMissing and missing.dll!Anything missing
    assert!(out.contains("3 images, 7 imports: 2 from the package, 2 HLE, 3 missing; provided: 57.1%"), "{out}");
    assert!(out.contains("| game.exe | 6 | 1 | 2 | 3 | 50.0% |"), "{out}");
    assert!(out.contains("| engine.dll | 1 | 1 | 0 | 0 | 100.0% |"), "{out}");
    assert!(out.contains("| kernel32 | 1 | 1 |") && out.contains("| missing | 1 | 1 |") && out.contains("| engine | 1 | 1 |"), "{out}");
    assert!(out.contains("| engine | EngineMissing | 1 |"), "{out}");
    assert!(out.contains("- notes.dll:"), "{out}");
    assert_eq!(std::fs::read_to_string(&md).unwrap(), out);
}
