//! Multi-module guests: a main EXE importing from a guest DLL that is found
//! through the VFS and initialised (DllMain) before the entry point, plus
//! GetModuleHandle/GetProcAddress at run time on HLE and guest modules.

use scarlett_hle::{module, register_all, vfs, Registry};
use scarlett_loader::{install_dynamic_loading, load, Binding, LoadOptions};
use scarlett_pe::builder::{Layout, PeBuilder, DATA_RVA, TEXT_RVA};

fn rip(c: &mut Vec<u8>, op: &[u8], target_rva: u32) {
    let next = TEXT_RVA + c.len() as u32 + op.len() as u32 + 4;
    c.extend_from_slice(op);
    c.extend_from_slice(&((target_rva as i64 - next as i64) as i32).to_le_bytes());
}

fn call_iat(c: &mut Vec<u8>, l: &Layout, dll: &str, sym: &str) {
    let next = TEXT_RVA + c.len() as u32 + 6;
    c.extend_from_slice(&[0xFF, 0x15]);
    c.extend_from_slice(&((l.iat(dll, sym) as i64 - next as i64) as i32).to_le_bytes());
}

/// engine.dll: `Mul3(x) = 3x` at .text+0; DllMain counts calls per
/// reason in `.data[reason]` (u32 each).
fn engine_dll() -> Vec<u8> {
    PeBuilder::new(0x1_D000_0000)
        .dll()
        .export("Mul3", 0)
        .data(&[0; 16])
        .entry_offset(4)
        .build(|_| {
            let mut c = vec![0x8D, 0x04, 0x49, 0xC3]; // lea eax, [rcx+rcx*2]; ret
            rip(&mut c, &[0x48, 0x8D, 0x05], DATA_RVA); // lea rax, [counts]
            c.extend_from_slice(&[0xFF, 0x04, 0x90]); // inc dword [rax+rdx*4]
            c.extend_from_slice(&[0xB8, 1, 0, 0, 0, 0xC3]); // mov eax, 1; ret
            c
        })
}

const R: u32 = DATA_RVA; // results: 4 qwords
const S_K32: u32 = DATA_RVA + 0x40;
const S_TID: u32 = DATA_RVA + 0x50;
const S_ENG: u32 = DATA_RVA + 0x70;
const S_MUL: u32 = DATA_RVA + 0x80;
const S_NONE: u32 = DATA_RVA + 0x90;

fn main_exe() -> Vec<u8> {
    let mut data = vec![0u8; 0xA0];
    for (off, s) in [
        (S_K32, "kernel32"),
        (S_TID, "GetCurrentThreadId"),
        (S_ENG, "engine.dll"),
        (S_MUL, "Mul3"),
        (S_NONE, "NoSuchFunction"),
    ] {
        let o = (off - DATA_RVA) as usize;
        data[o..o + s.len()].copy_from_slice(s.as_bytes());
    }
    PeBuilder::new(0x1_D100_0000)
        .import("engine.dll", &["Mul3"])
        .import(
            "kernel32.dll",
            &[
                "GetModuleHandleA",
                "GetProcAddress",
                "CreateThread",
                "WaitForSingleObject",
                "CloseHandle",
            ],
        )
        .data(&data)
        .entry_offset(3)
        .build(|l| {
            // worker thread at .text+0: return 0
            let mut c = vec![0x31, 0xC0, 0xC3];
            c.extend_from_slice(&[0x53, 0x48, 0x83, 0xEC, 0x30]); // push rbx; sub rsp, 0x30
            c.extend_from_slice(&[0xB9, 14, 0, 0, 0]); // mov ecx, 14
            call_iat(&mut c, l, "engine.dll", "Mul3");
            rip(&mut c, &[0x48, 0x89, 0x05], R); // mov [r0], rax

            rip(&mut c, &[0x48, 0x8D, 0x0D], S_K32); // lea rcx, "kernel32"
            call_iat(&mut c, l, "kernel32.dll", "GetModuleHandleA");
            c.extend_from_slice(&[0x48, 0x89, 0xC3]); // mov rbx, rax
            c.extend_from_slice(&[0x48, 0x89, 0xD9]); // mov rcx, rbx
            rip(&mut c, &[0x48, 0x8D, 0x15], S_TID); // lea rdx, "GetCurrentThreadId"
            call_iat(&mut c, l, "kernel32.dll", "GetProcAddress");
            c.extend_from_slice(&[0xFF, 0xD0]); // call rax
            rip(&mut c, &[0x48, 0x89, 0x05], R + 8); // mov [r1], rax

            rip(&mut c, &[0x48, 0x8D, 0x0D], S_ENG); // lea rcx, "engine.dll"
            call_iat(&mut c, l, "kernel32.dll", "GetModuleHandleA");
            c.extend_from_slice(&[0x48, 0x89, 0xC1]); // mov rcx, rax
            rip(&mut c, &[0x48, 0x8D, 0x15], S_MUL); // lea rdx, "Mul3"
            call_iat(&mut c, l, "kernel32.dll", "GetProcAddress");
            c.extend_from_slice(&[0xB9, 5, 0, 0, 0, 0xFF, 0xD0]); // mov ecx, 5; call rax
            rip(&mut c, &[0x48, 0x89, 0x05], R + 16); // mov [r2], rax

            c.extend_from_slice(&[0x48, 0x89, 0xD9]); // mov rcx, rbx
            rip(&mut c, &[0x48, 0x8D, 0x15], S_NONE); // lea rdx, "NoSuchFunction"
            call_iat(&mut c, l, "kernel32.dll", "GetProcAddress");
            rip(&mut c, &[0x48, 0x89, 0x05], R + 24); // mov [r3], rax

            // A thread, so engine.dll's DllMain sees THREAD_ATTACH/DETACH.
            c.extend_from_slice(&[0x31, 0xC9, 0x31, 0xD2]); // xor ecx, ecx; xor edx, edx
            rip(&mut c, &[0x4C, 0x8D, 0x05], TEXT_RVA); // lea r8, [worker]
            c.extend_from_slice(&[0x45, 0x31, 0xC9]); // xor r9d, r9d
            c.extend_from_slice(&[0x48, 0xC7, 0x44, 0x24, 0x20, 0, 0, 0, 0]); // flags = 0
            c.extend_from_slice(&[0x48, 0xC7, 0x44, 0x24, 0x28, 0, 0, 0, 0]); // tid out = NULL
            call_iat(&mut c, l, "kernel32.dll", "CreateThread");
            c.extend_from_slice(&[0x48, 0x89, 0xC3, 0x48, 0x89, 0xC1]); // mov rbx, rax; mov rcx, rax
            c.extend_from_slice(&[0xBA, 0xFF, 0xFF, 0xFF, 0xFF]); // mov edx, INFINITE
            call_iat(&mut c, l, "kernel32.dll", "WaitForSingleObject");
            c.extend_from_slice(&[0x48, 0x89, 0xD9]); // mov rcx, rbx
            call_iat(&mut c, l, "kernel32.dll", "CloseHandle");

            c.extend_from_slice(&[0x31, 0xC0, 0x48, 0x83, 0xC4, 0x30, 0x5B, 0xC3]); // xor eax,eax; add rsp,0x30; pop rbx; ret
            c
        })
}

#[test]
fn exe_with_guest_dll_dependency_and_dynamic_lookups() {
    let dir = std::env::temp_dir().join(format!("scarlett-mod-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("Engine.dll"), engine_dll()).unwrap();
    vfs::mount("M", &dir);
    vfs::set_current_dir("M:\\");

    let mut reg = Registry::new();
    register_all(&mut reg, &["kernel32"]);
    let opts = LoadOptions {
        name: Some("game.exe".into()),
        guest_path: Some("M:\\game.exe".into()),
        ..Default::default()
    };
    install_dynamic_loading(reg.clone(), opts.clone());

    let img = load(&main_exe(), &reg, &opts).unwrap();
    let mul3 = img.imports.iter().find(|i| i.name == "Mul3").unwrap();
    assert_eq!(
        mul3.binding,
        Binding::Guest,
        "bound straight to engine.dll's export"
    );
    let engine = module::find_guest("ENGINE").expect("dependency registered");
    assert_eq!(engine.guest_path, "M:\\engine.dll");

    assert_eq!(unsafe { img.run() }.unwrap(), 0);
    let r = |i: usize| img.region().read_u64((R as usize) + i * 8).unwrap();
    assert_eq!(r(0), 42, "Mul3(14) through the IAT");
    assert_ne!(
        r(1),
        0,
        "GetProcAddress(kernel32, GetCurrentThreadId) minted a working thunk"
    );
    assert_eq!(r(2), 15, "GetProcAddress on a guest DLL");
    assert_eq!(r(3), 0, "unknown export is NULL");
    assert!(module::proc_misses()
        .iter()
        .any(|(_, n)| n == "NoSuchFunction"));
    // DllMain: PROCESS_ATTACH before the entry point, then one
    // THREAD_ATTACH and one THREAD_DETACH for the guest-created thread.
    let counts: [u32; 4] = unsafe { ((engine.base + DATA_RVA as u64) as *const [u32; 4]).read() };
    assert_eq!(counts, [0, 1, 1, 1], "per-reason DllMain calls");
    std::fs::remove_dir_all(&dir).unwrap();
}
