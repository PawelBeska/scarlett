//! HLE handler calling back into guest code, which itself calls an HLE
//! import: host → guest → host → guest nesting (gs switching on macOS).

use scarlett_hle::{sync::register_sync, threading::register_threading, Registry};
use scarlett_loader::{load, LoadOptions};
use scarlett_pe::builder::{Layout, PeBuilder, DATA_RVA, TEXT_RVA};

fn call_iat(c: &mut Vec<u8>, l: &Layout, sym: &str) {
    let next = TEXT_RVA + c.len() as u32 + 6;
    c.extend_from_slice(&[0xFF, 0x15]);
    c.extend_from_slice(&((l.iat("kernel32.dll", sym) as i64 - next as i64) as i32).to_le_bytes());
}

/// `op [rip + target_rva]`, where `op` is the opcode+ModRM prefix.
fn rip(c: &mut Vec<u8>, op: &[u8], target_rva: u32) {
    let next = TEXT_RVA + c.len() as u32 + op.len() as u32 + 4;
    c.extend_from_slice(op);
    c.extend_from_slice(&((target_rva as i64 - next as i64) as i32).to_le_bytes());
}

const ONCE: u32 = DATA_RVA;
const CTX: u32 = DATA_RVA + 8;
const COUNT: u32 = DATA_RVA + 16;
const TID: u32 = DATA_RVA + 24;

#[test]
fn init_once_runs_guest_callback_once_with_nested_hle_call() {
    let mut reg = Registry::new();
    register_sync(&mut reg, &["kernel32"]);
    register_threading(&mut reg, &["kernel32"]);
    let cb_len = std::cell::Cell::new(0u32);
    let pe_for = |entry: u32| {
        PeBuilder::new(0x1_B000_0000)
            .import(
                "kernel32.dll",
                &["InitOnceExecuteOnce", "GetCurrentThreadId"],
            )
            .data(&[0; 32])
            .entry_offset(entry)
            .build(|l| {
                // callback(once, param, ctx*) at .text+0
                let mut c = vec![0x53, 0x48, 0x83, 0xEC, 0x20]; // push rbx; sub rsp, 32
                c.extend_from_slice(&[0x4C, 0x89, 0xC3]); // mov rbx, r8
                c.extend_from_slice(&[0x48, 0x8D, 0x04, 0x95, 0, 0, 0, 0]); // lea rax, [rdx*4]
                c.extend_from_slice(&[0x48, 0x89, 0x03]); // mov [rbx], rax
                rip(&mut c, &[0x48, 0xFF, 0x05], COUNT); // inc qword [count]
                call_iat(&mut c, l, "GetCurrentThreadId");
                rip(&mut c, &[0x89, 0x05], TID); // mov [tid], eax
                c.extend_from_slice(&[0xB8, 1, 0, 0, 0]); // mov eax, 1
                c.extend_from_slice(&[0x48, 0x83, 0xC4, 0x20, 0x5B, 0xC3]); // add rsp,32; pop rbx; ret
                cb_len.set(c.len() as u32);

                // entry: InitOnceExecuteOnce(&once, callback, 0x77, &ctx) twice
                c.extend_from_slice(&[0x48, 0x83, 0xEC, 0x28]); // sub rsp, 40
                for _ in 0..2 {
                    rip(&mut c, &[0x48, 0x8D, 0x0D], ONCE); // lea rcx, [once]
                    rip(&mut c, &[0x48, 0x8D, 0x15], TEXT_RVA); // lea rdx, [callback]
                    c.extend_from_slice(&[0x41, 0xB8, 0x77, 0, 0, 0]); // mov r8d, 0x77
                    rip(&mut c, &[0x4C, 0x8D, 0x0D], CTX); // lea r9, [ctx]
                    call_iat(&mut c, l, "InitOnceExecuteOnce");
                }
                rip(&mut c, &[0x48, 0x8B, 0x05], COUNT); // mov rax, [count]
                c.extend_from_slice(&[0x48, 0xC1, 0xE0, 0x10]); // shl rax, 16
                rip(&mut c, &[0x48, 0x03, 0x05], CTX); // add rax, [ctx]
                c.extend_from_slice(&[0x48, 0x83, 0xC4, 0x28, 0xC3]); // add rsp, 40; ret
                c
            })
    };
    pe_for(0); // first pass only measures the callback
    let pe = pe_for(cb_len.get());
    let img = load(&pe, &reg, &LoadOptions::default()).unwrap();
    // callback ran once; context = 0x77 << 2
    assert_eq!(unsafe { img.run() }.unwrap(), (1 << 16) + 0x77 * 4);
    let tid = img.region().read_u32(TID as usize).unwrap();
    assert_ne!(tid, 0, "nested GetCurrentThreadId reached the handler");
}
