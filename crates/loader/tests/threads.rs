//! Guest-created threads: CreateThread, events, waits on thread handles,
//! ExitThread from inside the thread procedure.

use scarlett_hle::{register_all, Registry};
use scarlett_loader::{load, LoadOptions};
use scarlett_pe::builder::{Layout, PeBuilder, DATA_RVA, TEXT_RVA};
use std::cell::Cell;

const SYMS: [&str; 7] = [
    "CreateEventW",
    "SetEvent",
    "CreateThread",
    "WaitForSingleObject",
    "GetExitCodeThread",
    "CloseHandle",
    "ExitThread",
];

fn call_iat(c: &mut Vec<u8>, l: &Layout, sym: &str) {
    let next = TEXT_RVA + c.len() as u32 + 6;
    c.extend_from_slice(&[0xFF, 0x15]);
    c.extend_from_slice(&((l.iat("kernel32.dll", sym) as i64 - next as i64) as i32).to_le_bytes());
}

fn rip(c: &mut Vec<u8>, op: &[u8], target_rva: u32) {
    let next = TEXT_RVA + c.len() as u32 + op.len() as u32 + 4;
    c.extend_from_slice(op);
    c.extend_from_slice(&((target_rva as i64 - next as i64) as i32).to_le_bytes());
}

const COUNT: u32 = DATA_RVA;
const FLAG: u32 = DATA_RVA + 8;
const CODE: u32 = DATA_RVA + 16;
const TID: u32 = DATA_RVA + 24;

fn program(entry: u32, worker_len: &Cell<u32>) -> Vec<u8> {
    PeBuilder::new(0x1_C000_0000)
        .import("kernel32.dll", &SYMS)
        .data(&[0; 32])
        .entry_offset(entry)
        .build(|l| {
            // worker(event) at .text+0: count++, SetEvent, ExitThread(0x77), flag = 1
            let mut c = vec![0x53, 0x48, 0x83, 0xEC, 0x20]; // push rbx; sub rsp, 32
            c.extend_from_slice(&[0x48, 0x89, 0xCB]); // mov rbx, rcx
            rip(&mut c, &[0x48, 0xFF, 0x05], COUNT); // inc qword [count]
            c.extend_from_slice(&[0x48, 0x89, 0xD9]); // mov rcx, rbx
            call_iat(&mut c, l, "SetEvent");
            c.extend_from_slice(&[0xB9, 0x77, 0, 0, 0]); // mov ecx, 0x77
            call_iat(&mut c, l, "ExitThread");
            c.extend_from_slice(&[0xB8, 1, 0, 0, 0]); // mov eax, 1   (must never run)
            rip(&mut c, &[0x89, 0x05], FLAG); // mov [flag], eax
            c.extend_from_slice(&[0x48, 0x83, 0xC4, 0x20, 0x5B, 0xC3]); // add rsp,32; pop rbx; ret
            worker_len.set(c.len() as u32);

            // entry
            c.extend_from_slice(&[0x53, 0x56, 0x48, 0x83, 0xEC, 0x38]); // push rbx; push rsi; sub rsp, 0x38
            c.extend_from_slice(&[0x31, 0xC9, 0x31, 0xD2, 0x45, 0x31, 0xC0, 0x45, 0x31, 0xC9]); // zero rcx,rdx,r8,r9
            call_iat(&mut c, l, "CreateEventW"); // auto-reset, unsignaled
            c.extend_from_slice(&[0x48, 0x89, 0xC3]); // mov rbx, rax
            c.extend_from_slice(&[0x31, 0xC9, 0x31, 0xD2]); // xor ecx, ecx; xor edx, edx
            rip(&mut c, &[0x4C, 0x8D, 0x05], TEXT_RVA); // lea r8, [worker]
            c.extend_from_slice(&[0x49, 0x89, 0xD9]); // mov r9, rbx
            c.extend_from_slice(&[0x48, 0xC7, 0x44, 0x24, 0x20, 0, 0, 0, 0]); // flags = 0
            rip(&mut c, &[0x48, 0x8D, 0x05], TID); // lea rax, [tid]
            c.extend_from_slice(&[0x48, 0x89, 0x44, 0x24, 0x28]); // mov [rsp+0x28], rax
            call_iat(&mut c, l, "CreateThread");
            c.extend_from_slice(&[0x48, 0x89, 0xC6]); // mov rsi, rax
            for h in [[0x48, 0x89, 0xD9], [0x48, 0x89, 0xF1]] {
                // WaitForSingleObject(event, INFINITE), then (thread, INFINITE)
                c.extend_from_slice(&h);
                c.extend_from_slice(&[0xBA, 0xFF, 0xFF, 0xFF, 0xFF]);
                call_iat(&mut c, l, "WaitForSingleObject");
            }
            c.extend_from_slice(&[0x48, 0x89, 0xF1]); // mov rcx, rsi
            rip(&mut c, &[0x48, 0x8D, 0x15], CODE); // lea rdx, [code]
            call_iat(&mut c, l, "GetExitCodeThread");
            for h in [[0x48, 0x89, 0xF1], [0x48, 0x89, 0xD9]] {
                c.extend_from_slice(&h);
                call_iat(&mut c, l, "CloseHandle");
            }
            rip(&mut c, &[0x8B, 0x05], CODE); // mov eax, [code]
            rip(&mut c, &[0x48, 0x8B, 0x0D], COUNT); // mov rcx, [count]
            c.extend_from_slice(&[0x48, 0xC1, 0xE1, 0x10, 0x48, 0x01, 0xC8]); // shl rcx,16; add rax,rcx
            rip(&mut c, &[0x48, 0x8B, 0x0D], FLAG); // mov rcx, [flag]
            c.extend_from_slice(&[0x48, 0xC1, 0xE1, 0x20, 0x48, 0x01, 0xC8]); // shl rcx,32; add rax,rcx
            c.extend_from_slice(&[0x48, 0x83, 0xC4, 0x38, 0x5E, 0x5B, 0xC3]); // add rsp,0x38; pop rsi; pop rbx; ret
            c
        })
}

#[test]
fn create_thread_event_wait_and_exit_thread() {
    let mut reg = Registry::new();
    register_all(&mut reg, &["kernel32"]);
    let len = Cell::new(0);
    program(0, &len);
    let pe = program(len.get(), &len);
    let img = load(&pe, &reg, &LoadOptions::default()).unwrap();
    for _ in 0..3 {
        img.region().write(COUNT as usize, &[0; 32]).unwrap();
        // exit code 0x77, worker ran once, code after ExitThread did not run
        assert_eq!(unsafe { img.run() }.unwrap(), 0x77 + (1 << 16));
        let tid = img.region().read_u32(TID as usize).unwrap();
        assert!(
            tid != 0 && (tid.is_multiple_of(4) || cfg!(windows)),
            "tid {tid:#x}"
        );
    }
}
