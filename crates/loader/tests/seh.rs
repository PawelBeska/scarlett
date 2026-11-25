//! Structured exception handling driven by the guest's own unwind tables
//! (non-Windows hosts; on Windows the OS does this natively).
//!
//! F (has a language handler H) calls G, G clobbers rbx and calls
//! RaiseException. Dispatch unwinds G (restoring F's rbx from G's push
//! slot), finds H for F's frame; H calls RtlUnwindEx(F's frame, target in
//! F, code), so F continues at `target` with rax = exception code.
#![cfg(not(windows))]

use scarlett_hle::{register_all, Registry};
use scarlett_loader::{load, LoadOptions};
use scarlett_pe::builder::{Layout, PeBuilder, DATA_RVA, TEXT_RVA};
use std::cell::Cell;

const CODE: u32 = 0xE000_1234;

fn call_iat(c: &mut Vec<u8>, l: &Layout, sym: &str) {
    let next = TEXT_RVA + c.len() as u32 + 6;
    c.extend_from_slice(&[0xFF, 0x15]);
    c.extend_from_slice(&((l.iat("kernel32.dll", sym) as i64 - next as i64) as i32).to_le_bytes());
}

#[derive(Default, Clone, Copy)]
struct Offs {
    g_end: u32,
    f: u32,
    f_end: u32,
    h: u32,
}

fn image(o: Offs) -> (Vec<u8>, Offs) {
    // .data: pdata @0 (2 entries), xdata G @0x20, xdata F @0x30
    let mut data = vec![0u8; 0x40];
    let put = |d: &mut Vec<u8>, at: usize, v: u32| d[at..at + 4].copy_from_slice(&v.to_le_bytes());
    put(&mut data, 0, TEXT_RVA);
    put(&mut data, 4, TEXT_RVA + o.g_end);
    put(&mut data, 8, DATA_RVA + 0x20);
    put(&mut data, 12, TEXT_RVA + o.f);
    put(&mut data, 16, TEXT_RVA + o.f_end);
    put(&mut data, 20, DATA_RVA + 0x30);
    // push rbx (ends +1); sub rsp, 0x20 (ends +5): ALLOC_SMALL 3, PUSH_NONVOL rbx
    let codes = [0x05, 0x32, 0x01, 0x30];
    data[0x20..0x24].copy_from_slice(&[0x01, 5, 2, 0]);
    data[0x24..0x28].copy_from_slice(&codes);
    data[0x30..0x34].copy_from_slice(&[0x01 | (1 << 3), 5, 2, 0]); // UNW_FLAG_EHANDLER
    data[0x34..0x38].copy_from_slice(&codes);
    put(&mut data, 0x38, TEXT_RVA + o.h);

    let found = Cell::new(Offs::default());
    let pe = PeBuilder::new(0x1_E000_0000)
        .import("kernel32.dll", &["RaiseException", "RtlUnwindEx"])
        .data(&data)
        .directory(3, DATA_RVA, 24)
        .entry_offset(o.f)
        .build(|l| {
            let mut f = Offs::default();
            // G
            let mut c = vec![0x53, 0x48, 0x83, 0xEC, 0x20]; // push rbx; sub rsp, 0x20
            c.extend_from_slice(&[0xBB, 0x99, 0x99, 0, 0]); // mov ebx, 0x9999 (clobber)
            c.push(0xB9);
            c.extend_from_slice(&CODE.to_le_bytes()); // mov ecx, CODE
            c.extend_from_slice(&[0x31, 0xD2, 0x45, 0x31, 0xC0, 0x45, 0x31, 0xC9]); // flags, nargs, args = 0
            call_iat(&mut c, l, "RaiseException");
            c.extend_from_slice(&[0x48, 0x83, 0xC4, 0x20, 0x5B, 0xC3]);
            f.g_end = c.len() as u32;
            // F
            f.f = c.len() as u32;
            c.extend_from_slice(&[0x53, 0x48, 0x83, 0xEC, 0x20]); // push rbx; sub rsp, 0x20
            c.extend_from_slice(&[0xBB, 0x34, 0x12, 0, 0]); // mov ebx, 0x1234
            let rel = 0i64 - (c.len() as i64 + 5);
            c.push(0xE8);
            c.extend_from_slice(&(rel as i32).to_le_bytes()); // call G
            c.extend_from_slice(&[0xB8, 0xAD, 0x0B, 0, 0, 0xEB, 0x02]); // mov eax, 0xBAD; jmp end
            let target = c.len() as u32;
            c.extend_from_slice(&[0x01, 0xD8]); // target: add eax, ebx
            c.extend_from_slice(&[0x48, 0x83, 0xC4, 0x20, 0x5B, 0xC3]); // end: epilog
            f.f_end = c.len() as u32;
            // H(record, establisher, context, dispatcher)
            f.h = c.len() as u32;
            c.extend_from_slice(&[0x48, 0x83, 0xEC, 0x38]); // sub rsp, 0x38
            c.extend_from_slice(&[0x4C, 0x89, 0xC0, 0x48, 0x89, 0x44, 0x24, 0x20]); // [rsp+0x20] = context
            c.extend_from_slice(&[0x48, 0xC7, 0x44, 0x24, 0x28, 0, 0, 0, 0]); // [rsp+0x28] = history = 0
            c.extend_from_slice(&[0x44, 0x8B, 0x09]); // mov r9d, [rcx]   return value = code
            c.extend_from_slice(&[0x49, 0x89, 0xC8]); // mov r8, rcx      record
            c.extend_from_slice(&[0x48, 0x89, 0xD1]); // mov rcx, rdx     target frame = establisher
            let next = TEXT_RVA + c.len() as u32 + 7;
            c.extend_from_slice(&[0x48, 0x8D, 0x15]); // lea rdx, [target]
            c.extend_from_slice(&((TEXT_RVA + target) as i64 - next as i64).to_le_bytes()[..4]);
            call_iat(&mut c, l, "RtlUnwindEx");
            c.extend_from_slice(&[0xB8, 1, 0, 0, 0, 0x48, 0x83, 0xC4, 0x38, 0xC3]); // not reached
            found.set(f);
            c
        });
    (pe, found.get())
}

#[test]
fn raise_dispatch_unwind_to_handler_target() {
    let (_, offs) = image(Offs::default());
    let (pe, again) = image(offs);
    assert_eq!(again.h, offs.h);
    let mut reg = Registry::new();
    register_all(&mut reg, &["kernel32"]);
    let img = load(&pe, &reg, &LoadOptions::default()).unwrap();
    for _ in 0..3 {
        // rax = code from RtlUnwindEx's ReturnValue, rbx = F's value restored by unwinding G
        assert_eq!(unsafe { img.run() }.unwrap(), (CODE + 0x1234) as u64);
    }
}
