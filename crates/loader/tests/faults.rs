//! Hardware faults in guest code become SEH exceptions (non-Windows hosts):
//! a null dereference caught by `__try/__except(EXCEPTION_EXECUTE_HANDLER)`
//! and a divide-by-zero whose filter fixes the CONTEXT and returns
//! EXCEPTION_CONTINUE_EXECUTION, both through `__C_specific_handler`.
#![cfg(not(windows))]

use scarlett_hle::{register_all, Registry};
use scarlett_loader::{load, LoadOptions};
use scarlett_pe::builder::{PeBuilder, DATA_RVA, TEXT_RVA};
use std::cell::Cell;

#[derive(Default, Clone, Copy)]
struct Offs {
    f1: u32,
    try1: u32,
    try1_end: u32,
    target1: u32,
    f1_end: u32,
    f2: u32,
    try2: u32,
    try2_end: u32,
    f2_end: u32,
    filter: u32,
    entry: u32,
}

fn image(o: Offs) -> (Vec<u8>, Offs) {
    // .data: pdata @0 (2 entries), xdata F1 @0x20, xdata F2 @0x50
    let mut d = vec![0u8; 0x80];
    let put = |d: &mut Vec<u8>, at: usize, v: u32| d[at..at + 4].copy_from_slice(&v.to_le_bytes());
    for (i, (b, e, x)) in [(o.f1, o.f1_end, 0x20), (o.f2, o.f2_end, 0x50)]
        .into_iter()
        .enumerate()
    {
        put(&mut d, i * 12, TEXT_RVA + b);
        put(&mut d, i * 12 + 4, TEXT_RVA + e);
        put(&mut d, i * 12 + 8, DATA_RVA + x);
    }
    // sub rsp, 0x28 (ends +4): ALLOC_SMALL 4; 1 code padded to 2 slots; EHANDLER = stub at .text+0
    for (x, scope) in [
        (0x20, [o.try1, o.try1_end, 1, o.target1]), // __except(EXCEPTION_EXECUTE_HANDLER)
        (0x50, [o.try2, o.try2_end, o.filter, o.try2_end]), // __except(filter())
    ] {
        d[x..x + 8].copy_from_slice(&[0x01 | (1 << 3), 4, 1, 0, 0x04, 0x42, 0, 0]);
        put(&mut d, x + 8, TEXT_RVA); // handler: jmp [__C_specific_handler]
        put(&mut d, x + 12, 1); // scope count
        for (k, v) in scope.into_iter().enumerate() {
            let v = if k == 2 && v == 1 { 1 } else { TEXT_RVA + v };
            put(&mut d, x + 16 + 4 * k, v);
        }
    }

    let found = Cell::new(Offs::default());
    let pe = PeBuilder::new(0x1_F000_0000)
        .import("vcruntime140.dll", &["__C_specific_handler"])
        .data(&d)
        .directory(3, DATA_RVA, 24)
        .entry_offset(o.entry)
        .build(|l| {
            let mut f = Offs::default();
            let mut c = Vec::new();
            // stub: jmp [rip + IAT(__C_specific_handler)]
            let next = TEXT_RVA + 6;
            c.extend_from_slice(&[0xFF, 0x25]);
            c.extend_from_slice(
                &((l.iat("vcruntime140.dll", "__C_specific_handler") as i64 - next as i64) as i32)
                    .to_le_bytes(),
            );
            // F1: null dereference
            f.f1 = c.len() as u32;
            c.extend_from_slice(&[0x48, 0x83, 0xEC, 0x28]);
            f.try1 = c.len() as u32;
            c.extend_from_slice(&[0x31, 0xC0, 0x8B, 0x00]); // xor eax, eax; mov eax, [rax]
            f.try1_end = c.len() as u32;
            c.extend_from_slice(&[0xB8, 0xAD, 0x0B, 0, 0]); // mov eax, 0xBAD (not reached)
            f.target1 = c.len() as u32; // __except block: eax = exception code
            c.extend_from_slice(&[0x48, 0x83, 0xC4, 0x28, 0xC3]);
            f.f1_end = c.len() as u32;
            // F2: divide by zero
            f.f2 = c.len() as u32;
            c.extend_from_slice(&[0x48, 0x83, 0xEC, 0x28]);
            f.try2 = c.len() as u32;
            c.extend_from_slice(&[0x31, 0xD2, 0x31, 0xC9, 0xB8, 10, 0, 0, 0, 0xF7, 0xF1]); // edx=ecx=0; eax=10; div ecx
            f.try2_end = c.len() as u32;
            c.extend_from_slice(&[0x48, 0x83, 0xC4, 0x28, 0xC3]);
            f.f2_end = c.len() as u32;
            // filter(EXCEPTION_POINTERS*, establisher): ctx.Rip += 2, ctx.Rax = 77, CONTINUE_EXECUTION
            f.filter = c.len() as u32;
            c.extend_from_slice(&[0x48, 0x8B, 0x41, 0x08]); // mov rax, [rcx+8]
            c.extend_from_slice(&[0x48, 0x83, 0x80, 0xF8, 0, 0, 0, 2]); // add qword [rax+0xF8], 2
            c.extend_from_slice(&[0x48, 0xC7, 0x40, 0x78, 77, 0, 0, 0]); // mov qword [rax+0x78], 77
            c.extend_from_slice(&[0xB8, 0xFF, 0xFF, 0xFF, 0xFF, 0xC3]); // mov eax, -1; ret
                                                                        // entry: (F2() << 32) | F1()
            f.entry = c.len() as u32;
            c.extend_from_slice(&[0x53, 0x48, 0x83, 0xEC, 0x20]);
            for target in [f.f1, f.f2] {
                let rel = target as i64 - (c.len() as i64 + 5);
                c.push(0xE8);
                c.extend_from_slice(&(rel as i32).to_le_bytes());
                if target == f.f1 {
                    c.extend_from_slice(&[0x89, 0xC3]); // mov ebx, eax
                }
            }
            c.extend_from_slice(&[0x48, 0xC1, 0xE0, 0x20, 0x48, 0x09, 0xD8]); // shl rax, 32; or rax, rbx
            c.extend_from_slice(&[0x48, 0x83, 0xC4, 0x20, 0x5B, 0xC3]);
            found.set(f);
            c
        });
    (pe, found.get())
}

#[test]
fn access_violation_and_divide_by_zero_handled_by_guest() {
    let (_, o) = image(Offs::default());
    let (pe, o2) = image(o);
    assert_eq!(o2.entry, o.entry);
    let mut reg = Registry::new();
    register_all(&mut reg, &["kernel32", "vcruntime140"]);
    let img = load(&pe, &reg, &LoadOptions::default()).unwrap();
    for _ in 0..3 {
        let r = unsafe { img.run() }.unwrap();
        assert_eq!(
            r & 0xFFFF_FFFF,
            0xC000_0005,
            "F1: STATUS_ACCESS_VIOLATION in __except"
        );
        assert_eq!(r >> 32, 77, "F2: filter skipped the div and set rax");
    }
}

/// Vectored handler at .text+0: if the code is STATUS_ILLEGAL_INSTRUCTION,
/// store it in ctx.Rax, skip the 2-byte `ud2` and continue; else search on.
/// (It must not claim other exceptions: the test above faults in parallel.)
const VEH: [u8; 38] = [
    0x48, 0x8B, 0x41, 0x08, // mov rax, [rcx+8]        ContextRecord
    0x48, 0x8B, 0x11, //       mov rdx, [rcx]          ExceptionRecord
    0x8B, 0x12, //             mov edx, [rdx]          code
    0x81, 0xFA, 0x1D, 0x00, 0x00, 0xC0, // cmp edx, STATUS_ILLEGAL_INSTRUCTION
    0x75, 0x12, //             jne search
    0x48, 0x89, 0x50, 0x78, // mov [rax+0x78], rdx     ctx.Rax = code
    0x48, 0x83, 0x80, 0xF8, 0, 0, 0, 2, // add qword [rax+0xF8], 2
    0xB8, 0xFF, 0xFF, 0xFF, 0xFF, 0xC3, // mov eax, -1; ret  (CONTINUE_EXECUTION)
    0x31, 0xC0, 0xC3, //       search: xor eax, eax; ret
];

/// No unwind info anywhere: a vectored handler is the only way out of `ud2`.
#[test]
fn vectored_handler_continues_after_illegal_instruction() {
    let pe = PeBuilder::new(0x1_F100_0000)
        .import(
            "kernel32.dll",
            &[
                "AddVectoredExceptionHandler",
                "RemoveVectoredExceptionHandler",
            ],
        )
        .entry_offset(VEH.len() as u32)
        .build(|l| {
            let call = |c: &mut Vec<u8>, sym: &str| {
                let next = TEXT_RVA + c.len() as u32 + 6;
                c.extend_from_slice(&[0xFF, 0x15]);
                c.extend_from_slice(
                    &((l.iat("kernel32.dll", sym) as i64 - next as i64) as i32).to_le_bytes(),
                );
            };
            let mut c = VEH.to_vec();
            c.extend_from_slice(&[0x53, 0x48, 0x83, 0xEC, 0x20, 0xB9, 1, 0, 0, 0]); // push rbx; sub rsp,32; ecx = 1 (first)
            let next = TEXT_RVA + c.len() as u32 + 7;
            c.extend_from_slice(&[0x48, 0x8D, 0x15]); // lea rdx, [handler]
            c.extend_from_slice(&((TEXT_RVA as i64 - next as i64) as i32).to_le_bytes());
            call(&mut c, "AddVectoredExceptionHandler");
            c.extend_from_slice(&[0x48, 0x89, 0xC3, 0x0F, 0x0B]); // mov rbx, rax; ud2
            c.extend_from_slice(&[0x48, 0x93, 0x48, 0x89, 0xC1]); // xchg rax, rbx (rbx = code); mov rcx, rax
            call(&mut c, "RemoveVectoredExceptionHandler");
            c.extend_from_slice(&[0x48, 0xC1, 0xE0, 0x20, 0x48, 0x09, 0xD8]); // shl rax, 32; or rax, rbx
            c.extend_from_slice(&[0x48, 0x83, 0xC4, 0x20, 0x5B, 0xC3]);
            c
        });
    let mut reg = Registry::new();
    register_all(&mut reg, &["kernel32"]);
    let img = load(&pe, &reg, &LoadOptions::default()).unwrap();
    let r = unsafe { img.run() }.unwrap();
    assert_eq!(
        r & 0xFFFF_FFFF,
        0xC000_001D,
        "handler saw STATUS_ILLEGAL_INSTRUCTION"
    );
    assert_eq!(
        r >> 32,
        1,
        "RemoveVectoredExceptionHandler found the registration"
    );
}
