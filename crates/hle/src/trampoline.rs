//! The Win64-ABI trampoline every import thunk jumps to.
//!
//! On entry (from the guest's `call [iat]`, through the thunk's `jmp`):
//!
//! ```text
//! r10        = thunk id
//! rcx..r9    = integer args 0..3      xmm0..xmm3 = float args 0..3
//! [rsp]      = return address into guest
//! [rsp+8]    = 32-byte shadow space owned by us
//! [rsp+40]   = arg 4, arg 5, ...
//! ```
//!
//! Win64 assigns argument *positions*, not per-class counters: argument 2 is
//! in r8 if it is an integer and in xmm2 if it is a float. The trampoline
//! spills both sets, so a handler can read either view of any position.
//! A float or `__m128` result is returned in xmm0, which the trampoline
//! loads from [`Frame::xmm_ret`] after the handler runs (xmm0 is volatile,
//! so loading it for integer-returning functions is harmless).
//!
//! Frame below the saved rbp (rsp ≡ 0 mod 16 after `push rbp`, and every
//! frame size is a multiple of 16):
//!
//! ```text
//! [rsp+0  .. 32)   shadow space for dispatch_from_asm
//! [rsp+32 .. 384)  Frame (see field docs)
//! macOS only:
//! [rsp+384]        thunk id, later the integer return value
//! [rsp+392]        guest TEB
//! [rsp+400]        saved rdi (callee-saved in Win64, used by the trap)
//! ```
//!
//! The frame also records the caller's callee-saved registers, return
//! address and stack pointer, so handlers such as `RaiseException` or
//! `RtlCaptureContext` can build the guest's `CONTEXT` at the call site.
//!
//! On macOS `gs` belongs to libpthread, so the macOS variant switches it to
//! the host TSD before calling Rust and back to the guest TEB afterwards
//! (see [`crate::teb`]). Elsewhere the guest `gs` is valid for host code too.
//! The trap preserves xmm registers, and `dispatch_from_asm` is `win64`, so
//! the compiler preserves xmm6-15 for the guest.

use crate::{dispatch, Call};

/// Register spill area the trampoline builds at `[rsp+32]`.
#[repr(C)]
pub struct Frame {
    /// rcx, rdx, r8, r9
    pub gpr: [u64; 4],
    /// xmm0-3
    pub xmm: [[u64; 2]; 4],
    pub xmm_ret: [u64; 2],
    /// rbx, rbp, rsi, rdi, r12, r13, r14, r15 as the caller had them.
    pub nonvol: [u64; 8],
    /// Caller's rsp after the call returns (just above the return address).
    pub caller_rsp: u64,
    /// Return address into the caller.
    pub caller_rip: u64,
    /// xmm6-15 as the caller had them.
    pub xmm_nonvol: [[u64; 2]; 10],
}

const _: () = assert!(std::mem::size_of::<Frame>() == 352);
const _: () = assert!(std::mem::offset_of!(Frame, nonvol) == 112);
const _: () = assert!(std::mem::offset_of!(Frame, xmm_nonvol) == 192);

/// Spill the frame (`[rsp+32..]`); rbp = trampoline frame pointer.
macro_rules! spill {
    () => {
        concat!(
            "mov [rsp+32], rcx\n",
            "mov [rsp+40], rdx\n",
            "mov [rsp+48], r8\n",
            "mov [rsp+56], r9\n",
            "movaps [rsp+64], xmm0\n",
            "movaps [rsp+80], xmm1\n",
            "movaps [rsp+96], xmm2\n",
            "movaps [rsp+112], xmm3\n",
            "mov [rsp+144], rbx\n",
            "mov rax, [rbp]\n",
            "mov [rsp+152], rax\n",
            "mov [rsp+160], rsi\n",
            "mov [rsp+168], rdi\n",
            "mov [rsp+176], r12\n",
            "mov [rsp+184], r13\n",
            "mov [rsp+192], r14\n",
            "mov [rsp+200], r15\n",
            "lea rax, [rbp+16]\n",
            "mov [rsp+208], rax\n",
            "mov rax, [rbp+8]\n",
            "mov [rsp+216], rax\n",
            "movaps [rsp+224], xmm6\n",
            "movaps [rsp+240], xmm7\n",
            "movaps [rsp+256], xmm8\n",
            "movaps [rsp+272], xmm9\n",
            "movaps [rsp+288], xmm10\n",
            "movaps [rsp+304], xmm11\n",
            "movaps [rsp+320], xmm12\n",
            "movaps [rsp+336], xmm13\n",
            "movaps [rsp+352], xmm14\n",
            "movaps [rsp+368], xmm15",
        )
    };
}

extern "win64" fn dispatch_from_asm(id: u64, frame: *mut Frame, stack: *const u64) -> u64 {
    // SAFETY: `frame` points at the spill area the trampoline just wrote and
    // `stack` at the guest's 5th argument slot (see module docs).
    let frame = unsafe { &mut *frame };
    let mut call = unsafe { Call::with_xmm(frame.gpr, frame.xmm, stack) };
    call.frame = frame as *const Frame;
    let ret = dispatch(id, &call);
    frame.xmm_ret = call.xmm_ret();
    ret
}

/// Target of every HLE thunk. Not callable from Rust; take its address with
/// [`trampoline_address`].
///
/// # Safety
/// Only ever entered via an HLE thunk, which sets r10 to a valid thunk id.
#[cfg(not(target_os = "macos"))]
#[unsafe(naked)]
pub unsafe extern "win64" fn hle_trampoline() {
    core::arch::naked_asm!(
        "push rbp",
        "mov rbp, rsp",
        "sub rsp, 384",
        spill!(),
        // dispatch_from_asm(id, &frame, &stack_args)
        "mov rcx, r10",
        "lea rdx, [rsp+32]",
        "lea r8, [rbp+48]",
        "call {dispatch}",
        "movaps xmm0, [rsp+128]",
        "mov rsp, rbp",
        "pop rbp",
        "ret",
        dispatch = sym dispatch_from_asm,
    )
}

/// macOS variant: as above, plus `gs` switching around the Rust call.
///
/// # Safety
/// As the generic variant; additionally `gs` must hold a fake TEB built by
/// [`crate::teb::GuestThread`].
#[cfg(target_os = "macos")]
#[unsafe(naked)]
pub unsafe extern "win64" fn hle_trampoline() {
    core::arch::naked_asm!(
        "push rbp",
        "mov rbp, rsp",
        "sub rsp, 416",
        spill!(),
        "mov [rsp+384], r10",
        "mov [rsp+400], rdi",
        "mov rax, gs:[{self_off}]",
        "mov [rsp+392], rax",
        "mov rdi, [rax+{host_gs}]",
        "mov eax, {trap}",
        "syscall",
        "mov rcx, [rsp+384]",
        "lea rdx, [rsp+32]",
        "lea r8, [rbp+48]",
        "call {dispatch}",
        "mov [rsp+384], rax",
        "mov rdi, [rsp+392]",
        "mov eax, {trap}",
        "syscall",
        "mov rax, [rsp+384]",
        "mov rdi, [rsp+400]",
        "movaps xmm0, [rsp+128]",
        "mov rsp, rbp",
        "pop rbp",
        "ret",
        dispatch = sym dispatch_from_asm,
        self_off = const crate::teb::off::SELF,
        host_gs = const crate::teb::off::HOST_GS,
        trap = const crate::teb::SET_GS_TRAP,
    )
}

pub fn trampoline_address() -> u64 {
    hle_trampoline as *const () as usize as u64
}
