//! The Win64-ABI trampoline every import thunk jumps to.
//!
//! On entry (from the guest's `call [iat]`, through the thunk's `jmp`):
//!
//! ```text
//! r10        = thunk id
//! rcx..r9    = args 0..3
//! [rsp]      = return address into guest
//! [rsp+8]    = 32-byte shadow space owned by us
//! [rsp+40]   = arg 4, arg 5, ...
//! ```
//!
//! Stack alignment: rsp ≡ 8 (mod 16) on entry; after `push rbp` it is ≡ 0,
//! and `sub rsp, 64` keeps it there for the inner call.

use crate::{dispatch, Call};

extern "win64" fn dispatch_from_asm(id: u64, regs: *const [u64; 4], stack: *const u64) -> u64 {
    // SAFETY: `regs` points at the spill area the trampoline just wrote and
    // `stack` at the guest's 5th argument slot (see module docs).
    let call = unsafe { Call::new(*regs, stack) };
    dispatch(id, &call)
}

/// Target of every HLE thunk. Not callable from Rust; take its address with
/// [`trampoline_address`].
///
/// # Safety
/// Only ever entered via an HLE thunk, which sets r10 to a valid thunk id.
#[unsafe(naked)]
pub unsafe extern "win64" fn hle_trampoline() {
    core::arch::naked_asm!(
        "push rbp",
        "mov rbp, rsp",
        "sub rsp, 64",
        // spill args 0..3 into [rsp+32 .. rsp+64)
        "mov [rsp+32], rcx",
        "mov [rsp+40], rdx",
        "mov [rsp+48], r8",
        "mov [rsp+56], r9",
        // dispatch_from_asm(id, &regs, &stack_args); [rsp .. rsp+32) is its shadow space
        "mov rcx, r10",
        "lea rdx, [rsp+32]",
        "lea r8, [rbp+48]",
        "call {dispatch}",
        "mov rsp, rbp",
        "pop rbp",
        "ret",
        dispatch = sym dispatch_from_asm,
    )
}

pub fn trampoline_address() -> u64 {
    hle_trampoline as *const () as usize as u64
}
