//! Entering guest code from Rust, and leaving it early (`ExitThread`).
//!
//! [`enter`] calls a guest function with the Win64 ABI. The outermost entry
//! on a thread records its stack pointer in [`EnterCtx::exit_rsp`];
//! [`exit`] jumps straight back to that frame and makes it return a given
//! value, skipping every guest and host frame in between, like `ExitThread`
//! does on Windows (no unwinding, no destructors; skipped Rust frames must
//! not hold locks, which HLE handlers don't across a call to `exit`).
//!
//! Because frames in between may have clobbered Win64 callee-saved state,
//! the entry frame saves rbx, rbp, rdi, rsi, r12-r15 and xmm6-15 itself
//! instead of trusting the callee.
//!
//! On macOS the entry also switches `gs` to the guest TEB and back (see
//! [`crate::teb`]); after [`exit`] it is switched back too.
//!
//! Entry frame (below the 8 pushed GPRs; rsp ≡ 0 mod 16):
//!
//! ```text
//! [rsp+0  .. 32)   shadow space for the guest callee
//! [rsp+32 .. 192)  xmm6 .. xmm15
//! [rsp+192]        previous exit_rsp (nested entries)
//! ```

/// Per-thread entry state, referenced from asm by fixed offsets.
#[repr(C)]
pub struct EnterCtx {
    /// Guest TEB (used for `gs` switching on macOS).
    pub teb: u64,
    /// rsp of the outermost active entry frame; 0 when not in guest code.
    pub exit_rsp: u64,
}

const _: () = assert!(std::mem::offset_of!(EnterCtx, teb) == 0);
const _: () = assert!(std::mem::offset_of!(EnterCtx, exit_rsp) == 8);

#[cfg(target_os = "macos")]
const _: () = {
    assert!(crate::teb::SET_GS_TRAP == 0x300_0003);
    assert!(crate::teb::off::HOST_GS == 0x1F00);
};

/// `gs` ← guest TEB (`r15` = ctx). Clobbers rax, rcx, rdi, r11.
#[cfg(target_os = "macos")]
macro_rules! gs_to_guest {
    () => {
        "mov rdi, [r15]\nmov eax, 0x3000003\nsyscall"
    };
}
/// `gs` ← host TSD (`r15` = ctx). Clobbers rax, rcx, rdi, r11.
#[cfg(target_os = "macos")]
macro_rules! gs_to_host {
    () => {
        "mov rdi, [r15]\nmov rdi, [rdi+0x1F00]\nmov eax, 0x3000003\nsyscall"
    };
}
#[cfg(not(target_os = "macos"))]
macro_rules! gs_to_guest {
    () => {
        ""
    };
}
#[cfg(not(target_os = "macos"))]
macro_rules! gs_to_host {
    () => {
        ""
    };
}

/// Common tail of `enter_asm` and `exit_asm`: rsp at the entry frame,
/// r15 = ctx, r12 = return value.
macro_rules! epilogue {
    () => {
        concat!(
            gs_to_host!(),
            "\n",
            "mov rax, [rsp+192]\n",
            "mov [r15+8], rax\n",
            "movaps xmm6,  [rsp+32]\n",
            "movaps xmm7,  [rsp+48]\n",
            "movaps xmm8,  [rsp+64]\n",
            "movaps xmm9,  [rsp+80]\n",
            "movaps xmm10, [rsp+96]\n",
            "movaps xmm11, [rsp+112]\n",
            "movaps xmm12, [rsp+128]\n",
            "movaps xmm13, [rsp+144]\n",
            "movaps xmm14, [rsp+160]\n",
            "movaps xmm15, [rsp+176]\n",
            "mov rax, r12\n",
            "add rsp, 200\n",
            "pop r15\n",
            "pop r14\n",
            "pop r13\n",
            "pop r12\n",
            "pop rsi\n",
            "pop rdi\n",
            "pop rbx\n",
            "pop rbp\n",
            "ret",
        )
    };
}

/// Win64: rcx=f, rdx=&[a0, a1, a2, a3], r8=ctx.
#[unsafe(naked)]
unsafe extern "win64" fn enter_asm(_f: u64, _args: *const [u64; 4], _ctx: *mut EnterCtx) -> u64 {
    core::arch::naked_asm!(
        "push rbp",
        "mov rbp, rsp",
        "push rbx",
        "push rdi",
        "push rsi",
        "push r12",
        "push r13",
        "push r14",
        "push r15",
        // entry rsp ≡ 8; +8 pushes ≡ 8 → sub 200 gives ≡ 0
        "sub rsp, 200",
        "mov r15, r8",
        "movaps [rsp+32],  xmm6",
        "movaps [rsp+48],  xmm7",
        "movaps [rsp+64],  xmm8",
        "movaps [rsp+80],  xmm9",
        "movaps [rsp+96],  xmm10",
        "movaps [rsp+112], xmm11",
        "movaps [rsp+128], xmm12",
        "movaps [rsp+144], xmm13",
        "movaps [rsp+160], xmm14",
        "movaps [rsp+176], xmm15",
        "mov rax, [r15+8]",
        "mov [rsp+192], rax",
        "test rax, rax",
        "jnz 2f",
        "mov [r15+8], rsp", // outermost entry: ExitThread lands here
        "2:",
        "mov rbx, rcx",
        "mov r12, rdx",
        gs_to_guest!(),
        "mov rcx, [r12]",
        "mov rdx, [r12+8]",
        "mov r8,  [r12+16]",
        "mov r9,  [r12+24]",
        "call rbx",
        "mov r12, rax",
        epilogue!(),
    )
}

/// Win64: rcx=ctx, rdx=value. Never returns to its caller.
#[unsafe(naked)]
unsafe extern "win64" fn exit_asm(_ctx: *mut EnterCtx, _value: u64) -> ! {
    core::arch::naked_asm!(
        "mov r15, rcx",
        "mov r12, rdx",
        "mov rsp, [r15+8]",
        // Unwinding to the outermost frame: no outer entry left to restore.
        "mov qword ptr [rsp+192], 0",
        epilogue!(),
    )
}

/// Call guest `f(args[0], .., args[3])`.
///
/// # Safety
/// `f` must be guest code safe to run natively; `ctx` must be this
/// thread's live context.
pub unsafe fn enter(f: u64, args: [u64; 4], ctx: *mut EnterCtx) -> u64 {
    unsafe { enter_asm(f, &args, ctx) }
}

/// Load a full Win64 `CONTEXT` (`ctx`, 16-byte aligned; offsets as in
/// [`crate::seh::Context`]) and continue at its Rip with its Rsp, never
/// returning. `teb` is the guest TEB (`gs` is switched to it on macOS).
/// Restores GPRs, xmm0-15 and MXCSR; flags are not restored.
#[unsafe(naked)]
unsafe extern "win64" fn restore_asm(_ctx: *const u8, _teb: u64) -> ! {
    core::arch::naked_asm!(
        "mov rbx, rcx",
        "push rdx",
        "mov r15, rsp", // gs_to_guest reads the TEB from [r15], like EnterCtx.teb
        gs_to_guest!(),
        "mov rcx, rbx",
        "ldmxcsr [rcx+0x34]",
        "movaps xmm0,  [rcx+0x1A0]",
        "movaps xmm1,  [rcx+0x1B0]",
        "movaps xmm2,  [rcx+0x1C0]",
        "movaps xmm3,  [rcx+0x1D0]",
        "movaps xmm4,  [rcx+0x1E0]",
        "movaps xmm5,  [rcx+0x1F0]",
        "movaps xmm6,  [rcx+0x200]",
        "movaps xmm7,  [rcx+0x210]",
        "movaps xmm8,  [rcx+0x220]",
        "movaps xmm9,  [rcx+0x230]",
        "movaps xmm10, [rcx+0x240]",
        "movaps xmm11, [rcx+0x250]",
        "movaps xmm12, [rcx+0x260]",
        "movaps xmm13, [rcx+0x270]",
        "movaps xmm14, [rcx+0x280]",
        "movaps xmm15, [rcx+0x290]",
        // Target stack: Rsp - 8 holds Rip for the final `ret`.
        "mov rax, [rcx+0x98]",
        "sub rax, 8",
        "mov rdx, [rcx+0xF8]",
        "mov [rax], rdx",
        "mov rdx, [rcx+0x88]",
        "mov rbx, [rcx+0x90]",
        "mov rbp, [rcx+0xA0]",
        "mov rsi, [rcx+0xA8]",
        "mov rdi, [rcx+0xB0]",
        "mov r8,  [rcx+0xB8]",
        "mov r9,  [rcx+0xC0]",
        "mov r10, [rcx+0xC8]",
        "mov r11, [rcx+0xD0]",
        "mov r12, [rcx+0xD8]",
        "mov r13, [rcx+0xE0]",
        "mov r14, [rcx+0xE8]",
        "mov r15, [rcx+0xF0]",
        "mov rsp, rax",
        "mov rax, [rcx+0x78]",
        "mov rcx, [rcx+0x80]",
        "ret",
    )
}

/// Continue guest execution from a saved `CONTEXT`.
///
/// # Safety
/// `ctx` must describe a valid guest state on this thread's stack, the
/// memory must not lie in the stack area being abandoned, and every frame
/// between here and the target must be disposable (see [`exit`]).
pub unsafe fn restore(ctx: *const u8, teb: u64) -> ! {
    debug_assert!((ctx as usize).is_multiple_of(16));
    unsafe { restore_asm(ctx, teb) }
}

/// Abandon all guest frames on this thread: the outermost [`enter`] returns
/// `value`.
///
/// # Safety
/// `ctx` must be this thread's context with an active entry
/// (`exit_rsp != 0`), and no frame being skipped may own resources that
/// need dropping.
pub unsafe fn exit(ctx: *mut EnterCtx, value: u64) -> ! {
    debug_assert_ne!(unsafe { (*ctx).exit_rsp }, 0);
    unsafe { exit_asm(ctx, value) }
}
