//! x64 structured exception handling for non-Windows hosts.
//!
//! On Windows the loader registers `.pdata` with `RtlAddFunctionTable` and
//! the real OS dispatches exceptions. Elsewhere this module provides the
//! same machinery, driven by the guest's own unwind tables:
//!
//! * [`lookup_function_entry`] / [`virtual_unwind`]: table-based unwinding
//!   (`RUNTIME_FUNCTION` + `UNWIND_INFO`, all unwind codes, chained info,
//!   epilog detection) over a Win64 [`Context`];
//! * `RaiseException` / `RtlRaiseException`: build the `CONTEXT` at the
//!   call site (from the trampoline [`Frame`]) and walk the frames, calling
//!   each language handler (`UNW_FLAG_EHANDLER`) like `RtlDispatchException`;
//! * `RtlUnwindEx` / `RtlUnwind`: call termination handlers
//!   (`UNW_FLAG_UHANDLER`) up to the target frame, then continue at the
//!   target IP via [`crate::enter::restore`], abandoning the dispatcher's
//!   own (host) frames;
//! * `RtlCaptureContext`, `RtlLookupFunctionEntry`, `RtlVirtualUnwind`,
//!   `RtlRestoreContext`, `SetUnhandledExceptionFilter`.
//!
//! Frames of host code (HLE handlers, the dispatcher) have no entries in
//! guest tables. Unwinding during dispatch therefore always starts from the
//! context of the raise site, never from inside the dispatcher.
//!
//! * hardware faults: a signal handler turns SIGSEGV/SIGBUS/SIGILL/SIGFPE/
//!   SIGTRAP in guest code into `STATUS_*` exceptions and resumes the
//!   thread in the dispatcher (on the guest stack, outside signal context);
//! * `__C_specific_handler` for `__try`/`__except`/`__finally`.
//!
//! * vectored exception and continue handlers.
//!
//! Not yet: `__CxxFrameHandler*` (C++ EH; works when the
//! game links its CRT statically, since the handler is then guest code).

use crate::module;
use crate::teb::{self, current_teb};
use crate::trampoline::Frame;
use crate::{Call, Registry};
use std::cell::RefCell;

// ---------------------------------------------------------------- CONTEXT

/// Win64 `CONTEXT` (0x4D0 bytes, 16-byte aligned).
#[repr(C, align(16))]
#[derive(Clone)]
pub struct Context(pub [u8; CONTEXT_SIZE]);

pub const CONTEXT_SIZE: usize = 0x4D0;
pub const CONTEXT_AMD64: u32 = 0x0010_0000;
pub const CONTEXT_FULL: u32 = CONTEXT_AMD64 | 0x1 | 0x2 | 0x8; // control, integer, floating point

mod ctx_off {
    pub const FLAGS: usize = 0x30;
    pub const MXCSR: usize = 0x34;
    pub const SEG_CS: usize = 0x38;
    pub const SEG_SS: usize = 0x42;
    pub const EFLAGS: usize = 0x44;
    pub const GPR: usize = 0x78; // rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi, r8..r15
    pub const RIP: usize = 0xF8;
    pub const FLT_MXCSR: usize = 0x100 + 0x18;
    pub const XMM: usize = 0x1A0;
}

pub const RAX: usize = 0;
pub const RCX: usize = 1;
pub const RDX: usize = 2;
pub const RBX: usize = 3;
pub const RSP: usize = 4;
pub const RBP: usize = 5;
pub const RSI: usize = 6;
pub const RDI: usize = 7;

impl Default for Context {
    fn default() -> Self {
        let mut c = Context([0; CONTEXT_SIZE]);
        c.put_u32(ctx_off::FLAGS, CONTEXT_FULL);
        c.put_u32(ctx_off::MXCSR, 0x1F80);
        c.put_u32(ctx_off::FLT_MXCSR, 0x1F80);
        c.0[ctx_off::SEG_CS] = 0x33;
        c.0[ctx_off::SEG_SS] = 0x2B;
        c.put_u32(ctx_off::EFLAGS, 0x202);
        c
    }
}

impl Context {
    fn get_u64(&self, o: usize) -> u64 {
        u64::from_le_bytes(self.0[o..o + 8].try_into().unwrap())
    }
    fn put_u64(&mut self, o: usize, v: u64) {
        self.0[o..o + 8].copy_from_slice(&v.to_le_bytes());
    }
    fn put_u32(&mut self, o: usize, v: u32) {
        self.0[o..o + 4].copy_from_slice(&v.to_le_bytes());
    }
    /// General register by x86 number (0 = rax … 15 = r15).
    pub fn reg(&self, n: usize) -> u64 {
        self.get_u64(ctx_off::GPR + 8 * n)
    }
    pub fn set_reg(&mut self, n: usize, v: u64) {
        self.put_u64(ctx_off::GPR + 8 * n, v)
    }
    pub fn rip(&self) -> u64 {
        self.get_u64(ctx_off::RIP)
    }
    pub fn set_rip(&mut self, v: u64) {
        self.put_u64(ctx_off::RIP, v)
    }
    pub fn rsp(&self) -> u64 {
        self.reg(RSP)
    }
    pub fn set_rsp(&mut self, v: u64) {
        self.set_reg(RSP, v)
    }
    pub fn xmm(&self, n: usize) -> [u64; 2] {
        let o = ctx_off::XMM + 16 * n;
        [self.get_u64(o), self.get_u64(o + 8)]
    }
    pub fn set_xmm(&mut self, n: usize, v: [u64; 2]) {
        let o = ctx_off::XMM + 16 * n;
        self.put_u64(o, v[0]);
        self.put_u64(o + 8, v[1]);
    }

    /// Context of the guest at an HLE call site: Rip = return address,
    /// Rsp = stack pointer after the return, nonvolatile registers as the
    /// caller had them, argument registers as passed.
    pub fn from_frame(f: &Frame) -> Context {
        let mut c = Context::default();
        c.set_reg(RCX, f.gpr[0]);
        c.set_reg(RDX, f.gpr[1]);
        c.set_reg(8, f.gpr[2]);
        c.set_reg(9, f.gpr[3]);
        for (i, &r) in [RBX, RBP, RSI, RDI, 12, 13, 14, 15].iter().enumerate() {
            c.set_reg(r, f.nonvol[i]);
        }
        c.set_rsp(f.caller_rsp);
        c.set_rip(f.caller_rip);
        for i in 0..4 {
            c.set_xmm(i, f.xmm[i]);
        }
        for i in 0..10 {
            c.set_xmm(6 + i, f.xmm_nonvol[i]);
        }
        c
    }

    /// # Safety
    /// `p` must point at a guest `CONTEXT`.
    pub unsafe fn read_from(p: *const u8) -> Context {
        let mut c = Context([0; CONTEXT_SIZE]);
        unsafe { std::ptr::copy_nonoverlapping(p, c.0.as_mut_ptr(), CONTEXT_SIZE) };
        c
    }

    /// # Safety
    /// `p` must point at a writable guest `CONTEXT`.
    pub unsafe fn write_to(&self, p: *mut u8) {
        unsafe { std::ptr::copy_nonoverlapping(self.0.as_ptr(), p, CONTEXT_SIZE) };
    }
}

// ---------------------------------------------------------------- EXCEPTION_RECORD

pub const EXCEPTION_RECORD_SIZE: usize = 152;
pub const EXCEPTION_MAXIMUM_PARAMETERS: usize = 15;
pub const EXCEPTION_NONCONTINUABLE: u32 = 0x1;
pub const EXCEPTION_UNWINDING: u32 = 0x2;
pub const EXCEPTION_EXIT_UNWIND: u32 = 0x4;
pub const EXCEPTION_TARGET_UNWIND: u32 = 0x20;
pub const STATUS_UNWIND: u32 = 0xC000_0027;
pub const STATUS_NONCONTINUABLE_EXCEPTION: u32 = 0xC000_0025;

pub const EXCEPTION_CONTINUE_EXECUTION: u32 = 0;
pub const EXCEPTION_CONTINUE_SEARCH: u32 = 1;

/// `EXCEPTION_RECORD` as raw bytes (guest layout).
#[repr(C, align(8))]
#[derive(Clone)]
pub struct Record(pub [u8; EXCEPTION_RECORD_SIZE]);

impl Record {
    pub fn new(code: u32, flags: u32, address: u64, params: &[u64]) -> Record {
        let mut r = Record([0; EXCEPTION_RECORD_SIZE]);
        r.0[0..4].copy_from_slice(&code.to_le_bytes());
        r.0[4..8].copy_from_slice(&flags.to_le_bytes());
        r.0[16..24].copy_from_slice(&address.to_le_bytes());
        let n = params.len().min(EXCEPTION_MAXIMUM_PARAMETERS);
        r.0[24..28].copy_from_slice(&(n as u32).to_le_bytes());
        for (i, p) in params[..n].iter().enumerate() {
            r.0[32 + 8 * i..40 + 8 * i].copy_from_slice(&p.to_le_bytes());
        }
        r
    }
    pub fn code(&self) -> u32 {
        u32::from_le_bytes(self.0[0..4].try_into().unwrap())
    }
    pub fn flags(&self) -> u32 {
        u32::from_le_bytes(self.0[4..8].try_into().unwrap())
    }
    pub fn set_flags(&mut self, f: u32) {
        self.0[4..8].copy_from_slice(&f.to_le_bytes());
    }
}

// ---------------------------------------------------------------- unwinding

/// `UNW_FLAG_*`
pub const UNW_FLAG_NHANDLER: u32 = 0;
pub const UNW_FLAG_EHANDLER: u32 = 1;
pub const UNW_FLAG_UHANDLER: u32 = 2;
pub const UNW_FLAG_CHAININFO: u32 = 4;

unsafe fn rd<T: Copy>(a: u64) -> T {
    unsafe { (a as *const T).read_unaligned() }
}

/// `RtlLookupFunctionEntry`: (image base, address of the guest
/// `RUNTIME_FUNCTION` covering `pc`).
pub fn lookup_function_entry(pc: u64) -> Option<(u64, u64)> {
    let m = module::guest_containing(pc)?;
    if m.pdata_count == 0 {
        return None;
    }
    let rva = (pc - m.base) as u32;
    let table = m.base + m.pdata_rva as u64;
    let (mut lo, mut hi) = (0u32, m.pdata_count);
    while lo < hi {
        let mid = (lo + hi) / 2;
        let e = table + mid as u64 * 12;
        // SAFETY: entry inside the mapped exception directory.
        let (begin, end) = unsafe { (rd::<u32>(e), rd::<u32>(e + 4)) };
        if rva < begin {
            hi = mid;
        } else if rva >= end {
            lo = mid + 1;
        } else {
            return Some((m.base, e));
        }
    }
    None
}

/// Follow `RUNTIME_FUNCTION` indirection (unwind RVA with bit 0 set).
unsafe fn resolve_entry(base: u64, mut fe: u64) -> u64 {
    for _ in 0..8 {
        let unwind: u32 = unsafe { rd(fe + 8) };
        if unwind & 1 == 0 {
            break;
        }
        fe = base + (unwind & !1) as u64;
    }
    fe
}

/// What [`virtual_unwind`] found besides the unwound context.
#[derive(Debug, Clone, Copy, Default)]
pub struct Unwound {
    /// Language handler (absolute), if the requested kind is present and
    /// the pc is in the body (not prolog/epilog).
    pub handler: Option<u64>,
    /// Address of the handler data following the handler RVA.
    pub handler_data: u64,
    pub establisher_frame: u64,
}

fn pop(ctx: &mut Context) -> u64 {
    let sp = ctx.rsp();
    ctx.set_rsp(sp + 8);
    // SAFETY: guest stack.
    unsafe { rd(sp) }
}

/// If `pc` sits in an epilog (optional rsp adjust, pops, `ret`/`jmp`),
/// emulate the rest of it and return true.
fn try_epilog(pc: u64, ctx: &mut Context, begin: u64, end: u64) -> bool {
    // SAFETY (reads): bounded to the function's own code range.
    let byte = |a: u64| -> Option<u8> { (a < end).then(|| unsafe { rd::<u8>(a) }) };
    let mut p = pc;
    let mut c = ctx.clone();
    // add rsp, imm8 / imm32 ; lea rsp, [reg+disp]
    match (byte(p), byte(p + 1), byte(p + 2)) {
        (Some(0x48), Some(0x83), Some(0xC4)) => {
            let imm = byte(p + 3).unwrap_or(0) as i8 as i64;
            c.set_rsp((c.rsp() as i64 + imm) as u64);
            p += 4;
        }
        (Some(0x48), Some(0x81), Some(0xC4)) => {
            // SAFETY: inside function.
            let imm: i32 = unsafe { rd(p + 3) };
            c.set_rsp((c.rsp() as i64 + imm as i64) as u64);
            p += 7;
        }
        (Some(rex @ (0x48 | 0x49)), Some(0x8D), Some(modrm)) if modrm >> 3 & 7 == 4 => {
            let base = (modrm & 7) as usize + if rex == 0x49 { 8 } else { 0 };
            let (disp, len) = match modrm >> 6 {
                1 => (byte(p + 3).unwrap_or(0) as i8 as i64, 4),
                // SAFETY: inside function.
                2 => (unsafe { rd::<i32>(p + 3) } as i64, 7),
                _ => (0, 3),
            };
            if modrm & 7 == 4 {
                return false; // SIB forms: not generated for epilogs
            }
            c.set_rsp((c.reg(base) as i64 + disp) as u64);
            p += len;
        }
        _ => {}
    }
    loop {
        match byte(p) {
            Some(op @ 0x58..=0x5F) => {
                let v = pop(&mut c);
                c.set_reg((op - 0x58) as usize, v);
                p += 1;
            }
            Some(0x41) if matches!(byte(p + 1), Some(0x58..=0x5F)) => {
                let v = pop(&mut c);
                c.set_reg((byte(p + 1).unwrap() - 0x58) as usize + 8, v);
                p += 2;
            }
            Some(0xC3) | Some(0xC2) => break,
            // Tail-call jumps out of the function.
            Some(0xE9) | Some(0xEB) => {
                let target = if byte(p) == Some(0xEB) {
                    (p as i64 + 2 + byte(p + 1).unwrap_or(0) as i8 as i64) as u64
                } else {
                    // SAFETY: inside function.
                    (p as i64 + 5 + unsafe { rd::<i32>(p + 1) } as i64) as u64
                };
                if (begin..end).contains(&target) {
                    return false; // jump within the function: not an epilog
                }
                break;
            }
            Some(0x48) if byte(p + 1) == Some(0xFF) => break, // rex.w jmp [..]
            _ => return false,
        }
    }
    let rip = pop(&mut c);
    c.set_rip(rip);
    *ctx = c;
    true
}

/// `RtlVirtualUnwind`: unwind one frame of `ctx` (at `pc` in the function
/// described by `fe`), returning the handler of kind `kind` if any.
///
/// # Safety
/// `fe` must be a `RUNTIME_FUNCTION` of the image at `base`, and `ctx`'s
/// stack must be readable guest memory.
pub unsafe fn virtual_unwind(kind: u32, base: u64, pc: u64, fe: u64, ctx: &mut Context) -> Unwound {
    let fe = unsafe { resolve_entry(base, fe) };
    let (begin, end, mut info) = unsafe {
        (
            base + rd::<u32>(fe) as u64,
            base + rd::<u32>(fe + 4) as u64,
            base + rd::<u32>(fe + 8) as u64,
        )
    };
    let mut out = Unwound::default();
    let first_info = info;

    // Frame base used by SAVE_* codes and returned as the establisher.
    let (v0, prolog_size, frame_reg_byte) =
        unsafe { (rd::<u8>(info), rd::<u8>(info + 1), rd::<u8>(info + 3)) };
    let flags = (v0 >> 3) as u32;
    let frame_reg = (frame_reg_byte & 0xF) as usize;
    let frame_off = (frame_reg_byte >> 4) as u64 * 16;
    let offset = pc.wrapping_sub(begin);
    let in_prolog = offset < prolog_size as u64;

    // Has SET_FPREG already executed at this pc?
    let mut frame = ctx.rsp();
    if frame_reg != 0 {
        let count = unsafe { rd::<u8>(info + 2) } as u64;
        let mut fp_set = !in_prolog;
        if in_prolog {
            for i in 0..count {
                let code: u16 = unsafe { rd(info + 4 + 2 * i) };
                if code >> 8 & 0xF == 3 && (code & 0xFF) as u64 <= offset {
                    fp_set = true;
                }
            }
        }
        if fp_set {
            frame = ctx.reg(frame_reg) - frame_off;
        }
    }
    out.establisher_frame = frame;

    if !in_prolog && try_epilog(pc, ctx, begin, end) {
        return out; // no handler is called for a frame in its epilog
    }

    let mut limit_offset = if in_prolog { Some(offset) } else { None };
    let mut machframe = false;
    loop {
        let count = unsafe { rd::<u8>(info + 2) } as u64;
        let mut i = 0;
        while i < count {
            let code: u16 = unsafe { rd(info + 4 + 2 * i) };
            let at = (code & 0xFF) as u64;
            let op = (code >> 8) & 0xF;
            let op_info = (code >> 12) as usize;
            let slot = |k: u64| -> u16 { unsafe { rd(info + 4 + 2 * (i + k)) } };
            let slots = match op {
                1 => {
                    if op_info == 0 {
                        2
                    } else {
                        3
                    }
                }
                4 | 8 => 2,
                5 | 9 => 3,
                6 if unsafe { rd::<u8>(info) } & 7 == 2 => 2, // v2 EPILOG
                _ => 1,
            };
            if limit_offset.is_some_and(|l| at > l) {
                i += slots;
                continue; // this prolog instruction has not run yet
            }
            match op {
                0 => {
                    let v = pop(ctx);
                    ctx.set_reg(op_info, v);
                }
                1 => {
                    let size = if op_info == 0 {
                        slot(1) as u64 * 8
                    } else {
                        slot(1) as u64 | (slot(2) as u64) << 16
                    };
                    ctx.set_rsp(ctx.rsp() + size);
                }
                2 => ctx.set_rsp(ctx.rsp() + op_info as u64 * 8 + 8),
                3 => ctx.set_rsp(ctx.reg(frame_reg) - frame_off),
                4 => ctx.set_reg(op_info, unsafe { rd(frame + slot(1) as u64 * 8) }),
                5 => ctx.set_reg(op_info, unsafe {
                    rd(frame + (slot(1) as u64 | (slot(2) as u64) << 16))
                }),
                8 => {
                    let a = frame + slot(1) as u64 * 16;
                    ctx.set_xmm(op_info, unsafe { [rd(a), rd(a + 8)] });
                }
                9 => {
                    let a = frame + (slot(1) as u64 | (slot(2) as u64) << 16);
                    ctx.set_xmm(op_info, unsafe { [rd(a), rd(a + 8)] });
                }
                10 => {
                    // PUSH_MACHFRAME: [rsp] = (error code), rip, cs, eflags, rsp, ss
                    let sp = ctx.rsp() + if op_info != 0 { 8 } else { 0 };
                    ctx.set_rip(unsafe { rd(sp) });
                    ctx.set_rsp(unsafe { rd(sp + 24) });
                    machframe = true;
                }
                _ => {} // v2 epilog descriptors, spare
            }
            i += slots;
        }
        let v = unsafe { rd::<u8>(info) };
        if (v >> 3) as u32 & UNW_FLAG_CHAININFO == 0 {
            break;
        }
        // Chained RUNTIME_FUNCTION follows the (even-padded) code array.
        let chained = info + 4 + 2 * count.div_ceil(2) * 2;
        let chained = unsafe { resolve_entry(base, chained) };
        info = base + unsafe { rd::<u32>(chained + 8) } as u64;
        limit_offset = None;
    }

    if !machframe {
        let rip = pop(ctx);
        ctx.set_rip(rip);
    }
    if !in_prolog && flags & kind & (UNW_FLAG_EHANDLER | UNW_FLAG_UHANDLER) != 0 {
        let count = unsafe { rd::<u8>(first_info + 2) } as u64;
        let h = first_info + 4 + 2 * count.div_ceil(2) * 2;
        out.handler = Some(base + unsafe { rd::<u32>(h) } as u64);
        out.handler_data = h + 4;
    }
    out
}

/// Unwind one frame of `ctx` whatever it is: table-driven when `rip` has
/// unwind info, otherwise treated as a leaf (return address at `[rsp]`).
/// Returns the table-driven result, if any.
///
/// # Safety
/// As [`virtual_unwind`].
unsafe fn step(kind: u32, ctx: &mut Context) -> (Option<(u64, u64)>, Unwound) {
    match lookup_function_entry(ctx.rip()) {
        Some((base, fe)) => {
            let pc = ctx.rip();
            let u = unsafe { virtual_unwind(kind, base, pc, fe, ctx) };
            (Some((base, fe)), u)
        }
        None => {
            let u = Unwound {
                establisher_frame: ctx.rsp(),
                ..Default::default()
            };
            let rip = pop(ctx);
            ctx.set_rip(rip);
            (None, u)
        }
    }
}

// ---------------------------------------------------------------- dispatch

/// `DISPATCHER_CONTEXT` (x64): ControlPc @0, ImageBase @8, FunctionEntry
/// @16, EstablisherFrame @24, TargetIp @32, ContextRecord @40,
/// LanguageHandler @48, HandlerData @56, HistoryTable @64, ScopeIndex @72.
#[repr(C, align(8))]
struct DispatcherContext([u64; 10]);

thread_local! {
    /// Contexts of exceptions being dispatched on this thread (innermost
    /// last). `RtlUnwindEx` called from a handler unwinds from here.
    static ACTIVE: RefCell<Vec<Context>> = const { RefCell::new(Vec::new()) };
    /// Context handed to `restore` (must outlive the jump, and must not
    /// live on the stack being abandoned).
    static JUMP: RefCell<Box<Context>> = RefCell::new(Box::default());
}

/// Continue at `ctx` on this guest thread. Clears the active-dispatch
/// stack (every dispatch below this point is abandoned).
fn resume(ctx: &Context) -> ! {
    let teb = current_teb().expect("exception dispatch outside a guest thread") as u64;
    ACTIVE.with(|a| a.borrow_mut().clear());
    let p = JUMP.with(|j| {
        let mut j = j.borrow_mut();
        **j = ctx.clone();
        &**j as *const Context as *const u8
    });
    // SAFETY: p points at thread-local heap memory; ctx is a guest state
    // produced by unwinding this thread's guest stack.
    unsafe { crate::enter::restore(p, teb) }
}

static UNHANDLED_FILTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Vectored exception / continue handlers: (registration handle, function).
static VEH: std::sync::Mutex<Vec<(u64, u64)>> = std::sync::Mutex::new(Vec::new());
static VCH: std::sync::Mutex<Vec<(u64, u64)>> = std::sync::Mutex::new(Vec::new());
static NEXT_VEH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0x7E00_0000);

/// Call each vectored handler in `list` with `EXCEPTION_POINTERS`; true
/// if one returned EXCEPTION_CONTINUE_EXECUTION (-1).
fn call_vectored(
    list: &std::sync::Mutex<Vec<(u64, u64)>>,
    rec: *mut Record,
    ctx: *mut Context,
) -> bool {
    let handlers: Vec<u64> = list.lock().unwrap().iter().map(|&(_, f)| f).collect();
    let mut pointers = [rec as u64, ctx as u64];
    handlers.into_iter().any(|f| {
        // SAFETY: guest handler: LONG f(EXCEPTION_POINTERS*).
        unsafe { teb::call_guest(f, pointers.as_mut_ptr() as u64, 0, 0) as i32 == -1 }
    })
}

/// Continue handlers run before resuming after CONTINUE_EXECUTION.
fn continue_execution(rec: &mut Record, ctx: &mut Context) -> ! {
    call_vectored(&VCH, rec, ctx);
    resume(ctx)
}

fn add_vectored(list: &std::sync::Mutex<Vec<(u64, u64)>>, first: bool, f: u64) -> u64 {
    let h = NEXT_VEH.fetch_add(0x10, std::sync::atomic::Ordering::Relaxed);
    let mut l = list.lock().unwrap();
    if first {
        l.insert(0, (h, f));
    } else {
        l.push((h, f));
    }
    h
}

fn remove_vectored(list: &std::sync::Mutex<Vec<(u64, u64)>>, h: u64) -> u64 {
    let mut l = list.lock().unwrap();
    let before = l.len();
    l.retain(|&(x, _)| x != h);
    (l.len() != before) as u64
}

fn add_vectored_exception_handler(c: &Call) -> u64 {
    add_vectored(&VEH, c.arg(0) as u32 != 0, c.arg(1))
}
fn remove_vectored_exception_handler(c: &Call) -> u64 {
    remove_vectored(&VEH, c.arg(0))
}
fn add_vectored_continue_handler(c: &Call) -> u64 {
    add_vectored(&VCH, c.arg(0) as u32 != 0, c.arg(1))
}
fn remove_vectored_continue_handler(c: &Call) -> u64 {
    remove_vectored(&VCH, c.arg(0))
}

/// `RtlDispatchException` + what the kernel does when nobody handles it.
fn dispatch(mut rec: Record, ctx: Context) -> ! {
    ACTIVE.with(|a| a.borrow_mut().push(ctx.clone()));
    let mut original = ctx.clone();
    if call_vectored(&VEH, &mut rec, &mut original) {
        continue_execution(&mut rec, &mut original);
    }
    let mut cur = ctx;
    for _ in 0..1024 {
        let mut next = cur.clone();
        // SAFETY: walking this thread's guest stack.
        let (entry, u) = unsafe { step(UNW_FLAG_EHANDLER, &mut next) };
        if next.rip() == 0 {
            break; // ran off the guest stack (returned into the entry frame)
        }
        if let (Some((base, fe)), Some(handler)) = (entry, u.handler) {
            let mut dc = DispatcherContext([0; 10]);
            dc.0[0] = cur.rip();
            dc.0[1] = base;
            dc.0[2] = fe;
            dc.0[3] = u.establisher_frame;
            dc.0[5] = &mut original as *mut Context as u64;
            dc.0[6] = handler;
            dc.0[7] = u.handler_data;
            // SAFETY: guest language handler, documented signature.
            let disposition = unsafe {
                teb::call_guest4(
                    handler,
                    [
                        &mut rec as *mut Record as u64,
                        u.establisher_frame,
                        &mut original as *mut Context as u64,
                        &mut dc as *mut DispatcherContext as u64,
                    ],
                )
            } as u32;
            match disposition {
                EXCEPTION_CONTINUE_EXECUTION => {
                    if rec.flags() & EXCEPTION_NONCONTINUABLE != 0 {
                        log::error!(
                            "handler continued a noncontinuable exception {:#x}",
                            rec.code()
                        );
                        std::process::exit(STATUS_NONCONTINUABLE_EXCEPTION as i32);
                    }
                    continue_execution(&mut rec, &mut original);
                }
                EXCEPTION_CONTINUE_SEARCH => {}
                d => log::warn!("exception handler returned disposition {d}; continuing search"),
            }
        }
        if entry.is_none() && module::guest_containing(cur.rip()).is_none() {
            break; // left guest code: nothing further up can handle it
        }
        cur = next;
    }
    ACTIVE.with(|a| a.borrow_mut().pop());
    let filter = UNHANDLED_FILTER.load(std::sync::atomic::Ordering::Relaxed);
    if filter != 0 {
        // EXCEPTION_POINTERS { record, context }
        let mut pointers = [
            &mut rec as *mut Record as u64,
            &mut original as *mut Context as u64,
        ];
        // SAFETY: guest filter function.
        let r = unsafe { teb::call_guest(filter, pointers.as_mut_ptr() as u64, 0, 0) } as i32;
        if r == -1 {
            resume(&original); // EXCEPTION_CONTINUE_EXECUTION
        }
    }
    log::error!(
        "unhandled guest exception {:#010x} at {:#x}; terminating",
        rec.code(),
        u64::from_le_bytes(rec.0[16..24].try_into().unwrap())
    );
    std::process::exit(rec.code() as i32)
}

fn caller_context(c: &Call) -> Context {
    Context::from_frame(c.frame().expect("SEH API called outside guest code"))
}

/// `RaiseException(code, flags, nargs, args)`.
fn raise_exception(c: &Call) -> u64 {
    let ctx = caller_context(c);
    let n = (c.arg(2) as usize).min(EXCEPTION_MAXIMUM_PARAMETERS);
    let args = c.arg_ptr::<u64>(3);
    let params: Vec<u64> = if args.is_null() {
        Vec::new()
    } else {
        // SAFETY: guest array of `n` ULONG_PTRs.
        unsafe { std::slice::from_raw_parts(args, n) }.to_vec()
    };
    let rec = Record::new(
        c.arg(0) as u32,
        c.arg(1) as u32 & EXCEPTION_NONCONTINUABLE,
        ctx.rip(),
        &params,
    );
    dispatch(rec, ctx)
}

/// `RtlRaiseException(record)`.
fn rtl_raise_exception(c: &Call) -> u64 {
    let ctx = caller_context(c);
    let mut rec = Record([0; EXCEPTION_RECORD_SIZE]);
    // SAFETY: guest EXCEPTION_RECORD.
    unsafe {
        std::ptr::copy_nonoverlapping(
            c.arg_ptr::<u8>(0),
            rec.0.as_mut_ptr(),
            EXCEPTION_RECORD_SIZE,
        )
    };
    dispatch(rec, ctx)
}

/// `RtlUnwindEx(target_frame, target_ip, record, return_value, context, history)`.
fn unwind_ex(
    target_frame: u64,
    target_ip: u64,
    rec_in: *const u8,
    retval: u64,
    start: Context,
) -> ! {
    let mut rec = if rec_in.is_null() {
        Record::new(STATUS_UNWIND, 0, start.rip(), &[])
    } else {
        let mut r = Record([0; EXCEPTION_RECORD_SIZE]);
        // SAFETY: guest EXCEPTION_RECORD.
        unsafe { std::ptr::copy_nonoverlapping(rec_in, r.0.as_mut_ptr(), EXCEPTION_RECORD_SIZE) };
        r
    };
    let mut flags = rec.flags() | EXCEPTION_UNWINDING;
    if target_frame == 0 {
        flags |= EXCEPTION_EXIT_UNWIND;
    }
    let mut cur = start;
    for _ in 0..1024 {
        let mut next = cur.clone();
        // SAFETY: walking this thread's guest stack.
        let (entry, u) = unsafe { step(UNW_FLAG_UHANDLER, &mut next) };
        let est = u.establisher_frame;
        if target_frame != 0 && est > target_frame {
            log::error!(
                "RtlUnwindEx: target frame {target_frame:#x} not on the stack (at {est:#x})"
            );
            std::process::exit(STATUS_UNWIND as i32);
        }
        let at_target = est == target_frame;
        if let (Some((base, fe)), Some(handler)) = (entry, u.handler) {
            rec.set_flags(
                flags
                    | if at_target {
                        EXCEPTION_TARGET_UNWIND
                    } else {
                        0
                    },
            );
            let mut dc = DispatcherContext([0; 10]);
            dc.0[0] = cur.rip();
            dc.0[1] = base;
            dc.0[2] = fe;
            dc.0[3] = est;
            dc.0[4] = target_ip;
            dc.0[5] = &mut cur as *mut Context as u64;
            dc.0[6] = handler;
            dc.0[7] = u.handler_data;
            // SAFETY: guest termination handler.
            unsafe {
                teb::call_guest4(
                    handler,
                    [
                        &mut rec as *mut Record as u64,
                        est,
                        &mut cur as *mut Context as u64,
                        &mut dc as *mut DispatcherContext as u64,
                    ],
                )
            };
        }
        if at_target {
            break;
        }
        if next.rip() == 0 {
            log::error!("RtlUnwindEx: ran off the stack looking for {target_frame:#x}");
            std::process::exit(STATUS_UNWIND as i32);
        }
        cur = next;
    }
    cur.set_rip(target_ip);
    cur.set_reg(RAX, retval);
    resume(&cur)
}

/// Where unwinding starts: the raise site of the innermost exception being
/// dispatched (the frames in between are the dispatcher's own), else the
/// caller of the unwind API.
fn unwind_start(c: &Call) -> Context {
    ACTIVE
        .with(|a| a.borrow().last().cloned())
        .unwrap_or_else(|| caller_context(c))
}

fn rtl_unwind_ex(c: &Call) -> u64 {
    let start = unwind_start(c);
    unwind_ex(c.arg(0), c.arg(1), c.arg_ptr(2), c.arg(3), start)
}

/// `RtlUnwind(target_frame, target_ip, record, return_value)`.
fn rtl_unwind(c: &Call) -> u64 {
    let start = unwind_start(c);
    unwind_ex(c.arg(0), c.arg(1), c.arg_ptr(2), c.arg(3), start)
}

/// `RtlCaptureContext(context)`: the caller's state at the call.
fn rtl_capture_context(c: &Call) -> u64 {
    let ctx = caller_context(c);
    // SAFETY: guest CONTEXT buffer.
    unsafe { ctx.write_to(c.arg_ptr(0)) };
    0
}

/// `RtlRestoreContext(context, record)`.
fn rtl_restore_context(c: &Call) -> u64 {
    // SAFETY: guest CONTEXT.
    let ctx = unsafe { Context::read_from(c.arg_ptr(0)) };
    resume(&ctx)
}

/// `RtlLookupFunctionEntry(pc, *image_base, history)`.
fn rtl_lookup_function_entry(c: &Call) -> u64 {
    let out = c.arg_ptr::<u64>(1);
    match lookup_function_entry(c.arg(0)) {
        Some((base, fe)) => {
            if !out.is_null() {
                // SAFETY: guest out pointer.
                unsafe { out.write_unaligned(base) };
            }
            fe
        }
        None => 0,
    }
}

/// `RtlVirtualUnwind(type, base, pc, fe, context, *handler_data, *establisher, ctx_pointers)`.
fn rtl_virtual_unwind(c: &Call) -> u64 {
    let (kind, base, pc, fe, ctxp) = (
        c.arg(0) as u32,
        c.arg(1),
        c.arg(2),
        c.arg(3),
        c.arg_ptr::<u8>(4),
    );
    let (data_out, est_out) = (c.arg_ptr::<u64>(5), c.arg_ptr::<u64>(6));
    if ctxp.is_null() || fe == 0 {
        return 0;
    }
    // SAFETY: guest CONTEXT and RUNTIME_FUNCTION.
    unsafe {
        let mut ctx = Context::read_from(ctxp);
        let u = virtual_unwind(kind, base, pc, fe, &mut ctx);
        ctx.write_to(ctxp);
        if !data_out.is_null() {
            data_out.write_unaligned(u.handler_data);
        }
        if !est_out.is_null() {
            est_out.write_unaligned(u.establisher_frame);
        }
        u.handler.unwrap_or(0)
    }
}

fn set_unhandled_exception_filter(c: &Call) -> u64 {
    UNHANDLED_FILTER.swap(c.arg(0), std::sync::atomic::Ordering::Relaxed)
}

/// Filters use their own constants: EXCEPTION_CONTINUE_SEARCH is 0 there.
fn unhandled_exception_filter(_: &Call) -> u64 {
    0
}

// ---------------------------------------------------------------- __C_specific_handler

pub const EXCEPTION_EXECUTE_HANDLER: u64 = 1;

/// `__C_specific_handler(record, establisher, context, dispatcher)`: the
/// language handler MSVC emits for `__try`/`__except`/`__finally`. Its
/// handler data is a `SCOPE_TABLE`: `Count`, then `Count` entries of
/// `{BeginAddress, EndAddress, HandlerAddress, JumpTarget}` (RVAs).
/// `JumpTarget == 0` marks a `__finally` (HandlerAddress = termination
/// handler); otherwise HandlerAddress is the filter (or the constant 1 for
/// `EXCEPTION_EXECUTE_HANDLER`) and JumpTarget the `__except` block.
fn c_specific_handler(c: &Call) -> u64 {
    let (rec_p, establisher, ctx_p, dc_p) =
        (c.arg_ptr::<u8>(0), c.arg(1), c.arg(2), c.arg_ptr::<u64>(3));
    // SAFETY: arguments come from the dispatcher (ours) per the handler ABI.
    unsafe {
        let flags: u32 = rd(rec_p as u64 + 4);
        let base = dc_p.add(1).read();
        let pc = dc_p.read().wrapping_sub(base) as u32;
        let table = dc_p.add(7).read();
        let scope_index = dc_p.add(9).cast::<u32>();
        let count: u32 = rd(table);
        let entry = |i: u32| -> [u32; 4] {
            let e = table + 4 + 16 * i as u64;
            [rd(e), rd(e + 4), rd(e + 8), rd(e + 12)]
        };
        if flags & (EXCEPTION_UNWINDING | EXCEPTION_EXIT_UNWIND) == 0 {
            for i in scope_index.read()..count {
                let [begin, end, handler, jump] = entry(i);
                if !(begin..end).contains(&pc) || jump == 0 {
                    continue;
                }
                let verdict = if handler as u64 == EXCEPTION_EXECUTE_HANDLER {
                    1
                } else {
                    let mut pointers = [rec_p as u64, ctx_p];
                    teb::call_guest(
                        base + handler as u64,
                        pointers.as_mut_ptr() as u64,
                        establisher,
                        0,
                    ) as i32
                };
                if verdict < 0 {
                    return EXCEPTION_CONTINUE_EXECUTION as u64;
                }
                if verdict > 0 {
                    let code: u32 = rd(rec_p as u64);
                    let start = unwind_start(c);
                    unwind_ex(establisher, base + jump as u64, rec_p, code as u64, start);
                }
            }
        } else {
            let target = dc_p.add(4).read().wrapping_sub(base) as u32;
            for i in scope_index.read()..count {
                let [begin, end, handler, jump] = entry(i);
                if !(begin..end).contains(&pc) {
                    continue;
                }
                if flags & EXCEPTION_TARGET_UNWIND != 0 && jump == target {
                    break;
                }
                if jump != 0 {
                    continue; // __except blocks do not run during unwind
                }
                scope_index.write(i + 1);
                // void __finally(BOOLEAN abnormal_termination, establisher)
                teb::call_guest(base + handler as u64, 1, establisher, 0);
            }
        }
    }
    EXCEPTION_CONTINUE_SEARCH as u64
}

// ---------------------------------------------------------------- hardware faults

pub const STATUS_ACCESS_VIOLATION: u32 = 0xC000_0005;
pub const STATUS_ILLEGAL_INSTRUCTION: u32 = 0xC000_001D;
pub const STATUS_INTEGER_DIVIDE_BY_ZERO: u32 = 0xC000_0094;
pub const STATUS_INTEGER_OVERFLOW: u32 = 0xC000_0095;
pub const STATUS_FLOAT_INVALID_OPERATION: u32 = 0xC000_0090;
pub const STATUS_BREAKPOINT: u32 = 0x8000_0003;

/// Entered (by rewriting the signal frame) on the guest thread after a
/// fault in guest code, with host `gs`, on the guest stack. `ctx` and
/// `rec` were copied onto that stack by the signal handler.
#[cfg(unix)]
extern "win64" fn fault_entry(ctx: *const Context, rec: *const Record) -> ! {
    // SAFETY: written by `fault::handler` just below the guest's rsp.
    let (ctx, rec) = unsafe { ((*ctx).clone(), (*rec).clone()) };
    dispatch(rec, ctx)
}

/// Install the fault handlers (once per process). Faults whose pc is not
/// in guest code go to whatever handler was installed before (e.g. Rust's
/// stack-overflow detection).
#[cfg(unix)]
pub fn install_fault_handlers() {
    fault::install();
}

#[cfg(unix)]
mod fault {
    use super::*;
    use std::sync::Once;

    const SIGNALS: [libc::c_int; 5] = [
        libc::SIGSEGV,
        libc::SIGBUS,
        libc::SIGILL,
        libc::SIGFPE,
        libc::SIGTRAP,
    ];
    static INSTALL: Once = Once::new();

    // si_code values for SIGFPE (not exported by the libc crate).
    #[cfg(target_os = "linux")]
    const FPE_INTDIV: libc::c_int = 1;
    #[cfg(target_os = "linux")]
    const FPE_INTOVF: libc::c_int = 2;
    #[cfg(target_os = "macos")]
    const FPE_INTDIV: libc::c_int = 7;
    #[cfg(target_os = "macos")]
    const FPE_INTOVF: libc::c_int = 8;
    static mut PREVIOUS: [Option<libc::sigaction>; 5] = [None; 5];

    pub fn install() {
        INSTALL.call_once(|| {
            for (i, &sig) in SIGNALS.iter().enumerate() {
                // SAFETY: plain sigaction; PREVIOUS is written only here, once.
                unsafe {
                    let mut sa: libc::sigaction = std::mem::zeroed();
                    sa.sa_sigaction = handler as *const () as usize;
                    sa.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK | libc::SA_NODEFER;
                    libc::sigemptyset(&mut sa.sa_mask);
                    let mut old: libc::sigaction = std::mem::zeroed();
                    if libc::sigaction(sig, &sa, &mut old) == 0 {
                        (*std::ptr::addr_of_mut!(PREVIOUS))[i] = Some(old);
                    }
                }
            }
        });
    }

    /// Register state of the interrupted code, host-independent.
    struct Regs {
        gpr: [u64; 16],
        rip: u64,
        eflags: u32,
        mxcsr: u32,
        xmm: [[u64; 2]; 16],
        err: u64,
    }

    #[cfg(target_os = "linux")]
    unsafe fn read_regs(uc: *mut libc::ucontext_t) -> Regs {
        use libc::*;
        let g = unsafe { &(*uc).uc_mcontext.gregs };
        let r = |i: c_int| g[i as usize] as u64;
        let gpr = [
            r(REG_RAX),
            r(REG_RCX),
            r(REG_RDX),
            r(REG_RBX),
            r(REG_RSP),
            r(REG_RBP),
            r(REG_RSI),
            r(REG_RDI),
            r(REG_R8),
            r(REG_R9),
            r(REG_R10),
            r(REG_R11),
            r(REG_R12),
            r(REG_R13),
            r(REG_R14),
            r(REG_R15),
        ];
        let mut regs = Regs {
            gpr,
            rip: r(REG_RIP),
            eflags: r(REG_EFL) as u32,
            mxcsr: 0x1F80,
            xmm: [[0; 2]; 16],
            err: r(REG_ERR),
        };
        let fp = unsafe { (*uc).uc_mcontext.fpregs };
        if !fp.is_null() {
            let fp = unsafe { &*fp };
            regs.mxcsr = fp.mxcsr;
            for i in 0..16 {
                let e = fp._xmm[i].element;
                regs.xmm[i] = [
                    e[0] as u64 | (e[1] as u64) << 32,
                    e[2] as u64 | (e[3] as u64) << 32,
                ];
            }
        }
        regs
    }

    #[cfg(target_os = "linux")]
    unsafe fn redirect(uc: *mut libc::ucontext_t, rip: u64, rsp: u64, rcx: u64, rdx: u64) {
        let g = unsafe { &mut (*uc).uc_mcontext.gregs };
        g[libc::REG_RIP as usize] = rip as i64;
        g[libc::REG_RSP as usize] = rsp as i64;
        g[libc::REG_RCX as usize] = rcx as i64;
        g[libc::REG_RDX as usize] = rdx as i64;
    }

    #[cfg(target_os = "linux")]
    unsafe fn fault_addr(info: *mut libc::siginfo_t) -> u64 {
        unsafe { (*info).si_addr() as u64 }
    }

    #[cfg(target_os = "macos")]
    unsafe fn read_regs(uc: *mut libc::ucontext_t) -> Regs {
        let m = unsafe { &*(*uc).uc_mcontext };
        let s = &m.__ss;
        let gpr = [
            s.__rax, s.__rcx, s.__rdx, s.__rbx, s.__rsp, s.__rbp, s.__rsi, s.__rdi, s.__r8, s.__r9,
            s.__r10, s.__r11, s.__r12, s.__r13, s.__r14, s.__r15,
        ];
        let mut xmm = [[0u64; 2]; 16];
        let base = &m.__fs.__fpu_xmm0 as *const libc::__darwin_xmm_reg as *const u64;
        for (i, x) in xmm.iter_mut().enumerate() {
            // SAFETY: __fpu_xmm0..15 are 16 consecutive 16-byte registers.
            unsafe {
                *x = [
                    base.add(2 * i).read_unaligned(),
                    base.add(2 * i + 1).read_unaligned(),
                ]
            };
        }
        Regs {
            gpr,
            rip: s.__rip,
            eflags: s.__rflags as u32,
            mxcsr: m.__fs.__fpu_mxcsr,
            xmm,
            err: m.__es.__err as u64,
        }
    }

    #[cfg(target_os = "macos")]
    unsafe fn redirect(uc: *mut libc::ucontext_t, rip: u64, rsp: u64, rcx: u64, rdx: u64) {
        let s = unsafe { &mut (*(*uc).uc_mcontext).__ss };
        s.__rip = rip;
        s.__rsp = rsp;
        s.__rcx = rcx;
        s.__rdx = rdx;
    }

    #[cfg(target_os = "macos")]
    unsafe fn fault_addr(info: *mut libc::siginfo_t) -> u64 {
        unsafe { (*info).si_addr as u64 }
    }

    /// Is `pc` in a mapped guest image? Lock-free-ish: gives up (treats the
    /// fault as foreign) if the module table is busy.
    fn in_guest(pc: u64) -> bool {
        module::guest_containing_try(pc).unwrap_or(false)
    }

    unsafe fn chain(sig: libc::c_int, info: *mut libc::siginfo_t, uc: *mut libc::c_void) {
        let i = SIGNALS.iter().position(|&s| s == sig).unwrap();
        // SAFETY: PREVIOUS is only written during install.
        let prev = unsafe { (*std::ptr::addr_of!(PREVIOUS))[i] };
        match prev {
            Some(p) if p.sa_sigaction != libc::SIG_DFL && p.sa_sigaction != libc::SIG_IGN => {
                if p.sa_flags & libc::SA_SIGINFO != 0 {
                    let f: extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void) =
                        unsafe { std::mem::transmute(p.sa_sigaction) };
                    f(sig, info, uc);
                } else {
                    let f: extern "C" fn(libc::c_int) =
                        unsafe { std::mem::transmute(p.sa_sigaction) };
                    f(sig);
                }
            }
            _ => unsafe {
                // Default action: restore it and let the fault happen again.
                let mut sa: libc::sigaction = std::mem::zeroed();
                sa.sa_sigaction = libc::SIG_DFL;
                libc::sigaction(sig, &sa, std::ptr::null_mut());
            },
        }
    }

    extern "C" fn handler(sig: libc::c_int, info: *mut libc::siginfo_t, ucv: *mut libc::c_void) {
        let uc = ucv as *mut libc::ucontext_t;
        // SAFETY: the kernel passes a valid ucontext for SA_SIGINFO handlers.
        let regs = unsafe { read_regs(uc) };
        if !in_guest(regs.rip) {
            unsafe { chain(sig, info, ucv) };
            return;
        }
        // The fault came from guest code, so on macOS gs is the guest TEB:
        // switch to the host TSD before touching thread-locals.
        #[cfg(target_os = "macos")]
        unsafe {
            let teb: u64;
            core::arch::asm!("mov {}, gs:[0x30]", out(reg) teb, options(nostack, readonly));
            let host = ((teb + crate::teb::off::HOST_GS as u64) as *const u64).read();
            core::arch::asm!("syscall", inlateout("rax") crate::teb::SET_GS_TRAP => _, in("rdi") host,
                lateout("rcx") _, lateout("r11") _, options(nostack));
        }
        if current_teb().is_none() {
            unsafe { chain(sig, info, ucv) };
            return;
        }

        let addr = unsafe { fault_addr(info) };
        let code = unsafe { (*info).si_code };
        let (status, rip, params): (u32, u64, Vec<u64>) = match sig {
            libc::SIGSEGV | libc::SIGBUS => {
                let kind = if addr == regs.rip {
                    8
                } else if regs.err & 2 != 0 {
                    1
                } else {
                    0
                };
                (STATUS_ACCESS_VIOLATION, regs.rip, vec![kind, addr])
            }
            libc::SIGILL => (STATUS_ILLEGAL_INSTRUCTION, regs.rip, vec![]),
            libc::SIGFPE => (
                match code {
                    FPE_INTDIV => STATUS_INTEGER_DIVIDE_BY_ZERO,
                    FPE_INTOVF => STATUS_INTEGER_OVERFLOW,
                    _ => STATUS_FLOAT_INVALID_OPERATION,
                },
                regs.rip,
                vec![],
            ),
            // int3 traps after the instruction; Windows reports the int3 itself.
            _ => (STATUS_BREAKPOINT, regs.rip.wrapping_sub(1), vec![0]),
        };

        let mut ctx = Context::default();
        for (i, v) in regs.gpr.iter().enumerate() {
            ctx.set_reg(i, *v);
        }
        ctx.set_rip(rip);
        ctx.put_u32(ctx_off::EFLAGS, regs.eflags);
        ctx.put_u32(ctx_off::MXCSR, regs.mxcsr);
        ctx.put_u32(ctx_off::FLT_MXCSR, regs.mxcsr);
        for (i, x) in regs.xmm.iter().enumerate() {
            ctx.set_xmm(i, *x);
        }
        let rec = Record::new(status, 0, rip, &params);

        // Copy both onto the guest stack, well below its rsp, and resume
        // there in `fault_entry(ctx, rec)` as if it had been called.
        let area = (regs.gpr[RSP] - 0x100 - (CONTEXT_SIZE + EXCEPTION_RECORD_SIZE) as u64) & !0xF;
        // SAFETY: free guest stack below the interrupted rsp.
        unsafe {
            std::ptr::copy_nonoverlapping(ctx.0.as_ptr(), area as *mut u8, CONTEXT_SIZE);
            std::ptr::copy_nonoverlapping(
                rec.0.as_ptr(),
                (area + CONTEXT_SIZE as u64) as *mut u8,
                EXCEPTION_RECORD_SIZE,
            );
            let rsp = area - 32 - 8; // shadow space + fake return address slot
            (rsp as *mut u64).write(0);
            redirect(
                uc,
                fault_entry as *const () as usize as u64,
                rsp,
                area,
                area + CONTEXT_SIZE as u64,
            );
        }
    }
}

/// Register the SEH APIs. On Windows hosts nothing is registered: the
/// real OS handles SEH for guest code (use `--host-forward`).
pub fn register_seh(reg: &mut Registry, dlls: &[&str]) {
    if cfg!(windows) {
        return;
    }
    for dll in dlls {
        reg.register(dll, "RaiseException", raise_exception)
            .register(dll, "RtlRaiseException", rtl_raise_exception)
            .register(dll, "RtlUnwindEx", rtl_unwind_ex)
            .register(dll, "RtlUnwind", rtl_unwind)
            .register(dll, "RtlCaptureContext", rtl_capture_context)
            .register(dll, "RtlRestoreContext", rtl_restore_context)
            .register(dll, "RtlLookupFunctionEntry", rtl_lookup_function_entry)
            .register(dll, "RtlVirtualUnwind", rtl_virtual_unwind)
            .register(
                dll,
                "SetUnhandledExceptionFilter",
                set_unhandled_exception_filter,
            )
            .register(dll, "UnhandledExceptionFilter", unhandled_exception_filter)
            .register(dll, "__C_specific_handler", c_specific_handler)
            .register(
                dll,
                "AddVectoredExceptionHandler",
                add_vectored_exception_handler,
            )
            .register(
                dll,
                "RemoveVectoredExceptionHandler",
                remove_vectored_exception_handler,
            )
            .register(
                dll,
                "AddVectoredContinueHandler",
                add_vectored_continue_handler,
            )
            .register(
                dll,
                "RemoveVectoredContinueHandler",
                remove_vectored_continue_handler,
            );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unwind a synthetic frame: `push rbp` (ends at +2), `sub rsp, 0x28`
    /// (+6), `movaps [rsp+0x10], xmm6` (+11), from the body and the prolog.
    #[test]
    fn virtual_unwind_pushes_alloc_and_saves() {
        // "image": RUNTIME_FUNCTION at 0, UNWIND_INFO at 0x10, code at 0x100..0x200.
        let mut img = vec![0u8; 0x200];
        img[0..4].copy_from_slice(&0x100u32.to_le_bytes());
        img[4..8].copy_from_slice(&0x200u32.to_le_bytes());
        img[8..12].copy_from_slice(&0x10u32.to_le_bytes());
        // version 1, no flags, prolog 11 bytes, 4 slots, no frame reg
        img[0x10..0x14].copy_from_slice(&[0x01, 11, 4, 0]);
        // codes in reverse prolog order
        let codes: [u16; 4] = [
            11 | (8 << 8) | (6 << 12), // SAVE_XMM128 xmm6 at [frame + 1*16]
            1,                         //   slot: offset/16
            6 | (2 << 8) | (4 << 12),  // ALLOC_SMALL 4*8+8 = 0x28
            2 | (5 << 12),             // PUSH_NONVOL (op 0) rbp
        ];
        for (i, c) in codes.iter().enumerate() {
            img[0x14 + 2 * i..0x16 + 2 * i].copy_from_slice(&c.to_le_bytes());
        }
        let base = img.as_ptr() as u64;

        // Stack: [rsp+0x10] = xmm6, [rsp+0x28] = rbp, [rsp+0x30] = ret.
        let mut stack = [0u64; 16];
        stack[2] = 0x6666;
        stack[3] = 0x6667;
        stack[5] = 0xBBBB; // saved rbp
        stack[6] = 0x4242_0000; // return address
        let rsp = stack.as_ptr() as u64;
        let mut ctx = Context::default();
        ctx.set_rsp(rsp);
        ctx.set_rip(base + 0x150);
        let u = unsafe { virtual_unwind(UNW_FLAG_NHANDLER, base, base + 0x150, base, &mut ctx) };
        assert_eq!(u.establisher_frame, rsp);
        assert_eq!(ctx.xmm(6), [0x6666, 0x6667]);
        assert_eq!(ctx.reg(RBP), 0xBBBB);
        assert_eq!(ctx.rip(), 0x4242_0000);
        assert_eq!(ctx.rsp(), rsp + 0x38);
        assert!(u.handler.is_none());

        // In the prolog right after `push rbp` (offset 2): only that push is undone.
        let mut stack2 = [0xCCCCu64, 0x5151_0000];
        let mut ctx2 = Context::default();
        ctx2.set_rsp(stack2.as_mut_ptr() as u64);
        unsafe { virtual_unwind(UNW_FLAG_NHANDLER, base, base + 0x102, base, &mut ctx2) };
        assert_eq!(ctx2.reg(RBP), 0xCCCC);
        assert_eq!(ctx2.rip(), 0x5151_0000);
    }

    #[test]
    fn epilog_is_emulated() {
        // add rsp, 0x10; pop rbx; ret
        let mut img = [0u8; 0x40];
        img[0x20..0x26].copy_from_slice(&[0x48, 0x83, 0xC4, 0x10, 0x5B, 0xC3]);
        let base = img.as_ptr() as u64;
        let stack = [0u64, 0, 0x7777, 0x9999_0000];
        let mut ctx = Context::default();
        ctx.set_rsp(stack.as_ptr() as u64);
        assert!(try_epilog(base + 0x20, &mut ctx, base, base + 0x40));
        assert_eq!(ctx.reg(RBX), 0x7777);
        assert_eq!(ctx.rip(), 0x9999_0000);
        assert_eq!(ctx.rsp(), stack.as_ptr() as u64 + 32);
        // Not an epilog: mov eax, 1
        img[0x30..0x35].copy_from_slice(&[0xB8, 1, 0, 0, 0]);
        let mut c2 = Context::default();
        assert!(!try_epilog(base + 0x30, &mut c2, base, base + 0x40));
    }
}
