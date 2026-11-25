//! Executes command-processor dispatches on the reference interpreter:
//! the compute shader named by `COMPUTE_PGM_LO/HI` runs over GPU memory
//! with the state the packets programmed. Field layouts are from Mesa's
//! `gfx10.json`, encodings from RADV (`radv_shader.c`,
//! `radv_cmd_buffer.c`).

use crate::cp::{Dispatch, Memory};
use crate::interp::{self, ExecError, WaveSize};
use crate::regs;
use crate::translate::{self, ComputeIo, CrossLane, Io, Options, TranslateError, Window};

/// DISPATCH_INITIATOR bits.
pub const COMPUTE_SHADER_EN: u32 = 1;
pub const USE_THREAD_DIMENSIONS: u32 = 1 << 5;
pub const CS_W32_EN: u32 = 1 << 15;

/// Largest shader fetched, in dwords.
pub const MAX_CODE_DWORDS: usize = 0x4000;

fn refuse(what: &str) -> ExecError {
    ExecError::Unsupported { at: 0, what: what.into() }
}

/// Wave size, VGPRs and LDS dwords a dispatch asks for. VGPRs: RSRC1
/// VGPRS = (count - 1) / (8 in wave32, 4 in wave64), as RADV encodes it;
/// LDS: RSRC2 LDS_SIZE in 512-byte units (Mesa `lds_encode_granularity`
/// for GFX7+ compute).
pub fn shape(d: &Dispatch) -> (WaveSize, usize, usize) {
    let wave32 = d.initiator & CS_W32_EN != 0;
    let size = if wave32 { WaveSize::W32 } else { WaveSize::W64 };
    let rsrc1 = d.regs.get(regs::COMPUTE_PGM_RSRC1);
    let vgprs = ((rsrc1 & 0x3F) as usize + 1) * if wave32 { 8 } else { 4 };
    let rsrc2 = d.regs.get(regs::COMPUTE_PGM_RSRC2);
    let lds = ((rsrc2 >> 15) & 0x1FF) as usize * 128;
    (size, vgprs, lds)
}

/// Fetch the shader: dwords from `addr` up to the first unmapped 4 KiB
/// page or [`MAX_CODE_DWORDS`].
pub fn fetch_code(mem: &dyn Memory, addr: u64) -> Vec<u32> {
    let mut code = Vec::new();
    let mut a = addr;
    while code.len() < MAX_CODE_DWORDS {
        let n = ((0x1000 - (a & 0xFFF)) / 4) as usize;
        let mut chunk = vec![0u32; n.min(MAX_CODE_DWORDS - code.len())];
        if !mem.read(a, &mut chunk) {
            break;
        }
        a += 4 * chunk.len() as u64;
        code.extend(chunk);
    }
    code
}

/// What [`run_dispatch`] and [`translate_dispatch`] both derive from a
/// dispatch: the shader I/O contract and its user SGPRs.
fn setup(d: &Dispatch) -> Result<(ComputeIo, Vec<u32>), ExecError> {
    if d.initiator & COMPUTE_SHADER_EN == 0 {
        return Err(refuse("COMPUTE_SHADER_EN clear"));
    }
    if d.initiator & USE_THREAD_DIMENSIONS != 0 {
        return Err(refuse("USE_THREAD_DIMENSIONS"));
    }
    if [regs::COMPUTE_START_X, regs::COMPUTE_START_Y, regs::COMPUTE_START_Z]
        .iter()
        .any(|&r| d.regs.get(r) != 0)
    {
        return Err(refuse("COMPUTE_START_* not 0"));
    }
    let rsrc2 = d.regs.get(regs::COMPUTE_PGM_RSRC2);
    if rsrc2 & 1 != 0 {
        return Err(refuse("scratch (SCRATCH_EN)"));
    }
    let io = ComputeIo::from_rsrc2(rsrc2, d.threads);
    if io.user_sgprs > 16 {
        return Err(refuse("more than 16 user SGPRs"));
    }
    let user: Vec<u32> = (0..io.user_sgprs).map(|i| d.regs.get(regs::COMPUTE_USER_DATA_0 + i)).collect();
    Ok((io, user))
}

/// Dwords of `code` reachable from dword 0 by falling through and taking
/// branches, i.e. the shader without whatever follows its last reachable
/// instruction (the fetch reads to the end of a page). None if a reachable
/// instruction does not decode or a branch leaves the code.
pub fn reachable_len(code: &[u32]) -> Option<usize> {
    let mut seen = vec![false; code.len()];
    let mut todo = vec![0usize];
    let mut end = 0;
    while let Some(mut at) = todo.pop() {
        loop {
            if at >= code.len() {
                return None;
            }
            if seen[at] {
                break;
            }
            seen[at] = true;
            let inst = crate::rdna::decode(code, at).ok()?;
            end = end.max(at + inst.len());
            if let Some(t) = inst.branch_target(at) {
                todo.push(t);
            }
            if inst.ends_block() {
                break;
            }
            at += inst.len();
        }
    }
    Some(end)
}

/// Run `d` on the interpreter. Refused: dispatches that do not start at
/// group 0, USE_THREAD_DIMENSIONS, scratch, more than 16 user SGPRs.
pub fn run_dispatch(d: &Dispatch, mem: &mut dyn Memory, max_steps: usize) -> Result<(), ExecError> {
    let (io, user) = setup(d)?;
    let (size, vgprs, lds) = shape(d);
    let code = fetch_code(mem, d.program);
    if code.is_empty() {
        return Err(ExecError::MemoryFault { at: 0, addr: d.program });
    }
    interp::dispatch_compute_at(&code, d.program, &io, &user, d.groups, size, vgprs, lds, mem, max_steps)
}

/// A dispatch translated to SPIR-V: the module, the options it was built
/// with, and the push constants (user SGPRs) to run it with.
pub struct Compiled {
    pub module: Vec<u32>,
    pub options: Options,
    pub push: Vec<u32>,
}

#[derive(Debug)]
pub enum CompileError {
    Setup(ExecError),
    Translate(TranslateError),
}

/// Translate `d`'s shader for a GPU memory window `window` (binding 1),
/// with the same state [`run_dispatch`] uses: one subgroup per wave,
/// every SGPR, the VGPRs and LDS the dispatch allocates, the code address
/// for `s_getpc_b64`.
pub fn translate_dispatch(d: &Dispatch, mem: &dyn Memory, window: Window) -> Result<Compiled, CompileError> {
    let (io, push) = setup(d).map_err(CompileError::Setup)?;
    let (size, vgprs, lds) = shape(d);
    let mut code = fetch_code(mem, d.program);
    let n = reachable_len(&code)
        .ok_or(CompileError::Setup(ExecError::MemoryFault { at: 0, addr: d.program }))?;
    code.truncate(n);
    let options = Options {
        io: Io::Compute(io),
        num_sgprs: interp::NUM_SGPRS as u32,
        num_vgprs: vgprs as u32,
        lanes: size.lanes() as u32,
        memory: Some(window),
        cross_lane: CrossLane::Subgroup,
        lds_words: lds as u32,
        code_base: d.program,
    };
    let module = translate::translate(&code, &options).map_err(CompileError::Translate)?;
    Ok(Compiled { module, options, push })
}
