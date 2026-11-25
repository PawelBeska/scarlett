//! A small SPIR-V evaluator for the modules the translator emits.
//!
//! It runs one subgroup of a `GLCompute` entry point: all invocations
//! together, instruction by instruction, in lockstep. That is exactly the
//! shape the translator produces (control flow uniform across the
//! subgroup, divergence expressed with predication), and it lets subgroup
//! operations (`OpGroupNonUniformBallot`, `OpGroupNonUniformShuffle`) be
//! evaluated across invocations. A branch whose target differs between
//! invocations is reported as [`EvalError::Divergent`], which for the
//! translator's output is a bug.
//!
//! Supported: loads, stores and access chains on Private / Function /
//! Input / StorageBuffer variables, 32-bit integer and float arithmetic,
//! bit operations, comparisons, `OpSelect`, `OpBitcast`, conversions,
//! structured branches and the two subgroup operations above. Anything
//! else is [`EvalError::Unsupported`]. Floats follow IEEE-754 binary32 as
//! Rust computes it (no denormal flushing), as [`crate::interp`] does, so
//! the two can be compared bit for bit.

use crate::spirv::{builtin, decoration, op, parse, scope, storage, Id, Instruction, Invalid};
use std::collections::HashMap;

#[derive(Debug, PartialEq)]
pub enum EvalError {
    Invalid(Invalid),
    Unsupported { at: usize, opcode: u16, why: &'static str },
    Divergent { at: usize },
    /// Two invocations touched the same workgroup word, at least one of
    /// them writing, with no barrier in between.
    Race { at: usize, var: Id, index: u32 },
    NoEntryPoint,
    StepLimit,
}

impl From<Invalid> for EvalError {
    fn from(e: Invalid) -> Self {
        EvalError::Invalid(e)
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Val {
    Word(u32),
    Bool(bool),
    Vec(Vec<u32>),
    /// Pointer into variable `var` at word `index` (None = whole value).
    Ptr { var: Id, index: Option<u32> },
    /// An undefined result (e.g. an out-of-range conversion). Fine to
    /// compute and to discard through `OpSelect`; any other use is an error.
    Undef,
}

#[derive(Clone, Debug)]
enum Store {
    Word(u32),
    Bool(bool),
    Vec(Vec<u32>),
    /// A storage buffer: struct { u32 data[]; } flattened to words. Shared
    /// by all invocations, so it lives outside the per-invocation memory.
    Buffer,
    /// A Workgroup u32 array, shared by all invocations (kept with the
    /// buffers).
    Shared,
    /// An Input array of vec4 (per-vertex attributes), flattened.
    Vec4Array(Vec<u32>),
}

/// Descriptor-bound storage buffers by (set, binding).
pub type Buffers = HashMap<(u32, u32), Vec<u32>>;

struct Module {
    insts: Vec<Instruction>,
    entry: Id,
    constants: HashMap<Id, Val>,
    /// (variable, pointer type, storage class)
    globals: Vec<(Id, u32, u32)>,
    builtins: HashMap<Id, u32>,
    bindings: HashMap<Id, (u32, u32)>,
    ptr_to_bool: HashMap<Id, bool>,
    ptr_to_vec: HashMap<Id, bool>,
    /// Workgroup pointer type → array length (for u32 arrays).
    ptr_to_shared_len: HashMap<Id, u32>,
    /// Private pointer type → array length (for u32 arrays).
    ptr_to_private_len: HashMap<Id, u32>,
    /// Pointer types to arrays of vec4.
    ptr_to_vec4_array: Vec<Id>,
    labels: HashMap<Id, usize>,
    /// Ids of GLSL.std.450 imports.
    glsl: Vec<Id>,
    /// 64-bit float types; their values are two words, low first (the
    /// OpBitcast mapping of uvec2 <-> double).
    f64_types: Vec<Id>,
    /// Output variable → Location.
    locations: HashMap<Id, u32>,
    /// Is the entry point a fragment shader?
    fragment: bool,
    /// Is the entry point a vertex shader?
    vertex: bool,
    /// The Position output, if any.
    position: Option<Id>,
}

fn load_module(words: &[u32]) -> Result<Module, EvalError> {
    let (_, insts) = parse(words)?;
    let mut m = Module {
        insts: Vec::new(),
        entry: 0,
        constants: HashMap::new(),
        globals: Vec::new(),
        builtins: HashMap::new(),
        bindings: HashMap::new(),
        ptr_to_bool: HashMap::new(),
        ptr_to_vec: HashMap::new(),
        ptr_to_shared_len: HashMap::new(),
        ptr_to_private_len: HashMap::new(),
        ptr_to_vec4_array: Vec::new(),
        labels: HashMap::new(),
        glsl: Vec::new(),
        f64_types: Vec::new(),
        locations: HashMap::new(),
        fragment: false,
        vertex: false,
        position: None,
    };
    let mut array_len: HashMap<Id, Id> = HashMap::new();
    let (mut bool_types, mut vec_types) = (Vec::new(), Vec::new());
    let mut vec4_arrays: Vec<Id> = Vec::new();
    let mut sets: HashMap<Id, u32> = HashMap::new();
    let mut binds: HashMap<Id, u32> = HashMap::new();
    let mut found_entry = false;
    for (n, i) in insts.iter().enumerate() {
        let o = &i.operands;
        match i.opcode {
            op::ENTRY_POINT => {
                m.entry = o[1];
                m.fragment = o[0] == crate::spirv::execution_model::FRAGMENT;
                m.vertex = o[0] == crate::spirv::execution_model::VERTEX;
                found_entry = true;
            }
            op::EXT_INST_IMPORT => {
                let bytes: Vec<u8> = o[1..].iter().flat_map(|w| w.to_le_bytes()).collect();
                let name = bytes.split(|b| *b == 0).next().unwrap_or(&[]);
                if name == crate::spirv::glsl::NAME.as_bytes() {
                    m.glsl.push(o[0]);
                }
            }
            op::TYPE_FLOAT if o[1] == 64 => m.f64_types.push(o[0]),
            op::TYPE_BOOL => bool_types.push(o[0]),
            op::TYPE_VECTOR => vec_types.push(o[0]),
            op::TYPE_ARRAY => {
                array_len.insert(o[0], o[2]);
                if vec_types.contains(&o[1]) {
                    vec4_arrays.push(o[0]);
                }
            }
            op::TYPE_POINTER => {
                if vec4_arrays.contains(&o[2]) {
                    m.ptr_to_vec4_array.push(o[0]);
                }
                m.ptr_to_bool.insert(o[0], bool_types.contains(&o[2]));
                m.ptr_to_vec.insert(o[0], vec_types.contains(&o[2]));
                if let Some(Val::Word(n)) = array_len.get(&o[2]).and_then(|c| m.constants.get(c)) {
                    match o[1] {
                        storage::WORKGROUP => {
                            m.ptr_to_shared_len.insert(o[0], *n);
                        }
                        storage::PRIVATE => {
                            m.ptr_to_private_len.insert(o[0], *n);
                        }
                        _ => {}
                    }
                }
            }
            op::CONSTANT => {
                m.constants.insert(o[1], Val::Word(o[2]));
            }
            op::CONSTANT_COMPOSITE => {
                let parts: Option<Vec<u32>> = o[2..]
                    .iter()
                    .map(|id| match m.constants.get(id) {
                        Some(Val::Word(w)) => Some(*w),
                        _ => None,
                    })
                    .collect();
                if let Some(p) = parts {
                    m.constants.insert(o[1], Val::Vec(p));
                }
            }
            op::CONSTANT_TRUE => {
                m.constants.insert(o[1], Val::Bool(true));
            }
            op::CONSTANT_FALSE => {
                m.constants.insert(o[1], Val::Bool(false));
            }
            op::DECORATE => match o[1] {
                decoration::BUILT_IN => {
                    m.builtins.insert(o[0], o[2]);
                    if o[2] == builtin::POSITION {
                        m.position = Some(o[0]);
                    }
                }
                decoration::DESCRIPTOR_SET => {
                    sets.insert(o[0], o[2]);
                }
                decoration::BINDING => {
                    binds.insert(o[0], o[2]);
                }
                decoration::LOCATION => {
                    m.locations.insert(o[0], o[2]);
                }
                _ => {}
            },
            op::VARIABLE if o[2] != storage::FUNCTION => m.globals.push((o[1], o[0], o[2])),
            op::LABEL => {
                m.labels.insert(o[0], n);
            }
            _ => {}
        }
    }
    for (&var, &set) in &sets {
        if let Some(&b) = binds.get(&var) {
            m.bindings.insert(var, (set, b));
        }
    }
    if !found_entry {
        return Err(EvalError::NoEntryPoint);
    }
    m.insts = insts;
    Ok(m)
}

/// Barrier-epoch tracking of workgroup memory, to catch missing barriers
/// that lockstep execution would otherwise hide. Two accesses to the same
/// word by different invocations, at least one a write, are ordered only
/// by a workgroup barrier between them, or (same subgroup) by a subgroup
/// barrier between them.
#[derive(Default)]
struct Races {
    /// Workgroup barrier epoch.
    epoch: u64,
    /// Subgroup barrier epoch per subgroup.
    sg_epoch: Vec<u64>,
    /// Invocations per subgroup.
    sg_size: usize,
    words: HashMap<(Id, u32), WordTrack>,
}

/// One workgroup word: (last write: (wg epoch, invocation, sg epoch), wg
/// epoch of the reads, reads in that epoch: (invocation, sg epoch)).
/// The fourth element: atomics in that epoch (invocation, sg epoch,
/// order-sensitive).
type WordTrack = (Option<(u64, usize, u64)>, u64, Vec<(usize, u64)>, Vec<(usize, u64, bool)>);

impl Races {
    fn sg(&self, inv: usize) -> usize {
        inv / self.sg_size.max(1)
    }

    /// Is a prior access (wg epoch `e`, invocation `other`, its subgroup
    /// epoch `se`) unordered with one by `inv` now?
    fn conflicts(&self, e: u64, other: usize, se: u64, inv: usize) -> bool {
        if e != self.epoch || other == inv {
            return false;
        }
        let sg = self.sg(other);
        sg != self.sg(inv) || self.sg_epoch.get(sg).copied().unwrap_or(0) == se
    }

    fn read(&mut self, at: usize, var: Id, index: u32, inv: usize) -> Result<(), EvalError> {
        let my_se = self.sg_epoch.get(self.sg(inv)).copied().unwrap_or(0);
        let t = self.words.entry((var, index)).or_default().clone();
        if let Some((e, w, se)) = t.0 {
            if self.conflicts(e, w, se, inv) {
                return Err(EvalError::Race { at, var, index });
            }
        }
        if t.1 == self.epoch && t.3.iter().any(|&(a, se, _)| self.conflicts(self.epoch, a, se, inv)) {
            return Err(EvalError::Race { at, var, index });
        }
        let epoch = self.epoch;
        let t = self.words.get_mut(&(var, index)).unwrap();
        if t.1 != epoch {
            t.1 = epoch;
            t.2.clear();
            t.3.clear();
        }
        t.2.push((inv, my_se));
        Ok(())
    }

    /// An atomic: unordered with plain accesses is a race, and with
    /// another atomic when either is order-sensitive.
    fn atomic(&mut self, at: usize, var: Id, index: u32, inv: usize, sensitive: bool) -> Result<(), EvalError> {
        let my_se = self.sg_epoch.get(self.sg(inv)).copied().unwrap_or(0);
        let t = self.words.entry((var, index)).or_default().clone();
        if let Some((e, w, se)) = t.0 {
            if self.conflicts(e, w, se, inv) {
                return Err(EvalError::Race { at, var, index });
            }
        }
        if t.1 == self.epoch
            && (t.2.iter().any(|&(r, se)| self.conflicts(self.epoch, r, se, inv))
                || t.3
                    .iter()
                    .any(|&(a, se, s)| (s || sensitive) && self.conflicts(self.epoch, a, se, inv)))
        {
            return Err(EvalError::Race { at, var, index });
        }
        let epoch = self.epoch;
        let t = self.words.get_mut(&(var, index)).unwrap();
        if t.1 != epoch {
            t.1 = epoch;
            t.2.clear();
            t.3.clear();
        }
        t.3.push((inv, my_se, sensitive));
        Ok(())
    }

    fn write(&mut self, at: usize, var: Id, index: u32, inv: usize) -> Result<(), EvalError> {
        let my_se = self.sg_epoch.get(self.sg(inv)).copied().unwrap_or(0);
        let t = self.words.entry((var, index)).or_default().clone();
        if let Some((e, w, se)) = t.0 {
            if self.conflicts(e, w, se, inv) {
                return Err(EvalError::Race { at, var, index });
            }
        }
        if t.1 == self.epoch
            && (t.2.iter().any(|&(r, se)| self.conflicts(self.epoch, r, se, inv))
                || t.3.iter().any(|&(a, se, _)| self.conflicts(self.epoch, a, se, inv)))
        {
            return Err(EvalError::Race { at, var, index });
        }
        let epoch = self.epoch;
        self.words.get_mut(&(var, index)).unwrap().0 = Some((epoch, inv, my_se));
        Ok(())
    }
}

struct Invocation {
    vals: HashMap<Id, Val>,
    mem: HashMap<Id, Store>,
    /// Discarded by OpKill (fragment shaders).
    killed: bool,
}

/// Run one subgroup of `invocations` invocations (invocation `i` has
/// `GlobalInvocationId = (i, 0, 0)` and subgroup invocation id `i`) in
/// lockstep against `buffers`.
pub fn run_compute(
    words: &[u32],
    invocations: u32,
    buffers: &mut Buffers,
    max_steps: usize,
) -> Result<(), EvalError> {
    run_dispatch(words, [1, 1, 1], [invocations, 1, 1], buffers, &[], max_steps)
}

/// [`run_dispatch_subgroups`] with one subgroup per work-group.
pub fn run_dispatch(
    words: &[u32],
    groups: [u32; 3],
    local: [u32; 3],
    buffers: &mut Buffers,
    push: &[u32],
    max_steps: usize,
) -> Result<(), EvalError> {
    let n = local[0] * local[1] * local[2];
    run_dispatch_subgroups(words, groups, local, n, buffers, push, max_steps)
}

/// Run a dispatch of `groups` work-groups of `local` invocations each, one
/// work-group after another, `subgroup` invocations per subgroup (local
/// index `i` is invocation `i % subgroup` of subgroup `i / subgroup`).
/// `push` is the push-constant block (u32 words). Workgroup memory is
/// fresh per group.
pub fn run_dispatch_subgroups(
    words: &[u32],
    groups: [u32; 3],
    local: [u32; 3],
    subgroup: u32,
    buffers: &mut Buffers,
    push: &[u32],
    max_steps: usize,
) -> Result<(), EvalError> {
    run_groups(words, groups, local, subgroup, buffers, push, None, max_steps).map(|_| ())
}

/// One fragment invocation's result: None if it was discarded, else its
/// outputs by Location (four words each).
pub type FragmentOut = Option<HashMap<u32, Vec<u32>>>;

/// Run a fragment shader over `frag_coord.len()` pixels as one subgroup
/// (invocation `i` has `FragCoord = frag_coord[i]`), in lockstep. A
/// branch that diverges is allowed only into a block that is `OpKill`.
pub fn run_fragment(
    words: &[u32],
    frag_coord: &[[f32; 4]],
    buffers: &mut Buffers,
    push: &[u32],
    max_steps: usize,
) -> Result<Vec<FragmentOut>, EvalError> {
    let input = FragmentInput { frag_coord, bary: &[], front: &[], attrs: &[] };
    run_fragment_with(words, &input, buffers, push, max_steps)
}

/// Key of the Position output in a vertex run's outputs.
pub const POSITION_KEY: u32 = u32::MAX;

/// One vertex invocation's draw-parameter inputs.
#[derive(Clone, Copy, Debug)]
pub struct VertexInput {
    pub vertex_index: u32,
    pub base_vertex: u32,
    pub instance_index: u32,
    pub base_instance: u32,
}

/// Run a vertex shader over `verts` as one subgroup; each invocation's
/// outputs by Location, Position under [`POSITION_KEY`].
pub fn run_vertex(
    words: &[u32],
    verts: &[VertexInput],
    buffers: &mut Buffers,
    push: &[u32],
    max_steps: usize,
) -> Result<Vec<HashMap<u32, Vec<u32>>>, EvalError> {
    let input = FragmentInput { frag_coord: &[], bary: &[], front: &[], attrs: &[] };
    let n = verts.len() as u32;
    let out = run_groups_v(words, [n, 1, 1], n, buffers, push, Some(&input), Some(verts), max_steps)?;
    Ok(out.into_iter().map(|o| o.unwrap_or_default()).collect())
}

/// Per-pixel and per-primitive inputs of a fragment run.
pub struct FragmentInput<'a> {
    pub frag_coord: &'a [[f32; 4]],
    /// BaryCoordKHR per pixel.
    pub bary: &'a [[f32; 3]],
    /// FrontFacing per pixel (missing: true).
    pub front: &'a [bool],
    /// Per-vertex attributes: (Location, three vertices' xyzw).
    pub attrs: &'a [(u32, [[f32; 4]; 3])],
}

pub fn run_fragment_with(
    words: &[u32],
    input: &FragmentInput,
    buffers: &mut Buffers,
    push: &[u32],
    max_steps: usize,
) -> Result<Vec<FragmentOut>, EvalError> {
    let n = input.frag_coord.len() as u32;
    run_groups(words, [1, 1, 1], [n, 1, 1], n, buffers, push, Some(input), max_steps)
}

#[allow(clippy::too_many_arguments)]
fn run_groups(
    words: &[u32],
    groups: [u32; 3],
    local: [u32; 3],
    subgroup: u32,
    buffers: &mut Buffers,
    push: &[u32],
    frag: Option<&FragmentInput>,
    max_steps: usize,
) -> Result<Vec<FragmentOut>, EvalError> {
    run_groups_inner(words, groups, local, subgroup, buffers, push, frag, None, max_steps)
}

#[allow(clippy::too_many_arguments)]
fn run_groups_v(
    words: &[u32],
    local: [u32; 3],
    subgroup: u32,
    buffers: &mut Buffers,
    push: &[u32],
    frag: Option<&FragmentInput>,
    verts: Option<&[VertexInput]>,
    max_steps: usize,
) -> Result<Vec<FragmentOut>, EvalError> {
    run_groups_inner(words, [1, 1, 1], local, subgroup, buffers, push, frag, verts, max_steps)
}

#[allow(clippy::too_many_arguments)]
fn run_groups_inner(
    words: &[u32],
    groups: [u32; 3],
    local: [u32; 3],
    subgroup: u32,
    buffers: &mut Buffers,
    push: &[u32],
    frag: Option<&FragmentInput>,
    verts: Option<&[VertexInput]>,
    max_steps: usize,
) -> Result<Vec<FragmentOut>, EvalError> {
    let mut frag_out = Vec::new();
    let m = load_module(words)?;
    let start = m
        .insts
        .iter()
        .position(|i| i.opcode == op::FUNCTION && i.operands[1] == m.entry)
        .ok_or(EvalError::NoEntryPoint)?;
    let invocations = local[0] * local[1] * local[2];
    for gz in 0..groups[2] {
        for gy in 0..groups[1] {
            for gx in 0..groups[0] {
                let wg = [gx, gy, gz];
                let mut bufs: HashMap<Id, Vec<u32>> = HashMap::new();
                for (var, key) in &m.bindings {
                    bufs.insert(*var, buffers.remove(key).unwrap_or_default());
                }
                for &(var, ptr_ty, sc) in &m.globals {
                    if let Some(&n) = m.ptr_to_shared_len.get(&ptr_ty) {
                        bufs.insert(var, vec![0; n as usize]);
                    }
                    if sc == storage::PUSH_CONSTANT {
                        bufs.insert(var, push.to_vec());
                    }
                }
                let mut invs: Vec<Invocation> = (0..invocations)
                    .map(|inv| {
                        let lid = [inv % local[0], inv / local[0] % local[1], inv / (local[0] * local[1])];
                        let mut mem = HashMap::new();
                        let mut vals = m.constants.clone();
                        for &(var, ptr_ty, sc) in &m.globals {
                            let s = if let Some(&b) = m.builtins.get(&var) {
                                match b {
                                    builtin::GLOBAL_INVOCATION_ID => Store::Vec(
                                        (0..3).map(|d| wg[d] * local[d] + lid[d]).collect(),
                                    ),
                                    builtin::WORKGROUP_ID => Store::Vec(wg.to_vec()),
                                    builtin::LOCAL_INVOCATION_ID => Store::Vec(lid.to_vec()),
                                    builtin::LOCAL_INVOCATION_INDEX => Store::Word(inv),
                                    builtin::SUBGROUP_LOCAL_INVOCATION_ID => Store::Word(inv % subgroup),
                                    builtin::SUBGROUP_ID => Store::Word(inv / subgroup),
                                    builtin::NUM_SUBGROUPS => Store::Word(invocations.div_ceil(subgroup)),
                                    builtin::SUBGROUP_SIZE => Store::Word(subgroup),
                                    builtin::FRAG_COORD => Store::Vec(
                                        frag.and_then(|f| f.frag_coord.get(inv as usize))
                                            .map(|c| c.iter().map(|x| x.to_bits()).collect())
                                            .unwrap_or(vec![0; 4]),
                                    ),
                                    builtin::POSITION => Store::Vec(vec![0; 4]),
                                    builtin::FRONT_FACING => Store::Bool(
                                        frag.is_none_or(|f| f.front.get(inv as usize).copied().unwrap_or(true)),
                                    ),
                                    builtin::VERTEX_INDEX
                                    | builtin::BASE_VERTEX
                                    | builtin::INSTANCE_INDEX
                                    | builtin::BASE_INSTANCE => {
                                        let v = verts.and_then(|v| v.get(inv as usize));
                                        Store::Word(v.map_or(0, |v| match b {
                                            builtin::VERTEX_INDEX => v.vertex_index,
                                            builtin::BASE_VERTEX => v.base_vertex,
                                            builtin::INSTANCE_INDEX => v.instance_index,
                                            _ => v.base_instance,
                                        }))
                                    }
                                    builtin::BARY_COORD_KHR => Store::Vec(
                                        frag.and_then(|f| f.bary.get(inv as usize))
                                            .map(|c| c.iter().map(|x| x.to_bits()).collect())
                                            .unwrap_or(vec![0; 3]),
                                    ),
                                    _ => Store::Vec(vec![0, 0, 0]),
                                }
                            } else if m.bindings.contains_key(&var) || sc == storage::PUSH_CONSTANT {
                                Store::Buffer
                            } else if m.ptr_to_vec4_array.contains(&ptr_ty) {
                                let loc = m.locations.get(&var).copied();
                                let data = frag
                                    .and_then(|f| f.attrs.iter().find(|(l, _)| Some(*l) == loc))
                                    .map(|(_, a)| a.iter().flatten().map(|x| x.to_bits()).collect())
                                    .unwrap_or(vec![0; 12]);
                                Store::Vec4Array(data)
                            } else if m.ptr_to_shared_len.contains_key(&ptr_ty) {
                                Store::Shared
                            } else if let Some(&n) = m.ptr_to_private_len.get(&ptr_ty) {
                                Store::Vec(vec![0; n as usize])
                            } else if m.ptr_to_bool.get(&ptr_ty) == Some(&true) {
                                Store::Bool(false)
                            } else if m.ptr_to_vec.get(&ptr_ty) == Some(&true) {
                                Store::Vec(vec![0; 4])
                            } else {
                                Store::Word(0)
                            };
                            mem.insert(var, s);
                            vals.insert(var, Val::Ptr { var, index: None });
                        }
                        Invocation { vals, mem, killed: false }
                    })
                    .collect();
                let r = run(&m, start, &mut invs, &mut bufs, subgroup as usize, max_steps);
                for (var, key) in &m.bindings {
                    if let Some(b) = bufs.remove(var) {
                        buffers.insert(*key, b);
                    }
                }
                r?;
                if m.fragment || m.vertex {
                    for inv in &invs {
                        frag_out.push((!inv.killed).then(|| {
                            let mut o: HashMap<u32, Vec<u32>> = m
                                .locations
                                .iter()
                                .filter_map(|(var, loc)| match inv.mem.get(var) {
                                    Some(Store::Vec(v)) => Some((*loc, v.clone())),
                                    _ => None,
                                })
                                .collect();
                            if let Some(Store::Vec(p)) = m.position.and_then(|p| inv.mem.get(&p)) {
                                o.insert(POSITION_KEY, p.clone());
                            }
                            o
                        }));
                    }
                }
            }
        }
    }
    Ok(frag_out)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SgState {
    Running,
    AtBarrier,
    Done,
}

/// Run the invocations of one work-group, `sg_size` per subgroup. Each
/// subgroup runs in lockstep with its own program counter; a
/// workgroup-scope `OpControlBarrier` suspends it until every subgroup has
/// arrived (a subgroup that returns while others wait is an error).
fn run(
    m: &Module,
    start: usize,
    invs: &mut [Invocation],
    bufs: &mut HashMap<Id, Vec<u32>>,
    sg_size: usize,
    max_steps: usize,
) -> Result<(), EvalError> {
    let n_sg = invs.len().div_ceil(sg_size.max(1));
    let mut races = Races {
        sg_epoch: vec![0; n_sg],
        sg_size,
        ..Default::default()
    };
    let mut pcs = vec![start + 1; n_sg];
    let mut states = vec![SgState::Running; n_sg];
    let mut steps = 0usize;
    loop {
        for g in 0..n_sg {
            if states[g] != SgState::Running {
                continue;
            }
            let lo = g * sg_size;
            let hi = (lo + sg_size).min(invs.len());
            states[g] = run_subgroup(
                m,
                &mut pcs[g],
                &mut invs[lo..hi],
                lo,
                g,
                bufs,
                &mut races,
                &mut steps,
                max_steps,
            )?;
        }
        if states.iter().all(|s| *s == SgState::Done) {
            return Ok(());
        }
        if states.contains(&SgState::Done) {
            return Err(EvalError::Unsupported {
                at: 0,
                opcode: op::CONTROL_BARRIER,
                why: "workgroup barrier not reached by every subgroup",
            });
        }
        // Everyone is at the barrier: release.
        races.epoch += 1;
        for s in states.iter_mut() {
            *s = SgState::Running;
        }
    }
}

/// Run one subgroup until it returns or reaches a workgroup barrier.
#[allow(clippy::too_many_arguments)]
fn run_subgroup(
    m: &Module,
    pc: &mut usize,
    invs: &mut [Invocation],
    first: usize,
    sg: usize,
    bufs: &mut HashMap<Id, Vec<u32>>,
    races: &mut Races,
    steps: &mut usize,
    max_steps: usize,
) -> Result<SgState, EvalError> {
    loop {
        *steps += 1;
        if *steps > max_steps {
            return Err(EvalError::StepLimit);
        }
        let i = &m.insts[*pc];
        let o = &i.operands;
        let unsupported = |why| EvalError::Unsupported {
            at: i.at,
            opcode: i.opcode,
            why,
        };
        *pc += 1;
        match i.opcode {
            op::LABEL | op::SELECTION_MERGE | op::LOOP_MERGE => continue,
            op::CONTROL_BARRIER | op::MEMORY_BARRIER => {
                let exec_scope = match invs[0].vals.get(&o[0]) {
                    Some(Val::Word(x)) => *x,
                    _ => return Err(unsupported("barrier scope")),
                };
                if i.opcode == op::CONTROL_BARRIER && exec_scope == scope::WORKGROUP {
                    return Ok(SgState::AtBarrier);
                }
                // Subgroup-scope barrier (or a memory barrier): orders this
                // subgroup's accesses; lockstep already synchronises it.
                races.sg_epoch[sg] += 1;
                continue;
            }
            op::RETURN => return Ok(SgState::Done),
            op::KILL => {
                for inv in invs.iter_mut() {
                    inv.killed = true;
                }
                return Ok(SgState::Done);
            }
            op::BRANCH | op::BRANCH_CONDITIONAL => {
                // A divergent branch is allowed only into an OpKill block:
                // those invocations are discarded, the rest go on.
                if i.opcode == op::BRANCH_CONDITIONAL {
                    let is_kill = |t: Id| {
                        m.labels
                            .get(&t)
                            .and_then(|&l| m.insts.get(l + 1))
                            .is_some_and(|x| x.opcode == op::KILL)
                    };
                    let mut to = [false, false];
                    for inv in invs.iter().filter(|x| !x.killed) {
                        match inv.vals.get(&o[0]) {
                            Some(Val::Bool(true)) => to[0] = true,
                            Some(Val::Bool(false)) => to[1] = true,
                            _ => return Err(unsupported("branch condition")),
                        }
                    }
                    if to[0] && to[1] {
                        let (kill_on, other) = if is_kill(o[2]) {
                            (false, o[1])
                        } else if is_kill(o[1]) {
                            (true, o[2])
                        } else {
                            return Err(EvalError::Divergent { at: i.at });
                        };
                        for inv in invs.iter_mut().filter(|x| !x.killed) {
                            if inv.vals.get(&o[0]) == Some(&Val::Bool(kill_on)) {
                                inv.killed = true;
                            }
                        }
                        *pc = *m.labels.get(&other).ok_or(unsupported("branch target"))?;
                        continue;
                    }
                }
                let mut target = None;
                for inv in invs.iter().filter(|x| !x.killed) {
                    let t = if i.opcode == op::BRANCH {
                        o[0]
                    } else {
                        match inv.vals.get(&o[0]) {
                            Some(Val::Bool(true)) => o[1],
                            Some(Val::Bool(false)) => o[2],
                            _ => return Err(unsupported("branch condition")),
                        }
                    };
                    if target.is_some_and(|x| x != t) {
                        return Err(EvalError::Divergent { at: i.at });
                    }
                    target = Some(t);
                }
                let t = target.ok_or(unsupported("no invocations"))?;
                *pc = *m.labels.get(&t).ok_or(unsupported("branch target"))?;
                continue;
            }
            op::GROUP_NON_UNIFORM_BALLOT => {
                check_scope(&invs[0], o[2]).map_err(|_| unsupported("scope"))?;
                let mut mask = [0u32; 4];
                for (k, inv) in invs.iter().enumerate() {
                    match inv.vals.get(&o[3]) {
                        Some(Val::Bool(true)) => mask[k / 32] |= 1 << (k % 32),
                        Some(Val::Bool(false)) => {}
                        _ => return Err(unsupported("ballot predicate")),
                    }
                }
                for inv in invs.iter_mut() {
                    inv.vals.insert(o[1], Val::Vec(mask.to_vec()));
                }
                continue;
            }
            op::GROUP_NON_UNIFORM_SHUFFLE => {
                check_scope(&invs[0], o[2]).map_err(|_| unsupported("scope"))?;
                let values: Vec<Val> = invs
                    .iter()
                    .map(|inv| inv.vals.get(&o[3]).cloned())
                    .collect::<Option<_>>()
                    .ok_or(unsupported("shuffle value"))?;
                for inv in invs.iter_mut() {
                    let Some(Val::Word(idx)) = inv.vals.get(&o[4]) else {
                        return Err(unsupported("shuffle index"));
                    };
                    // Out-of-range ids are undefined in SPIR-V; refuse them.
                    let v = values
                        .get(*idx as usize)
                        .cloned()
                        .ok_or(unsupported("shuffle index out of range"))?;
                    inv.vals.insert(o[1], v);
                }
                continue;
            }
            _ => {}
        }
        for (k, inv) in invs.iter_mut().enumerate() {
            if !inv.killed {
                step(m, i, inv, bufs, races, first + k)?;
            }
        }
    }
}

fn check_scope(inv: &Invocation, id: Id) -> Result<(), ()> {
    match inv.vals.get(&id) {
        Some(Val::Word(scope::SUBGROUP)) => Ok(()),
        _ => Err(()),
    }
}

fn step(
    m: &Module,
    i: &Instruction,
    inv: &mut Invocation,
    bufs: &mut HashMap<Id, Vec<u32>>,
    races: &mut Races,
    me: usize,
) -> Result<(), EvalError> {
    let o = &i.operands;
    let unsupported = |why| EvalError::Unsupported {
        at: i.at,
        opcode: i.opcode,
        why,
    };
    let vals = &inv.vals;
    let get = |id: Id| vals.get(&id).cloned().ok_or(unsupported("value not computed"));
    let word = |id: Id| match get(id)? {
        Val::Word(w) => Ok(w),
        Val::Undef => Err(unsupported("use of an undefined value")),
        _ => Err(unsupported("expected a 32-bit value")),
    };
    let boolean = |id: Id| match get(id)? {
        Val::Bool(b) => Ok(b),
        _ => Err(unsupported("expected a bool")),
    };
    let f = |id: Id| word(id).map(f32::from_bits);
    let w2 = |g: fn(u32, u32) -> u32| Ok::<_, EvalError>(Some(Val::Word(g(word(o[2])?, word(o[3])?))));
    let b2 = |g: fn(u32, u32) -> bool| Ok::<_, EvalError>(Some(Val::Bool(g(word(o[2])?, word(o[3])?))));
    let fb = |g: fn(f32, f32) -> bool| Ok::<_, EvalError>(Some(Val::Bool(g(f(o[2])?, f(o[3])?))));
    let ff = |g: fn(f32, f32) -> f32| {
        Ok::<_, EvalError>(Some(Val::Word(g(f(o[2])?, f(o[3])?).to_bits())))
    };
    let double = |id: Id| match get(id)? {
        Val::Vec(v) if v.len() == 2 => Ok(f64::from_bits(v[0] as u64 | (v[1] as u64) << 32)),
        _ => Err(unsupported("expected a double")),
    };
    let dval = |x: f64| {
        let r = x.to_bits();
        Val::Vec(vec![r as u32, (r >> 32) as u32])
    };
    if m.f64_types.contains(&o[0]) && matches!(i.opcode, op::CONVERT_S_TO_F | op::CONVERT_U_TO_F | op::F_CONVERT) {
        let x = word(o[2])?;
        let r = match i.opcode {
            op::CONVERT_S_TO_F => x as i32 as f64,
            op::CONVERT_U_TO_F => x as f64,
            _ => f32::from_bits(x) as f64,
        };
        inv.vals.insert(o[1], dval(r));
        return Ok(());
    }
    if matches!(i.opcode, op::CONVERT_F_TO_S | op::CONVERT_F_TO_U | op::F_CONVERT) && matches!(get(o[2])?, Val::Vec(_)) {
        let x = double(o[2])?;
        let r = match i.opcode {
            // round to nearest even, as Vulkan's correctly rounded OpFConvert
            op::F_CONVERT => Val::Word((x as f32).to_bits()),
            op::CONVERT_F_TO_S if x > -2_147_483_649.0 && x < 2_147_483_648.0 => Val::Word(x as i32 as u32),
            op::CONVERT_F_TO_U if x > -1.0 && x < 4_294_967_296.0 => Val::Word(x as u32),
            _ => Val::Undef,
        };
        inv.vals.insert(o[1], r);
        return Ok(());
    }
    if m.f64_types.contains(&o[0]) && matches!(i.opcode, op::F_ADD | op::F_MUL | op::EXT_INST) {
        let d = |id: Id| match get(id)? {
            Val::Vec(v) if v.len() == 2 => Ok(f64::from_bits(v[0] as u64 | (v[1] as u64) << 32)),
            _ => Err(unsupported("expected a double")),
        };
        let r = match i.opcode {
            op::F_ADD => d(o[2])? + d(o[3])?,
            op::F_MUL => d(o[2])? * d(o[3])?,
            _ => {
                if !m.glsl.contains(&o[2]) || o[3] != crate::spirv::glsl::FMA || o.len() != 7 {
                    return Err(unsupported("f64 extended instruction"));
                }
                // ASSUMPTION as for f32 Fma: the driver fuses.
                d(o[4])?.mul_add(d(o[5])?, d(o[6])?)
            }
        }
        .to_bits();
        inv.vals.insert(o[1], Val::Vec(vec![r as u32, (r >> 32) as u32]));
        return Ok(());
    }
    let result = match i.opcode {
        op::VARIABLE => {
            // Function-local variables (globals were set up at start).
            let s = Store::Word(0);
            inv.mem.insert(o[1], s);
            Some(Val::Ptr { var: o[1], index: None })
        }
        op::ACCESS_CHAIN => {
            let idx = &o[3..];
            let shared = matches!(inv.mem.get(&o[2]), Some(Store::Shared | Store::Vec(_) | Store::Vec4Array(_)));
            let index = match (shared, idx) {
                (true, [i]) => word(*i)?,
                (false, [z, i]) if word(*z)? == 0 => word(*i)?,
                _ => return Err(unsupported("access chain shape")),
            };
            Some(Val::Ptr {
                var: o[2],
                index: Some(index),
            })
        }
        op::LOAD => {
            let Val::Ptr { var, index } = get(o[2])? else {
                return Err(unsupported("load through a non-pointer"));
            };
            let s = inv.mem.get(&var).ok_or(unsupported("unknown variable"))?;
            Some(match (s, index) {
                (Store::Word(w), None) => Val::Word(*w),
                (Store::Bool(b), None) => Val::Bool(*b),
                (Store::Vec(v), None) => Val::Vec(v.clone()),
                (Store::Vec4Array(v), Some(k)) => Val::Vec(
                    v.get(4 * k as usize..4 * k as usize + 4)
                        .ok_or(unsupported("array read out of bounds"))?
                        .to_vec(),
                ),
                (Store::Vec(v), Some(k)) => {
                    Val::Word(*v.get(k as usize).ok_or(unsupported("array read out of bounds"))?)
                }
                (Store::Buffer | Store::Shared, Some(k)) => {
                    if matches!(s, Store::Shared) {
                        races.read(i.at, var, k, me)?;
                    }
                    let b = bufs.get(&var).ok_or(unsupported("unbound buffer"))?;
                    Val::Word(*b.get(k as usize).ok_or(unsupported("buffer read out of bounds"))?)
                }
                _ => return Err(unsupported("load shape")),
            })
        }
        op::STORE => {
            let Val::Ptr { var, index } = get(o[0])? else {
                return Err(unsupported("store through a non-pointer"));
            };
            let v = get(o[1])?;
            if v == Val::Undef {
                return Err(unsupported("store of an undefined value"));
            }
            let s = inv.mem.get_mut(&var).ok_or(unsupported("unknown variable"))?;
            match (s, index, v) {
                (Store::Word(w), None, Val::Word(x)) => *w = x,
                (Store::Bool(b), None, Val::Bool(x)) => *b = x,
                (Store::Vec(v), None, Val::Vec(x)) if v.len() == x.len() => *v = x,
                (Store::Vec(v), Some(k), Val::Word(x)) => {
                    *v.get_mut(k as usize).ok_or(unsupported("array write out of bounds"))? = x
                }
                (st @ (Store::Buffer | Store::Shared), Some(k), Val::Word(x)) => {
                    if matches!(st, Store::Shared) {
                        races.write(i.at, var, k, me)?;
                    }
                    let b = bufs.get_mut(&var).ok_or(unsupported("unbound buffer"))?;
                    *b.get_mut(k as usize)
                        .ok_or(unsupported("buffer write out of bounds"))? = x
                }
                _ => return Err(unsupported("store shape")),
            }
            None
        }
        op::ATOMIC_EXCHANGE..=op::ATOMIC_COMPARE_EXCHANGE | op::ATOMIC_I_ADD..=op::ATOMIC_XOR => {
            let Val::Ptr { var, index: Some(k) } = get(o[2])? else {
                return Err(unsupported("atomic through a non-pointer"));
            };
            match inv.mem.get(&var) {
                Some(Store::Buffer) => {}
                // Atomics on one word are ordered among themselves, except
                // that exchange / compare-exchange results depend on it.
                Some(Store::Shared) => races.atomic(
                    i.at,
                    var,
                    k,
                    me,
                    matches!(i.opcode, op::ATOMIC_EXCHANGE | op::ATOMIC_COMPARE_EXCHANGE),
                )?,
                _ => return Err(unsupported("atomic outside a storage buffer or workgroup memory")),
            }
            let cmpx = i.opcode == op::ATOMIC_COMPARE_EXCHANGE;
            let (v, cmp) = if cmpx { (word(o[6])?, word(o[7])?) } else { (word(o[5])?, 0) };
            let b = bufs.get_mut(&var).ok_or(unsupported("unbound buffer"))?;
            let slot = b.get_mut(k as usize).ok_or(unsupported("buffer atomic out of bounds"))?;
            let old = *slot;
            *slot = match i.opcode {
                op::ATOMIC_EXCHANGE => v,
                op::ATOMIC_COMPARE_EXCHANGE => if old == cmp { v } else { old },
                op::ATOMIC_I_ADD => old.wrapping_add(v),
                op::ATOMIC_I_SUB => old.wrapping_sub(v),
                op::ATOMIC_S_MIN => (old as i32).min(v as i32) as u32,
                op::ATOMIC_U_MIN => old.min(v),
                op::ATOMIC_S_MAX => (old as i32).max(v as i32) as u32,
                op::ATOMIC_U_MAX => old.max(v),
                op::ATOMIC_AND => old & v,
                op::ATOMIC_OR => old | v,
                op::ATOMIC_XOR => old ^ v,
                _ => return Err(unsupported("atomic opcode")),
            };
            Some(Val::Word(old))
        }
        op::COMPOSITE_CONSTRUCT => Some(Val::Vec(
            o[2..].iter().map(|&id| word(id)).collect::<Result<_, _>>()?,
        )),
        op::COMPOSITE_EXTRACT => match get(o[2])? {
            Val::Vec(v) => Some(Val::Word(
                *v.get(o[3] as usize).ok_or(unsupported("extract index"))?,
            )),
            _ => return Err(unsupported("extract from a non-vector")),
        },
        // Bitcast passes an undefined value through unchanged.
        op::BITCAST => Some(match get(o[2])? {
            Val::Undef => Val::Undef,
            // uvec2 <-> double: the same two words.
            Val::Vec(v) if v.len() == 2 => Val::Vec(v),
            _ => Val::Word(word(o[2])?),
        }),
        op::CONVERT_S_TO_F => Some(Val::Word((word(o[2])? as i32 as f32).to_bits())),
        op::CONVERT_U_TO_F => Some(Val::Word((word(o[2])? as f32).to_bits())),
        // Out-of-range float-to-int conversions give an undefined value in
        // SPIR-V: model it as Undef rather than picking a result.
        op::CONVERT_F_TO_U => {
            let x = f(o[2])?;
            Some(if x > -1.0 && x < 4_294_967_296.0 {
                Val::Word(x as u32)
            } else {
                Val::Undef
            })
        }
        op::CONVERT_F_TO_S => {
            let x = f(o[2])?;
            Some(if x > -2_147_483_649.0 && x < 2_147_483_648.0 {
                Val::Word(x as i32 as u32)
            } else {
                Val::Undef
            })
        }
        op::I_ADD => w2(u32::wrapping_add)?,
        op::I_SUB => w2(u32::wrapping_sub)?,
        op::I_MUL => w2(u32::wrapping_mul)?,
        op::U_DIV => {
            let (a, b) = (word(o[2])?, word(o[3])?);
            if b == 0 {
                return Err(unsupported("division by zero"));
            }
            Some(Val::Word(a / b))
        }
        op::BITWISE_AND => w2(|a, b| a & b)?,
        op::BITWISE_OR => w2(|a, b| a | b)?,
        op::BITWISE_XOR => w2(|a, b| a ^ b)?,
        // Shifts by >= 32 are undefined in SPIR-V; refuse rather than pick.
        op::SHIFT_LEFT_LOGICAL | op::SHIFT_RIGHT_LOGICAL | op::SHIFT_RIGHT_ARITHMETIC => {
            let (a, s) = (word(o[2])?, word(o[3])?);
            if s >= 32 {
                return Err(unsupported("shift amount >= 32"));
            }
            Some(Val::Word(match i.opcode {
                op::SHIFT_LEFT_LOGICAL => a << s,
                op::SHIFT_RIGHT_LOGICAL => a >> s,
                _ => ((a as i32) >> s) as u32,
            }))
        }
        op::NOT => Some(Val::Word(!word(o[2])?)),
        op::BIT_COUNT => Some(Val::Word(word(o[2])?.count_ones())),
        op::F_NEGATE => Some(Val::Word((-f(o[2])?).to_bits())),
        op::F_ADD => ff(|a, b| a + b)?,
        op::F_SUB => ff(|a, b| a - b)?,
        op::F_MUL => ff(|a, b| a * b)?,
        op::F_ORD_LESS_THAN => fb(|a, b| a < b)?,
        op::F_ORD_EQUAL => fb(|a, b| a == b)?,
        op::F_ORD_GREATER_THAN => fb(|a, b| a > b)?,
        op::F_ORD_GREATER_THAN_EQUAL => fb(|a, b| a >= b)?,
        // Ordered: false if either is NaN. Unordered: true if either is
        // NaN (Rust's `!=` already is the unordered form).
        op::F_ORD_NOT_EQUAL => fb(|a, b| !a.is_nan() && !b.is_nan() && a != b)?,
        op::F_UNORD_NOT_EQUAL => fb(|a, b| a != b)?,
        op::F_ORD_LESS_THAN_EQUAL => fb(|a, b| a <= b)?,
        op::S_LESS_THAN => b2(|a, b| (a as i32) < (b as i32))?,
        op::S_LESS_THAN_EQUAL => b2(|a, b| (a as i32) <= (b as i32))?,
        op::S_GREATER_THAN => b2(|a, b| (a as i32) > (b as i32))?,
        op::S_GREATER_THAN_EQUAL => b2(|a, b| (a as i32) >= (b as i32))?,
        op::U_LESS_THAN => b2(|a, b| a < b)?,
        op::U_LESS_THAN_EQUAL => b2(|a, b| a <= b)?,
        op::U_GREATER_THAN => b2(|a, b| a > b)?,
        op::U_GREATER_THAN_EQUAL => b2(|a, b| a >= b)?,
        op::I_EQUAL => b2(|a, b| a == b)?,
        op::I_NOT_EQUAL => b2(|a, b| a != b)?,
        op::LOGICAL_NOT => Some(Val::Bool(!boolean(o[2])?)),
        op::LOGICAL_AND => Some(Val::Bool(boolean(o[2])? & boolean(o[3])?)),
        op::LOGICAL_OR => Some(Val::Bool(boolean(o[2])? | boolean(o[3])?)),
        op::SELECT => Some(if boolean(o[2])? { get(o[3])? } else { get(o[4])? }),
        op::EXT_INST => {
            use crate::spirv::glsl;
            if !m.glsl.contains(&o[2]) {
                return Err(unsupported("extended instruction set"));
            }
            if o[3] == glsl::FMA && o.len() == 7 {
                // ASSUMPTION: the driver fuses (single rounding), as the
                // hardware v_fma does; Vulkan only guarantees mul + add
                // precision for Fma.
                let r = f(o[4])?.mul_add(f(o[5])?, f(o[6])?);
                inv.vals.insert(o[1], Val::Word(r.to_bits()));
                return Ok(());
            }
            if o.len() != 5 {
                return Err(unsupported("extended instruction arity"));
            }
            let x = f(o[4])?;
            let r = match o[3] {
                glsl::FLOOR => x.floor(),
                glsl::CEIL => x.ceil(),
                glsl::TRUNC => x.trunc(),
                glsl::ROUND_EVEN => x.round_ties_even(),
                _ => return Err(unsupported("GLSL.std.450 instruction")),
            };
            Some(Val::Word(r.to_bits()))
        }
        op::UNREACHABLE => return Err(unsupported("reached OpUnreachable")),
        _ => return Err(unsupported("opcode")),
    };
    if let Some(v) = result {
        inv.vals.insert(o[1], v);
    }
    Ok(())
}
