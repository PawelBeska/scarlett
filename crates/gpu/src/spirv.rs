//! SPIR-V module builder and a structural validator for its output.
//!
//! The shader translator emits through [`Builder`]: ids are allocated
//! here, types and constants are deduplicated, and instructions land in the
//! section the logical layout requires (capabilities, memory model, entry
//! points, execution modes, debug names, annotations, types/constants/
//! globals, functions), whatever order the translator produces them in.
//!
//! [`validate`] re-reads a finished module and checks what can go wrong in
//! a hand-written emitter: header, word counts, id bound, single
//! definition and prior declaration of every id, section order, block
//! structure (every block opens with `OpLabel` and ends with exactly one
//! terminator) and function-local variables at the top of the entry
//! block. It is no replacement for `spirv-val`, which also checks types
//! and the Vulkan environment rules; run that when it is available.
//!
//! Every numeric value here was read from Khronos' `spirv.h`
//! (SPIRV-Headers, `include/spirv/unified1`).

use std::collections::HashMap;

pub const MAGIC: u32 = 0x0723_0203;
/// SPIR-V 1.3 (`major << 16 | minor << 8`), the first version with the
/// `GroupNonUniform*` instructions in core; Vulkan 1.1 accepts it.
pub const VERSION_1_3: u32 = 0x0001_0300;
/// Generator magic: 0 = unregistered tool.
pub const GENERATOR: u32 = 0;
pub const WORD_COUNT_SHIFT: u32 = 16;
pub const OPCODE_MASK: u32 = 0xFFFF;

/// Opcodes (`SpvOp*`).
pub mod op {
    pub const NOP: u16 = 0;
    pub const NAME: u16 = 5;
    pub const EXTENSION: u16 = 10;
    pub const MEMBER_NAME: u16 = 6;
    pub const EXT_INST_IMPORT: u16 = 11;
    pub const EXT_INST: u16 = 12;
    pub const MEMORY_MODEL: u16 = 14;
    pub const ENTRY_POINT: u16 = 15;
    pub const EXECUTION_MODE: u16 = 16;
    pub const CAPABILITY: u16 = 17;
    pub const TYPE_VOID: u16 = 19;
    pub const TYPE_BOOL: u16 = 20;
    pub const TYPE_INT: u16 = 21;
    pub const TYPE_FLOAT: u16 = 22;
    pub const TYPE_VECTOR: u16 = 23;
    pub const TYPE_ARRAY: u16 = 28;
    pub const TYPE_RUNTIME_ARRAY: u16 = 29;
    pub const TYPE_STRUCT: u16 = 30;
    pub const TYPE_POINTER: u16 = 32;
    pub const TYPE_FUNCTION: u16 = 33;
    pub const CONSTANT_TRUE: u16 = 41;
    pub const CONSTANT_FALSE: u16 = 42;
    pub const CONSTANT: u16 = 43;
    pub const CONSTANT_COMPOSITE: u16 = 44;
    pub const FUNCTION: u16 = 54;
    pub const FUNCTION_PARAMETER: u16 = 55;
    pub const FUNCTION_END: u16 = 56;
    pub const FUNCTION_CALL: u16 = 57;
    pub const VARIABLE: u16 = 59;
    pub const LOAD: u16 = 61;
    pub const STORE: u16 = 62;
    pub const ACCESS_CHAIN: u16 = 65;
    pub const DECORATE: u16 = 71;
    pub const MEMBER_DECORATE: u16 = 72;
    pub const COMPOSITE_CONSTRUCT: u16 = 80;
    pub const COMPOSITE_EXTRACT: u16 = 81;
    pub const CONVERT_F_TO_U: u16 = 109;
    pub const CONVERT_F_TO_S: u16 = 110;
    pub const F_CONVERT: u16 = 115;
    pub const CONVERT_S_TO_F: u16 = 111;
    pub const CONVERT_U_TO_F: u16 = 112;
    pub const BITCAST: u16 = 124;
    pub const S_NEGATE: u16 = 126;
    pub const F_NEGATE: u16 = 127;
    pub const I_ADD: u16 = 128;
    pub const F_ADD: u16 = 129;
    pub const I_SUB: u16 = 130;
    pub const F_SUB: u16 = 131;
    pub const I_MUL: u16 = 132;
    pub const F_MUL: u16 = 133;
    pub const U_DIV: u16 = 134;
    pub const S_DIV: u16 = 135;
    pub const F_DIV: u16 = 136;
    pub const I_ADD_CARRY: u16 = 149;
    pub const LOGICAL_OR: u16 = 166;
    pub const LOGICAL_AND: u16 = 167;
    pub const LOGICAL_NOT: u16 = 168;
    pub const SELECT: u16 = 169;
    pub const I_EQUAL: u16 = 170;
    pub const I_NOT_EQUAL: u16 = 171;
    pub const U_GREATER_THAN: u16 = 172;
    pub const S_GREATER_THAN: u16 = 173;
    pub const U_GREATER_THAN_EQUAL: u16 = 174;
    pub const S_GREATER_THAN_EQUAL: u16 = 175;
    pub const U_LESS_THAN: u16 = 176;
    pub const S_LESS_THAN: u16 = 177;
    pub const U_LESS_THAN_EQUAL: u16 = 178;
    pub const S_LESS_THAN_EQUAL: u16 = 179;
    pub const F_ORD_EQUAL: u16 = 180;
    pub const F_ORD_NOT_EQUAL: u16 = 182;
    pub const F_UNORD_NOT_EQUAL: u16 = 183;
    pub const F_ORD_LESS_THAN: u16 = 184;
    pub const F_ORD_GREATER_THAN: u16 = 186;
    pub const F_ORD_LESS_THAN_EQUAL: u16 = 188;
    pub const F_ORD_GREATER_THAN_EQUAL: u16 = 190;
    pub const SHIFT_RIGHT_LOGICAL: u16 = 194;
    pub const SHIFT_RIGHT_ARITHMETIC: u16 = 195;
    pub const SHIFT_LEFT_LOGICAL: u16 = 196;
    pub const BITWISE_OR: u16 = 197;
    pub const BITWISE_XOR: u16 = 198;
    pub const BITWISE_AND: u16 = 199;
    pub const NOT: u16 = 200;
    pub const BIT_COUNT: u16 = 205;
    pub const CONTROL_BARRIER: u16 = 224;
    pub const MEMORY_BARRIER: u16 = 225;
    pub const PHI: u16 = 245;
    pub const LOOP_MERGE: u16 = 246;
    pub const SELECTION_MERGE: u16 = 247;
    pub const LABEL: u16 = 248;
    pub const BRANCH: u16 = 249;
    pub const BRANCH_CONDITIONAL: u16 = 250;
    pub const RETURN: u16 = 253;
    pub const RETURN_VALUE: u16 = 254;
    pub const KILL: u16 = 252;
    pub const ATOMIC_EXCHANGE: u16 = 229;
    pub const ATOMIC_COMPARE_EXCHANGE: u16 = 230;
    pub const ATOMIC_I_ADD: u16 = 234;
    pub const ATOMIC_I_SUB: u16 = 235;
    pub const ATOMIC_S_MIN: u16 = 236;
    pub const ATOMIC_U_MIN: u16 = 237;
    pub const ATOMIC_S_MAX: u16 = 238;
    pub const ATOMIC_U_MAX: u16 = 239;
    pub const ATOMIC_AND: u16 = 240;
    pub const ATOMIC_OR: u16 = 241;
    pub const ATOMIC_XOR: u16 = 242;
    pub const UNREACHABLE: u16 = 255;
    pub const GROUP_NON_UNIFORM_ELECT: u16 = 333;
    pub const GROUP_NON_UNIFORM_BROADCAST: u16 = 337;
    pub const GROUP_NON_UNIFORM_BROADCAST_FIRST: u16 = 338;
    pub const GROUP_NON_UNIFORM_BALLOT: u16 = 339;
    pub const GROUP_NON_UNIFORM_SHUFFLE: u16 = 345;
}

/// GLSL.std.450 extended instructions (SPIRV-Headers `GLSL.std.450.h`).
pub mod glsl {
    pub const NAME: &str = "GLSL.std.450";
    pub const ROUND_EVEN: u32 = 2;
    pub const TRUNC: u32 = 3;
    pub const FLOOR: u32 = 8;
    pub const CEIL: u32 = 9;
    pub const FMA: u32 = 50;
}

pub mod capability {
    pub const SHADER: u32 = 1;
    pub const FLOAT64: u32 = 10;
    pub const INT64: u32 = 11;
    pub const GROUP_NON_UNIFORM: u32 = 61;
    pub const GROUP_NON_UNIFORM_BALLOT: u32 = 64;
    pub const GROUP_NON_UNIFORM_SHUFFLE: u32 = 65;
    pub const FRAGMENT_BARYCENTRIC_KHR: u32 = 5284;
    pub const DRAW_PARAMETERS: u32 = 4427;
}

pub mod addressing {
    pub const LOGICAL: u32 = 0;
}

pub mod memory_model {
    pub const GLSL450: u32 = 1;
    pub const VULKAN: u32 = 3;
}

pub mod execution_model {
    pub const VERTEX: u32 = 0;
    pub const FRAGMENT: u32 = 4;
    pub const GL_COMPUTE: u32 = 5;
}

pub mod execution_mode {
    pub const ORIGIN_UPPER_LEFT: u32 = 7;
    pub const LOCAL_SIZE: u32 = 17;
}

pub mod storage {
    pub const UNIFORM_CONSTANT: u32 = 0;
    pub const INPUT: u32 = 1;
    pub const UNIFORM: u32 = 2;
    pub const OUTPUT: u32 = 3;
    pub const WORKGROUP: u32 = 4;
    pub const PRIVATE: u32 = 6;
    pub const FUNCTION: u32 = 7;
    pub const PUSH_CONSTANT: u32 = 9;
    pub const STORAGE_BUFFER: u32 = 12;
}

pub mod decoration {
    pub const BLOCK: u32 = 2;
    pub const ARRAY_STRIDE: u32 = 6;
    pub const BUILT_IN: u32 = 11;
    pub const NON_WRITABLE: u32 = 24;
    pub const NO_CONTRACTION: u32 = 42;
    pub const PER_VERTEX_KHR: u32 = 5285;
    pub const LOCATION: u32 = 30;
    pub const BINDING: u32 = 33;
    pub const DESCRIPTOR_SET: u32 = 34;
    pub const OFFSET: u32 = 35;
}

pub mod builtin {
    pub const POSITION: u32 = 0;
    pub const FRAG_COORD: u32 = 15;
    pub const WORKGROUP_ID: u32 = 26;
    pub const LOCAL_INVOCATION_ID: u32 = 27;
    pub const GLOBAL_INVOCATION_ID: u32 = 28;
    pub const SUBGROUP_SIZE: u32 = 36;
    pub const SUBGROUP_LOCAL_INVOCATION_ID: u32 = 41;
    pub const VERTEX_INDEX: u32 = 42;
    pub const INSTANCE_INDEX: u32 = 43;
    pub const LOCAL_INVOCATION_INDEX: u32 = 29;
    pub const NUM_SUBGROUPS: u32 = 38;
    pub const SUBGROUP_ID: u32 = 40;
    pub const BARY_COORD_KHR: u32 = 5286;
    pub const FRONT_FACING: u32 = 17;
    pub const BASE_VERTEX: u32 = 4424;
    pub const BASE_INSTANCE: u32 = 4425;
}

pub mod semantics {
    pub const NONE: u32 = 0;
    pub const ACQUIRE: u32 = 0x2;
    pub const RELEASE: u32 = 0x4;
    pub const ACQUIRE_RELEASE: u32 = 0x8;
    pub const SEQUENTIALLY_CONSISTENT: u32 = 0x10;
    pub const UNIFORM_MEMORY: u32 = 0x40;
    pub const WORKGROUP_MEMORY: u32 = 0x100;
}

pub mod scope {
    pub const WORKGROUP: u32 = 2;
    pub const SUBGROUP: u32 = 3;
}

/// A result id.
pub type Id = u32;

/// Logical-layout sections, in module order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Section {
    Capability,
    Extension,
    ExtInstImport,
    MemoryModel,
    EntryPoint,
    ExecutionMode,
    Debug,
    Annotation,
    Global,
    Function,
}

const SECTIONS: usize = 10;

/// Encode a literal string: UTF-8, NUL-terminated, zero-padded to words.
pub fn string_words(s: &str) -> Vec<u32> {
    let mut b = s.as_bytes().to_vec();
    b.push(0);
    b.resize(b.len().div_ceil(4) * 4, 0);
    b.chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn encode(opcode: u16, operands: &[u32]) -> Vec<u32> {
    let mut v = Vec::with_capacity(operands.len() + 1);
    v.push(((operands.len() as u32 + 1) << WORD_COUNT_SHIFT) | opcode as u32);
    v.extend_from_slice(operands);
    v
}

#[derive(Default)]
struct FunctionBuf {
    /// `OpFunction` and parameters.
    head: Vec<u32>,
    /// `OpVariable`s with Function storage, placed after the first label.
    vars: Vec<u32>,
    /// First `OpLabel`.
    entry_label: Vec<u32>,
    body: Vec<u32>,
}

pub struct Builder {
    next_id: Id,
    sections: [Vec<u32>; SECTIONS],
    /// Deduplicated types and constants: encoding without the result id.
    interned: HashMap<(u16, Vec<u32>), Id>,
    capabilities: Vec<u32>,
    func: Option<FunctionBuf>,
    /// Is the current block terminated?
    terminated: bool,
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl Builder {
    pub fn new() -> Self {
        Builder {
            next_id: 1,
            sections: Default::default(),
            interned: HashMap::new(),
            capabilities: Vec::new(),
            func: None,
            terminated: true,
        }
    }

    pub fn id(&mut self) -> Id {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn push(&mut self, s: Section, opcode: u16, operands: &[u32]) {
        self.sections[s as usize].extend(encode(opcode, operands));
    }

    pub fn capability(&mut self, cap: u32) {
        if !self.capabilities.contains(&cap) {
            self.capabilities.push(cap);
            self.push(Section::Capability, op::CAPABILITY, &[cap]);
        }
    }

    pub fn extension(&mut self, name: &str) {
        let ops = string_words(name);
        self.push(Section::Extension, op::EXTENSION, &ops);
    }

    pub fn ext_inst_import(&mut self, name: &str) -> Id {
        let id = self.id();
        let mut ops = vec![id];
        ops.extend(string_words(name));
        self.push(Section::ExtInstImport, op::EXT_INST_IMPORT, &ops);
        id
    }

    pub fn memory_model(&mut self, addressing: u32, model: u32) {
        self.sections[Section::MemoryModel as usize].clear();
        self.push(Section::MemoryModel, op::MEMORY_MODEL, &[addressing, model]);
    }

    pub fn entry_point(&mut self, model: u32, function: Id, name: &str, interface: &[Id]) {
        let mut ops = vec![model, function];
        ops.extend(string_words(name));
        ops.extend_from_slice(interface);
        self.push(Section::EntryPoint, op::ENTRY_POINT, &ops);
    }

    pub fn execution_mode(&mut self, function: Id, mode: u32, literals: &[u32]) {
        let mut ops = vec![function, mode];
        ops.extend_from_slice(literals);
        self.push(Section::ExecutionMode, op::EXECUTION_MODE, &ops);
    }

    pub fn name(&mut self, target: Id, name: &str) {
        let mut ops = vec![target];
        ops.extend(string_words(name));
        self.push(Section::Debug, op::NAME, &ops);
    }

    pub fn decorate(&mut self, target: Id, decoration: u32, literals: &[u32]) {
        let mut ops = vec![target, decoration];
        ops.extend_from_slice(literals);
        self.push(Section::Annotation, op::DECORATE, &ops);
    }

    pub fn member_decorate(&mut self, ty: Id, member: u32, decoration: u32, literals: &[u32]) {
        let mut ops = vec![ty, member, decoration];
        ops.extend_from_slice(literals);
        self.push(Section::Annotation, op::MEMBER_DECORATE, &ops);
    }

    /// A type or constant, deduplicated. `operands` excludes the result id
    /// (and for constants, `operands[0]` is the result type).
    fn intern(&mut self, opcode: u16, operands: &[u32], has_type: bool) -> Id {
        let key = (opcode, operands.to_vec());
        if let Some(&id) = self.interned.get(&key) {
            return id;
        }
        let id = self.id();
        let mut ops = Vec::with_capacity(operands.len() + 1);
        if has_type {
            ops.push(operands[0]);
            ops.push(id);
            ops.extend_from_slice(&operands[1..]);
        } else {
            ops.push(id);
            ops.extend_from_slice(operands);
        }
        self.push(Section::Global, opcode, &ops);
        self.interned.insert(key, id);
        id
    }

    pub fn type_void(&mut self) -> Id {
        self.intern(op::TYPE_VOID, &[], false)
    }
    pub fn type_bool(&mut self) -> Id {
        self.intern(op::TYPE_BOOL, &[], false)
    }
    pub fn type_int(&mut self, width: u32, signed: bool) -> Id {
        self.intern(op::TYPE_INT, &[width, signed as u32], false)
    }
    pub fn type_u32(&mut self) -> Id {
        self.type_int(32, false)
    }
    pub fn type_f32(&mut self) -> Id {
        self.type_float(32)
    }
    pub fn type_float(&mut self, width: u32) -> Id {
        self.intern(op::TYPE_FLOAT, &[width], false)
    }
    pub fn type_vector(&mut self, component: Id, count: u32) -> Id {
        self.intern(op::TYPE_VECTOR, &[component, count], false)
    }
    pub fn type_array(&mut self, element: Id, length_const: Id) -> Id {
        self.intern(op::TYPE_ARRAY, &[element, length_const], false)
    }
    pub fn type_runtime_array(&mut self, element: Id) -> Id {
        // Not deduplicated: each one usually gets its own ArrayStride.
        let id = self.id();
        self.push(Section::Global, op::TYPE_RUNTIME_ARRAY, &[id, element]);
        id
    }
    /// Struct types are never deduplicated (decorations are per type).
    pub fn type_struct(&mut self, members: &[Id]) -> Id {
        let id = self.id();
        let mut ops = vec![id];
        ops.extend_from_slice(members);
        self.push(Section::Global, op::TYPE_STRUCT, &ops);
        id
    }
    pub fn type_pointer(&mut self, storage: u32, pointee: Id) -> Id {
        self.intern(op::TYPE_POINTER, &[storage, pointee], false)
    }
    pub fn type_function(&mut self, ret: Id, params: &[Id]) -> Id {
        let mut ops = vec![ret];
        ops.extend_from_slice(params);
        self.intern(op::TYPE_FUNCTION, &ops, false)
    }

    pub fn constant_u32(&mut self, v: u32) -> Id {
        let t = self.type_u32();
        self.intern(op::CONSTANT, &[t, v], true)
    }
    pub fn constant_f32(&mut self, v: f32) -> Id {
        let t = self.type_f32();
        self.intern(op::CONSTANT, &[t, v.to_bits()], true)
    }
    pub fn constant_bool(&mut self, v: bool) -> Id {
        let t = self.type_bool();
        let opcode = if v { op::CONSTANT_TRUE } else { op::CONSTANT_FALSE };
        self.intern(opcode, &[t], true)
    }
    pub fn constant_composite(&mut self, ty: Id, parts: &[Id]) -> Id {
        let mut ops = vec![ty];
        ops.extend_from_slice(parts);
        self.intern(op::CONSTANT_COMPOSITE, &ops, true)
    }

    /// A module-scope variable (`ptr_type` must be a pointer type).
    pub fn global_variable(&mut self, ptr_type: Id, storage: u32) -> Id {
        assert_ne!(storage, storage::FUNCTION, "use local_variable");
        let id = self.id();
        self.push(Section::Global, op::VARIABLE, &[ptr_type, id, storage]);
        id
    }

    /// Start a function; its entry block is opened too.
    pub fn begin_function(&mut self, ret: Id, fn_type: Id) -> Id {
        assert!(self.func.is_none(), "nested function");
        let id = self.id();
        let label = self.id();
        self.func = Some(FunctionBuf {
            head: encode(op::FUNCTION, &[ret, id, 0, fn_type]),
            vars: Vec::new(),
            entry_label: encode(op::LABEL, &[label]),
            body: Vec::new(),
        });
        self.terminated = false;
        id
    }

    fn body(&mut self) -> &mut Vec<u32> {
        &mut self.func.as_mut().expect("no current function").body
    }

    /// A Function-storage variable, hoisted to the top of the entry block.
    pub fn local_variable(&mut self, ptr_type: Id) -> Id {
        let id = self.id();
        let f = self.func.as_mut().expect("no current function");
        f.vars
            .extend(encode(op::VARIABLE, &[ptr_type, id, storage::FUNCTION]));
        id
    }

    /// An instruction with a result: `opcode result_type result operands…`.
    pub fn op(&mut self, result_type: Id, opcode: u16, operands: &[u32]) -> Id {
        assert!(!self.terminated, "instruction after a block terminator");
        let id = self.id();
        let mut ops = vec![result_type, id];
        ops.extend_from_slice(operands);
        let w = encode(opcode, &ops);
        self.body().extend(w);
        id
    }

    /// An instruction without a result (`OpStore`, `OpSelectionMerge`…).
    pub fn op_void(&mut self, opcode: u16, operands: &[u32]) {
        assert!(!self.terminated, "instruction after a block terminator");
        let w = encode(opcode, operands);
        self.body().extend(w);
    }

    pub fn load(&mut self, ty: Id, ptr: Id) -> Id {
        self.op(ty, op::LOAD, &[ptr])
    }

    pub fn store(&mut self, ptr: Id, value: Id) {
        self.op_void(op::STORE, &[ptr, value]);
    }

    fn terminate(&mut self, opcode: u16, operands: &[u32]) {
        self.op_void(opcode, operands);
        self.terminated = true;
    }

    pub fn branch(&mut self, target: Id) {
        self.terminate(op::BRANCH, &[target]);
    }

    pub fn branch_conditional(&mut self, cond: Id, yes: Id, no: Id) {
        self.terminate(op::BRANCH_CONDITIONAL, &[cond, yes, no]);
    }

    pub fn ret(&mut self) {
        self.terminate(op::RETURN, &[]);
    }

    pub fn unreachable(&mut self) {
        self.terminate(op::UNREACHABLE, &[]);
    }

    pub fn kill(&mut self) {
        self.terminate(op::KILL, &[]);
    }

    /// Open the block `label` (from [`Builder::id`]). The previous block
    /// must have been terminated.
    pub fn label(&mut self, label: Id) {
        assert!(self.terminated, "previous block has no terminator");
        let w = encode(op::LABEL, &[label]);
        self.body().extend(w);
        self.terminated = false;
    }

    pub fn end_function(&mut self) {
        assert!(self.terminated, "last block has no terminator");
        let f = self.func.take().expect("no current function");
        let out = &mut self.sections[Section::Function as usize];
        out.extend(f.head);
        out.extend(f.entry_label);
        out.extend(f.vars);
        out.extend(f.body);
        out.extend(encode(op::FUNCTION_END, &[]));
    }

    /// The finished module.
    pub fn finish(self) -> Vec<u32> {
        assert!(self.func.is_none(), "unfinished function");
        let mut out = vec![MAGIC, VERSION_1_3, GENERATOR, self.next_id, 0];
        for s in self.sections {
            out.extend(s);
        }
        out
    }
}

// ---------------------------------------------------------------- validation

#[derive(Debug, PartialEq)]
pub enum Invalid {
    Header(&'static str),
    WordCount { at: usize },
    UnknownOpcode { at: usize, opcode: u16 },
    IdOutOfBound { at: usize, id: Id },
    Redefined { at: usize, id: Id },
    Undefined { at: usize, id: Id },
    SectionOrder { at: usize, opcode: u16 },
    Block { at: usize, what: &'static str },
}

/// How the operands after the result (if any) are made up.
#[derive(Clone, Copy)]
enum Ops {
    /// Every operand is an id that must already be defined.
    Ids,
    /// Every operand is an id; forward references are allowed (labels).
    ForwardIds,
    /// Leading ids, then literals.
    IdsThenLiterals(usize),
    /// Not checked beyond the result.
    Unchecked,
}

struct Shape {
    section: Section,
    has_type: bool,
    has_result: bool,
    ops: Ops,
}

fn shape(opcode: u16) -> Option<Shape> {
    use Section::*;
    let s = |section, has_type, has_result, ops| {
        Some(Shape {
            section,
            has_type,
            has_result,
            ops,
        })
    };
    match opcode {
        op::CAPABILITY => s(Capability, false, false, Ops::Unchecked),
        op::EXTENSION => s(Extension, false, false, Ops::Unchecked),
        op::EXT_INST_IMPORT => s(ExtInstImport, false, true, Ops::Unchecked),
        op::MEMORY_MODEL => s(MemoryModel, false, false, Ops::Unchecked),
        // Interface ids may name variables declared later.
        op::ENTRY_POINT => s(EntryPoint, false, false, Ops::Unchecked),
        op::EXECUTION_MODE => s(ExecutionMode, false, false, Ops::Unchecked),
        op::NAME | op::MEMBER_NAME => s(Debug, false, false, Ops::Unchecked),
        op::DECORATE | op::MEMBER_DECORATE => s(Annotation, false, false, Ops::Unchecked),
        op::TYPE_VOID | op::TYPE_BOOL => s(Global, false, true, Ops::Ids),
        op::TYPE_INT | op::TYPE_FLOAT => s(Global, false, true, Ops::Unchecked),
        op::TYPE_VECTOR => s(Global, false, true, Ops::IdsThenLiterals(1)),
        op::TYPE_ARRAY | op::TYPE_RUNTIME_ARRAY | op::TYPE_STRUCT | op::TYPE_FUNCTION => {
            s(Global, false, true, Ops::Ids)
        }
        op::TYPE_POINTER => s(Global, false, true, Ops::Unchecked),
        op::CONSTANT_TRUE | op::CONSTANT_FALSE | op::CONSTANT_COMPOSITE => {
            s(Global, true, true, Ops::Ids)
        }
        op::CONSTANT => s(Global, true, true, Ops::Unchecked),
        // Global or local; the validator decides by storage class.
        op::VARIABLE => s(Global, true, true, Ops::Unchecked),
        op::FUNCTION => s(Function, true, true, Ops::Unchecked),
        op::FUNCTION_PARAMETER => s(Function, true, true, Ops::Ids),
        op::FUNCTION_END => s(Function, false, false, Ops::Ids),
        op::FUNCTION_CALL => s(Function, true, true, Ops::ForwardIds),
        op::LABEL => s(Function, false, true, Ops::Ids),
        op::BRANCH | op::BRANCH_CONDITIONAL => s(Function, false, false, Ops::ForwardIds),
        op::SELECTION_MERGE => s(Function, false, false, Ops::Unchecked),
        op::LOOP_MERGE => s(Function, false, false, Ops::Unchecked),
        op::RETURN | op::UNREACHABLE | op::RETURN_VALUE | op::KILL => s(Function, false, false, Ops::Ids),
        op::STORE => s(Function, false, false, Ops::Ids),
        op::CONTROL_BARRIER | op::MEMORY_BARRIER => s(Function, false, false, Ops::Ids),
        op::COMPOSITE_EXTRACT => s(Function, true, true, Ops::IdsThenLiterals(1)),
        op::PHI => s(Function, true, true, Ops::ForwardIds),
        op::EXT_INST => s(Function, true, true, Ops::Unchecked),
        op::LOAD | op::ACCESS_CHAIN | op::COMPOSITE_CONSTRUCT | op::CONVERT_F_TO_U
        | op::CONVERT_F_TO_S | op::CONVERT_S_TO_F | op::CONVERT_U_TO_F | op::F_CONVERT | op::BITCAST
        | op::S_NEGATE | op::F_NEGATE | op::I_ADD | op::F_ADD | op::I_SUB | op::F_SUB
        | op::I_MUL | op::F_MUL | op::U_DIV | op::S_DIV | op::F_DIV | op::I_ADD_CARRY
        | op::LOGICAL_OR | op::LOGICAL_AND | op::LOGICAL_NOT | op::SELECT | op::I_EQUAL
        | op::I_NOT_EQUAL | op::U_GREATER_THAN | op::S_GREATER_THAN
        | op::U_GREATER_THAN_EQUAL | op::S_GREATER_THAN_EQUAL | op::U_LESS_THAN
        | op::S_LESS_THAN | op::U_LESS_THAN_EQUAL | op::S_LESS_THAN_EQUAL | op::F_ORD_EQUAL
        | op::F_ORD_NOT_EQUAL | op::F_UNORD_NOT_EQUAL | op::F_ORD_LESS_THAN
        | op::F_ORD_GREATER_THAN | op::F_ORD_LESS_THAN_EQUAL | op::F_ORD_GREATER_THAN_EQUAL
        | op::SHIFT_RIGHT_LOGICAL | op::SHIFT_RIGHT_ARITHMETIC | op::SHIFT_LEFT_LOGICAL
        | op::BITWISE_OR | op::BITWISE_XOR | op::BITWISE_AND | op::NOT | op::BIT_COUNT
        | op::GROUP_NON_UNIFORM_ELECT | op::GROUP_NON_UNIFORM_BROADCAST
        | op::GROUP_NON_UNIFORM_BROADCAST_FIRST | op::GROUP_NON_UNIFORM_BALLOT
        | op::GROUP_NON_UNIFORM_SHUFFLE | op::ATOMIC_EXCHANGE | op::ATOMIC_COMPARE_EXCHANGE
        | op::ATOMIC_I_ADD | op::ATOMIC_I_SUB | op::ATOMIC_S_MIN | op::ATOMIC_U_MIN | op::ATOMIC_S_MAX
        | op::ATOMIC_U_MAX | op::ATOMIC_AND | op::ATOMIC_OR | op::ATOMIC_XOR => s(Function, true, true, Ops::Ids),
        _ => None,
    }
}

fn is_terminator(opcode: u16) -> bool {
    matches!(
        opcode,
        op::BRANCH | op::BRANCH_CONDITIONAL | op::RETURN | op::RETURN_VALUE | op::UNREACHABLE | op::KILL
    )
}

/// One instruction of a parsed module.
#[derive(Debug, Clone, PartialEq)]
pub struct Instruction {
    pub at: usize,
    pub opcode: u16,
    pub operands: Vec<u32>,
}

/// Split a module into instructions (header checked, word counts checked).
pub fn parse(words: &[u32]) -> Result<(u32, Vec<Instruction>), Invalid> {
    if words.len() < 5 {
        return Err(Invalid::Header("shorter than the 5-word header"));
    }
    if words[0] != MAGIC {
        return Err(Invalid::Header("bad magic"));
    }
    if words[4] != 0 {
        return Err(Invalid::Header("schema must be 0"));
    }
    let bound = words[3];
    let mut out = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> WORD_COUNT_SHIFT) as usize;
        if count == 0 || at + count > words.len() {
            return Err(Invalid::WordCount { at });
        }
        out.push(Instruction {
            at,
            opcode: (words[at] & OPCODE_MASK) as u16,
            operands: words[at + 1..at + count].to_vec(),
        });
        at += count;
    }
    Ok((bound, out))
}

/// Structural checks (see the module docs).
pub fn validate(words: &[u32]) -> Result<(), Invalid> {
    let (bound, insts) = parse(words)?;
    let mut defined = vec![false; bound as usize];
    let mut section = Section::Capability;
    // Block state inside a function: None = between functions.
    let mut in_function = false;
    let mut block_open = false;
    let mut first_block = false;
    let mut seen_non_var = false;

    let check_bound = |at, id: Id| {
        if id == 0 || id >= bound {
            Err(Invalid::IdOutOfBound { at, id })
        } else {
            Ok(())
        }
    };

    // Forward references are allowed for labels and functions; collect all
    // definitions first for those.
    let mut all_defined = vec![false; bound as usize];
    for i in &insts {
        if let Some(sh) = shape(i.opcode) {
            if sh.has_result {
                let id = i.operands[sh.has_type as usize];
                check_bound(i.at, id)?;
                all_defined[id as usize] = true;
            }
        }
    }

    for i in &insts {
        let at = i.at;
        let sh = shape(i.opcode).ok_or(Invalid::UnknownOpcode {
            at,
            opcode: i.opcode,
        })?;
        let mut target = sh.section;
        if i.opcode == op::VARIABLE && i.operands.get(2) == Some(&storage::FUNCTION) {
            target = Section::Function;
        }
        if target < section {
            return Err(Invalid::SectionOrder {
                at,
                opcode: i.opcode,
            });
        }
        section = target;

        let need = sh.has_type as usize + sh.has_result as usize;
        if i.operands.len() < need {
            return Err(Invalid::WordCount { at });
        }
        if sh.has_type {
            let t = i.operands[0];
            check_bound(at, t)?;
            if !defined[t as usize] {
                return Err(Invalid::Undefined { at, id: t });
            }
        }
        let rest = &i.operands[need..];
        let ids: &[u32] = match sh.ops {
            Ops::Ids | Ops::ForwardIds => rest,
            Ops::IdsThenLiterals(n) => &rest[..n.min(rest.len())],
            Ops::Unchecked => &[],
        };
        for &id in ids {
            check_bound(at, id)?;
            let ok = match sh.ops {
                Ops::ForwardIds => all_defined[id as usize],
                _ => defined[id as usize],
            };
            if !ok {
                return Err(Invalid::Undefined { at, id });
            }
        }
        if sh.has_result {
            let id = i.operands[sh.has_type as usize];
            if defined[id as usize] {
                return Err(Invalid::Redefined { at, id });
            }
            defined[id as usize] = true;
        }

        // Function and block structure.
        match i.opcode {
            op::FUNCTION => {
                if in_function {
                    return Err(Invalid::Block { at, what: "nested OpFunction" });
                }
                in_function = true;
                first_block = true;
            }
            op::FUNCTION_PARAMETER if block_open => {
                return Err(Invalid::Block { at, what: "parameter after the first label" });
            }
            op::FUNCTION_PARAMETER => {}
            op::LABEL => {
                if !in_function || block_open {
                    return Err(Invalid::Block { at, what: "label inside an open block" });
                }
                block_open = true;
                seen_non_var = false;
            }
            op::FUNCTION_END => {
                if block_open || first_block {
                    return Err(Invalid::Block { at, what: "function ends inside a block" });
                }
                in_function = false;
            }
            _ if in_function => {
                if !block_open {
                    return Err(Invalid::Block { at, what: "instruction outside a block" });
                }
                if i.opcode == op::VARIABLE {
                    if !first_block || seen_non_var {
                        return Err(Invalid::Block {
                            at,
                            what: "local variable not at the top of the entry block",
                        });
                    }
                } else {
                    seen_non_var = true;
                }
                if is_terminator(i.opcode) {
                    block_open = false;
                    first_block = false;
                }
            }
            _ => {}
        }
    }
    if in_function {
        return Err(Invalid::Block {
            at: words.len(),
            what: "missing OpFunctionEnd",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strings_are_nul_terminated_and_padded() {
        assert_eq!(string_words("main"), vec![0x6E69_616D, 0]);
        assert_eq!(string_words("abc"), vec![0x0063_6261]);
        assert_eq!(string_words(""), vec![0]);
    }
}
