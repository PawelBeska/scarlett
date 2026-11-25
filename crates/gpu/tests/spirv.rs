//! The SPIR-V builder on a small compute shader, and the validator on
//! broken modules.

use scarlett_gpu::spirv::{
    self, addressing, builtin, capability, decoration, execution_mode, execution_model,
    memory_model, op, storage, Builder, Invalid,
};

/// buf[gid.x] = gid.x < 10 ? gid.x * 2 + 1 : 0
fn compute_module() -> Vec<u32> {
    let mut b = Builder::new();
    b.capability(capability::SHADER);
    b.memory_model(addressing::LOGICAL, memory_model::GLSL450);

    let void = b.type_void();
    let u32t = b.type_u32();
    let boolt = b.type_bool();
    let uvec3 = b.type_vector(u32t, 3);
    let arr = b.type_runtime_array(u32t);
    b.decorate(arr, decoration::ARRAY_STRIDE, &[4]);
    let block = b.type_struct(&[arr]);
    b.decorate(block, decoration::BLOCK, &[]);
    b.member_decorate(block, 0, decoration::OFFSET, &[0]);
    let p_block = b.type_pointer(storage::STORAGE_BUFFER, block);
    let p_u32_sb = b.type_pointer(storage::STORAGE_BUFFER, u32t);
    let p_uvec3_in = b.type_pointer(storage::INPUT, uvec3);
    let p_u32_fn = b.type_pointer(storage::FUNCTION, u32t);

    let buf = b.global_variable(p_block, storage::STORAGE_BUFFER);
    b.decorate(buf, decoration::DESCRIPTOR_SET, &[0]);
    b.decorate(buf, decoration::BINDING, &[0]);
    let gid = b.global_variable(p_uvec3_in, storage::INPUT);
    b.decorate(gid, decoration::BUILT_IN, &[builtin::GLOBAL_INVOCATION_ID]);

    let fn_ty = b.type_function(void, &[]);
    let main = b.begin_function(void, fn_ty);
    b.name(main, "main");
    let tmp = b.local_variable(p_u32_fn);
    let zero = b.constant_u32(0);
    let one = b.constant_u32(1);
    let two = b.constant_u32(2);
    let ten = b.constant_u32(10);

    let g = b.load(uvec3, gid);
    let x = b.op(u32t, op::COMPOSITE_EXTRACT, &[g, 0]);
    b.store(tmp, zero);
    let small = b.op(boolt, op::U_LESS_THAN, &[x, ten]);
    let (then, merge) = (b.id(), b.id());
    b.op_void(op::SELECTION_MERGE, &[merge, 0]);
    b.branch_conditional(small, then, merge);
    b.label(then);
    let x2 = b.op(u32t, op::I_MUL, &[x, two]);
    let v = b.op(u32t, op::I_ADD, &[x2, one]);
    b.store(tmp, v);
    b.branch(merge);
    b.label(merge);
    let r = b.load(u32t, tmp);
    let slot = b.op(p_u32_sb, op::ACCESS_CHAIN, &[buf, zero, x]);
    b.store(slot, r);
    b.ret();
    b.end_function();

    b.entry_point(execution_model::GL_COMPUTE, main, "main", &[buf, gid]);
    b.execution_mode(main, execution_mode::LOCAL_SIZE, &[64, 1, 1]);
    b.finish()
}

#[test]
fn compute_shader_is_structurally_valid() {
    let m = compute_module();
    assert_eq!(m[0], spirv::MAGIC);
    assert_eq!(m[1], spirv::VERSION_1_3);
    spirv::validate(&m).unwrap();

    // Sections came out in layout order even though the entry point and
    // execution mode were added last.
    let (_, insts) = spirv::parse(&m).unwrap();
    let order: Vec<u16> = insts.iter().map(|i| i.opcode).collect();
    let pos = |o| order.iter().position(|&x| x == o).unwrap();
    assert!(pos(op::CAPABILITY) < pos(op::MEMORY_MODEL));
    assert!(pos(op::MEMORY_MODEL) < pos(op::ENTRY_POINT));
    assert!(pos(op::ENTRY_POINT) < pos(op::EXECUTION_MODE));
    assert!(pos(op::EXECUTION_MODE) < pos(op::NAME));
    assert!(pos(op::NAME) < pos(op::DECORATE));
    assert!(pos(op::DECORATE) < pos(op::TYPE_VOID));
    assert!(pos(op::TYPE_VOID) < pos(op::FUNCTION));
    // The local variable was hoisted right after the entry label.
    assert_eq!(order[pos(op::LABEL) + 1], op::VARIABLE);
    // Types and constants are deduplicated.
    assert_eq!(order.iter().filter(|&&o| o == op::TYPE_INT).count(), 1);
}

#[test]
fn builder_deduplicates() {
    let mut b = Builder::new();
    let a = b.type_u32();
    assert_eq!(a, b.type_int(32, false));
    assert_ne!(a, b.type_int(32, true));
    assert_eq!(b.constant_u32(7), b.constant_u32(7));
    assert_ne!(b.constant_u32(7), b.constant_f32(7.0));
}

fn find(m: &[u32], opcode: u16) -> usize {
    let (_, insts) = spirv::parse(m).unwrap();
    insts.iter().find(|i| i.opcode == opcode).unwrap().at
}

fn remove(m: &mut Vec<u32>, at: usize) {
    let n = (m[at] >> 16) as usize;
    m.drain(at..at + n);
}

#[test]
fn validator_rejects_broken_modules() {
    let good = compute_module();

    let mut m = good.clone();
    m[0] = 0;
    assert_eq!(spirv::validate(&m), Err(Invalid::Header("bad magic")));

    let mut m = good.clone();
    let at = find(&m, op::I_ADD);
    m[at] = (99 << 16) | op::I_ADD as u32;
    assert_eq!(spirv::validate(&m), Err(Invalid::WordCount { at }));

    // Use of an id beyond the bound.
    let mut m = good.clone();
    let at = find(&m, op::I_ADD);
    m[at + 3] = m[3] + 5;
    assert!(matches!(spirv::validate(&m), Err(Invalid::IdOutOfBound { .. })));

    // A block without its terminator.
    let mut m = good.clone();
    let at = find(&m, op::BRANCH);
    remove(&mut m, at);
    assert!(matches!(spirv::validate(&m), Err(Invalid::Block { .. })));

    // Use before definition: drop the multiply feeding the add.
    let mut m = good.clone();
    let at = find(&m, op::I_MUL);
    remove(&mut m, at);
    assert!(matches!(spirv::validate(&m), Err(Invalid::Undefined { .. })));

    // A decoration after the types.
    let mut m = good.clone();
    let at = find(&m, op::DECORATE);
    let n = (m[at] >> 16) as usize;
    let deco: Vec<u32> = m.drain(at..at + n).collect();
    let f = find(&m, op::FUNCTION);
    m.splice(f..f, deco);
    assert!(matches!(spirv::validate(&m), Err(Invalid::SectionOrder { .. })));
}
