//! GDK runtime classes through `QueryApiImpl`, called from guest code the
//! way `xgameruntime.lib`'s thunks do: query, call one vtable slot, Release.

use scarlett_hle::com::{self, ComClass, Guid};
use scarlett_hle::{xgameruntime, Call, Registry};
use scarlett_loader::{load, LoadOptions};
use scarlett_pe::builder::{Layout, PeBuilder, DATA_RVA, TEXT_RVA};
use std::sync::Arc;

const SERVED: &str = "11111111-2222-3333-4455-66778899aabb";
const UNKNOWN: &str = "deadbeef-0000-0000-0000-000000000001";

fn call_iat(c: &mut Vec<u8>, l: &Layout, dll: &str, sym: &str) {
    let next = TEXT_RVA + c.len() as u32 + 6;
    let disp = l.iat(dll, sym) as i64 - next as i64;
    c.extend_from_slice(&[0xFF, 0x15]);
    c.extend_from_slice(&(disp as i32).to_le_bytes());
}

/// `lea <reg>, [rip + DATA_RVA]` with `modrm_reg` = 1 (rcx) or 2 (rdx).
fn lea_data(c: &mut Vec<u8>, modrm_reg: u8) {
    let next = TEXT_RVA + c.len() as u32 + 7;
    c.extend_from_slice(&[0x48, 0x8D, 0x05 | (modrm_reg << 3)]);
    c.extend_from_slice(&((DATA_RVA as i64 - next as i64) as i32).to_le_bytes());
}

/// hr = QueryApiImpl(&g, &g, &obj); if FAILED(hr) return hr;
/// r = obj->slot3(40, 2); obj->Release(); return r
fn program(l: &Layout) -> Vec<u8> {
    let mut c = vec![0x48, 0x83, 0xEC, 0x38]; // sub rsp, 0x38
    lea_data(&mut c, 1);
    lea_data(&mut c, 2);
    c.extend_from_slice(&[0x4C, 0x8D, 0x44, 0x24, 0x28]); // lea r8, [rsp+0x28]
    call_iat(&mut c, l, "xgameruntime.dll", "QueryApiImpl");
    c.extend_from_slice(&[0x85, 0xC0]); // test eax, eax
    c.extend_from_slice(&[0x75, 0x00]); // jnz done (patched)
    let jnz = c.len();
    c.extend_from_slice(&[0x48, 0x8B, 0x4C, 0x24, 0x28]); // mov rcx, [rsp+0x28]
    c.extend_from_slice(&[0x48, 0x8B, 0x01]); // mov rax, [rcx]
    c.extend_from_slice(&[0xBA, 40, 0, 0, 0]); // mov edx, 40
    c.extend_from_slice(&[0x41, 0xB8, 2, 0, 0, 0]); // mov r8d, 2
    c.extend_from_slice(&[0xFF, 0x50, 0x18]); // call [rax+0x18]
    c.extend_from_slice(&[0x48, 0x89, 0x44, 0x24, 0x30]); // mov [rsp+0x30], rax
    c.extend_from_slice(&[0x48, 0x8B, 0x4C, 0x24, 0x28]); // mov rcx, [rsp+0x28]
    c.extend_from_slice(&[0x48, 0x8B, 0x01]); // mov rax, [rcx]
    c.extend_from_slice(&[0xFF, 0x50, 0x10]); // call [rax+0x10] (Release)
    c.extend_from_slice(&[0x48, 0x8B, 0x44, 0x24, 0x30]); // mov rax, [rsp+0x30]
    c[jnz - 1] = (c.len() - jnz) as u8;
    c.extend_from_slice(&[0x48, 0x83, 0xC4, 0x38, 0xC3]); // done: add rsp, 0x38; ret
    c
}

struct Counter(std::sync::atomic::AtomicU64);

fn add(c: &Call) -> u64 {
    let s = com::state::<Counter>(c.arg(0)).expect("this is a live Counter object");
    s.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    c.arg(1) + c.arg(2)
}

fn run(clsid: &str) -> u64 {
    let mut reg = Registry::new();
    xgameruntime::register_xgameruntime(&mut reg);
    let pe = PeBuilder::new(0x1_4000_0000)
        .import("xgameruntime.dll", &["QueryApiImpl"])
        .data(&Guid::parse(clsid).unwrap().0)
        .build(program);
    let img = load(&pe, &reg, &LoadOptions::default()).unwrap();
    unsafe { img.run() }.unwrap()
}

#[test]
fn query_api_impl_serves_registered_classes_and_reports_the_rest() {
    let class = com::register_class(ComClass {
        name: "TestAdder",
        iids: vec![],
        methods: vec![("Add", add)],
    });
    xgameruntime::register_runtime_class(Guid::parse(SERVED).unwrap(), class, || {
        Arc::new(Counter(Default::default()))
    });

    assert_eq!(run(SERVED), 42);
    assert_eq!(run(UNKNOWN) as u32, xgameruntime::E_NOT_SUPPORTED);

    let req = xgameruntime::requested_classes();
    let s = Guid::parse(SERVED).unwrap();
    let u = Guid::parse(UNKNOWN).unwrap();
    assert!(req.contains(&(s, s, true)), "{req:?}");
    assert!(req.contains(&(u, u, false)), "{req:?}");
}
