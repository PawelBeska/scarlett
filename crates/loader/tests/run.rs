//! End-to-end: build a real PE with hand-assembled x86-64, load it, run it
//! natively, and check that calls land in Rust HLE handlers.

use scarlett_hle::{Call, Registry};
use scarlett_loader::{load, mem::Region, Binding, LoadError, LoadOptions};
use scarlett_pe::builder::{Layout, PeBuilder, DATA_RVA, TEXT_RVA};

/// `call [rip+disp32]` targeting the IAT slot of `dll!sym`, placed at `at`.
fn call_iat(code: &mut Vec<u8>, l: &Layout, dll: &str, sym: &str) {
    let next = TEXT_RVA + code.len() as u32 + 6;
    let disp = l.iat(dll, sym) as i64 - next as i64;
    code.extend_from_slice(&[0xFF, 0x15]);
    code.extend_from_slice(&(disp as i32).to_le_bytes());
}

/// entry: return sum6(1, 2, 3, 4, 5, 6) + add2(rax, 100)
fn sum_program(l: &Layout) -> Vec<u8> {
    let mut c = vec![0x48, 0x83, 0xEC, 0x38]; // sub rsp, 0x38
    c.extend_from_slice(&[0xB9, 1, 0, 0, 0]); // mov ecx, 1
    c.extend_from_slice(&[0xBA, 2, 0, 0, 0]); // mov edx, 2
    c.extend_from_slice(&[0x41, 0xB8, 3, 0, 0, 0]); // mov r8d, 3
    c.extend_from_slice(&[0x41, 0xB9, 4, 0, 0, 0]); // mov r9d, 4
    c.extend_from_slice(&[0x48, 0xC7, 0x44, 0x24, 0x20, 5, 0, 0, 0]); // mov qword [rsp+0x20], 5
    c.extend_from_slice(&[0x48, 0xC7, 0x44, 0x24, 0x28, 6, 0, 0, 0]); // mov qword [rsp+0x28], 6
    call_iat(&mut c, l, "testx.dll", "Sum6");
    c.extend_from_slice(&[0x48, 0x89, 0xC1]); // mov rcx, rax
    c.extend_from_slice(&[0xBA, 100, 0, 0, 0]); // mov edx, 100
    call_iat(&mut c, l, "testx.dll", "Add2");
    c.extend_from_slice(&[0x48, 0x83, 0xC4, 0x38, 0xC3]); // add rsp, 0x38; ret
    c
}

fn sum6(c: &Call) -> u64 {
    (0..6).map(|i| c.arg(i)).sum()
}

fn add2(c: &Call) -> u64 {
    c.arg(0) + c.arg(1)
}

fn registry() -> Registry {
    let mut r = Registry::new();
    r.register("TESTX", "Sum6", sum6)
        .register("testx.dll", "Add2", add2);
    r
}

#[test]
fn runs_native_code_through_hle_trampoline() {
    let pe = PeBuilder::new(0x1_4000_0000)
        .import("testx.dll", &["Sum6", "Add2"])
        .build(sum_program);
    let img = load(&pe, &registry(), &LoadOptions::default()).unwrap();
    assert!(img.imports.iter().all(|i| i.binding == Binding::Hle));
    let ret = unsafe { img.run() }.unwrap();
    assert_eq!(ret, 21 + 100);
}

#[test]
fn relocates_when_not_at_preferred_base() {
    let pe = PeBuilder::new(0x1_4000_0000)
        .import("testx.dll", &["Sum6", "Add2"])
        .data_pointer(0, TEXT_RVA)
        .data_pointer(8, DATA_RVA)
        .build(sum_program);
    let opts = LoadOptions {
        prefer_image_base: false,
        ..Default::default()
    };
    let img = load(&pe, &registry(), &opts).unwrap();
    assert_ne!(img.base(), img.preferred_base);
    assert_eq!(img.relocations_applied, 2);
    let r = img.region();
    assert_eq!(
        r.read_u64(DATA_RVA as usize).unwrap(),
        img.base() + TEXT_RVA as u64
    );
    assert_eq!(
        r.read_u64(DATA_RVA as usize + 8).unwrap(),
        img.base() + DATA_RVA as u64
    );
    // still runs after relocation
    assert_eq!(unsafe { img.run() }.unwrap(), 121);
}

#[test]
fn unimplemented_import_is_stubbed_and_returns_zero() {
    // entry: sub rsp,40; call [Missing]; add rax, 7; add rsp,40; ret
    let pe = PeBuilder::new(0x1_5000_0000)
        .import("nowhere.dll", &["Missing"])
        .build(|l| {
            let mut c = vec![0x48, 0x83, 0xEC, 0x28];
            call_iat(&mut c, l, "nowhere.dll", "Missing");
            c.extend_from_slice(&[0x48, 0x83, 0xC0, 0x07, 0x48, 0x83, 0xC4, 0x28, 0xC3]);
            c
        });
    let img = load(&pe, &Registry::new(), &LoadOptions::default()).unwrap();
    assert_eq!(img.imports[0].binding, Binding::Stub);
    assert_eq!(unsafe { img.run() }.unwrap(), 7);
    let stats = scarlett_hle::thunk_stats();
    assert!(stats
        .iter()
        .any(|(d, n, implemented, calls)| d == "nowhere.dll"
            && n == "Missing"
            && !implemented
            && *calls == 1));
}

#[test]
fn fixed_image_fails_cleanly_when_base_is_taken() {
    let base = 0x1_6000_0000;
    let squatter = Region::alloc(0x10000, Some(base)).unwrap();
    assert_eq!(
        squatter.base(),
        base,
        "test setup: could not reserve the base"
    );
    let pe = PeBuilder::new(base).strip_relocs().build(|_| vec![0xC3]);
    match load(&pe, &Registry::new(), &LoadOptions::default()) {
        Err(LoadError::BaseUnavailable { preferred }) => assert_eq!(preferred, base),
        other => panic!("expected BaseUnavailable, got {:?}", other.map(|_| ())),
    }
}

/// Section protections: .text r-x, .rdata r--, .data rw-.
#[cfg(target_os = "linux")]
#[test]
fn section_protections() {
    let pe = PeBuilder::new(0x1_7000_0000)
        .data(&[1; 8])
        .build(|_| vec![0xC3]);
    let img = load(&pe, &Registry::new(), &LoadOptions::default()).unwrap();
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
    let perms = |addr: u64| -> String {
        maps.lines()
            .find_map(|l| {
                let (range, rest) = l.split_once(' ')?;
                let (lo, hi) = range.split_once('-')?;
                let (lo, hi) = (
                    u64::from_str_radix(lo, 16).ok()?,
                    u64::from_str_radix(hi, 16).ok()?,
                );
                (lo..hi).contains(&addr).then(|| rest[..3].to_string())
            })
            .unwrap()
    };
    assert_eq!(perms(img.base()), "r--");
    assert_eq!(perms(img.base() + TEXT_RVA as u64), "r-x");
    assert_eq!(perms(img.base() + 0x2000), "r--");
    assert_eq!(perms(img.base() + DATA_RVA as u64), "rw-");
}
