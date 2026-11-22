use scarlett_pe::builder::{PeBuilder, DATA_RVA, TEXT_RVA};
use scarlett_pe::*;

const BASE: u64 = 0x1_4000_0000;

fn sample() -> Vec<u8> {
    PeBuilder::new(BASE)
        .import("kernelx.dll", &["CreateEventW", "CloseHandle"])
        .import("d3d12_x.dll", &["D3D12CreateDevice"])
        .export("beta", 4)
        .export("alpha", 0)
        .data(&[0xAA; 16])
        .data_pointer(0, TEXT_RVA)
        .data_pointer(8, DATA_RVA + 8)
        .entry_offset(0)
        .build(|_| vec![0xC3; 8])
}

#[test]
fn headers_and_sections() {
    let bytes = sample();
    let pe = PeFile::parse(&bytes).unwrap();
    assert_eq!(pe.headers.machine, MACHINE_AMD64);
    assert_eq!(pe.headers.image_base, BASE);
    assert_eq!(pe.headers.entry_point, TEXT_RVA);
    assert!(!pe.headers.is_dll());
    let names: Vec<_> = pe.sections.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, [".text", ".rdata", ".data", ".reloc"]);
    assert!(pe.sections[0].is_exec() && !pe.sections[0].is_write());
    assert!(pe.sections[2].is_write());
}

#[test]
fn imports() {
    let bytes = sample();
    let pe = PeFile::parse(&bytes).unwrap();
    let imps = pe.imports().unwrap();
    assert_eq!(imps.len(), 2);
    assert_eq!(imps[0].dll, "kernelx.dll");
    let syms: Vec<_> = imps[0].symbols.iter().map(|s| s.name.to_string()).collect();
    assert_eq!(syms, ["CreateEventW", "CloseHandle"]);
    assert_eq!(imps[0].symbols[1].iat_rva, imps[0].symbols[0].iat_rva + 8);
    assert_eq!(imps[1].symbols[0].name.to_string(), "D3D12CreateDevice");
}

#[test]
fn exports_are_sorted_by_name_but_keep_ordinals() {
    let bytes = sample();
    let pe = PeFile::parse(&bytes).unwrap();
    let ex = pe.exports().unwrap();
    assert_eq!(ex.len(), 2);
    let beta = ex
        .iter()
        .find(|e| e.name.as_deref() == Some("beta"))
        .unwrap();
    assert_eq!(beta.rva, TEXT_RVA + 4);
    assert_eq!(beta.ordinal, 1);
    assert!(beta.forwarder.is_none());
}

#[test]
fn relocations() {
    let bytes = sample();
    let pe = PeFile::parse(&bytes).unwrap();
    let rel = pe.relocations().unwrap();
    assert_eq!(
        rel,
        vec![
            Reloc {
                rva: DATA_RVA,
                kind: REL_BASED_DIR64
            },
            Reloc {
                rva: DATA_RVA + 8,
                kind: REL_BASED_DIR64
            },
        ]
    );
}

#[test]
fn rejects_garbage() {
    assert!(matches!(PeFile::parse(b"M"), Err(Error::Truncated { .. })));
    assert_eq!(PeFile::parse(b"nope").err(), Some(Error::BadMagic("MZ")));
    let mut bytes = sample();
    bytes[0] = b'X';
    assert_eq!(PeFile::parse(&bytes).err(), Some(Error::BadMagic("MZ")));
    let mut bytes = sample();
    bytes[0x84] = 0x4C; // machine -> i386
    bytes[0x85] = 0x01;
    assert_eq!(
        PeFile::parse(&bytes).err(),
        Some(Error::UnsupportedMachine(0x014C))
    );
}

#[test]
fn truncated_file_never_panics() {
    let bytes = sample();
    for len in (0..bytes.len()).step_by(7) {
        if let Ok(pe) = PeFile::parse(&bytes[..len]) {
            let _ = pe.imports();
            let _ = pe.exports();
            let _ = pe.relocations();
        }
    }
}
