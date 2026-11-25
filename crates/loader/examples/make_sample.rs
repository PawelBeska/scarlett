//! Writes `sample.exe`: prints via OutputDebugStringA, calls one
//! unimplemented import, returns 42. Smoke test for `scarlett run`.
//!
//!     cargo run -p scarlett-loader --example make_sample
//!     cargo run -p scarlett -- run sample.exe

use scarlett_pe::builder::{PeBuilder, DATA_RVA, TEXT_RVA};

fn main() {
    let pe = PeBuilder::new(0x1_4000_0000)
        .import("kernel32.dll", &["OutputDebugStringA", "GetTickCount64"])
        .data(b"hello from guest code\0")
        .build(|l| {
            let mut c = vec![0x48, 0x83, 0xEC, 0x28]; // sub rsp, 40
                                                      // lea rcx, [rip + msg]
            let next = TEXT_RVA + c.len() as u32 + 7;
            c.extend_from_slice(&[0x48, 0x8D, 0x0D]);
            c.extend_from_slice(&((DATA_RVA as i64 - next as i64) as i32).to_le_bytes());
            for sym in ["OutputDebugStringA", "GetTickCount64"] {
                let next = TEXT_RVA + c.len() as u32 + 6;
                c.extend_from_slice(&[0xFF, 0x15]);
                c.extend_from_slice(
                    &((l.iat("kernel32.dll", sym) as i64 - next as i64) as i32).to_le_bytes(),
                );
            }
            c.extend_from_slice(&[0xB8, 42, 0, 0, 0]); // mov eax, 42
            c.extend_from_slice(&[0x48, 0x83, 0xC4, 0x28, 0xC3]); // add rsp, 40; ret
            c
        });
    std::fs::write("sample.exe", pe).expect("write sample.exe");
    println!("wrote sample.exe");
}
