//! Float arguments and results through the HLE trampoline, from real guest
//! code: Win64 passes argument *positions* in rcx/xmm0, rdx/xmm1, ...

use scarlett_hle::{Call, Registry};
use scarlett_loader::{load, LoadOptions};
use scarlett_pe::builder::{PeBuilder, TEXT_RVA};

/// double Mix(double a, int b, float c, int d, double e) = a*b + c*d + e
fn mix(c: &Call) -> u64 {
    let v = c.arg_f64(0) * c.arg(1) as f64 + c.arg_f32(2) as f64 * c.arg(3) as f64 + c.arg_f64(4);
    c.ret_f64(v);
    0xBAD // rax is don't-care for a double return
}

#[test]
fn mixed_int_float_args_and_xmm0_return() {
    let mut reg = Registry::new();
    reg.register("mathx.dll", "Mix", mix);
    let pe = PeBuilder::new(0x1_A000_0000)
        .import("mathx.dll", &["Mix"])
        .build(|l| {
            let mut c = vec![0x48, 0x83, 0xEC, 0x38]; // sub rsp, 0x38
            c.extend_from_slice(&[0x48, 0xB8]); // mov rax, 2.5
            c.extend_from_slice(&2.5f64.to_bits().to_le_bytes());
            c.extend_from_slice(&[0x66, 0x48, 0x0F, 0x6E, 0xC0]); // movq xmm0, rax
            c.extend_from_slice(&[0xBA, 4, 0, 0, 0]); // mov edx, 4
            c.push(0xB8); // mov eax, 3.0f
            c.extend_from_slice(&3.0f32.to_bits().to_le_bytes());
            c.extend_from_slice(&[0x66, 0x0F, 0x6E, 0xD0]); // movd xmm2, eax
            c.extend_from_slice(&[0x41, 0xB9, 7, 0, 0, 0]); // mov r9d, 7
            c.extend_from_slice(&[0x48, 0xB8]); // mov rax, 0.5
            c.extend_from_slice(&0.5f64.to_bits().to_le_bytes());
            c.extend_from_slice(&[0x48, 0x89, 0x44, 0x24, 0x20]); // mov [rsp+0x20], rax
            let next = TEXT_RVA + c.len() as u32 + 6;
            c.extend_from_slice(&[0xFF, 0x15]); // call [Mix]
            c.extend_from_slice(
                &((l.iat("mathx.dll", "Mix") as i64 - next as i64) as i32).to_le_bytes(),
            );
            c.extend_from_slice(&[0xF2, 0x0F, 0x58, 0xC0]); // addsd xmm0, xmm0
            c.extend_from_slice(&[0xF2, 0x48, 0x0F, 0x2C, 0xC0]); // cvttsd2si rax, xmm0
            c.extend_from_slice(&[0x48, 0x83, 0xC4, 0x38, 0xC3]); // add rsp, 0x38; ret
            c
        });
    let img = load(&pe, &reg, &LoadOptions::default()).unwrap();
    // (2.5*4 + 3*7 + 0.5) * 2 = 63
    assert_eq!(unsafe { img.run() }.unwrap(), 63);
}
