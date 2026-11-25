//! Guest code reading its TEB through `gs`: static TLS (`__declspec(thread)`
//! as MSVC compiles it), dynamic TLS slots and the last-error field.

use scarlett_hle::threading::register_threading;
use scarlett_hle::{teb, Call, Registry};
use scarlett_loader::{load, LoadOptions};
use scarlett_pe::builder::{Layout, PeBuilder, DATA_RVA, TEXT_RVA};
use std::sync::Mutex;

fn call_iat(code: &mut Vec<u8>, l: &Layout, dll: &str, sym: &str) {
    let next = TEXT_RVA + code.len() as u32 + 6;
    let disp = l.iat(dll, sym) as i64 - next as i64;
    code.extend_from_slice(&[0xFF, 0x15]);
    code.extend_from_slice(&(disp as i32).to_le_bytes());
}

/// `op [rip + (DATA_RVA + data_off)]` where `op` is the opcode+ModRM prefix.
fn rip_data(code: &mut Vec<u8>, op: &[u8], data_off: u32) {
    let next = TEXT_RVA + code.len() as u32 + op.len() as u32 + 4;
    code.extend_from_slice(op);
    code.extend_from_slice(&((DATA_RVA + data_off) as i64 - next as i64).to_le_bytes()[..4]);
}

const TEMPLATE: u64 = 0x1122_3344_5566_7788;
const INDEX_OFF: u32 = 0x30;

/// .data layout: TLS directory at 0, `_tls_index` at 0x30, template at 0x40.
fn tls_image(base: u64) -> Vec<u8> {
    let mut data = vec![0u8; 0x48];
    data[0x20..0x24].copy_from_slice(&8u32.to_le_bytes()); // SizeOfZeroFill
    data[0x24..0x28].copy_from_slice(&0x0050_0000u32.to_le_bytes()); // ALIGN_16BYTES
    data[0x30..0x34].copy_from_slice(&0xDEADu32.to_le_bytes()); // overwritten by loader
    data[0x40..0x48].copy_from_slice(&TEMPLATE.to_le_bytes());
    PeBuilder::new(base)
        .import("testx.dll", &["Report"])
        .data(&data)
        .data_pointer(0x00, DATA_RVA + 0x40) // StartAddressOfRawData
        .data_pointer(0x08, DATA_RVA + 0x48) // EndAddressOfRawData
        .data_pointer(0x10, DATA_RVA + INDEX_OFF) // AddressOfIndex
        .tls_directory(0)
        .build(|l| {
            let mut c = vec![0x48, 0x83, 0xEC, 0x28]; // sub rsp, 40
            c.extend_from_slice(&[0x65, 0x48, 0x8B, 0x04, 0x25, 0x58, 0, 0, 0]); // mov rax, gs:[0x58]
            rip_data(&mut c, &[0x8B, 0x0D], INDEX_OFF); // mov ecx, [_tls_index]
            c.extend_from_slice(&[0x48, 0x8B, 0x04, 0xC8]); // mov rax, [rax+rcx*8]
            c.extend_from_slice(&[0x48, 0x8B, 0x08]); // mov rcx, [rax]      template
            c.extend_from_slice(&[0x48, 0x8B, 0x50, 0x08]); // mov rdx, [rax+8]  zero fill
            c.extend_from_slice(&[0x65, 0x4C, 0x8B, 0x04, 0x25, 0x30, 0, 0, 0]); // mov r8, gs:[0x30]
            c.extend_from_slice(&[0x49, 0x89, 0xC1]); // mov r9, rax          block address
            c.extend_from_slice(&[0x48, 0xFF, 0x00]); // inc qword [rax]
            call_iat(&mut c, l, "testx.dll", "Report");
            c.extend_from_slice(&[0x48, 0x83, 0xC4, 0x28, 0xC3]); // add rsp, 40; ret
            c
        })
}

#[derive(Debug, Clone, Copy)]
struct Seen {
    value: u64,
    zero_fill: u64,
    teb_self: u64,
    block: u64,
    handler_teb: u64,
}

static SEEN: Mutex<Vec<Seen>> = Mutex::new(Vec::new());

fn report(c: &Call) -> u64 {
    SEEN.lock().unwrap().push(Seen {
        value: c.arg(0),
        zero_fill: c.arg(1),
        teb_self: c.arg(2),
        block: c.arg(3),
        handler_teb: teb::current_teb().map_or(0, |t| t as u64),
    });
    c.arg(0)
}

#[test]
fn static_tls_block_per_thread_from_template() {
    let mut reg = Registry::new();
    reg.register("testx.dll", "Report", report);
    // Not at the preferred base, so the TLS directory's VAs need relocating.
    let opts = LoadOptions {
        prefer_image_base: false,
        ..Default::default()
    };
    let img = load(&tls_image(0x1_8000_0000), &reg, &opts).unwrap();
    let index = img.tls_index.expect("image has static TLS");
    assert_eq!(
        img.region()
            .read_u32((DATA_RVA + INDEX_OFF) as usize)
            .unwrap(),
        index
    );

    // Two runs = two guest threads; each must see a fresh copy.
    for _ in 0..2 {
        assert_eq!(unsafe { img.run() }.unwrap(), TEMPLATE);
    }
    let seen = SEEN.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    for s in &seen {
        assert_eq!(s.value, TEMPLATE, "{s:x?}");
        assert_eq!(s.zero_fill, 0, "{s:x?}");
        assert_ne!(s.teb_self, 0);
        assert_eq!(s.teb_self, s.handler_teb, "gs:[0x30] is the TEB HLE sees");
        assert_eq!(s.block % 16, 0, "ALIGN_16BYTES honoured");
    }
    // Template in the image is untouched by the guest's `inc`.
    assert_eq!(
        img.region().read_u64((DATA_RVA + 0x40) as usize).unwrap(),
        TEMPLATE
    );
}

#[test]
fn dynamic_tls_and_last_error_live_in_the_teb() {
    let mut reg = Registry::new();
    register_threading(&mut reg, &["kernel32"]);
    let syms = ["TlsAlloc", "TlsSetValue", "SetLastError"];
    let pe = PeBuilder::new(0x1_9000_0000)
        .import("kernel32.dll", &syms)
        .build(|l| {
            let mut c = vec![0x53, 0x48, 0x83, 0xEC, 0x20]; // push rbx; sub rsp, 32
            call_iat(&mut c, l, "kernel32.dll", "TlsAlloc");
            c.extend_from_slice(&[0x89, 0xC3]); // mov ebx, eax
            c.extend_from_slice(&[0x89, 0xD9]); // mov ecx, ebx
            c.extend_from_slice(&[0xBA, 0xCD, 0xAB, 0, 0]); // mov edx, 0xABCD
            call_iat(&mut c, l, "kernel32.dll", "TlsSetValue");
            // mov rax, gs:[0x1480 + rbx*8]   (TlsSlots[index], read directly)
            c.extend_from_slice(&[0x65, 0x48, 0x8B, 0x04, 0xDD, 0x80, 0x14, 0, 0]);
            c.extend_from_slice(&[0x48, 0x89, 0xC3]); // mov rbx, rax
            c.extend_from_slice(&[0xB9, 5, 0, 0, 0]); // mov ecx, 5
            call_iat(&mut c, l, "kernel32.dll", "SetLastError");
            c.extend_from_slice(&[0x65, 0x8B, 0x04, 0x25, 0x68, 0, 0, 0]); // mov eax, gs:[0x68]
            c.extend_from_slice(&[0x48, 0x01, 0xD8]); // add rax, rbx
            c.extend_from_slice(&[0x48, 0x83, 0xC4, 0x20, 0x5B, 0xC3]); // add rsp,32; pop rbx; ret
            c
        });
    let img = load(&pe, &reg, &LoadOptions::default()).unwrap();
    assert_eq!(unsafe { img.run() }.unwrap(), 0xABCD + 5);
}

#[test]
fn guest_thread_fields() {
    let t = teb::GuestThread::attach(0x2000_0000, 0x1000_0000);
    let teb = t.teb();
    assert_eq!(teb::current_teb(), Some(teb));
    let rd = |o: usize| unsafe { teb.add(o).cast::<u64>().read() };
    assert_eq!(rd(teb::off::SELF), teb as u64);
    assert_ne!(teb::current_thread_id(), 0);
    if !cfg!(windows) {
        assert_eq!(rd(teb::off::STACK_BASE), 0x2000_0000);
        assert_eq!(rd(teb::off::STACK_LIMIT), 0x1000_0000);
        assert_eq!(teb::current_process_id(), std::process::id());
    }
    drop(t);
    assert_eq!(teb::current_teb(), None);
}
