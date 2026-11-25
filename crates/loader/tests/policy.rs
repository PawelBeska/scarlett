//! The unimplemented-import policy, with real guest calls.

use scarlett_hle::{policy, Registry};
use scarlett_loader::{load, LoadOptions};
use scarlett_pe::builder::{Layout, PeBuilder, TEXT_RVA};

fn call_iat(code: &mut Vec<u8>, l: &Layout, dll: &str, sym: &str) {
    let next = TEXT_RVA + code.len() as u32 + 6;
    let disp = l.iat(dll, sym) as i64 - next as i64;
    code.extend_from_slice(&[0xFF, 0x15]);
    code.extend_from_slice(&(disp as i32).to_le_bytes());
}

/// entry: return dll!sym(0x11, 0x22, 0x33, 0x44)
fn image(dll: &'static str, sym: &'static str) -> Vec<u8> {
    PeBuilder::new(0x1_4000_0000)
        .import(dll, &[sym])
        .build(move |l| {
            let mut c = vec![0x48, 0x83, 0xEC, 0x28]; // sub rsp, 0x28
            c.extend_from_slice(&[0xB9, 0x11, 0, 0, 0]); // mov ecx, 0x11
            c.extend_from_slice(&[0xBA, 0x22, 0, 0, 0]); // mov edx, 0x22
            c.extend_from_slice(&[0x41, 0xB8, 0x33, 0, 0, 0]); // mov r8d, 0x33
            c.extend_from_slice(&[0x41, 0xB9, 0x44, 0, 0, 0]); // mov r9d, 0x44
            call_iat(&mut c, l, dll, sym);
            c.extend_from_slice(&[0x48, 0x83, 0xC4, 0x28, 0xC3]); // add rsp, 0x28; ret
            c
        })
}

fn run(dll: &'static str, sym: &'static str) -> u64 {
    let opts = LoadOptions {
        name: Some("policy.exe".into()),
        ..Default::default()
    };
    let img = load(&image(dll, sym), &Registry::new(), &opts).unwrap();
    unsafe { img.run() }.unwrap()
}

#[test]
fn log_and_fail_policies() {
    policy::set_unimplemented_policy(policy::Unimplemented::Log);
    assert_eq!(run("d3d12.dll", "D3D12CreateDevice"), 0);

    policy::set_unimplemented_policy(policy::Unimplemented::Fail);
    assert_eq!(
        run("d3d12.dll", "D3D12CreateDevice") as u32,
        policy::E_NOTIMPL
    );
    // Not known to return HRESULT: still 0.
    assert_eq!(run("kernel32.dll", "GetSomethingUnknown"), 0);
}

/// `stop` ends the process, so it runs in a child copy of this test binary.
#[test]
fn stop_policy_reports_caller_and_exits() {
    if std::env::var_os("SCARLETT_POLICY_CHILD").is_some() {
        policy::set_unimplemented_policy(policy::Unimplemented::Stop);
        run("xgameruntime.dll", "XGameRuntimeInitialize");
        unreachable!("stop policy returned");
    }
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "stop_policy_reports_caller_and_exits", "--nocapture"])
        .env("SCARLETT_POLICY_CHILD", "1")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(policy::STOP_EXIT_CODE), "{stderr}");
    assert!(
        stderr.contains("xgameruntime.dll!XGameRuntimeInitialize called from policy.exe+0x"),
        "{stderr}"
    );
    assert!(stderr.contains("rcx=0x11, rdx=0x22, r8=0x33, r9=0x44"), "{stderr}");
}
