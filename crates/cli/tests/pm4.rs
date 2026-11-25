//! `scarlett pm4`: a PM4 stream and a shader from files; the dispatch
//! writes a buffer that is dumped and checked.

use std::process::Command;

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

const fn pkt3(op: u32, body: u32) -> u32 {
    3 << 30 | (body - 1) << 16 | op << 8
}

#[test]
fn pm4_command_runs_a_dispatch() {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("pm4_cli");
    std::fs::create_dir_all(&dir).unwrap();
    // v1 = v0 * 3 + s2 (TGID.x); v2 = (s2 * 64 + v0) * 4; store v1 at s[0:1] + v2
    let shader = [
        0xD543_0001, 0x0009_0700, // v_mad_u32_u24 v1, v0, 3, s2
        0xD746_0002, 0x0401_0C02, // v_lshl_add_u32 v2, s2, 6, v0
        0x3404_0482,              // v_lshlrev_b32 v2, 2, v2
        0xDC70_8000, 0x0000_0102, // global_store_dword v2, v1, s[0:1]
        0xBF81_0000,
    ];
    let (code, out) = (0x30_0000u32, 0x10_0000u32);
    let sh = |reg: u32, v: &[u32]| {
        let mut w = vec![pkt3(0x76, 1 + v.len() as u32), reg - 0x2C00];
        w.extend_from_slice(v);
        w
    };
    let mut s = Vec::new();
    s.extend(sh(0x2E0C, &[code >> 8, 0])); // COMPUTE_PGM_LO/HI
    s.extend(sh(0x2E12, &[0, 2 << 1 | 1 << 7])); // RSRC1, RSRC2
    s.extend(sh(0x2E07, &[64, 1, 1])); // NUM_THREAD_X..Z
    s.extend(sh(0x2E40, &[out, 0])); // USER_DATA_0/1
    s.extend([pkt3(0x15, 4), 2, 1, 1, 1 | 1 << 15]); // DISPATCH_DIRECT, wave32
    let (sp, cp, dp) = (dir.join("s.pm4"), dir.join("code.bin"), dir.join("out.bin"));
    std::fs::write(&sp, bytes(&s)).unwrap();
    std::fs::write(&cp, bytes(&shader)).unwrap();
    let r = Command::new(env!("CARGO_BIN_EXE_scarlett"))
        .arg("pm4")
        .arg(&sp)
        .arg("--load")
        .arg(format!("{code:#x}={}", cp.display()))
        .arg("--alloc")
        .arg(format!("{out:#x}:512"))
        .arg("--dump")
        .arg(format!("{out:#x}:512={}", dp.display()))
        .arg("--spirv")
        .arg(&dir)
        .arg("--window")
        .arg(format!("{out:#x}:512"))
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&r.stdout);
    assert!(r.status.success(), "{stdout}\n{}", String::from_utf8_lossy(&r.stderr));
    assert!(stdout.contains(": ran"), "{stdout}");
    let got: Vec<u32> = std::fs::read(&dp)
        .unwrap()
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let want: Vec<u32> = (0..128u32).map(|i| (i % 64) * 3 + i / 64).collect();
    assert_eq!(got, want);
    let spv_path = stdout.split(" -> ").nth(1).and_then(|r| r.split(':').next()).expect("spv path printed");
    let spv = std::fs::read(spv_path.trim()).unwrap();
    assert_eq!(&spv[..4], &0x0723_0203u32.to_le_bytes());
}
