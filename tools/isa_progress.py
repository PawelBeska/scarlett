#!/usr/bin/env python3
"""ISA progress: GFX10 mnemonics from LLVM's assembler tests that scarlett's
decoder names (a mnemonic is added together with its interpreter and
translator implementation, so this approximates the implemented ISA).

usage: python3 -I tools/isa_progress.py <dir with LLVM's llvm/test/MC/AMDGPU>

Get the test files with a sparse clone, outside the repository:
  git clone --filter=blob:none --sparse https://github.com/llvm/llvm-project.git
  cd llvm-project && git sparse-checkout set llvm/test/MC/AMDGPU
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PREFIXES = ("s_", "v_", "ds_", "buffer_", "tbuffer_", "global_", "flat_", "scratch_", "image_", "exp")


def llvm_mnemonics(test_dir: pathlib.Path) -> set[str]:
    names = set()
    files = sorted(test_dir.glob("gfx10_asm_*.s")) + [test_dir / "flat-global.s"]
    for f in files:
        if not f.exists():
            continue
        for line in f.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line or line[0] in "/#;.":
                continue
            tok = re.sub(r"_(e32|e64|sdwa|dpp)$", "", line.split()[0])
            if tok.startswith(PREFIXES):
                names.add(tok)
    return names


def our_mnemonics() -> set[str]:
    src = (ROOT / "crates/gpu/src/rdna.rs").read_text()
    names = set(re.findall(r'=> "([a-z0-9_]+)"', src))
    # VOPC names generated from interp::Compare by rdna::Inst::mnemonic
    for x in ("", "x"):
        for c in "f lt eq le gt lg ge o u nge nlg ngt nle neq nlt tru".split():
            names |= {f"v_cmp{x}_{c}_{t}" for t in ("f32", "f64", "f16")}
        for c in "f lt eq le gt ne ge t".split():
            names |= {f"v_cmp{x}_{c}_{t}" for t in ("i32", "u32", "i64", "u64")}
        for c in "lt eq le gt ne ge".split():
            names |= {f"v_cmp{x}_{c}_{t}" for t in ("i16", "u16")}
        names |= {f"v_cmp{x}_class_{t}" for t in ("f32", "f64", "f16")}
    # flat_* are the global_* names in segment 0
    names |= {"flat_" + n[len("global_"):] for n in names if n.startswith("global_")}
    return names


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    llvm = llvm_mnemonics(pathlib.Path(sys.argv[1]))
    if not llvm:
        sys.exit("no gfx10_asm_*.s files found there")
    have = llvm & our_mnemonics()
    print(f"{len(have)} / {len(llvm)} = {100 * len(have) / len(llvm):.1f}%")
    for p in ("s", "v", "ds", "buffer", "tbuffer", "global", "flat", "image"):
        a = sum(n.startswith(p + "_") for n in llvm)
        h = sum(n.startswith(p + "_") for n in have)
        print(f"  {p}: {h}/{a}")


if __name__ == "__main__":
    main()
