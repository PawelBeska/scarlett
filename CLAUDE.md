# scarlett: instrukcje dla Claude

Clean-room emulator Xboxa Series w Ruście (kod gościa x86-64 wykonuje się natywnie, system
emulowany HLE; GPU: PM4, interpreter RDNA2, translator RDNA2 → SPIR-V). Stan i plan:
`README.md`, `docs/ROADMAP.md`. Odpowiadaj po polsku.

## Zasady, których się trzymamy

- **Clean-room.** Nie kopiujemy kodu z projektów GPL ani innych: Leonida, VIctory i jego forki
  (np. victory-amd), AnyXboxX, XCL. Ich README, tytuły PR i issue można czytać jako dane
  (co gra używa, co działa), nie jako źródło implementacji.
- **Semantyka tylko z publicznych źródeł**, wskazanych w komentarzu przy kodzie:
  - specyfikacja ISA RDNA2 od AMD w formie XML (`amdgpu_isa_rdna2.xml` z archiwum
    https://gpuopen.com/download/machine-readable-isa/latest/): opisy instrukcji, kolejność
    argumentów, warunki SCC, opcode'y;
  - LLVM AMDGPU (`llvm/lib/Target/AMDGPU/*.td`, `SIISelLowering.cpp`, `AMDGPUInstCombine*`,
    `ConstantFolding.cpp`) i testy asemblera `llvm/test/MC/AMDGPU/gfx10_*` (kodowania);
  - Mesa (ACO, NIR `nir_opcodes.py`, RADV), Khronos (SPIR-V, Vulkan).
- **Odmowa zamiast zgadywania.** Gdy źródła nie ustalają wyniku (bity NaN, tryb denormali,
  remis w `s_min`, adresy w aperturach FLAT…), interpreter zwraca błąd, a nie wymyśla wartość.
- **Pliki gry** tylko z kopii użytkownika, już czytelne na dysku, poza repozytorium. Nic z gry
  (shadery, zasoby, exe, raporty z nazwami z gry) nie trafia do repozytorium. Nie deszyfrujemy
  i nie obchodzimy zabezpieczeń.
- **Pobrany kod** (referencje LLVM/Mesa itd.) czytamy, nie uruchamiamy; trzymamy poza repo.

## Jak dodajemy instrukcje GPU

1. Źródło semantyki (patrz wyżej) i opcode sprawdzony na kodowaniu z testów LLVM.
2. Stała i mnemonik w `crates/gpu/src/rdna.rs`, wykonanie w `interp.rs`, tłumaczenie w
   `translate.rs` (bez typów, których ewaluator nie zna; f16/f64 bywają liczone na bitach).
3. Test w `crates/gpu/tests/translate.rs`: ręcznie policzone przypadki brzegowe plus test
   różnicowy `check` / `check_all` / `check_mem` (interpreter jako wyrocznia, translator przez
   `spirv_eval`, wave32 i wave64, pełna i rzadka maska `exec`).
4. Mutacje translatora: kopia pliku, podmiana fragmentu, uruchomienie testu, przywrócenie; każda
   ocalała mutacja to brakujący przypadek w teście albo mutacja równoważna (wtedy ją nazywamy).
5. Pełne testy i clippy: `cargo test --workspace`, `cargo clippy --workspace --all-targets`
   (na Macu arm64 z `--target x86_64-apple-darwin`; sprawdzamy też `x86_64-pc-windows-msvc` i
   `x86_64-unknown-linux-gnu`). README: opis instrukcji i liczba testów.

## Postęp

- Po każdej partii instrukcji podaj „Gotowość: X%” z
  `python3 -I tools/isa_progress.py <llvm-project>/llvm/test/MC/AMDGPU` i podziałem per prefiks.
- Importy gry: `scarlett inventory <katalog gry> --md <raport poza repo>`.

## Git

- Commituj, gdy użytkownik poprosi albo gdy pracujesz w pętli, którą zlecił; push tylko na
  wyraźną prośbę. Commit kończy się linią `Co-Authored-By` modelu.
- Żadnych sekretów ani plików gry w commitach.
