# scarlett

Clean-room emulator Xboxa Series w Ruście, host: Windows x86-64.

Kod gościa (x86-64, Zen 2) wykonuje się **natywnie**. Emulujemy system (HLE), nie CPU.
Importy obrazu PE trafiają do handlerów w Ruście przez trampolinę Win64.

## Status: etap 1–3 (fundament)

| crate | co robi |
|---|---|
| `scarlett-pe` | parser PE32+: nagłówki, sekcje, importy (+ delay-load), eksporty, relokacje, TLS, `.pdata`; bez `unsafe`, każdy odczyt sprawdza granice. Feature `builder` generuje syntetyczne PE do testów. |
| `scarlett-hle` | rejestr `dll!symbol → handler`, globalna tablica thunków, naked trampolina Win64 (4 argumenty w rejestrach i dowolnie dużo na stosie), tryb strict, statystyki wywołań. |
| `scarlett-loader` | mapowanie obrazu (najpierw pod preferowaną bazę), relokacje DIR64/HIGHLOW, wiązanie importów (HLE → host forward → stub), ochrona stron per sekcja (W^X), `RtlAddFunctionTable` dla wyjątków, uruchomienie entry/TLS callbacks na wątku ze stosem z nagłówka. |
| `scarlett` (cli) | `inspect` i `run`. |

```
cargo test
cargo run -p scarlett-loader --example make_sample
cargo run -p scarlett -- inspect sample.exe
cargo run -p scarlett -- run sample.exe [--strict] [--no-preferred-base] [--host-forward]
```

Po `run` CLI wypisuje listę niezaimplementowanych importów, które gra faktycznie wywołała,
posortowaną po liczbie wywołań. To jest twoja lista TODO.

### Testowane

15 testów, w tym end-to-end: zbudowany PE z ręcznie złożonym x86-64 ładuje się,
relokuje, wykonuje natywnie i woła handlery w Ruście (6 argumentów, w tym 2 ze stosu).

**Uwaga:** projekt był budowany i testowany na Linux x86-64 (ta sama ścieżka ABI Win64).
Kod `cfg(windows)` (`VirtualAlloc`, `RtlAddFunctionTable`, `LoadLibraryA`) ma sygnatury
zweryfikowane ze źródłami `windows-sys 0.59`, ale nie był jeszcze kompilowany pod MSVC.
Pierwsze `cargo test` na Windowsie jest więc też pierwszą weryfikacją tej części.

## Wejście

Rozpakowany plik PE z **legalnego** źródła: własne programy z Dev Mode, sample z GDK.
Rozpakowywanie i deszyfrowanie paczek konsolowych jest poza zakresem projektu.

## Roadmapa

4. **TLS i TEB**: statyczny TLS (`gs:[0x58]`), indeks TLS, `__declspec(thread)`. Teraz loader odrzuca takie obrazy.
5. **xmm w trampolinie**: przekazywanie argumentów i zwrotów float/SIMD.
6. **Moduły HLE** per DLL (wątki, sync, pamięć, pliki, czas) i mikrotesty z Dev Mode dla każdego API.
7. **GPU**: parser command bufferów D3D12, rekompilator ISA RDNA2 → SPIR-V, backend Vulkan.
8. **Audio, input, I/O**: dekompresja assetów w software.
