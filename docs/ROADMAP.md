# Plan: od emulatora do uruchomionej gry

Punkt wyjścia: architektura jak w projektach, które dochodzą do rozgrywki (kod gry
wykonuje się natywnie na x86-64, system jest emulowany wysokopoziomowo). Mamy loader
PE, trampolinę importów, TEB/TLS, SEH, wątki, synchronizację, pliki/VFS i zalążek
XGameRuntime; po stronie GPU dekoder i interpreter RDNA2 oraz translator do SPIR-V
sprawdzany różnicowo. Brakuje bibliotek konsoli, warstwy grafiki D3D12.X i pokrycia
prawdziwego korpusu shaderów; translator nie był jeszcze uruchamiany na prawdziwym GPU.

Zasady na cały plan: clean-room (bez kodu z projektów GPL: Leonida, VIctory i forki,
AnyXboxX; ich opisy co najwyżej jako dane), semantyka z publicznych źródeł (LLVM,
Mesa, specyfikacja ISA AMD, Khronos) albo z obserwacji zachowania gry na naszym
runtime; interpreter odmawia zamiast zgadywać. Pliki gry tylko z własnej kopii, już
czytelne na dysku, trzymane poza repozytorium; nic z gry nie trafia do repozytorium,
niczego nie deszyfrujemy ani nie obchodzimy zabezpieczeń.

## Etap 0: Windows i GPU (krótki)

- Repozytorium na PC z RTX: Rust (MSVC), Vulkan SDK (`spirv-val`), sterowniki.
- Uruchamianie przetłumaczonych shaderów na prawdziwym Vulkanie z porównaniem z
  interpreterem (ten sam test różnicowy co dziś, GPU zamiast ewaluatora).
- Kamień milowy: testy translatora przechodzą na RTX; różnice względem ewaluatora
  (fuzja Fma, denormale, wave64 na podgrupach 32-wątkowych) są spisane.

## Etap 1: inwentarz importów gry

- `scarlett inventory <katalog gry>`: wszystkie obrazy PE paczki, każdy import z
  dostawcą (moduł gościa w paczce, nasze HLE, host Windows, brak), liczniki na DLL i
  na moduł docelowy, raport tekstowy/Markdown poza repozytorium.
- Drugi wskaźnik postępu obok procentu ISA: odsetek importów z dostawcą.
- Kamień milowy: pełna lista brakujących funkcji z priorytetami.

## Etap 2: start gry do pierwszego wywołania grafiki

- `kernelx`: przekierowanie na `kernel32`/`kernelbase` tam, gdzie zachowanie jest
  identyczne; reszta w HLE.
- `xmem` (`XMemAlloc`, pamięć graficzna), GDK runtime (`XTaskQueue`/`XAsync`,
  lokalny `XUser`, `XSystem`, `XPackage`, `XGameSave`), DirectStorage
  (`DStorageGetFactory`) na wątkach roboczych przez VFS.
- Inicjalizacja CRT gry i jej DLL na naszym loaderze; każdy nieznany import
  logowany przez `policy`.
- Kamień milowy: gra dochodzi do `D3D12XboxCreateDevice`, log pokazuje następne braki.

## Etap 3: D3D12.X → Vulkan do pierwszej klatki menu

- Urządzenie, kolejki, listy poleceń, sterty deskryptorów, zasoby i pamięć (pamięć
  wspólna: shadery czytają adresy GPU wprost, więc mapowanie adresów gościa na bufory
  Vulkana).
- Pipeline'y: blob RDNA2 z opisu stanu → translator → SPIR-V, z cache na dysku.
- `PresentX` → swapchain.
- Konsolowe sloty interfejsów mierzymy sami (logowanie wywołań gry), nie przepisujemy
  ustaleń z projektów GPL.
- Kamień milowy: menu gry na ekranie.

## Etap 4: korpus shaderów (równolegle z 2–3)

- Narzędzie wyciągające programy RDNA2 z paczki, tłumaczenie całości, licznik odmów
  per instrukcja; kolejność prac nad ISA wynika z danych.
- Duże brakujące bloki: shadery mesh/NGG i geometry, `image_sample` i filtrowanie,
  formaty BCn i sRGB, kafelkowanie `_X`/`_D`.
- Kamień milowy: > 95% korpusu tłumaczy się i przechodzi `spirv-val`.

## Etap 5: świat gry

- Tekstury i streaming (układy pamięci), dispatche > 65 535 grup, wejście
  (`GameInput`), audio (XAudio2, dekoder XMA).
- Kamień milowy: pierwsza klatka świata i ruch postaci.

## Etap 6: grywalność

- Wydajność (równoległa kompilacja pipeline'ów, cache), śledzenie promieni (BVH),
  stabilność długich sesji.

## Ryzyka

- Dziesiątki tysięcy shaderów i pipeline'ów: cache i tłumaczenie w tle są konieczne.
- Shadery mesh wymagają `VK_EXT_mesh_shader` (RTX 4070 ma).
- Clean-room jest wolniejszy: każdą funkcję konsoli ustalamy sami.

## Postęp

- ISA RDNA2: odsetek mnemoników GFX10 (z testów asemblera LLVM) nazwanych przez
  dekoder, wprowadzanych razem z implementacją w interpreterze i translatorze.
- Importy: odsetek importów paczki z dostawcą (etap 1).
