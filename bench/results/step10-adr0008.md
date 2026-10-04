# Passo 10: `ObjectiveFunction` e caso `Coords`

Controllo di prestazioni del passo 10 di `CHECKLIST.md`. Si confronta con
`baseline-adr0008.md` e `step8-adr0008.md`, con le stesse condizioni.

- Commit: `6255827` più le modifiche di questo passo (`@inline` sugli `evaluate` 1D)
- Julia 1.13.0, Apple M4 Pro (14 thread), macOS
- Data: 2026-10-02

## Il gradiente 1D era bimodale

Prima di questo passo la riga gradiente di `astrofit_vs_handwritten` valeva 1.17x
oppure 1.38x a seconda del processo (33.9 µs oppure 39.8 µs), con codice identico:
lo si vede rilanciando due volte lo stesso stato. Non dipendeva da `Const2D` né da
altre modifiche (verificato con `src/` del passo 8 e con una definizione placebo).

Causa: con i duali di ForwardDiff il corpo di `evaluate(::Gaussian1D, x)` supera la
soglia di inlining, e il χ² chiamava `evaluate` fuori linea tre volte per punto
(`bl _j_evaluate` nel codice nativo). Una chiamata nel ciclo caldo rende il tempo
sensibile alla posizione del codice JIT. Correzione: `@inline` sugli `evaluate` 1D
non banali (Gaussian1D, Lorentzian1D, Voigt1D, PowerLaw1D, BlackBody1D,
BrokenPowerLaw1D, Exponential1D), come già per i 2D al passo 4.

## `bench/astrofit_vs_handwritten.jl` (3 esecuzioni, tutte uguali)

| | baseline | passo 8 | passo 10 | vincolo |
|---|---:|---:|---:|---|
| render | 0.80x | 0.80x | **0.80x** | ≤1.0x |
| chi2 | 0.95x | 0.82x | **0.82x** (0 alloc) | ≤1.0x |
| gradiente | 34.0 µs, 1.17x | 33.7 µs, 1.17x | **22.8 µs, 0.79x** | ≤ baseline |
| ottimizzazione (LBFGS) | 1.21x | 1.08x | 0.84x | |

## `bench/gradient_benchmark.jl`

| | chi2 | gradiente |
|---|---:|---:|
| AstroFit | 11167 ns (0) | 22875 ns (7) |
| senza `@fastmath` | 11625 ns | 24708 ns |

AstroFit / senza `@fastmath`: 0.92x sul gradiente.

## `bench/benchmarks.jl`

Hα + [NII] render 0.96x, `withparams` 3.8 ns, scaling N = 2 … 64: 0.90–0.97x
(invariati rispetto al passo 8).

## Suite `benchmark/` (mediana)

Nuovo gruppo `mixed/coords/100`: `Gaussian2D + Const2D` su 100×100 `Coords`, 7 liberi,
contro `hand_coords_chi2` (doppio ciclo sugli assi, stessa formula).

| chiave | AstroFit | scritto a mano | rapporto | vincolo |
|---|---:|---:|---:|---|
| coords/100 objective | 53.5 µs (0 alloc) | 60.0 µs | **0.89x** | ≤1.0x |
| coords/100 gradient | 224.9 µs | 193.4 µs | **1.16x** | ≤1.17x |

Il χ² 2D con i duali ha già `evaluate` in linea: le uniche chiamate fuori linea sono
il costruttore di `Gaussian2D` (una volta, in `withparams`) ed `exp`, che chiama anche
il codice scritto a mano.

Altre righe rispetto al passo 8 (ns):

| chiave | passo 8 | passo 10 |
|---|---:|---:|
| mixed/pointwise/512 gradient | 5916.7 | 4327.4 |
| mixed/pointwise/4096 gradient | 44667 | 32125 |
| mixed/pointwise/4096 objective | 15833 | 15750 |
| profiles Gaussian2D grid! / sum! | 291416 / 566417 | 290833 / 564792 |
| profiles Sersic2D grid! / sum! | 546646 / 1082333 | 547417 / 1086833 |
| profiles Voigt1D render! / sum! | 1975.0 / 3994.8 | 1975.0 / 4036.5 |
| withparams 1G / 8G / 64G | 3.8 / 4.5 / 47.0 | 3.8 / 4.3 / 47.0 |
| grid lib/hw | 1.00x (1G–4G), 0.88x (64G) | 1.00x (1G–4G), 0.88x (64G) |

Le righe `render/*` sono invariate (Gaussian1D 4601 ns = handwritten).
