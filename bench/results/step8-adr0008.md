# Passo 8: motore ADR-0008 contro la baseline

Controllo di prestazioni del passo 8 di `CHECKLIST.md`, dopo il cambio del motore
(passo 7). Si confronta con `baseline-adr0008.md`, con le stesse condizioni.

- Commit: `feb5c3f` più la migrazione dei benchmark di questo passo
- Julia 1.13.0, Apple M4 Pro (14 thread), macOS
- Data: 2026-10-02

## `bench/astrofit_vs_handwritten.jl`

| | baseline | passo 8 | vincolo |
|---|---:|---:|---|
| render | 10792 ns, 0.80x | 10792 ns, **0.80x** (3 alloc) | ≤1.0x |
| chi2 | 12917 ns, 0.95x | 11167 ns, **0.82x** (0 alloc) | ≤1.0x |
| gradiente | 34041 ns, 1.17x | 33667 ns, **1.17x** (7 alloc) | ≤ baseline |
| ottimizzazione (LBFGS) | 52.73 ms, 1.21x | 47.17 ms, 1.08x | |

## `bench/gradient_benchmark.jl`

| | baseline chi2 | passo 8 chi2 | baseline gradiente | passo 8 gradiente |
|---|---:|---:|---:|---:|
| AstroFit | 12917 ns (0) | 11167 ns (0) | 45000 ns (7) | 30500 ns (7) |
| senza `@fastmath` | 11625 ns | 11625 ns | 24667 ns | 24666 ns |

La riga del gradiente AstroFit è rumorosa tra un'esecuzione e l'altra (45.0 µs
nella baseline, 56.0 µs al passo 4, ora 30.5 µs): si giudica sulla riga di
`astrofit_vs_handwritten`.

## `bench/benchmarks.jl`

| | baseline | passo 8 |
|---|---:|---:|
| Hα + [NII], rapporto render | 0.96x | 0.96x |
| `withparams` | 3.5 ns | 3.6 ns |
| scaling N = 2 … 64, rapporto | 0.90–0.97x | 0.90–0.92x (N ≤ 16) |

## Suite `benchmark/` (mediana, ns)

`render!` è diventato `out .= render.(m, xs)`, mentre le chiavi restano le stesse.
`profiles` 2D usa `Coords(col, col)` al posto di colonna × riga.

| chiave | baseline | passo 8 |
|---|---:|---:|
| profiles/Gaussian2D/grid! | 287250 | 291416 |
| profiles/Gaussian2D/sum! | 1725000 | 566417 |
| profiles/Gaussian2D/compiled! (nuova) | — | 566500 |
| profiles/Sersic2D/grid! | 545500 | 546646 |
| profiles/Sersic2D/sum! | 2536541 | 1082333 |
| profiles/Sersic2D/compiled! (nuova) | — | 1082625 |
| profiles/Voigt1D/render! | 2050.9 | 1975.0 |
| profiles/Voigt1D/sum! | 48084 | 3994.8 |
| render/Gaussian1D astrofit / hw | 4607.1 / 4601.3 | 4601.1 / 4601.1 |
| render/Lorentzian1D astrofit / hw | 583.8 / 586.3 | 584.9 / 586.3 |
| render/BlackBody1D astrofit / hw | 4601.3 / 4601.3 | 4601.1 / 4601.3 |
| render/Voigt1D | 4696.4 | 4619.1 |
| render/BrokenPowerLaw1D | 5645.8 | 5201.3 |
| render/PowerLaw1D | 17250 | 17209 |
| render/Exponential1D | 4613.0 | 4607.1 |
| withparams 1G / 8G / 64G | 3.8 / 4.5 / 47.3 | 3.8 / 4.5 / 47.0 |
| mixed/pointwise/512 render! / objective / gradiente | 1970.8 / 1975.0 / 5791.7 | 1966.7 / 1975.0 / 5916.7 |
| mixed/pointwise/4096 render! / objective / gradiente | 15709 / 15833 / 43500 | 15708 / 15833 / 44667 |

`grid` (100 `withparams` + render in place): tutti i rapporti lib/hw sono uguali alla
baseline (1.00x per 1G/2G/4G, 0.88x per 64G).

Tutte le righe in place, `objective` e `profiles` hanno 0 allocazioni. `compiled!`
coincide con `sum!`, quindi i livelli `Leaf` e `CompiledModel` non costano niente
sul percorso sui punti. Le righe `mixed` con `GaussianPSF` della baseline non
esistono più (passo 7).

## Albero 2D reale contro il codice scritto a mano (caso `IMG` di `bench_fixtures.jl`)

Quattro foglie (2 × Gaussian2D, 2 × Sersic2D) con dei tie, 20 parametri liberi,
100×100 `Coords`, confrontate con `hand_img_render!` (un solo ciclo, tie risolti).
Nessuna baseline: prima del passo 8 il caso non veniva misurato.

| | AstroFit | scritto a mano | rapporto |
|---|---:|---:|---:|
| `out .= render.(withparams(cm, p), pts)` | 658.1 µs (0 alloc) | 653.4 µs | 1.007x |
| `f(p)` (χ²) | 682.0 µs (0 alloc) | — | |

## Controllo dell'uscita nel codice generato

`code_llvm` di `render(Gaussian1D(), 0.5)` e di `render(cm, (1.0, 2.0))` (Gaussian2D
+ Sersic2D in un `CompiledModel`) non contiene nessun `throw` di AstroFit. L'unico
`throw` è il `DomainError` di `sqrt` di Base dentro Sersic2D, presente anche nel
modello da solo.
