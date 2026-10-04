# Baseline prima del refactor ADR-0008

Punto di partenza per i controlli di prestazioni dei passi 4, 8, 10 e 13 di `CHECKLIST.md`.

- Commit: `6d33edc` (branch `refactor/render-redesign`), `src/` senza modifiche
- Julia 1.13.0, Apple M4 Pro (14 thread), macOS
- Data: 2026-10-02
- `Pkg.test()`: 456/456 passati

## `bench/astrofit_vs_handwritten.jl`

`julia --project=bench`. Hα + [NII], 5 parametri liberi, 1000 punti. Statistica: quella dello script.

| | AstroFit | Scritto a mano | Rapporto |
|---|---:|---:|---:|
| render | 10792 ns (3 alloc) | 13459 ns (3 alloc) | 0.80x |
| chi2 | 12917 ns (0 alloc) | 13667 ns (1 alloc) | 0.95x |
| gradiente | 34041 ns (7 alloc) | 29083 ns (9 alloc) | 1.17x |
| ottimizzazione (LBFGS) | 52.73 ms (22501 alloc) | 43.62 ms (23863 alloc) | 1.21x |

Gradiente e ottimizzazione partono già sopra 1.0x: per queste righe il vincolo realistico è "≤ baseline".

## `bench/gradient_benchmark.jl`

`julia --project=examples`. Stesso modello, 5 parametri liberi, 1000 punti.

| | chi2 | alloc | gradiente | alloc |
|---|---:|---:|---:|---:|
| AstroFit | 12917 ns | 0 | 45000 ns | 7 |
| monolitico | 11250 ns | 1 | 72000 ns | 9 |
| split | 11125 ns | 1 | 70666 ns | 9 |
| senza `@fastmath` | 11625 ns | 1 | 24667 ns | 9 |

Gradiente AstroFit / senza `@fastmath`: **1.82x**. Il gradiente AstroFit vale 45 µs qui e 34 µs nello script precedente: si confronta sempre con la riga dello stesso script.

## `bench/benchmarks.jl`

`julia --project=examples`. Hα + [NII], 5 parametri liberi, 151 punti:

| | |
|---|---:|
| solo `withparams` | 3.5 ns, 0 alloc |
| `render(withparams)` AstroFit | 1650.0 ns, 2 alloc |
| render scritto a mano | 1712.5 ns, 2 alloc |
| rapporto | 0.96x |

Scaling, N gaussiane con tutte le ampiezze legate:

| N | liberi | withparams | AstroFit | scritto a mano | rapporto |
|---:|---:|---:|---:|---:|---:|
| 2 | 5 | 3.0 ns | 2986.1 ns | 3218.8 ns | 0.93x |
| 4 | 9 | 3.9 ns | 5743.0 ns | 6300.0 ns | 0.91x |
| 8 | 17 | 5.0 ns | 11250.0 ns | 12417.0 ns | 0.91x |
| 16 | 33 | 16.3 ns | 22292.0 ns | 24750.0 ns | 0.90x |
| 32 | 65 | 34.5 ns | 44417.0 ns | 49375.0 ns | 0.90x |
| 64 | 129 | 70.1 ns | 98334.0 ns | 101458.0 ns | 0.97x |

## Suite `benchmark/` (`SUITE`)

Eseguita con `tune!` e `run`, parametri BenchmarkTools di default, in un ambiente temporaneo con `Pkg.develop(path=".")` più BenchmarkTools e ForwardDiff. Statistica: **mediana**. Tempi in ns.

### render (`X1`, 1201 punti, `render!`)

| modello | astrofit | scritto a mano | rapporto |
|---|---:|---:|---:|
| Gaussian1D | 4607.1 | 4601.3 | 1.00x |
| Lorentzian1D | 583.8 | 586.3 | 1.00x |
| BlackBody1D | 4601.3 | 4601.3 | 1.00x |
| Voigt1D | 4696.4 | — | |
| BrokenPowerLaw1D | 5645.8 | — | |
| Exponential1D | 4613.0 | — | |
| PowerLaw1D | 17250.0 | — | |

Tutte le righe: 0 alloc.

### withparams (0 alloc)

| 1G | 2G | 4G | 8G | 16G | 32G | 64G |
|---:|---:|---:|---:|---:|---:|---:|
| 3.8 | 3.8 | 3.8 | 4.5 | 9.0 | 16.4 | 47.3 |

### grid (100 `withparams` + `render!` con `p` casuale)

| caso | 128 punti lib / hw | 1024 punti lib / hw | 8192 punti lib / hw |
|---|---:|---:|---:|
| 1G | 52500 / 52958 (0.99x) | 400625 / 400250 (1.00x) | 3205625 / 3196417 (1.00x) |
| 2G | 99417 / 99250 (1.00x) | 762292 / 762063 (1.00x) | 6121479 / 6096083 (1.00x) |
| 4G | 195292 / 194458 (1.00x) | 1494833 / 1494125 (1.00x) | 11990917 / 11959979 (1.00x) |
| 64G | 3334833 / 3787458 (0.88x) | 24020000 / 27415625 (0.88x) | 189021750 / 214042771 (0.88x) |

Allocazioni uguali tra lib e hw: 2 per ogni chiamata, cioè il vettore `p`.

### profiles (2D e Voigt, 0 alloc)

| chiave | mediana |
|---|---:|
| Gaussian2D/grid! | 287250 |
| Gaussian2D/sum! | 1725000 |
| Sersic2D/grid! | 545500 |
| Sersic2D/sum! | 2536541 |
| Voigt1D/render! | 2050.9 |
| Voigt1D/sum! | 48084 |

Righe di riferimento per il passo 4 (migrazione a `_cache_`).

### mixed

| caso/n | render | render! | objective | gradiente |
|---|---:|---:|---:|---:|
| pointwise/512 | 2092.6 (3 alloc) | 1970.8 (0) | 1975.0 (0) | 5791.7 (7) |
| pointwise/4096 | 17000 (3) | 15709 (0) | 15833 (0) | 43500 (7) |
| convolved/512 † | 6566.6 (11) | 6450.0 (8) | 6483.4 (8) | 12500 (15) |
| convolved/4096 † | 50209 (11) | 50041 (8) | 50167 (8) | 91959 (15) |
| chained/512 † | 10667 (13) | 10708 (13) | 10750 (13) | 18208 (20) |
| chained/4096 † | 82458 (13) | 83208 (13) | 83500 (13) | 136625 (20) |
| transformed/512 † | 6566.6 (11) | 6441.8 (8) | 6516.6 (8) | 14542 (15) |
| transformed/4096 † | 50208 (11) | 49959 (8) | 50250 (8) | 110000 (15) |

† usa `GaussianPSF`: la riga sparisce al passo 7 ed è esclusa dai confronti successivi. Al passo 8 resta confrontabile solo `pointwise`, senza la colonna `render!`, che diventa `out .= render.(m, xs)`.
