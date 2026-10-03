# Passo 13: verifica finale del refactor ADR-0008

Controllo di prestazioni del passo 13 di `CHECKLIST.md`, contro la baseline del passo 1
(`baseline-adr0008.md`).

- Codice: `ea3f2a5` (dopo Runic), misurato in un worktree pulito; baseline `6d33edc`
- Julia 1.13.0, Apple M4 Pro (14 thread), macOS
- Data: 2026-10-02

**I tempi assoluti non sono confrontabili con il passo 1.** Dopo un riavvio della macchina
ogni tempo, AstroFit e scritto a mano, è circa il 35% più basso (`render/Gaussian1D/handwritten`,
codice invariato: 4601.3 → 2976.9 ns, fattore 0.647). Per questo:

- `astrofit_vs_handwritten` è stato rimisurato **sulla baseline e sul codice finale nelle stesse
  condizioni** (due worktree, esecuzioni alternate);
- le righe della suite `benchmark/` senza controparte scritta a mano sono confrontate con la
  baseline **dopo** la divisione per 0.647 (colonna "normalizzato").

## `bench/astrofit_vs_handwritten.jl`: baseline e finale nelle stesse condizioni

Esecuzioni alternate base, finale, base, finale, base, finale, ognuna con lo script e
l'ambiente `bench` del proprio albero. "Dopo LBFGS" è il gradiente rimisurato alla fine dello
stesso processo.

| | baseline `6d33edc` | finale `ea3f2a5` | vincolo |
|---|---|---|---|
| render | 0.80x, 0.80x, 0.80x | **0.82x, 0.80x, 0.80x** | ≤1.0x ✅ |
| χ² | 0.95x, 0.95x, 0.95x | **0.82x, 0.82x, 0.83x** | ≤1.0x ✅ |
| gradiente | 2.28x, 1.83x, 1.83x | **0.79x, 0.79x, 0.79x** (14.7 µs) | ≤ baseline ✅ |
| gradiente dopo LBFGS (AstroFit / scritto a mano) | 27.9 / 18.6, 22.0 / 18.6, 22.0 / 18.6 µs | 14.7 / 18.5, 14.8 / 18.6, 14.8 / 18.6 µs | |
| ottimizzazione (LBFGS) | 1.16x, 1.19x, 1.18x | 0.83x, 0.82x, 0.83x | |

La baseline varia tra un processo e l'altro e dentro lo stesso processo; il codice finale no,
in queste sei esecuzioni.

**Anomalia non spiegata.** Nell'albero di lavoro principale, con `src/` identico a `ea3f2a5`, tre
esecuzioni consecutive hanno dato gradiente 1.83x (33.9 µs) e una quarta 0.79x, con 37.7 µs alla
rimisura dopo LBFGS. In quel momento un'altra sessione stava modificando `test/` e `Project.toml`
nello stesso albero. Il codice nativo del χ² con i duali era lo stesso del worktree pulito
(`evaluate` in linea, solo `exp` chiamata fuori linea). Le misure di questo file vengono dal
worktree pulito.

## `bench/gradient_benchmark.jl` (albero principale, `src/` = `ea3f2a5`)

| | χ² | alloc | gradiente | alloc |
|---|---:|---:|---:|---:|
| AstroFit | 7302.2 ns | 0 | 15042.0 ns | 7 |
| monolitico | 7291.5 ns | 1 | 46541.0 ns | 9 |
| split | 7208.4 ns | 1 | 45750.0 ns | 9 |
| senza `@fastmath` | 7516.6 ns | 1 | 16000.0 ns | 9 |

Gradiente AstroFit / senza `@fastmath`: **0.94x** (baseline 1.82x).

## `bench/benchmarks.jl` (albero principale)

Rapporto render Hα + [NII] 0.96x (baseline 0.96x); `withparams` 2.3 ns, 0 alloc. Scaling,
rapporto AstroFit / scritto a mano: N = 2, 4, 8, 16, 32, 64 → 0.92, 0.91, 0.90, 0.90, 0.90, 0.97x
(baseline 0.93, 0.91, 0.91, 0.90, 0.90, 0.97x).

## Suite `benchmark/` (worktree pulito, mediana, ns)

### Righe con controparte scritta a mano: rapporti

| chiave | baseline | finale |
|---|---:|---:|
| render/Gaussian1D astrofit / hw | 1.00x | 1.00x |
| render/Lorentzian1D astrofit / hw | 1.00x | 1.00x |
| render/BlackBody1D astrofit / hw | 1.00x | 1.00x |
| grid 1G, 2G, 4G lib / hw | 0.99–1.00x | 0.99–1.01x |
| grid 64G lib / hw | 0.88x | 0.87–0.88x |
| mixed/coords/100 objective / hw (nuova) | — | 0.89x |
| mixed/coords/100 gradient / hw (nuova) | — | 1.16x |

### Righe senza controparte: normalizzate per 0.647

| chiave | baseline | finale | normalizzato | rispetto alla baseline |
|---|---:|---:|---:|---:|
| profiles/Gaussian2D/grid! | 287250 | 188167 | 290830 | 1.01x |
| profiles/Sersic2D/grid! | 545500 | 353042 | 545660 | 1.00x |
| profiles/Gaussian2D/sum! | 1725000 | 366292 | 566140 | 0.33x |
| profiles/Sersic2D/sum! | 2536541 | 701458 | 1084170 | 0.43x |
| profiles/Voigt1D/render! | 2050.9 | 1279.2 | 1977.1 | 0.96x |
| profiles/Voigt1D/sum! | 48084 | 2592.6 | 4007.1 | 0.08x |
| render/Voigt1D | 4696.4 | 2986.1 | 4615.3 | 0.98x |
| render/BrokenPowerLaw1D | 5645.8 | 3395.8 | 5248.5 | 0.93x |
| render/PowerLaw1D | 17250 | 11125 | 17194 | 1.00x |
| withparams/64G | 47.3 | 30.1 | 46.5 | 0.98x |
| mixed/pointwise/4096 objective | 15833 | 10167 | 15714 | 0.99x |
| mixed/pointwise/4096 gradient | 43500 | 20750 | 32071 | 0.74x |

`profiles/*/compiled!` coincide con `sum!` (366250 e 701209). Le righe `mixed` con `GaussianPSF`
della baseline non esistono più (passo 7).

### Allocazioni

0 in tutte le righe in place, `objective`, `profiles` e `withparams`; `grid` 2 per chiamata
(il vettore `p`, uguale tra lib e hw); i gradienti 7 (la configurazione di ForwardDiff).

## Altri controlli

- Runic: nessuna differenza in `src/`, `ext/` e nei test, tranne `test/pigeons_tests.jl`, un file
  vuoto che il lavoro in corso su `test/` cancella.
- `Base.Docs.hasdoc`: vero per i 19 simboli dell'ADR ed `evaluate`.
- `Pkg.test()` nel worktree pulito: 338/338.
