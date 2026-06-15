# Benchmarks — constraints are (almost) free

**Thesis.** AstroFit compiles each model *and its constraints* into straight-line
code at compile time (the `@generated` `free_lenses` / `_scatter` / `_resolve` in
`src/params.jl`). So **adding constraints — bounds, fixes, ties — does not
meaningfully change runtime**. A constrained model evaluates and differentiates as
fast as the same model with every parameter free, and both match a hand-written
kernel with the constraints baked directly into the code.

The mechanical reason: **ties are resolved inside `withparams`** (the `_resolve`
step), while **`render` walks the identical model tree regardless of constraints**.
So `withparams` is exactly where any constraint overhead would appear — that is the
headline measurement.

## Run

```bash
cd /home/matteo/julia/AstroFit.jl
julia bench/benchmarks.jl
```

Uses the global Julia env (`~/.julia/environments/v1.12`), which already has
AstroFit (dev-linked), BenchmarkTools, ForwardDiff and Plots — there is no
bench-local `Project.toml`. The script prints timing tables and writes two figures.

## What it measures

Each benchmark uses the public API only (`@model`, `@constrain`, `withparams`,
`render`, `paramvector`). Before timing, the script asserts the handwritten kernels
produce the same output as the compiled models (`≈`), so the comparison is
apples-to-apples.

- **Part A — fixed models, 4-way comparison** (`AstroFit free` / `AstroFit
  constrained` / `handwritten free` / `handwritten constrained`):
  - *Small*: `Linear1D + Gaussian1D`.
  - *Realistic*: the Hα + [NII] 4-Gaussian complex from
    `../examples/halpha_nii_fit.jl` (11 free params vs 5 once the [NII] doublet is
    tied to Hα).
  - Measures `withparams`, `render(withparams(cm,p),x)`, and a ForwardDiff gradient
    of the least-squares loss.
- **Part B — scaling sweep** (`AstroFit free` vs `AstroFit constrained`): a sum of
  `N ∈ {1,2,4,8,16}` Gaussians, built programmatically. Shows the two curves stay
  on top of each other as the model grows.

## Reading the results

- **`withparams`: ~2 ns and 0 allocations**, free *and* constrained, at every model
  size. This is the cleanest result: resolving ties is compiled away to nothing.
- **`render`: free ≈ constrained ≈ handwritten** (within noise) — the abstraction is
  zero-cost for evaluation; cost is set purely by how many components you draw.
- **`gradient`: the constrained model is *faster*, never slower.** Ties remove free
  parameters, so ForwardDiff carries fewer partials. AstroFit tracks the handwritten
  kernel closely; the small residual gap (most visible on the tiny model) is
  ForwardDiff threading `Dual`s through the optic-based scatter, not constraint
  overhead — it shrinks as the real per-point work grows.

Bottom line: **constraints are resolved at compile time, so they are essentially
free at runtime.** Use as many ties/bounds as the physics demands.

## Figures

- `scaling.png` — Part B: `withparams`, `render`, gradient vs `N`; free vs
  constrained overlap.
- `fixed_complex.png` — Part A: 4-way bars for the Hα/[NII] complex
  (blue = AstroFit, red = handwritten).
