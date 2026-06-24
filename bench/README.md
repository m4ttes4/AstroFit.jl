# Benchmarks — `withparams` overhead is nearly zero

**Thesis.** A handwritten function with constraints and ties hardcoded is the
natural performance baseline: it is fast, but it is hard to reuse and compose.
AstroFit should get very close to that baseline while keeping constraints in the
model layer.

The headline comparison is therefore:

```julia
render(withparams(cm, p), x)      # AstroFit constrained model
handwritten_constrained(p, x)     # same constraints hardcoded in a function
```

`withparams` is timed separately because it is where AstroFit scatters the flat
parameter vector back into the model and resolves tied parameters. Those steps
are compiled from the constraint spec by generated functions, so the hot path is
straight-line code rather than runtime constraint lookup.

## Run

```bash
julia --project=. bench/benchmarks.jl
```

The script expects AstroFit, BenchmarkTools and Plots to be available in the
active project environment. It prints tables with times,
allocations and AstroFit/handwritten ratios, then writes two figures under
`bench/results/<timestamp>` by default. Pass an output directory as the first
argument, or set `ASTROFIT_BENCH_OUTDIR`, to choose a stable path without
overwriting older results.

## What it measures

Each benchmark uses AstroFit's public authoring API (`@model`, `@constrain`,
`withparams`, `render`, `params`). Before timing, the script asserts that
the handwritten constrained kernel matches the AstroFit constrained model.

- **Fixed cases:**
  - `Linear1D + Gaussian1D` with a fixed continuum slope and bounded line
    parameters.
  - Hα + [NII], where the [NII] amplitudes, centroids and widths are tied to
    Hα.
- **Scaling case:**
  - a sum of `N ∈ {1,2,4,8,16}` Gaussians;
  - every component has free centroid and width;
  - all amplitudes are tied to one master amplitude.

For each case the script reports:

- `withparams(cm, p)` alone;
- `render(withparams(cm, p), x)`;
- the equivalent handwritten constrained render;
- the `AstroFit / handwritten` render ratio;
- a secondary AstroFit-only free-vs-constrained check.

## Reading the results

The ratio is the main number. A value near `1.0x` means that the ergonomic
AstroFit model evaluates about as fast as the hardcoded constrained function.

The `withparams` row isolates the abstraction cost. If it remains tiny and
allocation-free, ties are being resolved by compiled code rather than by a
runtime name lookup or constraint dispatch layer.

The free-vs-constrained rows are secondary. They answer a narrower question:
adding bounds, fixes and ties to an AstroFit model should not by itself create a
large runtime penalty.

## Figures

- `fixed_complex.png` — fixed-case ratio bars plus the isolated `withparams`
  timing.
- `scaling.png` — scaling ratio, isolated `withparams` time, and constrained
  render time for AstroFit vs the handwritten kernel.

# Optimization benchmarks — fit-level overhead

`optimization_benchmarks.jl` sits one layer above `benchmarks.jl`: instead of
`render`, it times the **fit**. For each case (the same `kernels.jl` models) it
compares AstroFit against a handwritten loss on three metrics:

- `objective f(p)` — `ObjectiveFunction(cm, x, y, err)` vs a handwritten `chi2`;
- `gradient ∇f(p)` — `ForwardDiff` through `withparams`+`render` vs through the
  handwritten loss. **This is the headline:** `AutoForwardDiff` pushes `Dual`
  numbers through `withparams` (tree rebuild + tie resolution every call), so an
  abstraction regression shows up here first;
- `solve(LBFGS)` — `OptimizationProblem(cm, …)` vs a hand `OptimizationFunction`.

The handwritten `chi2` kernels are scalar and non-allocating (mirroring
AstroFit's `chi2` loop) so the baseline does not pour `Dual`-array allocations
into the gradient and skew the ratio. Each case asserts `f(p0) ≈ hand(p0)` and
gradient equality before timing — the guard against constraints being hardcoded
differently in the hand kernel.

`solve` is the noisy metric: every case is bounded → `Fminbox(LBFGS)`, sensitive
to ULP-level objective differences, so the two solves can land on different
iteration counts. `iters_A vs iters_H` prints beside the ratio and it is flagged
`⚠` when they diverge; obj+grad stay the clean signal. The `solve` timing is
skipped above ~17 free parameters (a single bounded solve dwarfs the time budget,
giving a slow 1-sample median).

## Run

This script needs Optimization, OptimizationOptimJL and ForwardDiff, which are
weak/test deps — `--project=.` cannot load them. A dedicated `bench/`
environment (devs AstroFit + adds the five packages) keeps the numbers
reproducible across commits:

```bash
julia --project=bench bench/optimization_benchmarks.jl
```

It prints per-case tables and writes `optimization.csv` (medians + ratios +
iteration counts) under `bench/results/<timestamp>`. To track a `withparams`
change: snapshot the CSV on the current commit, apply the change, re-run, diff —
obj+grad are where a regression would surface.
