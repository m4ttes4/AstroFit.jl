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
cd /home/matteo/julia/AstroFit.jl
julia --project=/home/matteo/.julia/environments/v1.12 bench/benchmarks.jl
```

The script uses the Julia environment in `/home/matteo/.julia/environments/v1.12`,
which provides AstroFit, BenchmarkTools and Plots. It prints tables with times,
allocations and AstroFit/handwritten ratios, then writes two figures.

## What it measures

Each benchmark uses AstroFit's public authoring API (`@model`, `@constrain`,
`withparams`, `render`, `paramvector`). Before timing, the script asserts that
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
