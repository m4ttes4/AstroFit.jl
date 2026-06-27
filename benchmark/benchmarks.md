# AstroFit Benchmark Suite — Plan

Goal: a regression-tracking benchmark suite covering every hot path, runnable
locally and in CI via [AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl).

AirspeedVelocity discovers `benchmark/benchmarks.jl`, expects a top-level
`const SUITE = BenchmarkGroup()`, and runs it across two git revisions to flag
regressions. No bespoke runner — same `SUITE` works with `benchpkg` and with
plain `run(SUITE)`.

## Core principle: compare against a handwritten baseline

**Where possible, every benchmark ships a handwritten counterpart.** ASV
catches regressions *vs the previous AstroFit revision*; the handwritten
baseline catches a different, more dangerous failure: the abstraction silently
drifting away from the bare-loop floor across *all* revisions (a slow creep ASV
can't see because both sides regress together).

For each critical path, register two keys side by side:

- `…/astrofit` — the library call (`render!`, `chi2`, `f(p)`, `gradient(f,p)`)
- `…/handwritten` — the same math as a plain `@inbounds` loop / inlined kernel,
  no `CompiledModel`, no `withparams`, no `ObjectiveFunction`

The ratio `astrofit / handwritten` is the real KPI: it should sit at ~1.0
(zero abstraction tax — as the χ² spot-check already showed: 2625 ns vs 2666 ns
inline). A ratio that grows over revisions is the signal to act, even when the
absolute time looks flat. Add a CI assertion later (`@test ratio < 1.15`) if you
want it to fail the build, not just report.

Where a handwritten version is meaningless (e.g. `@model` macro, tree-walk
accessors with no scalar analogue) skip it — note `# no handwritten analogue`
so the omission reads as intent.

---

## Layout

```
benchmark/
  benchmarks.jl     # defines const SUITE; the only entry point ASV needs
  Project.toml      # deps: BenchmarkTools, Distributions, ForwardDiff, Optimization
benchmarks.md       # this plan
.github/workflows/benchmark.yml
```

`benchmark/Project.toml` deps the package under test (`AstroFit` as the parent
project) plus the weakdeps the suite exercises:

```toml
[deps]
AstroFit = "<uuid>"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
Optimization = "7f7a1694-90dd-40f0-9382-eb1efda571ba"
```

---

## Critical points (what to benchmark and why)

Grouped by `SUITE` key. Each line is one hot path; `→ why` names the risk a
regression would introduce.

### `render` — the inner kernel (throughput floor)
Single-point `render` + broadcast `render!`, one bench per model. Everything
above sits on this loop.

- **1D**: `Gaussian1D`, `Lorentzian1D`, `Voigt1D`, `Linear1D`, `Const1D`,
  `PowerLaw1D`, `BrokenPowerLaw1D`, `BlackBody1D`, `Exponential1D`
- **2D**: `Gaussian2D`, `Sersic2D`, `Moffat2D`, `Beta2D`
- `→ why`: pow/exp-heavy kernels (`Sersic2D` has `^(1/n)`, `Moffat2D`/`Beta2D`
  fractional powers, `BlackBody1D` `expm1`) dominate fit cost. A scalar→pow
  regression here multiplies through every objective eval.
- Cover both `render(m, x)` scalar and `render!(out, m, xs)` broadcast — the
  broadcast path has its own `@inbounds` loop (`models1d.jl:14`) that can drift
  from the scalar one.

### `compound` — tree evaluation
- `render` of a `+`-compound (e.g. `cont + g1 + g2`, and a 4-component 2D scene
  like the blended-galaxies example).
- `→ why`: recursive tree walk; a type-instability in the `+` node propagates
  to every fit. Bench 2-term and 4-term to expose super-linear growth.

### `withparams` — model reconstruction (called every eval & every AD step)
- `withparams(cm, p)` for a small (3-param) and a large (~18-param) model.
- `→ why`: rebuilt from the param vector on *every* objective call and *every*
  ForwardDiff step. Must stay type-stable and zero-alloc; the single most
  leveraged function in the library. A regression here taxes optimization and
  sampling alike.

### `chi2` — all four method specializations
- 1D weighted / 1D unweighted (fast path, `loss.jl:69`/`79`)
- ND weighted / ND unweighted (generic `map`/splat path, `loss.jl:48`/`57`)
- `→ why`: the 1D fast path exists precisely to dodge `map`/splat overhead;
  benchmark both so a refactor can't silently collapse 1D onto the slow path.

### `objective` — full `ObjectiveFunction` evaluation
- `f(p)` for `statistic ∈ {:chi2, :negloglikelihood, :logposterior}`.
- `→ why`: this is what optimizers/samplers actually call. `:logposterior`
  adds the bounds check (`loss.jl:212`) + prior; verify the branch is cheap and
  the `Val{statistic}` dispatch stays static.

### `gradient` — AD throughput (the Bayesian-critical path)
- `ForwardDiff.gradient(f, p)` of the objective, small and large param count.
- `→ why`: HMC/NUTS and gradient optimizers spend ~all their time here; cost
  scales ~P× the primal under ForwardDiff. This is the benchmark that catches
  AD-hostile changes (branches, `-Inf` walls, type instability through
  `withparams`). The most important single number for the Bayesian roadmap.

### `prior` — `logprior` evaluation (needs Distributions)
- `logprior(cm, p)` with a handful of priors set (`LogNormal`/`Normal`/`Beta`).
- `→ why`: per-param `logpdf` loop + `findfirst` name lookup
  (`AstroFitDistributionsExt.jl:39`). The `findfirst` is O(P) per param → O(P²);
  bench at P≈18 so that quadratic shows up before it bites a real fit.

### `params` — tree-walk accessors
- `params(cm)`, `bounds(cm)`, `paramnames(cm)` on a large model.
- `→ why`: cheap individually but called at setup and (some) per-eval; a DFS
  order regression also silently corrupts param↔slot mapping (`params.jl:2`).

> **Not benchmarked here**: `@model` macro expansion (compile-time, not a
> runtime path) and `solve(...)` end-to-end (dominated by the external solver,
> not AstroFit — track the objective + gradient instead).

### Handwritten baseline per group

| group | handwritten counterpart | target ratio |
|---|---|---|
| `render` (scalar/`render!`) | inlined kernel loop (the exact formula, no struct) | ~1.0 |
| `compound` | sum of inlined kernels in one fused loop | ~1.0 |
| `withparams` | construct the concrete model struct directly from `p` | ~1.0 |
| `chi2` | `@inbounds` residual loop over inlined `render` | ~1.0 |
| `objective` | hand-written `-0.5*chi2 + const` loop | ~1.0 |
| `gradient` | `ForwardDiff.gradient` of the handwritten objective | ~1.0–1.2 |
| `prior` | `sum(logpdf(dist, p[i]) for …)` with a precomputed index map | ≥1.0 (lib does `findfirst`) |
| `params`/`bounds`/`paramnames` | — *no handwritten analogue* | n/a |

The `prior` row is where a gap is *expected*: the library's `findfirst` name
lookup (O(P²)) will lose to a precomputed map — quantifying that gap is exactly
how you decide whether to cache the index map in the ext.

---

## SUITE skeleton (`benchmark/benchmarks.jl`)

Representative shape — fill each group from the list above. Interpolate every
runtime value with `$` so BenchmarkTools doesn't time global lookups.

```julia
using BenchmarkTools, AstroFit
using Distributions, ForwardDiff

const SUITE = BenchmarkGroup()

# ---- shared fixtures ----
const X1 = collect(0.0:0.01:12.0)                       # ~1200 pts
const c  = range(-8.0, 8.0; length = 100)
const X2 = [x for x in c, _ in c]
const Y2 = [y for _ in c, y in c]

# handwritten kernels — the bare-loop floor each library path is measured against
hw_gauss1d!(out, A, μ, σ, xs) = (@inbounds for i in eachindex(out, xs)
    out[i] = A * exp(-((xs[i] - μ) / σ)^2 / 2)
end; out)
function hw_chi2_1d(A, μ, σ, x, y, err)   # inlined render + residual loop
    acc = 0.0
    @inbounds for i in eachindex(y)
        m = A * exp(-((x[i] - μ) / σ)^2 / 2)
        acc += abs2((m - y[i]) / err[i])
    end
    acc
end

# ---- render: library vs handwritten, side by side ----
SUITE["render"] = BenchmarkGroup()
let m = Gaussian1D(8.0, 5.0, 0.6), out = similar(X1)
    SUITE["render"]["Gaussian1D/render!/astrofit"]    = @benchmarkable render!($out, $m, $X1)
    SUITE["render"]["Gaussian1D/render!/handwritten"] = @benchmarkable hw_gauss1d!($out, 8.0, 5.0, 0.6, $X1)
end
# … one paired (astrofit/handwritten) entry per model in the list

# ---- objective + gradient: paired ----
let cm = (@model begin g = Gaussian1D(amplitude=5.0, mean=4.5, sigma=0.8); g end),
    y  = render.(Ref(Gaussian1D(8.0,5.0,0.6)), X1), err = fill(0.3, length(X1))
    f  = ObjectiveFunction(cm, X1, y, err)
    p  = params(cm)
    hw = q -> hw_chi2_1d(q[1], q[2], q[3], X1, y, err)
    SUITE["chi2"]["1D/weighted/astrofit"]      = @benchmarkable $f($p)
    SUITE["chi2"]["1D/weighted/handwritten"]   = @benchmarkable $hw($p)
    SUITE["gradient"]["1D/chi2/astrofit"]      = @benchmarkable ForwardDiff.gradient($f, $p)
    SUITE["gradient"]["1D/chi2/handwritten"]   = @benchmarkable ForwardDiff.gradient($hw, $p)
    SUITE["withparams"]["small/astrofit"]      = @benchmarkable withparams($cm, $p)
    SUITE["withparams"]["small/handwritten"]   = @benchmarkable Gaussian1D($p[1], $p[2], $p[3])
end
# … large (~18-param) blended-galaxies fixture for withparams/gradient/prior
```

Keep fixtures `const` and module-level; ASV re-includes the file per revision,
so construction cost must stay out of the timed region.

---

## Run locally

```bash
# whole suite
julia -tauto --project=benchmark -e 'include("benchmark/benchmarks.jl"); run(SUITE)'

# one group while iterating
julia -tauto --project=benchmark -e 'include("benchmark/benchmarks.jl"); run(SUITE["gradient"])'

# tune sample budgets first (slower, more stable)
julia -tauto --project=benchmark -e 'include("benchmark/benchmarks.jl"); tune!(SUITE); run(SUITE)'
```

Compare two revisions with the ASV CLI:

```bash
benchpkg AstroFit --rev=main,HEAD --bench-on=main
```

---

## CI (`.github/workflows/benchmark.yml`)

```yaml
name: Benchmarks
on:
  pull_request_target:
    branches: [main]
permissions:
  pull-requests: write
jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: MilesCranmer/AirspeedVelocity.jl@action-v1
        with:
          julia-version: '1'
          tune: 'false'        # flip to 'true' once the suite is stable
```

This posts a PR comment with per-benchmark deltas vs the base branch. Add
`benchmark-filter: 'gradient'` to scope a noisy PR to one group.

---

## Conventions / gotchas

- **Interpolate everything** (`$x`) — un-interpolated globals time type-unstable
  lookups, not your code.
- **Report `minimum`/`median`, never `mean`** — GC and scheduler noise live in
  the tail; minimum is the truest single-thread cost.
- **Watch allocations, not just time** — `withparams`/`chi2`/`render!` should be
  zero-alloc; a non-zero `memory`/`allocs` in the report is the earliest signal
  of a type instability, often before wall-time moves.
- **Pin sizes in `const` fixtures** — changing array length between revisions
  makes deltas meaningless.
- **Gradient group is the canary** — if only one group runs in a fast CI lane,
  make it `gradient`: it transitively exercises `withparams`, `render`, `chi2`,
  and the `Val{statistic}` dispatch in a single number.
```
