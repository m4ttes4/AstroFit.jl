# AstroFit

Build, constrain, and fit parametric astrophysical models in Julia.

AstroFit is for workflows where the physics is full of constraints: shared line
centers, tied widths, fixed ratios, bounded amplitudes, reusable components, and
custom model pieces. Handwritten functions with those rules hardcoded are fast,
but they quickly become hard to reuse. AstroFit gives you composable models and
keeps the fitting hot path close to handwritten speed by compiling parameter
scatter and tie resolution into generated, straight-line code.

I started this because I missed the way [Astropy modeling](https://docs.astropy.org/en/stable/modeling/) and [lmfit](https://lmfit.github.io/lmfit-py/) let you snap models together, but I wanted that in Julia where the compiler can actually inline everything. [`AccessibleModels`](https://github.com/) was another reference point for the composable-model idea.

> [!WARNING]
> AstroFit is a working proof of concept, not a
> production-ready package. It works for the workflows I built it for, but the API,
> documentation, and test coverage should still be treated as experimental. I
> wrote and maintain the repository myself, and AI assistance played an important
> role while designing the generated-function internals that make
> `withparams` fast.

- Define reusable model components with clear names.
- Attach physical constraints with `@constrain`.
- Fit with a flat parameter vector through fast `withparams(cm, p)`.
- Extend the system with plain Julia structs and `render` methods.

---

## Contents

- [Motivation](#motivation)
- [Quick Start](#quick-start)
- [Building Models](#building-models)
- [Adding Constraints](#adding-constraints)
- [Working With Parameters](#working-with-parameters)
- [Fitting](#fitting)
- [Optimization.jl Integration](#optimizationjl-integration)
- [Future Progress](#future-progress)
- [Benchmarks](#benchmarks)
- [Real Examples](#real-examples)
- [Extending AstroFit](#extending-astrofit)
- [Internal Design](#internal-design)

---

## Motivation

In astrophysics, parameters are rarely independent: two emission lines share a
velocity width, a doublet has a fixed flux ratio, a redshift shifts the entire
rest-frame model. Constraints are the rule, not the exception.

You can write a monolithic Julia function that hardcodes everything. It's fast,
but the moment you change the setup — add a line, drop a constraint — you end up
rewriting half the code. Or you use a layer that resolves constraints with
runtime lookups, but you pay that cost on every fit iteration.

AstroFit tries to sit in the middle: write model components as reusable pieces,
declare constraints explicitly, and let Julia compile the resolved path. The
model stays inspectable and easy to modify, but the inner loop comes down to
`withparams(cm, p)` plus `render` — no lookups, no overhead.

Models are composed with binary operators (`+`, `*`, `|>`), a pattern common
across fitting libraries because it makes the structure of a model immediately
obvious: `continuum + line_ha + line_nii` reads like what it is. You see the
physics, not the plumbing.

---

## Quick Start

```julia
using AstroFit

# Define a two-component emission line model
spec = @model begin
    cont = Linear1D(0.0, 1.0)
    ha   = Gaussian1D(5.0, 6563.0, 2.0)
    cont + ha
end

# Add physical constraints
@constrain spec begin
    ha.mean = 6563.0
    ha.amplitude in (0, Inf)
    ha.sigma     in (0.1, Inf)
end

# Build a loss function and fit
wavelengths = collect(6540.0:0.5:6590.0)
observed = render(spec, wavelengths)

loss(p) = sum(abs2, render(withparams(spec, p), wavelengths) .- observed)

using ForwardDiff
ForwardDiff.gradient(loss, params(spec))
```

What happened:

- `@model` created a named, composable model tree.
- `@constrain` added bounds and a fix — auto-rebinding `spec` in place.
- `params(spec)` and `bounds(spec)` give the optimizer its starting point and box.
- `withparams(spec, p)` rebuilds the bare model tree from a flat parameter vector.

---

## Building Models

AstroFit models are immutable Julia structs. You evaluate them with `render`.

```julia
g = Gaussian1D(amplitude=1.0, mean=0.0, sigma=2.0)
render(g, 0.5)
render(g, -1:0.5:1)

c = Const1D(3.0)
l = Linear1D(slope=2.0, intercept=1.0)
```

### Composition operators

Components combine with ordinary operators — no macro needed:

| Expression | Meaning |
|------------|---------|
| `a + b` | Sum of two models |
| `a - b` | Difference |
| `a * b` | Product |
| `a / b` | Quotient |
| `a ∘ b` | Pipe: `a(b(x))` |
| `a \|> b` | Pipe: `b(a(x))` |

```julia
# Emission line on a flat continuum
m = Const1D(0.5) + Gaussian1D(1.0, 0.0, 1.0)
render(m, 0.0)   # 1.5

# Absorption line
absorption = Const1D(1.0) - Gaussian1D(0.4, 0.0, 1.0)
render(absorption, 0.0)   # 0.6
```

### Named models with `@model`

Use block form when you want named components:

```julia
spec = @model begin
    cont  = Linear1D(0.0, 1.0)
    ha    = Gaussian1D(5.0, 6563.0, 2.0)
    n6548 = Gaussian1D(1.0, 6548.0, 2.0)
    n6583 = Gaussian1D(3.0, 6583.0, 2.0)
    cont + ha + n6548 + n6583
end
```

The final expression is the composition. Every bound name must appear in it.
Each named component becomes a `Leaf` node in an annotated model tree, ready to
carry constraints. All parameters start free.

### Navigating the tree

Each name is a property returning the leaf — resolved from the type, so it is
exact and allocation-free:

```julia
spec.ha                # the Leaf tagged :ha
spec.ha.model          # Gaussian1D(5.0, 6563.0, 2.0)
spec.ha.constraints    # (Free(), Free(), Free())
```

---

## Adding Constraints

Each leaf field carries one of four constraint kinds:

| Kind | Meaning | Free slot? |
|------|---------|-----------|
| `Free()` | a free parameter | yes |
| `Bounded(lo, hi)` | free, but confined to `[lo, hi]` | yes |
| `Fixed(v)` | pinned to a constant | no |
| `Tied(paths, f)` | `value = f(master₁, …)` of other free params | no |

A `Tied` references one or more **free** masters by `(leaf, field)` path. Its
masters must be `Free` or `Bounded` — no chaining.

### Standalone constraint verbs

`@fix`, `@bound`, `@free`, `@tie`, and `@prior` each edit one parameter and
auto-rebind the model variable:

```julia
@fix   spec.ha.mean = 6563.0              # pin to a constant
@bound spec.ha.amplitude in (0, Inf)      # confine to [0, ∞)
@free  spec.cont.slope                    # release back to free
@tie   spec.n6583.amplitude -> 2.96 * spec.n6548.amplitude   # fixed ratio
```

### `@constrain` block

For more than one or two edits, `@constrain m begin … end` is the ergonomic
form: leaf names are bare, each constraint has its own operator (no `@verb`
prefix except `@free`), a parameter constrained twice is a compile-time error,
and the whole model is validated at the end:

```julia
@constrain spec begin
    ha.mean    = 6563.0                    # fix
    n6548.mean = 6548.0
    n6583.mean = 6583.0
    ha.amplitude    in (0, Inf)            # bound
    n6548.amplitude in (0, Inf)
    n6548.sigma     -> ha.sigma            # tie
    n6583.sigma     -> ha.sigma
    n6583.amplitude -> 2.96 * n6548.amplitude
end

nfree(spec)        # 5
paramnames(spec)   # [:cont_slope, :cont_intercept, :ha_amplitude, :ha_sigma, :n6548_amplitude]
```

### Low-level engine

The verbs lower to `setconstraint`, useful when building constraints
programmatically:

```julia
s = setconstraint(spec, :ha, :sigma, Bounded(0.5, 10.0))
s = setconstraint(s, :n6548, :sigma, Tied(((:ha, :sigma),), identity))
validate(s)   # checks all ties point at free masters; throws otherwise
```

---

## Working With Parameters

### Free parameters

```julia
nfree(spec)        # number of free parameters
params(spec)       # current free values (p₀ for the optimizer)
paramnames(spec)   # slot labels: [:cont_slope, :cont_intercept, …]
bounds(spec)       # (lower, upper) vectors aligned with params
```

All four accessors walk the tree in the same DFS order `withparams` assigns
slots, so they line up slot-for-slot.

### Rebuilding with `withparams`

```julia
model = withparams(spec, params(spec))
```

`withparams` scatters the flat parameter vector into the free positions,
re-resolves all tied parameters, and returns the **bare** model tree (Leaf
wrappers stripped). This is the function you call inside the fitting loop.

For example, if `n6583.amplitude -> 2.96 * n6548.amplitude`, the optimizer never
sees a separate `n6583_amplitude` slot. `withparams` rebuilds a plain model where
that field has already been computed from `n6548_amplitude`, so `render` does not
need to know about constraints.

---

## Fitting

AstroFit provides a built-in likelihood layer and a solver-agnostic `objective`
function. You can also write a plain loss function by hand — either way, the hot
path is `withparams` + `render`.

### Built-in objective

`objective(cm, x, y)` returns a closure `u -> -logposterior(cm, u, x, y, err)`
ready to **minimise** over the flat parameter vector. Without priors it is the
negative Gaussian log-likelihood; with `err = nothing` (the default) it assumes
unit variance (equivalent to least squares):

```julia
λ = collect(6540.0:0.5:6590.0)
y = render(withparams(fit, params(fit)), λ)

loss = objective(fit, λ, y)
loss(params(fit))   # minimum at the truth
```

Pass per-point standard deviations to get a weighted likelihood:

```julia
err = fill(0.1, length(y))
loss_w = objective(fit, λ, y; err)
```

The objective is fully differentiable — gradient-based optimizers and AD work
out of the box:

```julia
using ForwardDiff
ForwardDiff.gradient(loss, params(fit))   # 5-element gradient
```

### Manual loss function

If you need a custom objective (e.g. Cash statistic, regularisation), build it
directly from `withparams` + `render`:

```julia
loss_lsq(p) = sum(abs2, render(withparams(fit, p), λ) .- y)
```

### Choosing a solver

`params(fit)` gives the starting point, `bounds(fit)` gives the box. Hand them
to any Julia optimizer:

```julia
# using Optim
# lo, hi = bounds(fit)
# res = optimize(loss, lo, hi, params(fit), Fminbox(LBFGS()))
# best = withparams(fit, Optim.minimizer(res))
```

---

## Optimization.jl Integration

AstroFit ships a package extension for
[Optimization.jl](https://github.com/SciML/Optimization.jl). Loading
`Optimization` and `ForwardDiff` together activates it — no extra import needed.

First, some synthetic data to work with:

```julia
using AstroFit

λ = collect(-5.0:0.1:5.0)
true_model = Const1D(1.0) + Gaussian1D(5.0, 0.0, 1.0)
y = render(true_model, λ) .+ 0.01 .* randn(length(λ))
```

Now build a model with an initial guess, add constraints, and fit:

```julia
using Optimization, ForwardDiff, OptimizationOptimJL

spec = @model begin
    cont = Const1D(0.5)
    line = Gaussian1D(3.0, 0.2, 1.5)
    cont + line
end

@constrain spec begin
    line.amplitude in (0, Inf)
    line.sigma     in (0.1, Inf)
end

prob = OptimizationProblem(spec, λ, y)
sol  = solve(prob, Optim.Fminbox(Optim.LBFGS()))

best = withparams(spec, sol.u)
```

`OptimizationProblem(spec, λ, y)` extracts `params(spec)` as the starting point
and `bounds(spec)` as `lb`/`ub` automatically. If no parameter is bounded, the
box is omitted so unconstrained solvers (BFGS, NelderMead) work directly.

If you need to control the AD backend or build the problem manually, use
`OptimizationFunction` instead:

```julia
optf = OptimizationFunction(spec, λ, y; adtype = AutoForwardDiff())
prob = OptimizationProblem(optf, params(spec); lb, ub)
```

---

## Future Progress

Bayesian analysis is a natural next step for AstroFit, and it should not require
a different model layer. The core pieces are already present: parameters are a
flat vector, `loglikelihood` and `logposterior` work on that vector, and priors
are already supported through `Distributions.jl` objects:

```julia
using Distributions

@constrain spec begin
    line.sigma ~ LogNormal(0.0, 0.5)
end
```

What is still missing is integration glue for samplers. In practice that means
writing small extensions that expose AstroFit models to Pigeons.jl and similar
Bayesian libraries, mapping their parameter vectors into `logposterior(cm, p, x,
y, err)` and carrying `paramnames(cm)` through to the sampled output.

---

## Benchmarks

The benchmark asks one specific question:

> If a model has physical constraints, how much slower is AstroFit than the
> hand-written Julia function you would write for maximum speed?

```julia
render(withparams(cm, p), x)      # AstroFit
handwritten_constrained(p, x)     # hardcoded baseline
```

The hand-written baseline has no abstraction cost: the fixed values, bounds, and
ties are baked directly into the function body. AstroFit keeps the reusable model
representation, but resolves ties through compiled straight-line code rather than
runtime lookup.

![AstroFit benchmark scaling](bench/scaling.png)

The current result: constrained AstroFit rendering stays close to the
hand-written baseline as models grow, while `withparams` remains tiny and
allocation-free.

See [`bench/README.md`](bench/README.md) for the benchmark script, command, and
current numbers.

---

## Real Examples

Full working scripts are in the [`examples/`](examples/) directory.

### Double Gaussian + linear continuum (1D)

Two emission lines on a sloped continuum, fitted to synthetic noisy data. The
second Gaussian's width and amplitude are tied to the first (`g2.sigma = g1.sigma`,
`g2.amplitude = 0.5 * g1.amplitude`), reducing 9 model parameters to 6 free ones.

```julia
cm = @model begin
    cont = Linear1D(slope = 0.0, intercept = 0.5)
    g1   = Gaussian1D(amplitude = 5.0, mean = 4.5, sigma = 0.8)
    g2   = Gaussian1D(amplitude = 3.0, mean = 8.0, sigma = 0.8)
    cont + g1 + g2
end

@constrain cm begin
    g2.sigma     -> g1.sigma
    g2.amplitude -> 0.5 * g1.amplitude
end
```

![Double Gaussian fit](examples/double_gaussian_fit.png)

See [`examples/double_gaussian_fit.jl`](examples/double_gaussian_fit.jl) for the
full script.

### Blended galaxies bulge+disk decomposition (2D)

Two partially overlapping galaxies, each decomposed into a Gaussian bulge and an
exponential disk (Sersic n=1). All four components are elliptical (`q`, `theta`
free). Within each galaxy, the bulge center and position angle are tied to the
disk. Sersic indices are fixed. 18 free parameters total, fitted with
`Fminbox(LBFGS())` via Optimization.jl.

```julia
cm = @model begin
    bulge1 = Gaussian2D(amplitude = 20.0, x0 = -3.5, y0 = 0.5, sigma = 2.5, q = 1.0, theta = 0.0)
    disk1  = Sersic2D(amplitude = 8.0, x0 = -3.5, y0 = 0.5, r_eff = 5.0, n = 1.0, q = 0.9, theta = 0.0)
    bulge2 = Gaussian2D(amplitude = 15.0, x0 = 4.5, y0 = 0.0, sigma = 1.5, q = 1.0, theta = 0.0)
    disk2  = Sersic2D(amplitude = 5.0, x0 = 4.5, y0 = 0.0, r_eff = 4.5, n = 1.0, q = 0.9, theta = 0.0)
    bulge1 + disk1 + bulge2 + disk2
end

@constrain cm begin
    disk1.n
    disk2.n
    bulge1.x0    -> disk1.x0
    bulge1.y0    -> disk1.y0
    bulge1.theta -> disk1.theta
    bulge2.x0    -> disk2.x0
    bulge2.y0    -> disk2.y0
    bulge2.theta -> disk2.theta
    # ... bounds on amplitudes, sizes, q, theta
end
```

![Blended galaxies fit](examples/blended_galaxies_fit.png)

See [`examples/blended_galaxies_fit.jl`](examples/blended_galaxies_fit.jl) for
the full script.

### Redshifted galaxy spectrum flagship fit (1D)

A larger synthetic AGN host-galaxy spectrum with a visibly curved power-law
continuum, narrow Balmer lines, broad AGN Balmer components, [OIII], [NII],
[SII], and Na D absorption. The model uses a custom redshift coordinate
transform, redshift-dependent flux scaling, fixed atomic doublet ratios, shared
narrow-line widths, tied broad-line widths, fixed rest wavelengths, bounded
emission/absorption amplitudes, and a Gaussian likelihood through
Optimization.jl. The result compresses 43 raw model fields to 15 free fitted
parameters.

```julia
cm = @model begin
    cont = Linear1D(slope = cont_slope, intercept = cont_intercept)
    stellar = PowerLaw1D(norm = pl_norm, x_ref = L_REF, index = pl_index)

    hbeta = Gaussian1D(amplitude = ha_amplitude / 2.86, mean = L_HB, sigma = narrow_sigma)
    broad_hbeta = Gaussian1D(amplitude = broad_ha_amplitude / 3.1, mean = L_HB, sigma = broad_sigma)
    oiii_b = Gaussian1D(amplitude = oiii_blue_amplitude, mean = L_OIII_B, sigma = narrow_sigma)
    oiii_r = Gaussian1D(amplitude = 2.98 * oiii_blue_amplitude, mean = L_OIII_R, sigma = narrow_sigma)

    ha = Gaussian1D(amplitude = ha_amplitude, mean = L_HA, sigma = narrow_sigma)
    broad_ha = Gaussian1D(amplitude = broad_ha_amplitude, mean = L_HA, sigma = broad_sigma)
    nii_b = Gaussian1D(amplitude = nii_blue_amplitude, mean = L_NII_B, sigma = narrow_sigma)
    nii_r = Gaussian1D(amplitude = 3.06 * nii_blue_amplitude, mean = L_NII_R, sigma = narrow_sigma)
    sii_b = Gaussian1D(amplitude = sii_blue_amplitude, mean = L_SII_B, sigma = narrow_sigma)
    sii_r = Gaussian1D(amplitude = sii_red_amplitude, mean = L_SII_R, sigma = narrow_sigma)

    nad_d2 = Gaussian1D(amplitude = nad_d2_amplitude, mean = L_NAD_D2, sigma = nad_sigma)
    nad_d1 = Gaussian1D(amplitude = 0.65 * nad_d2_amplitude, mean = L_NAD_D1, sigma = nad_sigma)

    redshift = RedshiftAxis1D(z = z)
    flux_scale = RedshiftFluxScale1D(z = z)

    ((cont + stellar + hbeta + broad_hbeta + oiii_b + oiii_r + ha +
      broad_ha + nii_b + nii_r + sii_b + sii_r + nad_d2 + nad_d1) ∘ redshift) * flux_scale
end

@constrain cm begin
    stellar.x_ref
    hbeta.amplitude -> ha.amplitude / 2.86
    hbeta.mean
    hbeta.sigma -> ha.sigma
    broad_hbeta.amplitude -> broad_ha.amplitude / 3.1
    broad_hbeta.mean
    broad_hbeta.sigma -> broad_ha.sigma
    oiii_r.amplitude -> 2.98 * oiii_b.amplitude
    oiii_r.sigma -> ha.sigma
    nii_r.amplitude -> 3.06 * nii_b.amplitude
    nii_r.sigma -> ha.sigma
    nad_d1.amplitude -> 0.65 * nad_d2.amplitude
    nad_d1.sigma -> nad_d2.sigma
    flux_scale.z -> redshift.z
    # ... bounds on continuum, narrow/broad line amplitudes, widths, and redshift
end
```

![Complex galaxy spectrum fit](examples/complex_galaxy_spectrum_fit.png)

See [`examples/complex_galaxy_spectrum_fit.jl`](examples/complex_galaxy_spectrum_fit.jl)
for the full script.

---

## Extending AstroFit

Any Julia struct that subtypes `AbstractModel` can be an AstroFit model.

```julia
Base.@kwdef struct Redshift1D{T<:Real} <: AbstractModel
    z::T = 0.0
end

AstroFit.render(m::Redshift1D, λ::Number) = λ / (1 + m.z)
```

Then use it like any built-in component:

```julia
spec = @model begin
    line   = Gaussian1D(1.0, 5000.0, 10.0)
    zshift = Redshift1D(z=0.1)
    line ∘ zshift
end
```

Rules for custom models:

- subtype `AbstractModel`;
- define scalar `render(m::YourModel, x::Number)`;
- accept `Number`, not only `Float64`, so ForwardDiff dual values work;
- let AstroFit handle composition, naming, constraints, and `withparams`.

---

## Internal Design

### Structure

`CompiledModel` has two fields:

- `tree`: one annotated model tree.
- `priors`: optional statistical priors, stored separately from mechanical constraints.

The tree is built from the same compound operator nodes used by ordinary models
(`Sum`, `Difference`, `Product`, `Quotient`, `Pipe`). The leaves are
`Leaf{name}` wrappers. Each leaf stores the user component and a tuple of
constraints aligned with that component's fields.

```mermaid
flowchart TB
    CM["CompiledModel{T,P}"]
    CM --> TREE["tree::T"]
    CM --> PRIORS["priors::P"]

    TREE --> SUM["Sum / Difference / Product / Quotient / Pipe"]
    SUM --> LEFT["left subtree"]
    SUM --> RIGHT["right subtree"]

    LEFT --> LEAF1["Leaf{:cont}"]
    RIGHT --> LEAF2["Leaf{:ha}"]

    LEAF1 --> MODEL1["model::Linear1D"]
    LEAF1 --> CONS1["constraints::Tuple<br/>Free, Free"]

    LEAF2 --> MODEL2["model::Gaussian1D"]
    LEAF2 --> CONS2["constraints::Tuple<br/>Bounded, Fixed, Bounded"]
```

For a model like:

```julia
spec = @model begin
    cont = Linear1D(0.0, 1.0)
    ha   = Gaussian1D(5.0, 6563.0, 2.0)
    cont + ha
end
```

the stored tree is conceptually:

```text
CompiledModel
└─ tree = Sum(
       Leaf{:cont}(Linear1D(...), (Free(), Free())),
       Leaf{:ha}(Gaussian1D(...), (Free(), Free(), Free())),
   )
```

After constraints, only the leaf constraint tuples change; the algebraic tree
shape does not need a parallel specification object. This is the main invariant:
the model values and constraint metadata live in one structure, so there is no
separate registry/spec tree that can drift out of sync.

### Parameter Slots

`params`, `bounds`, and `paramnames` all walk the annotated tree in the same
left-to-right order:

1. Visit the left subtree before the right subtree.
2. Inside each leaf, visit fields in the order defined by the model struct.
3. Count only `Free` and `Bounded` fields as optimizer slots.

That gives one flat vector for optimizers:

```julia
p0 = params(spec)
lo, hi = bounds(spec)
names = paramnames(spec)
```

`Fixed` fields do not get slots. `Tied` fields also do not get slots; they are
computed from one or more free/bounded master parameters.

### Generated `withparams`

`withparams(cm, p)` is the hot path. It is an `@generated` function because the
tree type encodes the leaf names, model types, and constraint types. At
specialization time, AstroFit can inspect that type and emit straight-line code
for this exact model layout.

The generated function does two compile-time passes over the tree type:

1. Build a slot map:
   `(:ha, :amplitude) => 3`, `(:ha, :sigma) => 4`, and so on.
2. Emit reconstruction code for the bare model tree:
   - `Free` / `Bounded` fields become `p[k]`.
   - `Fixed` fields read the stored fixed value.
   - `Tied` fields call their stored function on the master slots.

Conceptually, this:

```julia
withparams(spec, p)
```

turns into code shaped like:

```julia
Sum(
    Linear1D(p[1], p[2]),
    Gaussian1D(p[3], 6563.0, p[4]),
)
```

for a model where `ha.mean` is fixed at `6563.0`. A tie such as:

```julia
n6583.amplitude -> 2.96 * n6548.amplitude
```

emits code equivalent to:

```julia
Gaussian1D(2.96 * p[k_n6548_amp], ...)
```

There is no runtime dictionary lookup, name resolution, or constraint dispatch
inside the fit loop. `withparams` returns the bare compound model tree, with
`Leaf` wrappers stripped, so the next call is normal Julia dispatch:

```julia
render(withparams(spec, p), x)
```

This is also why custom models should accept `Number` fields and coordinates:
ForwardDiff dual values flow through the generated reconstruction and into
`render` without special cases.

### Constraint Edits

Constraints are edited immutably. `setconstraint(cm, :ha, :sigma, Bounded(...))`
finds the target leaf, swaps one entry in that leaf's constraint tuple, and
rebuilds only the path from the root to that leaf. No parameter indices are
stored in constraints, so editing a constraint does not require renumbering the
whole model.

`validate(cm)` checks global rules after edits:

- every `Tied` master must exist;
- every `Tied` master must be free or bounded;
- ties cannot point to fixed or tied targets.

The macro layer runs validation once at the end of a `@constrain` block.

---
