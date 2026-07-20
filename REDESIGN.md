# Kernels as first-class models — what changed and why

A record of the change that made kernels (PSF convolution and anything else
whose value at one point depends on its neighbours) native to AstroFit, rather
than a special case handled around the edges of a pointwise core.

This document is written to be read *against the code*. Every section shows what
the code looked like before, what it looks like now, and what the difference
buys. The design conversation that led here is in [KERNELS.md](KERNELS.md); the
individual decisions are in [ADR-0001..0004](docs/adr/).

**Scope of the change**: 6 files modified, 3 added, ~280 lines. 261 tests pass.
The pointwise path is byte-identical in behaviour and identical in measured
performance.

---

## 1. The problem

Every model in AstroFit answered one question: *what is the value at this
coordinate?* That is the scalar `render(m, x::Number)`, and arrays came from
broadcasting it, from the top:

```julia
# src/model.jl — before
render(m::AbstractModel, xs::AbstractArray...) = render.(m, xs...)
```

This is a good design, and it is the reason the package is fast: a whole tree of
models collapses into **one** fused broadcast pass, with no intermediate arrays.
`Sum` never builds a left array and a right array — it adds two numbers at a
time.

A convolution cannot answer that question. The value at point `i` depends on the
input values *around* `i`, so there is no scalar `render(psf, x)` to write. This
isn't a variation on a pointwise model; it is a different kind of model.

The friction was structural, not local. Three things broke at once:

1. **`render`** — the array path was broadcast-of-scalar all the way down, so
   there was nowhere for a whole-array operation to happen.
2. **`chi2`** — every variant looped `render(model, x[i])` one point at a time.
   A kernel model wrapped in an `ObjectiveFunction` was not slow, it was
   *broken*.
3. **`_defaults`** — the `@model` macro marks every field of every leaf `Free`.
   For a kernel that is wrong twice: semantically (a PSF width is a calibration
   input, not a fit parameter) and mechanically (a free `Int` field makes
   ForwardDiff throw `InexactError: Int(Dual)`).

## 2. The design decision: a trait, not a predicate

The [KERNELS.md](KERNELS.md) session converged on a `_haskernel(model)`
predicate — a boolean asking *"does this tree contain a kernel?"* — used to
guard array-render methods on every compound node.

**That is not what shipped.** The implementation uses a trait instead:

```julia
# src/model.jl
struct Pointwise end
struct Domainwise end

evalstyle(m) = evalstyle(typeof(m))
evalstyle(::Type{<:AbstractModel}) = Pointwise()

@inline _combine(::Pointwise, ::Pointwise) = Pointwise()
@inline _combine(_, _) = Domainwise()
```

The two look interchangeable — both fold to a compile-time constant, both cost
nothing at runtime. They are not, and the difference is worth being precise
about, because it is the single most consequential choice in this change.

`_haskernel` asks *"does this subtree contain a special case?"*. It is a whole-
subtree property, so a guard written with it is true at the root and true at
every node beneath. In `(g1 + g2) |> psf`, `_haskernel` is `true` at the `Pipe`,
`true` at the `Sum`, and `true` at `g1` and `g2`'s parent — so every one of
those nodes takes the array branch, and **fusion is destroyed throughout the
tree**, even in the pointwise part that has nothing to do with the kernel.

`evalstyle` asks *"how is this evaluated?"*, and combines **upward** from the
leaves. The `Sum` node `g1 + g2` reports `Pointwise()`, because both its
children are pointwise — the kernel above it does not contaminate it. So `g1 +
g2` renders as one fused broadcast, and only its *result* is handed to the
kernel. Fusion breaks exactly at the kernel boundary and nowhere else.

The same reasoning, as a table:

| Node in `(g1 + g2) \|> psf` | `_haskernel` | `evalstyle` | Consequence |
|---|---|---|---|
| `Pipe` (root) | `true` | `Domainwise` | array path — correct, both agree |
| `Sum` (`g1 + g2`) | `true` | **`Pointwise`** | `_haskernel` breaks fusion here for no reason |
| `g1`, `g2` | `false` | `Pointwise` | both agree |

Secondary benefit: the trait describes a *property of the model*, not a fact
about a feature. Adding a second non-pointwise model kind later (pixel binning,
instrumental response) needs no new predicate — it declares `Domainwise` and
every existing path handles it.

## 3. The render layer

### Before

```julia
# src/model.jl — 3 lines, one path
render(m::AbstractModel, xs::AbstractArray...) = render.(m, xs...)

function render!(out::AbstractArray, m::AbstractModel, xs...)
    out .= render.(m, xs...)
    return out
end
```

### After

```julia
# src/model.jl
render(m::AbstractModel, xs::AbstractArray...) = _render(evalstyle(m), m, xs...)

# Pointwise — the pre-kernel path, unchanged on purpose: this is the fused
# single-pass broadcast the ≤1.0x-vs-handwritten benchmark rests on.
@inline _render(::Pointwise, m, xs::AbstractArray...) = render.(m, xs...)

# Domainwise — structural recursion (_arender, compound.jl). Pointwise subtrees
# inside still take the fused path above, so fusion breaks only at kernels.
@inline _render(::Domainwise, m, xs::AbstractArray...) = _arender(m, xs...)
```

**The key line is the `Pointwise` one.** It is the old body, verbatim. That is
not an accident of implementation — it is the mechanism by which the benchmark
guarantee survives. A kernel-free model does not take a *similar* path to
before; the trait resolves at compile time and the emitted code is the same
fused broadcast it always was. The measurement in §8 confirms it rather than
assuming it.

`render!` got the same two-way split, for a reason worth stating: writing
`out .= render(m, xs...)` for both cases would have been shorter but would
allocate a temporary on the pointwise path, silently regressing in-place
rendering.

## 4. Compound nodes: structural recursion

Five nodes (`Sum`, `Difference`, `Product`, `Quotient`, `Pipe`) each gained two
things — a trait rule and an array render:

```julia
# src/compound.jl
const _COMPOUND = Union{Sum, Difference, Product, Quotient, Pipe}

# A compound node is pointwise only if both sides are.
evalstyle(::Type{N}) where {N <: _COMPOUND} =
    _combine(evalstyle(N.parameters[1]), evalstyle(N.parameters[2]))

@inline _arender(m::Sum, xs::AbstractArray...) = render(m.left, xs...) .+ render(m.right, xs...)
@inline _arender(m::Difference, xs::AbstractArray...) = render(m.left, xs...) .- render(m.right, xs...)
@inline _arender(m::Product, xs::AbstractArray...) = render(m.left, xs...) .* render(m.right, xs...)
@inline _arender(m::Quotient, xs::AbstractArray...) = render(m.left, xs...) ./ render(m.right, xs...)

@inline _arender(m::Pipe, xs::AbstractArray...) = render(m.right, render(m.left, xs...))
```

Two details carry all the weight:

**Each side recurses through `render`, not through `_arender`.** That single
choice is what produces maximal fusion. `render(m.left, xs...)` re-consults the
trait, so a pointwise left branch takes the fused broadcast and a domainwise one
recurses further. The recursion is self-limiting.

**`Pipe` needed no special-casing for kernels.** Its scalar meaning is "feed the
left output into the right"; the array version is the same sentence with the
array as the unit. When the right side is a kernel this *is* convolution. When
it is a pointwise model it is ordinary composition. One line covers both — there
is no branch anywhere that asks "is this a kernel?".

All four arithmetic nodes got the method, not just `Sum`. The bodies differ only
by the operator, and "`+` composes with kernels but `*` doesn't" would read as a
bug rather than a decision — `(source |> psf) * transmission` is as physical as
the sum case.

### What this makes possible

Composition in any order, which is success criterion 2, falls out of the
recursion rather than being implemented:

```julia
(line |> psf) + cont          # convolved line on an unconvolved continuum
cont + (line |> psf)          # order of the sum is irrelevant
(line |> psf) * transmission
(line |> psf) - baseline
line |> psf1 |> psf2          # chained kernels
((a + b) |> psf) + c          # pointwise subtree feeds the kernel
```

Every one of these is asserted in
[test/kernel_tests.jl](test/kernel_tests.jl). None of them required code
specific to that shape.

## 5. The kernel type

```julia
# src/kernel.jl
abstract type AbstractKernel <: AbstractModel end

evalstyle(::Type{<:AbstractKernel}) = Domainwise()
```

That is the whole type. Writing a kernel is writing a model, with one
substitution — define the **array** render instead of the scalar one:

```julia
# a pointwise model                    # a kernel
struct Gaussian1D{T} <: AbstractModel  struct BoxKernel <: AbstractKernel
    amplitude::T                           width::Int
    mean::T                            end
    sigma::T
end

render(m::Gaussian1D, x::Number) =     render(k::BoxKernel, ys::AbstractVector) =
    ...                                    ...   # same size out as in
```

Everything else — `@model`, navigation, constraints, `withparams`, `params` —
treats it as an ordinary leaf, because it *is* one.

The contract has two rules, both from
[ADR-0001](docs/adr/0001-kernel-grid-contract.md):

- **Intensities in, intensities out.** The array a kernel receives is the values
  produced upstream, not coordinates. Array index is the grid, so widths are in
  samples and the grid is assumed uniform.
- **Size-preserving.** Output shape equals input shape. Edge handling is the
  kernel's own choice; the framework does not impose one.

### `GaussianPSF`

The concrete kernel that makes the machinery useful
([src/zoo/kernels.jl](src/zoo/kernels.jl)):

```julia
function render(k::GaussianPSF, ys::AbstractVector)
    σ = k.sigma
    σ > 0 || throw(ArgumentError("GaussianPSF: sigma must be positive, got $σ"))
    h = max(1, ceil(Int, 4σ))
    w = [exp(-abs2(d / σ) / 2) for d in (-h):h]

    out = similar(ys, promote_type(eltype(ys), eltype(w)))
    ...
        out[i] = acc / wsum          # renormalized over the weights that fit
end
```

Three choices in there worth naming:

**`acc / wsum`, not `acc / sum(w)`.** The normalization is recomputed per output
point over only the weights that fell inside the array. Without this, truncation
at the borders darkens the array ends; with it, a flat signal stays flat all the
way to the edge. There is a test asserting exactly that (`render(k, fill(3.0,
40)) ≈ flat`).

**`similar(ys, promote_type(eltype(ys), eltype(w)))`.** This is what lets
ForwardDiff through. Either input can be dual-typed — `ys` when an upstream model
parameter is being differentiated, `w` when `sigma` itself is free — and the
output type follows whichever it is.

**Direct convolution, no FFT.** An FFT would be faster for very wide kernels and
would break ForwardDiff. The `ponytail:` comment in the source names the O(N·σ)
ceiling and the upgrade path.

## 6. The loss layer

`chi2` was four methods that all looped point by point. It is now a dispatcher
plus two paths:

```julia
# src/fit/loss.jl
chi2(model, coords, y, err) = _chi2(evalstyle(model), model, coords, y, err)

@inline _chi2(::Pointwise, model, coords, y, err) = _chi2p(model, coords, y, err)

function _chi2(::Domainwise, model, coords, y, ::Nothing)
    μ = render(model, coords...)
    _checkpred(μ, y)
    return sum(i -> abs2(μ[i] - y[i]), eachindex(y))
end
```

The four original methods were **renamed** to `_chi2p`, not rewritten — the
scalar loop, the `@inbounds`, and the 1D fast path that skips `map`/splat are
all untouched. The pointwise path still allocates nothing, and the double
dispatch inlines away entirely (§8).

The domainwise path renders once for the whole array. The prediction array is an
unavoidable allocation — but it is one per objective evaluation, not one per
point, and the residual sum runs over a generator so the residuals themselves
are never materialized.

`_checkpred` is the one piece of defensive code in the change:

```julia
function _checkpred(μ, y)
    size(μ) == size(y) || throw(DimensionMismatch(
        "model prediction has size $(size(μ)) but data has size $(size(y)) — " *
        "a kernel must return an array the same size as its input"))
end
```

Without it, a kernel that shrinks its output (a `valid`-mode convolution instead
of `same`) would compare misaligned arrays. That fails in the worst possible
way: no exception, a fit that converges to nonsense. This is the class of bug
worth spending three lines on.

**Nothing in `ext/` needed changing.** `loglikelihood` is `-0.5 * chi2 + const`
and `logposterior` adds the prior sum, so every Bayesian entry point inherits
the split for free. Verified by test rather than by inspection, across all three
extensions: `logposterior`, the LogDensityProblems target
(`logdensity`/`dimension`), gradients of the log-posterior against finite
differences (including a free PSF width), and the Pigeons adapter's
initialization and reference distribution. A full Pigeons run on a
`(line |> psf) + cont` model recovers the generating parameters
(`[2.034, 0.505, 0.970, 0.396]` against a truth of `[2.0, 0.5, 1.0, 0.4]`); the
sampling run itself is not in the suite, only the adapter surface.

## 7. Kernel fields default to Fixed

One method, in [src/macro.jl](src/macro.jl):

```julia
_defaults(m) = ntuple(_ -> Free(), fieldcount(typeof(m)))

# Kernels are the exception: a kernel is normally a known calibration input, and
# a free integer field (a width in samples) breaks ForwardDiff outright.
_defaults(m::AbstractKernel) = ntuple(i -> Fixed(getfield(m, i)), fieldcount(typeof(m)))
```

Concretely, this is the difference between:

```julia
cm = @model begin
    line = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
    psf  = GaussianPSF(sigma = 1.5)
    line |> psf
end

nfree(cm)                # 3, not 4 — the PSF width is not an optimizer slot
paramnames(cm)           # [:line_amplitude, :line_mean, :line_sigma]
params(cm)               # Vector{Float64}, not Vector{Real}

cm = @free cm.psf.sigma  # opt in explicitly when you do want to fit it
nfree(cm)                # 4
```

The rejected alternative was fixing only `Integer` fields, which targets the
observed crash exactly. It was rejected because Int-vs-Float is a *proxy* for
"structural vs fittable" and the proxy breaks: a float structural field (an
oversampling factor, a truncation radius) would stay free and reintroduce the
same failure later, and harder to diagnose. Full reasoning in
[ADR-0004](docs/adr/0004-kernel-fields-fixed-by-default.md).

## 8. Verification

### The benchmark gate

The claim in §3 — that a kernel-free model emits the same code as before — is
structural, so it is testable rather than arguable:

```
                    allocations    min time
AstroFit chi2            0         3828.125 ns
handwritten loop         —         3828.125 ns
```

Identical, not comparable. There is a `@testitem` asserting `(@allocated f(p))
== 0` on the pointwise path, so the guarantee breaks loudly if someone disturbs
it.

### ForwardDiff

The hard case is a **free** `psf.sigma`, which pushes a `Dual` into the kernel's
own arithmetic. Automatic vs finite differences, all four parameters:

```
AD : [-7.081424, -9.365076, -4.684122, 0.169021]
FD : [-7.081424, -9.365076, -4.684122, 0.169021]
max relative error: 7.7e-9
```

The gradient is not merely finite — it is correct. Both are asserted in the test
suite.

### Optimization

`OptimizationProblem(cm, x, y)` with a `(line |> psf) + cont` model recovers the
generating parameters to `rtol = 1e-4` under LBFGS.

One honest note about a related test that is deliberately *not* written: with a
free PSF width, `psf.sigma` and `line.sigma` are degenerate — a convolution of
two gaussians is determined by their combination, so many pairs fit the data
equally well (χ² ≈ 1.6e-9 at a parameter pair that is not the generating one).
A test asserting parameter recovery there would fail for a reason that is not a
bug. The tests assert χ² ≈ 0 at the true parameters and AD == FD instead, which
is what actually distinguishes correct code from broken code.

### Coverage

261 tests pass, 83 of them new in
[test/kernel_tests.jl](test/kernel_tests.jl): user-defined kernels, the full
composition matrix, trait propagation, the allocation gate, `render!` on both
paths, PSF normalization and flat-signal conservation, `Fixed`-by-default and
`@free`, the size-mismatch error, AD-vs-FD, an Optimization round trip, the
Distributions path, and display.

## 9. One thing that had to change outside the render layer

`withparams` was listed above as untouched, and it nearly was. One line had to
change, and the reason is worth recording because it is the only place where
kernels forced a change to the machinery that predates them.

The `@generated` reconstruction rebuilt every leaf as:

```julia
constructorof(M)(promote(fields...)...)     # before
```

That `promote` is load-bearing: a `Gaussian1D{T<:Real}` whose `sigma` came from
the parameter vector as a `Dual` while its other fields are stored `Float64`
cannot be constructed unless the fields agree on one `T`. Promoting them
together is what makes ForwardDiff work at all.

It also promoted fields that have nothing to promote. Every model in the zoo has
all-`Real` fields, so this never showed. A **measured instrumental PSF** does
not: its defining field is the sampled kernel array.

```julia
struct ScaledPSF{V <: AbstractVector, T <: Real} <: AbstractKernel
    kernel::V      # the measured PSF — data, not a shape parameter
    scale::T       # a fittable scalar next to it
end
```

Reconstructing that leaf threw:

```
promotion of types Vector{Float64} and Float64 failed to change any arguments
```

The fix reads which fields are numeric from the model type at code-generation
time, and promotes only those:

```julia
function _ctorexpr(M, fields)
    num = findall(F -> F <: Number, collect(fieldtypes(M)))
    length(num) == fieldcount(M) && return :($ctor(promote($(fields...))...))   # unchanged
    length(num) <= 1 && return :($ctor($(fields...)))
    # mixed: promote the numeric fields among themselves, splice back in place
end
```

The all-numeric branch emits the *same expression as before*, so every existing
model is unaffected and the benchmark gate still reads 0 allocations and
3828.125 ns. The observable result on a mixed leaf is that a dual number lifts
only what it should:

```julia
withparams(cm, ForwardDiff.Dual.(p, 1.0)).psf.model
# ScaledPSF{Vector{Float64}, ForwardDiff.Dual{Nothing, Float64, 1}}
#           ^ array untouched  ^ scalar lifted
```

A 2D instrumental PSF — one whose field is the intensity *matrix* — works
through the same path, convolving the rendered image:

```julia
im = @model begin
    src  = Gauss2D(3.0, 0.0, 0.0, 1.0)
    ipsf = ImagePSF(psfmat)          # 3×3 measured PSF
    src |> ipsf
end
render(im, X, permutedims(Y))        # 13×13 grid in, 13×13 convolved image out
```

Both are covered by tests, with gradients checked against finite differences
away from the minimum (relative error ~1e-10) so that a zero gradient cannot
pass for a correct one.

## 10. What was planned and deliberately not built

This document originally proposed a `Domain` type (`PointCloud` vs
`GridDomain`) to make the evaluation grid explicit, motivated by a real bug: in
the fit path, 2D data is a *flat point list*, and two equal-length axis vectors
handed to `render.` broadcast elementwise and silently return the **diagonal**
instead of the image. Verified:

```julia
render(g, ax, ay)                    # (9,)   — the diagonal, silently wrong
render(g, ax, ay_different_length)   # DimensionMismatch
render.(g, ax, permutedims(ay))      # (9, 7) — the actual grid
```

It did not ship, because the success criteria were vector-based and vectors are
unambiguous. The `Domain` split only earns its complexity when kernels are
*fitted* in 2D, where flat-list-vs-grid becomes a real fork.
[ADR-0003](docs/adr/0003-grid-form-data-for-kernel-fits.md) therefore describes
a plan, not the current code — worth knowing before implementing 2D kernel
fitting, and worth not building before then.

Also deliberately absent: a workspace buffer on `ObjectiveFunction` to avoid
re-allocating the prediction array each iteration (do it when a profile says
so), and any kernel zoo beyond `GaussianPSF` (`MoffatPSF`, instrumental LSF —
add when a real fit needs one).

## 11. File-by-file summary

| File | Change |
|---|---|
| [src/model.jl](src/model.jl) | `Pointwise`/`Domainwise` traits, `_combine`, style-dispatched `render`/`render!` entry points |
| [src/kernel.jl](src/kernel.jl) | **new** — `AbstractKernel`, its trait, the documented contract, a clear error for unsupported array types |
| [src/compound.jl](src/compound.jl) | trait combination for the five nodes, `_arender` structural recursion |
| [src/compiled.jl](src/compiled.jl) | `Leaf` and `CompiledModel` forward the trait to what they wrap |
| [src/macro.jl](src/macro.jl) | `_defaults(::AbstractKernel)` → all `Fixed` |
| [src/fit/loss.jl](src/fit/loss.jl) | `chi2` dispatches on style; scalar loop renamed `_chi2p`, otherwise untouched; `_checkpred` |
| [src/withparams.jl](src/withparams.jl) | `_ctorexpr` — promote only the numeric fields (§9); the all-numeric case emits the previous expression verbatim |
| [src/zoo/kernels.jl](src/zoo/kernels.jl) | **new** — `GaussianPSF` |
| [test/kernel_tests.jl](test/kernel_tests.jl) | **new** — 16 test items, 83 assertions |
| [README.md](README.md) | "Kernels and PSF Convolution" section |

Untouched: `params.jl`, `constrain.jl`, `constraints.jl`, `priors.jl`,
`show.jl`, the existing zoo, and every file in `ext/`.

The one change to `withparams` is worth being precise about, because it is the
exception that proves the rule. Its `@generated` slot map — the part that makes
AstroFit fast — reads the tree *type* and still knows nothing about how the
rebuilt model is evaluated; that machinery was already factored along the seam
this change needed. What had to move was one line of field marshalling, and only
because a kernel can hold something that is not a number.
