# Kernel Models — Design Notes

Working notes from a design session on adding a `AbstractKernel <: AbstractModel`
family to AstroFit — models whose `render` is not evaluable at a single
coordinate (a PSF convolution being the motivating example).

> **Status: superseded by the implementation.** Kernels are now native — see
> [`src/kernel.jl`](src/kernel.jl), [`src/model.jl`](src/model.jl) and the
> "Kernels and PSF Convolution" section of the README. What shipped follows
> [REDESIGN.md](REDESIGN.md) rather than the `_haskernel` guards sketched
> below: the presence predicate became an `evalstyle` trait
> (`Pointwise`/`Domainwise`) computed from the model type, which subsumes §3
> and §4 here. The four ADRs still hold. This file is kept as the record of how
> the design was reached, including the alternatives that were rejected and
> why.

## 1. The core problem: AstroFit is pointwise, top to bottom

Every existing model bottoms out at `render(m, x::Number)`. Arrays are
supported only by broadcasting that scalar function from the top:
`render(m::AbstractModel, xs::AbstractArray...) = render.(m, xs...)`
([model.jl:5](src/model.jl:5)). This holds for compound nodes too — `Sum`,
`Pipe`, etc. define no array-specific `render`/`render!`, so even the
"vectorized" path is scalar-per-point under the hood.

Convolution breaks this invariant: the output at one point depends on
neighboring input values, not on the coordinate alone. It cannot be
expressed as a scalar `render(m, x::Number)` at all. A kernel is therefore a
genuinely different kind of model — `render(kernel, xs::AbstractArray) ->
AbstractArray`, whole array in, whole array out — not a variation on the
existing pointwise ones.

## 2. Decisions made

Two design forks were resolved via `AskUserQuestion`:

- **Grid contract**: a kernel takes intensities only (`render(k, arr)`),
  treating array index as the grid — no physical coordinates (`dx` in
  arcsec/Å/etc.) threaded through. Simple, self-contained, matches "PSF
  convolves a rendered image" use case.
  Grids are uniform in practice, so index space and physical space differ by a
  constant the user applies once when constructing the kernel
  ([ADR-0001](docs/adr/0001-kernel-grid-contract.md)). The array render is also
  **size-preserving** — output shape equals input shape — which is what lets
  `chi2` compare `μ` against `y` with no runtime shape check. Edge handling
  (clamping, zero-padding, ...) is each kernel's own business, not the
  framework's.
- **Composition scope**: **all five compound nodes get an array render, behind
  a `_haskernel` guard** ([ADR-0002](docs/adr/0002-kernel-composition-scope.md)).
  A tree with no kernel takes the existing fused scalar broadcast, bit for bit;
  the guard folds to a compile-time constant, so kernel-free models pay nothing.
  Rejected alternatives:
  - *A — top-level only*: user convolves the rendered array by hand, no
    tree-dispatch changes at all. Simplest, but doesn't compose.
  - *B — dedicated Pipe only*: array method on `Pipe` alone. Rejected because
    it makes `(source |> psf) + background` a `MethodError`, and a
    PSF-convolved source summed with an unconvolved background is a real
    modelling pattern.
  - *C — array-native compound nodes, unconditionally*: structural recursion
    everywhere with no guard. Replaces the fused single-pass scalar broadcast
    with per-node intermediate arrays — regresses the ≤1.0x-vs-handwritten
    benchmark guarantee (`bench/README.md`). The guarded form above is C where
    it is needed and the status quo everywhere else.
- **Coordinate arity**: variadic (`xs...`) throughout, because a PSF on an image
  is the motivating case and 2D models render on two coordinate arrays. The
  kernel itself still takes exactly one intensity array — the two arities are
  different and must not be conflated.
- **Data layout for fitting**: a model containing a kernel requires grid-form
  data ([ADR-0003](docs/adr/0003-grid-form-data-for-kernel-fits.md)) — see §4.

## 3. Core dispatch design

```julia
abstract type AbstractKernel <: AbstractModel end

# Direct navigation: render(cm.psf, xs) works on its own.
# More specific than the AbstractModel fallback -> wins, no ambiguity.
render(l::Leaf{name, K}, xs::AbstractArray) where {name, K <: AbstractKernel} =
    render(l.model, xs)

# model |> kernel: render left over the whole array, feed the array to the kernel.
# Variadic in the COORDINATES (2D models take two), single array into the kernel.
render(m::Pipe, xs::AbstractArray...) =
    _haskernel(m.right) ? render(m.right, render(m.left, xs...)) : render.(m, xs...)

# Arithmetic nodes: structural recursion only when a kernel is present.
# Same shape for Sum/Difference/Product/Quotient — the operator is the only diff.
@inline render(m::Sum, xs::AbstractArray...) =
    _haskernel(m) ? render(m.left, xs...) .+ render(m.right, xs...) : render.(m, xs...)
```

All four arithmetic nodes get this method, not just `Sum`. The bodies are
identical modulo the operator, and the asymmetry of "`+` composes with kernels
but `*` doesn't" reads as a bug, not a decision — `(source |> psf) * transmission`
is as physical as the sum case. Cost of the symmetry: three extra lines.

When the guard is active, the kernel-free branch of a node (`g2` in
`(g1 |> psf) + g2`) is rendered via `render.(g2, xs...)` — a scalar broadcast
that allocates an array. That is the only extra allocation versus today, and it
is accepted.

`render(l::Leaf, x::Number...)` is left untouched — calling a kernel-containing
model with a scalar `x` throws a clean `MethodError` (no scalar `render`
exists for a kernel), rather than silently computing something wrong. This
was verified empirically (see §5, item 6).

### `_haskernel` — the compile-time presence predicate

Needed to (a) generalize the `Pipe` override beyond an exact
`Pipe{L,Leaf{name,K,C}}` type match, and (b) route `chi2` (§4) to a
vectorized path when a kernel is present anywhere in the tree.

Mirrors the existing `_leafnames!` idiom ([macro.jl:82-83](src/macro.jl:82))
— value-based recursion over `.left`/`.right`, not the type-based walk
`_slotmap!` uses (that one only exists because it runs inside a
`@generated` function, before any value exists — not the case here):

```julia
_haskernel(l::Leaf{name, M}) where {name, M} = M <: AbstractKernel
_haskernel(m) = _haskernel(m.left) || _haskernel(m.right)
_haskernel(cm::CompiledModel) = _haskernel(getfield(cm, :tree))
```

Because it dispatches purely on a concrete value's type, `_haskernel(x)`
folds to a compile-time constant — the branch inside `render(::Pipe, ...)`
and inside `chi2` costs nothing at runtime.

Consequences of this generalization over the original shape-matched method:

- Composes to arbitrary depth on the right: `image |> (psf1 |> psf2)` works,
  since `render(m.right, ...)` recurses back into the same `Pipe` method.
- The `Pipe` guard keys on `_haskernel(m.right)`, the arithmetic guards on
  `_haskernel(m)`. Deliberate: a `Pipe` only routes to the array path when the
  kernel is on the *right*. `psf |> g` — convolve, then transform pointwise —
  takes the scalar branch and `MethodError`s at the psf leaf. Fails loud, but
  it is an asymmetry worth knowing about before reading the code.
- A kernel buried under an arithmetic node (`(g1 |> psf) + g2`) is both
  detected *and* supported, since those nodes now carry their own guarded
  array render. This is the change from the original scope-B shape.

## 4. The chi2 blocker (the real finding of this session)

Every `chi2` variant in [loss.jl](src/fit/loss.jl) loops
`render(model, x[i])` one point at a time — the entire fitting path assumes
scalar-per-point. A kernel model cannot be evaluated one point at a time, so
as-is, wrapping a kernel in an `ObjectiveFunction` and fitting it is not just
slow, it's broken.

Fix (agreed with the advisor, not yet implemented): don't touch `render`
further, don't build a `RenderMode` trait through it — `render` already
routes correctly. Split only at the `chi2` call site:

```julia
if _haskernel(model)
    μ = render(model, x)                 # one call, whole array
    # vectorized residual: sum(abs2, (μ .- y) ./ err)
else
    # existing scalar loop, untouched — stays zero-alloc
end
```

### Grid-form data is mandatory for kernel fits

A conflict found while reviewing this design against the code: the fit path
does **not** carry a grid. `check_data` requires `length(c) == length(y)` for
every coordinate, and [bayes_tests.jl:119](test/bayes_tests.jl:119) fits 2D data
as *flattened* `xs`/`ys` point lists. `render(m.left, xs, ys)` on that layout
returns a flat vector of length N — convolving it with a PSF is meaningless,
because spatial adjacency was destroyed by the flattening.

Resolution ([ADR-0003](docs/adr/0003-grid-form-data-for-kernel-fits.md)): when
`_haskernel(cm)`, `ObjectiveFunction` takes `y` as an array whose shape *is* the
grid, plus one coordinate axis vector per dimension (lengths `size(y, d)`, not
`N`). Those axes are stored flat and reshaped to broadcast shapes by the kernel
branch — axis `k` extends along dimension `k` — so the 2D render is
`render.(m, xaxis, permutedims(yaxis))` and returns a matrix. Verified in
scratch: two plain equal-length axis vectors broadcast elementwise and return
the **diagonal** (`size == (9,)`), unequal lengths throw `DimensionMismatch`,
and only the reshaped form gives the `(9, 7)` grid. Leaving that reshape to the
caller would be a silent-wrong trap, so it lives inside the branch.

`check_data` branches on the same predicate. The pointwise path keeps the
flat form untouched. Rejected: reshaping the flat vectors inside `chi2`, which
would assume a flattening order nobody ever promised, and restricting kernels to
1D fits, which excludes image fitting — the motivating case.

Non-kernel models keep the current zero-allocation scalar loop
(`DECISIONS.md` is explicit this is deliberate, and `CLAUDE.md` guards the
≤1.0x-vs-handwritten benchmark claim — a blanket vectorization would regress
it). Kernel models get a single vectorized `render` call instead —
unavoidably allocates the prediction array, but that's one allocation per
objective evaluation, not per point.

**Swept and found clean**: the only other per-datum render loop that might
have needed the same fix is the future Uncertainty Model `~` machinery
described in `DECISIONS.md` (`μ[i] -> render(m, x[i])`) — it isn't
implemented yet, so there's nothing to fix there today. Flag for whoever
builds it later: it will need the same pointwise/kernel distinction.

## 5. Rejected alternative: `AbstractModel{has_kernel}`

Considered and rejected: parametrizing the abstract type itself
(`AbstractModel{HasKernel}`) so dispatch routes on a type parameter instead
of the `_haskernel` predicate function.

Blast radius that killed it:

- Every existing leaf struct (~15 across `models1d.jl`/`models2d.jl`) would
  need the parameter added.
- `Leaf` would gain a 4th type parameter derived from `M` — breaks every
  `name, M, C = T.parameters` destructure in `withparams.jl` (`_slotmap!`,
  `_treeexpr`) and `_nav` in `compiled.jl`.
- Compound nodes can't compute a supertype parameter from a function in
  Julia — `Sum{L,R} <: AbstractModel{haskernel(L)||haskernel(R)}` is not
  legal. Would need explicit `Sum{L,R,HK}` with constructors computing `HK`
  — 5 structs and 5 constructors rewritten.

Against nearly the entire core for a single-author POC: a `_haskernel(T)`
predicate gives the *same* compositional coverage (arbitrary depth, since it
recurses the actual tree) for ~5 lines and zero changes to existing structs.
The type parameter would only buy cleaner method signatures — pure
readability, identical runtime cost (both fold to zero). The "kernels can
only be called on arrays" safety net the parameter was meant to provide
already exists for free: kernels define no `Number` method, so a scalar call
`MethodError`s cleanly regardless.

## 6. Experiments run (scratch scripts, not committed)

Two toy kernels, deliberately not expressible as scalar renders:

```julia
# Zero math — pure dispatch sanity check. Output at i depends on input at
# n+1-i, not on any local neighborhood — can't be written as broadcast.
struct ReverseKernel <: AbstractKernel end
render(::ReverseKernel, xs::AbstractArray) = reverse(xs)

# Closer to a real PSF: box smoothing via direct convolution, edges clamped
# to keep output the same length as input.
struct RunningMeanKernel <: AbstractKernel
    width::Int
end
function render(k::RunningMeanKernel, xs::AbstractVector)
    n = length(xs)
    h = k.width ÷ 2
    out = similar(xs, float(eltype(xs)))
    @inbounds for i in 1:n
        lo, hi = max(1, i - h), min(n, i + h)
        out[i] = sum(@view xs[lo:hi]) / (hi - lo + 1)
    end
    return out
end
```

Verified against a `@model` tree (`g = Gaussian1D(...); k = ReverseKernel();
g |> k`):

1. Bare kernel, no `@model` — direct call works.
2. Kernel as a `Leaf` inside `@model`, direct navigation (`cm.k`) — Leaf
   override fires correctly.
3. `model |> kernel` — Pipe override fires, output matches
   `reverse(render(model, xs))` exactly.
4. `RunningMeanKernel` smoothing a `Gaussian1D` via Pipe — sane output.
5. `paramnames`/`nfree` on a kernel-containing model — see §7, this is where
   the width-as-free bug was found.
6. Scalar `render` on a kernel-containing model — clean `MethodError`, not a
   wrong number.
7. Plain scalar composition (no kernel) — byte-for-byte unchanged, no
   regression.

### ForwardDiff check (the point 3 risk flagged by the advisor)

Convolution assumes real numbers moving through a fixed-size array — the
worry was whether a `Dual`-typed array from ForwardDiff survives the kernel
render cleanly, since that's what the vectorized `chi2` path (§4) will push
through during gradient-based fitting.

With the buggy Free `width::Int` (see §7) in the parameter vector:
```
ERROR: InexactError: Int(Int64, Dual{...}(3,0,0,0,1))
```
ForwardDiff tries to make every slot in `p` a `Dual`, including `width`;
`RunningMeanKernel`'s constructor can't hold a `Dual` in an `Int` field.

With `width` fixed out (`@fix m.smoother.width = 3`):
```
params(m)  = [2.0, 0.0, 1.0]                      # concrete Vector{Float64}
gradient   = [-0.100, -3.9e-18, -0.200]           # no NaN/Inf
eltype(μD) = ForwardDiff.Dual{Nothing, Float64, 1} # Dual propagates cleanly
alloc      = 576 bytes per loss call                # not zero-alloc, expected
```

Conclusion: the vectorized render-through-kernel path is ForwardDiff-safe
and correct, **conditional on kernel structural fields never entering the
free-parameter vector** (§7 — this makes that fix non-optional).

## 7. Kernel field defaults — resolved: all Fixed

The macro marks every field of every leaf Free by default
(`_defaults`, [macro.jl:44](src/macro.jl:44)). For a kernel like
`RunningMeanKernel(width::Int)`, that's wrong twice over: semantically
(`width` is a discrete shape hyperparameter, not something a least-squares
optimizer should perturb continuously) and now empirically (§6 —
it crashes ForwardDiff outright, and independently degrades `params(cm)` to
an abstract `Vector{Real}` instead of a concrete `Vector{Float64}`, which is
its own latent type-instability).

**Decided: (a)** — all fields of a leaf whose model is `<: AbstractKernel`
default Fixed; a fittable PSF `sigma` needs an explicit `@free`
([ADR-0004](docs/adr/0004-kernel-fields-fixed-by-default.md)).

Rejected: **(b)**, fixing only `Integer` fields. It uses Int-vs-Float as a
*proxy* for "structural vs fittable", and the proxy breaks on float structural
fields — an oversampling factor, a truncation radius, a kernel sampling step.
Those would stay Free and reintroduce the same crash later and harder to
diagnose. (b) treats the symptom seen in §6, (a) encodes the semantics: a
kernel is normally a known calibration input, and in practice the PSF here is
fixed almost always.

## 8. Touch-point checklist

| # | Item | Status |
|---|------|--------|
| 1 | `AbstractKernel` type | done, tested |
| 2 | `render(Leaf, xs::AbstractArray)` override | done, tested |
| 3 | `render(Pipe, xs::AbstractArray...)` override | prototyped with exact shape-match; needs generalizing to `_haskernel`-driven, variadic form (§3) |
| 4 | `_defaults` kernel-field policy ([macro.jl:44](src/macro.jl:44)) | decided (a) — §7, not implemented |
| 5 | Vectorized `chi2` ([loss.jl](src/fit/loss.jl)) | designed, not implemented — §4 |
| 6 | `_haskernel` predicate | designed, not implemented — §3 |
| 7 | Kernel at tree root in `chi2` (`CompiledModel{Leaf{...,K}}`, no `Pipe`) | untested |
| 8 | `show.jl` display of a kernel-containing model | untested |
| 9 | `render!` anywhere in the fit path | likely unaffected (`ext/` greps clean, `loss.jl` only calls `render`) — not yet confirmed |
| 10 | Nested Pipe (`g |> (psf1 |> psf2)`) | untested |
| 11 | Array render on the four arithmetic nodes (`(g1 \|> psf) + g2`) | decided — §3, not implemented |
| 12 | Grid-form `check_data`/`ObjectiveFunction` branch | designed, not implemented — §4 |
| 13 | Size-preserving contract, `size(render(k, arr)) == size(arr)` | needs one `@testitem` |
| 14 | 2D kernel render through a variadic `Pipe` | untested |
| 15 | Axis-vector reshaping in the grid branch (`permutedims` for axis 2, `reshape` for axis k) | designed — §4, not implemented |

## 9. Next steps

1. Validate items 7, 10, 14 in scratch before touching `src/`.
2. Implement items 3 (generalized + variadic), 4, 5, 6, 11, 12 in `src/`.
4. Add a concrete zoo kernel (e.g. `GaussianPSF`, direct convolution — no
   FFT, so it stays ForwardDiff-safe for a free `sigma`) once the machinery
   above is landed and tested.
5. One `@testitem` per landed piece, per project convention.
