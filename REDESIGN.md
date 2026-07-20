# Redesign — kernels as a first-class citizen

A plan for what AstroFit's evaluation layer would look like if it had been
designed with convolution-style models in mind from day one. Written after the
kernel design session recorded in [KERNELS.md](KERNELS.md) and
[ADR-0001..0004](docs/adr/), which reached a workable design by adding guards
around a pointwise core. This document asks the different question: what shape
removes the need for those guards entirely.

> **Status: implemented, in its 1D form.** The `evalstyle` trait spine (§3, §4),
> the kernel layer (§5) and the single-`chi2` loss (§7) shipped; the benchmark
> gate (§9) holds — the pointwise χ² still allocates nothing and matches the
> handwritten baseline. The `Domain` type (§2) did **not** ship: the success
> criteria were vector-based, and vectors are unambiguous, so the
> `PointCloud`/`GridDomain` split stays deferred until 2D kernel *fitting* is
> actually needed. [ADR-0003](docs/adr/0003-grid-form-data-for-kernel-fits.md)
> is therefore still a plan, not a description of the code.

## 1. Root cause: two concepts exist but are never named

Every friction point in the kernel session traces to the same thing — the
current design has two load-bearing concepts that live implicitly, spread across
signatures and conventions instead of being types:

**The evaluation grid.** Today it is "whatever you pass after the model": a
`Number` per coordinate, or a tuple of arrays, or — in the fit path — a flat
point list with `length(c) == length(y)`. Nothing distinguishes *N scattered
points* from *an N-point axis of a grid*. That ambiguity produced the worst
finding of the session: two equal-length axis vectors broadcast elementwise and
silently return the diagonal instead of the image (§4 of KERNELS.md, verified).
A wrong answer with no exception.

**Pointwise-ness.** The property "this model can be evaluated one coordinate at
a time, so its whole subtree can be fused into a single broadcast" is real,
compile-time knowable, and load-bearing for the benchmark. Today it is
*assumed*, not represented — which is why kernels needed a `_haskernel`
predicate bolted on afterwards to ask the question the type system should have
answered.

Make both explicit and the guards stop being guards: they become dispatch.

## 2. `Domain` — the evaluation grid as a type

One type, two concrete forms, chosen because the fit path genuinely needs both:

- **`PointCloud`** — N scattered coordinates, one entry per datum. Broadcast is
  elementwise. This is what the current flat-list fit path
  ([bayes_tests.jl:119](test/bayes_tests.jl:119)) already uses; it stays the
  representation for irregular data, and the pointwise fast path lives here.
- **`GridDomain`** — one axis vector per dimension plus the resulting shape.
  Broadcast is outer-product: the domain knows that axis `k` must extend along
  dimension `k`, and yields correctly-shaped broadcast axes on demand. The
  `permutedims`/`reshape` trap becomes one method, computed once, instead of an
  ad-hoc reshape buried in `chi2`.

This subsumes both grid ADRs. [ADR-0001](docs/adr/0001-kernel-grid-contract.md)'s
index-as-grid contract becomes a property `GridDomain` can carry and check
(uniform spacing), rather than an undocumented precondition. 
[ADR-0003](docs/adr/0003-grid-form-data-for-kernel-fits.md)'s "kernel fits need
grid-form data" stops being a runtime branch in `check_data` and becomes a
signature: a kernel renders on `GridDomain`, and nothing else. Handing scattered
points to a PSF is a `MethodError` at construction, structurally.

**The Domain is an argument to `render`, never a field of the model.** This is
the tempting mistake to avoid: a model that stores its grid can no longer be
rendered on a different one, which breaks oversampling, model-vs-data grids,
and plotting at higher resolution. Models stay grid-agnostic.

Deliberately out of scope: an N-D domain algebra, resampling, WCS. 1D spectra
and 2D images are the world this package lives in.

## 3. Evaluation trait — pointwise-ness as a compile-time answer

A two-state trait on the model type:

- `Pointwise()` — defines a scalar `render(m, x::Number...)`; everything in the
  current zoo.
- `Domainwise()` — defines only `render(m, xs, ::Domain)`; kernels.

Compound nodes *compute* their trait by combining children: pointwise ⊗
pointwise = pointwise, anything else = domainwise. Because it dispatches on
concrete types it folds to a compile-time constant, exactly as `_haskernel`
does — same cost, but it answers "how is this evaluated" instead of "does this
contain a special case", which is what makes it a spine rather than a patch.

Two states only. An open taxonomy of evaluation styles is the speculative
abstraction to refuse here.

## 4. Render layer — one entry point, fusion preserved by construction

```
render(model, domain)
├── trait == Pointwise    → render.(model, broadcast_axes(domain)...)
└── trait == Domainwise   → structural recursion, fusing pointwise subtrees
```

**Scalar `render(m, x::Number...)` methods stay, for leaves and for compound
nodes.** They are not legacy — they are the mechanism that keeps the benchmark
intact. The `Pointwise` branch routes straight back to today's fused broadcast
([model.jl:5](src/model.jl:5) plus the `@inline` scalar compound methods in
[compound.jl](src/compound.jl)), so a kernel-free model emits the *same code as
today*, not merely comparable code. This is the plan's acceptance gate, stated
concretely: `benchpkg AstroFit --rev=main,dirty` shows no regression on the
kernel-free suite, because the emitted path is identical.

The gain over the shipped `_haskernel` design is **maximal fusion in mixed
trees**. In `(g1 + g2) |> psf`, the subtree `g1 + g2` is pointwise, so it
renders as one fused broadcast pass and only then hands an array to the kernel.
Fusion breaks at kernel boundaries and nowhere else. The guarded design of
[ADR-0002](docs/adr/0002-kernel-composition-scope.md) breaks it at *every* node
of a kernel-bearing tree, because the guard tests the whole subtree rather than
locating the boundary.

## 5. Kernel as a first-class fittable component

"Native" means fittable, not merely non-crashing. A kernel is an ordinary leaf
flowing through `withparams` like any other, with four things the current sketch
does not have:

- **Differentiable normalization.** A sampled kernel must sum to 1, and the
  normalization must be ForwardDiff-safe so a free `sigma` produces a correct
  gradient — normalize by the sampled sum (differentiable) or by the analytic
  integral, never by a value captured outside the derivative path.
- **Edge policy as part of the kernel's own API**, not a framework decision —
  [ADR-0001](docs/adr/0001-kernel-grid-contract.md) already fixes size-preservation and leaves the
  edges open; this makes the choice (clamp / zero-pad / renormalize) an explicit
  field rather than an implicit convention.
- **Direct convolution, no FFT.** `ponytail:` O(N·k²) ceiling, upgrade to FFT
  only when a measured profile demands it; direct convolution keeps the whole
  path ForwardDiff-safe, which FFT does not.
- **A real zoo**: `GaussianPSF`, `MoffatPSF`, `Instrumental LSF` for spectra —
  the concrete components that make the machinery worth having.

[ADR-0004](docs/adr/0004-kernel-fields-fixed-by-default.md) (kernel fields
Fixed by default) carries over unchanged and matters more here, not less:
kernels being ordinary leaves is exactly why their fields would otherwise
default Free.

## 6. What does *not* change — the spine is already right

This is the part that makes the plan actionable. The core that makes AstroFit
what it is needs no redesign:

| Component | Why it survives |
|---|---|
| `withparams` ([withparams.jl](src/withparams.jl)) | `@generated` slot map over the tree *type*; knows nothing about how the rebuilt model is evaluated |
| `@model` / `_defaults` ([macro.jl](src/macro.jl)) | tree construction; only the kernel-field default policy changes (already decided) |
| `constrain.jl`, `params.jl`, `compiled.jl` | constraints, slot ordering, navigation — all orthogonal to evaluation |

The redesign lives at the render layer and in `loss.jl`. Everything above the
render call is ported as-is.

## 7. Loss layer — one `chi2`, no branch

`ObjectiveFunction` stores a `Domain` instead of a raw coordinate tuple. `chi2`
then has a single implementation — `render(model, domain)` compared against `y`
— with no `_haskernel` test, because the trait already chose the evaluation
strategy one layer down. The pointwise case still compiles to the current
zero-allocation scalar loop; that is a property of the fused branch, not of a
separate code path in `chi2`.

`check_data` becomes domain construction: validating that axes and data shapes
agree is what building a `GridDomain` *is*, so invalid data fails at
construction with no separate validator.

For the domainwise path, which unavoidably allocates a prediction array per
objective evaluation, a workspace buffer carried on `ObjectiveFunction` reduces
this to one allocation per fit instead of one per iteration — worth doing only
after a profile says so.

## 8. Rebuilt vs ported

**Rebuilt** (3 files): `model.jl` (trait + render entry point), `compound.jl`
(trait combination; scalar methods kept verbatim), `fit/loss.jl` (Domain-based,
single `chi2`).
**New** (2 files): `domain.jl`, `zoo/kernels.jl`.
**Ported unchanged**: `withparams.jl`, `macro.jl`, `constrain.jl`,
`constraints.jl`, `params.jl`, `compiled.jl`, `priors.jl`, the existing zoo, all
of `ext/`.

The honest summary: this is an evolution of three files plus two new ones, not a
fifteen-file rewrite. The `Domain` + trait spine drops onto the existing core
because that core was already factored along the right seam.

## 9. Verification

- **Benchmark gate, non-negotiable**: `benchpkg AstroFit --rev=main,dirty` on
  the kernel-free suite must show no regression. The claim is structural (§4),
  so a regression means the fused branch is not being reached — a bug, not a
  tradeoff.
- One `@testitem` per landed piece, per project convention: domain construction
  and shaped-axis broadcast; trait combination over compound nodes; maximal
  fusion in a mixed tree; size preservation; a `GaussianPSF` gradient check
  against finite differences with a free `sigma`.
- The scattered-points-to-a-kernel case must fail at construction. A test that
  asserts the error is the guard.

## 10. Would I actually do this?

Only if kernels become central. The shipped `_haskernel` design is correct and
cheaper; this one is better *shaped*, and pays off when the second and third
non-pointwise model type arrive (binning/pixel integration, instrumental
response, anything else where the answer at a point depends on a neighbourhood).
Two of those and the guards start multiplying — that is the signal to do this,
and doing it before that signal is speculative work on a single-author POC.
