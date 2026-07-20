# Plan — one type parameter per field

A complete, self-contained implementation plan. It assumes no knowledge of the
conversation that produced it. Read it end to end before editing anything; the
Traps section (§7) contains findings that were established empirically and will
cost you hours if rediscovered by debugging.

**Repository**: AstroFit.jl. **Branch**: `kernel-support` (based on `main`).
**Baseline at time of writing**: commit `4071dd9`, 293 tests passing.

---

## 1. What this changes, in one sentence

Model structs stop sharing a single type parameter across all their fields
(`Gaussian1D{T}` with three `::T` fields) and get **one type parameter per
field** (`Gaussian1D{A,M,S}`), which removes the need to `promote` fields to a
common type during model reconstruction — deleting a code path in the core, 12
hand-written constructors in the model zoo, and a documented field contract,
while making automatic differentiation measurably faster for models with a mix
of free and fixed parameters.

## 2. Why — the motivation, with numbers

### 2.1 How AstroFit rebuilds a model

`withparams(cm, p)` (src/withparams.jl) is the hot path of the package. It is an
`@generated` function: it reads the model tree's *type*, works out at compile
time which fields map to which slot of the flat parameter vector `p`, and emits
straight-line code that reconstructs the tree. Fields come from three places:

- `Free`/`Bounded` → read from `p[k]`
- `Fixed` → a constant stored in the tree
- `Tied` → computed from other slots by a stored function

ForwardDiff differentiates a fit by passing a `p` whose elements are
`ForwardDiff.Dual` numbers. So during AD, a leaf's `Free` fields arrive as
`Dual` while its `Fixed` fields are still the stored `Float64` constants.

### 2.2 The problem this creates today

`Gaussian1D{T<:Real}` declares all three fields as `::T`. They must therefore
agree on one type. Reconstructing it from a `Dual` amplitude and two `Float64`
constants is impossible:

```julia
julia> Gaussian1D(ForwardDiff.Dual(1.0, 1.0), 6.0, 0.5)
ERROR: MethodError: no method matching Gaussian1D(::ForwardDiff.Dual{…}, ::Float64, ::Float64)
```

Two separate mechanisms exist purely to work around this:

1. **In the core** — `_ctorexpr` (src/withparams.jl) promotes a leaf's parameter
   fields to a common type before calling the constructor.
2. **In the zoo** — each multi-field model carries a hand-written promoting
   constructor, e.g. src/zoo/models1d.jl:9

   ```julia
   Gaussian1D(amplitude::Real, mean::Real, sigma::Real) =
       Gaussian1D(promote(amplitude, mean, sigma)...)
   ```

Both exist only because the fields are forced to share `T`. Give each field its
own parameter and both become unnecessary: `Gaussian1D{Dual,Float64,Float64}` is
a perfectly good type, and the default constructor builds it with no help.

### 2.3 The performance consequence

Promoting to a common type means that during AD, **fixed fields become `Dual`
too**, so arithmetic that could have stayed in `Float64` — the `exp`, the
subtraction, the division in a gaussian — is computed in dual arithmetic
instead. Measured on the real AstroFit path (`ForwardDiff.gradient` over an
`ObjectiveFunction`, 1201-point grid, minimum of a `@benchmark`):

| Model | free/fields | current | per-field, no promote |
|---|---|---|---|
| `Gaussian1D{T}` | 3 of 3 | 9.67 μs | 9.75 μs |
| `Gaussian1D{T}` | 1 of 3 | 7.96 μs | 7.97 μs |
| per-field gaussian | 3 of 3 | 9.67 μs | 9.62 μs |
| per-field gaussian | 1 of 3 | 8.01 μs | **4.97 μs** |

Read this table carefully, because three of its four rows are the *absence* of
an effect and that is the point:

- **All parameters free → no gain.** Every field is `Dual` either way.
- **A leaf with all fields fixed → no gain.** Its fields are stored `Float64`
  constants, no `Dual` enters, `promote` is a no-op. (Verified: an all-fixed
  `GaussianPSF` leaf reconstructs as `GaussianPSF{Float64}` even mid-gradient.)
- **A leaf mixing free and fixed float fields → 1.6× on the whole gradient.**
  This is the only case that gains, and the gain scales with the model's total
  `nfree`, since that is the dual width the fixed fields no longer carry.

The middle two rows also show that per-field parameters alone change nothing
while `_ctorexpr` still promotes: 8.01 μs vs 8.01. **The struct change and the
`_ctorexpr` change must land together or the benefit does not appear.**

Whether this case matters is a judgement about workload: fixing a line centre to
a known wavelength while fitting amplitude and width is routine in spectroscopy;
fitting every parameter of every component is equally routine and gains nothing.

### 2.4 The simplification, which is the primary motivation

Independently of speed, this deletes:

- `_ctorexpr` in src/withparams.jl (~20 lines with its docstring)
- `_isparamfield` in src/model.jl and the "field contract" it documents (~35
  lines with its docstring), plus its README section
- 12 hand-written promoting constructors in the zoo

and replaces all of it with `constructorof(M)(fields...)`.

The field contract currently has to explain why a `Float64` field is a fittable
parameter while an `Int` field is an internal value, and why `Bool` — a `Number`
in Julia — counts as internal. Under per-field parameters **that whole question
disappears**: every field keeps whatever type it is given, nothing is coerced
into agreeing with anything, and a kernel holding a `Symbol`, a `Matrix` and an
`Int` needs no special handling because nothing was ever going to touch them.

## 3. The design

Every model struct declares one type parameter per field:

```julia
# before
Base.@kwdef struct Gaussian1D{T <: Real} <: AbstractModel
    amplitude::T = 1.0
    mean::T = 0.0
    sigma::T = 1.0
end

Gaussian1D(amplitude::Real, mean::Real, sigma::Real) =
    Gaussian1D(promote(amplitude, mean, sigma)...)      # delete this

# after
Base.@kwdef struct Gaussian1D{A <: Real, M <: Real, S <: Real} <: AbstractModel
    amplitude::A = 1.0
    mean::M = 0.0
    sigma::S = 1.0
end
```

Naming convention for the parameters: use initials of the field names where they
are unambiguous (`A`, `M`, `S`), otherwise `T1, T2, …`. Pick one convention and
apply it uniformly; the 2D models have six or seven fields and initials collide
(`Gaussian2D` has `x0`/`y0`, `Sersic2D` has `n`/`q`), so **use `T1…Tn` for every
model with more than four fields** and initials only for the small ones. Do not
mix conventions within one file.

Constraint on the bounds: keep `<: Real` on every parameter of a normal model.
It is what makes ForwardDiff duals legal in those slots, and dropping it would
allow nonsense like a `String` amplitude. Kernel fields that are *not* numeric
(a sampled kernel array, an edge policy) keep whatever bound they already have
(`V <: AbstractVector`, unconstrained `F` for a function, or none).

## 4. Files to change

### 4.1 src/withparams.jl — the core change

Delete `_ctorexpr` and its docstring entirely (currently around lines 53–75),
and change the single call site in `_treeexpr` so that the constructor is called
with the fields **as they are**:

```julia
# on main — promotes every field, throws on any non-numeric one
:(Leaf{$(QuoteNode(name))}($(constructorof(M))(promote($(fields...))...), ($acc).constraints))

# on this branch — _ctorexpr decides which fields to promote
:(Leaf{$(QuoteNode(name))}($(_ctorexpr(M, fields)), ($acc).constraints))

# after this change — no promotion at all
:(Leaf{$(QuoteNode(name))}($(constructorof(M))($(fields...)), ($acc).constraints))
```

**The `promote` call is the thing being removed, and `_ctorexpr` only goes away
as a consequence.** This is worth stating precisely because it is easy to get
backwards: per-field type parameters do *not* make `promote` safe on a
heterogeneous field set. `promote` acts on runtime values, not on declared
types, so it fails identically whatever the struct's parameters are:

```julia
struct K{V <: AbstractVector, S <: Real, N <: Integer}
    taps::V; scale::S; halfwidth::N
end

K(promote([0.1, 0.2, 0.1], 1.5, 3)...)
# ERROR: promotion of types Vector{Float64}, Float64 and Int64
#        failed to change any arguments

K([0.1, 0.2, 0.1], 1.5, 3)
# K{Vector{Float64}, Float64, Int64}([0.1, 0.2, 0.1], 1.5, 3)
```

Per-field parameters are what make the `promote` call *unnecessary* — they are
not what makes it harmless. The end state is one token shorter than `main`, not
longer.

`fields` may go back to being a generator rather than a `Vector` if you prefer;
it was made a `Vector` only because `_ctorexpr` indexed into it.

`constructorof` comes from ConstructionBase and returns the *base* constructor
(`Gaussian1D`, not `Gaussian1D{Float64}`), so the new type parameters are
re-inferred from the argument types. That is exactly what is wanted and is why
no explicit type parameters need to be threaded anywhere.

### 4.2 src/model.jl — delete the field contract

Remove `_isparamfield` and its docstring (the block introduced by the comment
banner `# The field contract`). Nothing else in the file changes. Check for
other references first:

```bash
grep -rn "_isparamfield" src test ext
```

At the time of writing there are four references: the definition and its
docstring in src/model.jl, the `findall(_isparamfield, ...)` call inside
`_ctorexpr` in src/withparams.jl (which §4.1 deletes anyway), and one test item
in test/kernel_tests.jl (see Trap 7.4 before deleting that one).

### 4.3 src/zoo/models1d.jl and models2d.jl — the zoo

Twelve models need the struct signature rewritten and their promoting
constructor deleted:

| Model | File | Fields | Promoting ctor at |
|---|---|---|---|
| `Gaussian1D` | models1d.jl | amplitude, mean, sigma | :9 |
| `Linear1D` | models1d.jl | slope, intercept | :42 |
| `Lorentzian1D` | models1d.jl | amplitude, mean, gamma | :61 |
| `Voigt1D` | models1d.jl | amplitude, mean, sigma, gamma | :82 |
| `PowerLaw1D` | models1d.jl | norm, x_ref, index | :122 |
| `BlackBody1D` | models1d.jl | amplitude, temperature | :140 |
| `BrokenPowerLaw1D` | models1d.jl | norm, x_break, index1, index2 | :161 |
| `Exponential1D` | models1d.jl | amplitude, tau | :181 |
| `Gaussian2D` | models2d.jl | amplitude, x0, y0, sigma, q, theta | :13 |
| `Sersic2D` | models2d.jl | amplitude, x0, y0, r_eff, n, q, theta | :49 |
| `Moffat2D` | models2d.jl | amplitude, x0, y0, alpha, beta, q, theta | :88 |
| `Beta2D` | models2d.jl | amplitude, x0, y0, r_core, beta, q, theta | :123 |

Line numbers are from commit `4071dd9`; verify with
`grep -n "promote(" src/zoo/*.jl` before editing.

Three models need **no change** — they have a single field, so one shared
parameter and one per-field parameter are the same thing, and they have no
promoting constructor:

- `Const1D` (models1d.jl), `Redshift1D` (models1d.jl), `GaussianPSF`
  (zoo/kernels.jl)

`src/zoo/recipes1d.jl` constructs models by keyword and needs no change, but
re-read it after the edit: it passes computed expressions such as
`ratio * amplitude` and `-abs(amplitude)`, whose types must still land in a
`<: Real` slot.

### 4.4 src/params.jl — required, see Trap 7.1

`params(cm)` must be made to return a concrete vector. See §7.1 for why this is
mandatory rather than cosmetic. The change:

```julia
# before
params(cm::CompiledModel) = collect(_params(getfield(cm, :tree)))

# after — the isempty guard is required, see below
function params(cm::CompiledModel)
    vals = _params(getfield(cm, :tree))
    return isempty(vals) ? Float64[] : collect(promote(vals...))
end
```

The guard is not defensive padding: `collect(promote())` returns
`Vector{Union{}}` (verified), which is a legal but useless empty vector whose
element type will surprise anything downstream that tries to write into it. A
model with every parameter fixed or tied is a real configuration — it is what
`nfree(cm) == 0` means — so this path is reachable.

### 4.5 README.md — the user-facing contract

Two sections need rewriting.

**"Extending AstroFit" → Step 1** currently teaches the shared-`T` pattern and
warns that `T` must be `<: Real`. Rewrite the struct example with per-field
parameters and keep the `<: Real` warning, which is still true and still the
reason ForwardDiff works.

**"Extending AstroFit" → "The field contract"** (the `#### The field contract`
subsection) should be **deleted**, and replaced by a couple of sentences stating
the new, weaker and simpler rule:

> Each field keeps its own type. Declare a field with its own `<: Real`
> parameter if you might ever want to fit it; give it a concrete type
> (`Int`, `Bool`, `Symbol`, an array) if it is an internal value. Nothing is
> coerced, so an internal field of any type is carried through untouched.

Also check the "Kernels and PSF Convolution" section: its `InstrumentalPSF`
example is written against the old contract and should be updated to per-field
form.

### 4.6 Docs of record

- `REDESIGN.md` §9 documents the promote/`_ctorexpr` story in detail and ends
  with the field contract. Rewrite it to describe the per-field design, keeping
  the history of *why* promote existed — it explains the design and a future
  reader will otherwise wonder why the zoo ever had those constructors.
- `docs/adr/` — this decision qualifies for an ADR (hard to reverse, surprising
  without context, a real trade-off). The directory currently holds `0001`–`0004`
  (all about the kernel layer), so this is `0005-per-field-type-parameters.md`;
  confirm the highest number before writing. Follow the existing format: 1–3
  sentences of context, decision, and the rejected alternative. The alternative
  worth recording is "keep the shared `T` plus promotion", and the reason to
  reject it is that it costs a promoting constructor per model and forces fixed
  parameters into dual arithmetic during AD.
- `CLAUDE.md` states under "Model zoo": *"Type parameter must be `T<:Real` and
  coordinates `Number` (not `Float64`)"*. Update the first half.

## 5. The breaking change, stated plainly

After this change, a model struct that shares one type parameter across several
fields **and has no promoting constructor of its own** will fail as soon as one
of its fields is fixed or tied and the model is differentiated:

```julia
Base.@kwdef struct Blackbody1D{T<:Real} <: AbstractModel   # the old README pattern
    temperature::T = 5000.0
    norm::T = 1.0
end

cm = @model begin; b = Blackbody1D(); b; end
cm = @fix cm.b.norm
ForwardDiff.gradient(ObjectiveFunction(cm, x, y), params(cm))
# ERROR: MethodError: no method matching Blackbody1D(::Dual{…}, ::Float64)
```

It will *not* fail when every field is free, which makes the break intermittent
and confusing — a model works until the user fixes a parameter. **This is the
main risk of the change and the reason the error message matters.**

Two mitigations, and it is worth doing both:

1. **Documentation** — the README rewrite in §4.5, plus a note in the release
   notes / commit message that user models must move to per-field parameters or
   add their own promoting constructor (the pattern the zoo is losing).
2. **Optional but recommended: a diagnostic.** `@model` can detect at macro
   expansion time that a struct has fewer type parameters than fields and warn,
   or `withparams` can catch the `MethodError` and rethrow with an explanation.
   The macro-time check is better: it fires once, at the point of the mistake,
   rather than deep inside a gradient. Sketch:

   ```
   ERROR: Blackbody1D declares 1 type parameter for 2 fields, so its fields are
   forced to share a type. AstroFit reconstructs a model field by field, and a
   fixed field keeps its Float64 value while a free one becomes a dual number
   during differentiation, so the fields will not always agree.
   Declare one parameter per field:  struct Blackbody1D{T1<:Real, T2<:Real}
   ```

   Treat the diagnostic as part of the change, not a follow-up: without it the
   failure mode is a bare `MethodError` from generated code.

## 6. Order of work

Land it as one commit — the intermediate states are broken — but do the edits in
this order so you can check your understanding as you go:

1. Rewrite **one** zoo model (`Gaussian1D`) to per-field, delete its promoting
   constructor. Run the suite: it should still pass, because `_ctorexpr` still
   promotes. This confirms the struct rewrite alone is safe.
2. Change `_treeexpr` to call `constructorof(M)(fields...)` and delete
   `_ctorexpr`. Run the suite: expect failures from the 11 zoo models not yet
   converted **only if** a test fixes one of their fields; many will pass, which
   is the intermittency described in §5.
3. Convert the remaining 11 models and delete their constructors.
4. Delete `_isparamfield` and its test item.
5. Fix `params` (§4.4).
6. Add the diagnostic (§5.2).
7. Docs (§4.5, §4.6).
8. Verification (§8).

## 7. Traps

These were all established by measurement or by hitting them. Do not re-derive.

### 7.1 `collect` on a mixed tuple gives `Vector{Real}` — this one bites silently

The zoo's promoting constructors do more than serve AD: they normalize user
input. Today `Gaussian1D(amplitude = 1, mean = 0.0, sigma = 1.0)` — an integer
literal — promotes to `Gaussian1D{Float64}`. Remove the constructor and it
becomes `Gaussian1D{Int64, Float64, Float64}`, and then:

```julia
julia> typeof(collect((1, 0.0, 1.0)))
Vector{Real}                       # ← abstract element type

julia> typeof(collect(promote(1, 0.0, 1.0)))
Vector{Float64}
```

`params(cm)` is `collect(_params(tree))`, so a user who writes `1` instead of
`1.0` would silently get a `Vector{Real}` parameter vector: type-unstable,
slower, and a plausible source of AD failures downstream. This is why §4.4 is
mandatory. Add a test asserting `params(cm) isa Vector{Float64}` for a model
built with integer literals.

### 7.2 The struct change and the `_ctorexpr` change must land together

Per-field parameters with `_ctorexpr` still promoting gives **no** speedup
(8.01 μs vs 8.01 μs, measured). If you convert the zoo first and benchmark
before touching the core, you will conclude the change is worthless.

### 7.3 `ForwardDiff.Dual` is `Real` but not `AbstractFloat`

Relevant if you write any type predicate during this work:

```julia
ForwardDiff.Dual{Nothing,Float64,1} <: AbstractFloat   # false
ForwardDiff.Dual{Nothing,Float64,1} <: Real            # true
```

Keeping `<: Real` as the bound on parameter type parameters is therefore
correct and necessary; `<: AbstractFloat` would reject duals outright.

### 7.4 Nested duals must keep working

Second-order AD (or `withparams` applied to an already-dual model) produces
`Dual{…, Dual{…}}`. There is an existing test for this in `test/kernel_tests.jl`
("the field contract" test item). When you delete `_isparamfield`, **keep the
nested-dual assertions** — move them into another test item rather than deleting
the whole block with the predicate assertions.

### 7.5 The benchmark gate is a hard requirement

`CLAUDE.md` documents a ≤1.0x-vs-handwritten performance guarantee, and it is
guarded by a test asserting the χ² loop allocates nothing. The current numbers
on this machine: **0 allocations, 3828.125 ns**, identical to a handwritten
loop. A regression here is a bug, not a trade-off. §8 says how to check.

### 7.6 Type-parameter count and compile time

Each distinct combination of field types is a distinct concrete type and gets
its own specialization of `render`, `withparams` and everything downstream. In
practice the combinations are few — during a fit, all free fields are `Dual`
together and all fixed fields are `Float64` together, giving two configurations
per model, not 2ⁿ. But `Sersic2D` now has seven type parameters, and if compile
latency regresses noticeably this is where it comes from. Measure `time_to_load`
and first-call latency in §8 rather than assuming either way.

### 7.7 `@kwdef` defaults are unaffected

`Base.@kwdef` with per-field parameters works unchanged; the defaults (`= 1.0`
etc.) stay as they are and still determine the inferred type parameters when the
user omits an argument.

## 8. Verification protocol

Run all of it. The first three are pass/fail; the fourth is a judgement call.

**Test suite** — must be 293 passing or more (never fewer, minus any test items
you deliberately removed with `_isparamfield`):

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

**Benchmark gate** — must show 0 allocations and parity with the handwritten
loop:

```julia
# in a temp env with AstroFit dev'd and BenchmarkTools added
cm = @model begin
    g = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
    c = Const1D(value = 0.5)
    g + c
end
x = collect(-5.0:0.01:5.0); y = render(cm, x)
f = ObjectiveFunction(cm, x, y); p = AstroFit.params(cm)
@assert (@allocated f(p)) == 0
minimum((@benchmark $f($p)).times)     # compare against a handwritten loop
```

**Full benchmark suite vs `main`** — every ratio should be ≈1.00. Note the
tooling quirks: AstroFit is not registered so `--path=.` is required, and the
output directory must exist beforehand or the run completes and then fails on
write:

```bash
mkdir -p /tmp/bres
benchpkg AstroFit --path=. --rev=main,<your-branch> --bench-on=<your-branch> \
    --output-dir=/tmp/bres
```

Treat any single outlier with suspicion before believing it: in a previous run
`render/BrokenPowerLaw1D/astrofit` read 0.926 ± 0.082 and was pure noise (1.00 ±
0.11 on re-run), and `render/Gaussian1D/astrofit` read 0.991 twice in a row and
turned out to be one tick of timer quantization — a direct A/B with two git
worktrees gave byte-identical medians. **Re-run before reporting a regression,
and confirm with a direct measurement.**

**The gain itself** — confirm the change did what it is for:

```julia
# a 3-field model with 2 fields fixed, gradient through ObjectiveFunction
# expect ≈5 μs where the old code gave ≈8 μs on the same machine
```

If this shows no improvement, check Trap 7.2 first.

## 9. Rollback

The change is one commit and touches no data formats or stored state, so
`git revert` is sufficient. The only consideration is that user models written
for the new API (per-field parameters) continue to work after a revert —
`_ctorexpr`'s promote handles them — so a revert is safe in both directions.

## 10. What is explicitly out of scope

- Do not change `render` or the `Pointwise`/`Domainwise` evaluation traits.
- Do not change the kernel layer (`src/kernel.jl`, `src/zoo/kernels.jl`) beyond
  what §4 requires; `GaussianPSF` has one field and needs no edit at all.
- Do not touch anything in `ext/`. Nothing there constructs models.
- Do not "improve" the zoo's `render` methods while you are in those files. The
  diff should contain struct signatures, deleted constructors, and nothing else.
