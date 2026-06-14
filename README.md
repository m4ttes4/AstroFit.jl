# AstroFit

A Julia library for building, constraining, and fitting parametric astrophysical models.

AstroFit lets you compose models from reusable building blocks, attach physical constraints, and extract a flat parameter vector suitable for any gradient-based or gradient-free optimizer — all with zero-allocation, type-stable hot paths and full ForwardDiff compatibility.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Building Models with `@model`](#building-models-with-model)
- [Adding Constraints with `@constrain`](#adding-constraints-with-constrain)
- [Working with Parameters](#working-with-parameters)
- [Fitting Loop](#fitting-loop)
- [Extending AstroFit](#extending-astrofit)
- [Internal Design](#internal-design)

---

## Installation

```julia
using Pkg
Pkg.add("AstroFit")
```

Requires Julia ≥ 1.9.

---

## Quick Start

```julia
using AstroFit

# 1. Build a composite model from named components
m = @model begin
    narrow = Gaussian1D(amplitude=2.0, mean=6563.0, sigma=1.5)
    broad  = Gaussian1D(amplitude=0.5, mean=6563.0, sigma=8.0)
    narrow + broad
end

# 2. Attach constraints
cm = @constrain m begin
    @bound narrow.amplitude in (0, Inf)
    @bound narrow.mean      in (6555.0, 6570.0)
    @bound narrow.sigma     in (0.1, Inf)
    @bound broad.amplitude  in (0, Inf)
    @tie   broad.mean       = narrow.mean     # always equal to narrow.mean
    @bound broad.sigma      in (1.0, Inf)
end

# 3. Evaluate
wavelengths = 6540.0:0.1:6590.0
flux = render(cm, wavelengths)

# 4. Extract parameters for an optimizer
p0         = paramvector(cm)          # Vector{Float64}, only free params
lo, hi     = bounds_vectors(cm.spec)  # bound vectors aligned with p0

# 5. Fit (example with any optimizer)
loss(p) = sum(abs2, render(withparams(cm, p), wavelengths) .- observed_flux)
p_fit   = your_optimizer(loss, p0, lo, hi)

# 6. Reconstruct the fitted model
cm_fit = withparams(cm, p_fit)
println("narrow amplitude = ", cm_fit.narrow.amplitude)
```

---

## Core Concepts

### Models

A **model** is an immutable struct that knows how to evaluate itself on inputs. Every model is a subtype of `AbstractModel{N}`, where `N` is the number of input dimensions. You evaluate a model with `render(model, x...)`.

Leaf models hold parameters as struct fields:

```julia
g = Gaussian1D(amplitude=1.0, mean=0.0, sigma=2.0)
render(g, 0.5)   # => 0.939...
render(g, -1:0.5:1)  # broadcast over array
```

### CompiledModel

A `CompiledModel` bundles a model tree together with its constraints and a component registry. It is the central object you interact with after calling `@model` or `@constrain`.

```julia
cm = @model begin
    a = Gaussian1D(amplitude=1.0, mean=0.0, sigma=1.0)
    b = Const1D(value=0.5)
    a + b
end
# cm is a CompiledModel

render(cm, 0.0)        # evaluate
cm.a.amplitude         # read a component's parameter
nfree(cm)              # number of free parameters
```

A `CompiledModel` carries four pieces of information:

| Field    | Type               | Contents                                         |
|----------|--------------------|--------------------------------------------------|
| `.model` | `M`                | The evaluated model tree (always tie-resolved)   |
| `.spec`  | `S <: Tuple`       | Tuple of `(optic, constraint)` pairs             |
| `.priors`| `P <: Tuple`       | Tuple of `(optic, distribution)` pairs           |
| `.names` | `R <: NamedTuple`  | Component name → optic (or `Registry`) mapping   |

**Invariant I1:** `.model` is always tie-resolved. You will never observe a stale `Tied` parameter. Every constructor path goes through `_compiled`, which calls `resolve` before storing the tree.

### Constraint Types

Four constraint types govern each parameter:

| Type         | Meaning                                           |
|--------------|---------------------------------------------------|
| `Free()`     | Parameter is unconstrained; included in fit       |
| `Fixed(v)`   | Parameter locked to value `v`; excluded from fit  |
| `Bounded(lo, hi)` | Parameter is free but bounded to `[lo, hi]`  |
| `Tied(f, masters)` | Parameter computed as `f(master1, master2, ...)`; excluded from fit |

All bare model parameters start as `Free` when you call `@model`. `@constrain` overrides specific entries.

### Operators

Models compose with arithmetic operators and pipes:

| Expression   | Type produced        | Evaluation                      |
|--------------|----------------------|---------------------------------|
| `a + b`      | `Sum{N,L,R}`         | `render(a,x) + render(b,x)`     |
| `a - b`      | `Difference{N,L,R}`  | `render(a,x) - render(b,x)`     |
| `a * b`      | `Product{N,L,R}`     | `render(a,x) * render(b,x)`     |
| `a / b`      | `Quotient{N,L,R}`    | `render(a,x) / render(b,x)`     |
| `a ∘ b`      | `Pipe{N,L,R}`        | `render(a, render(b,x))`        |
| `a \|> b`    | `Pipe{N,L,R}`        | `render(b, render(a,x))`        |

> Note: bare scalars are not allowed in model algebra. Use a named `Const1D` so the constant is fittable and addressable.

---

## Building Models with `@model`

`@model` constructs a `CompiledModel` from a composition of named components.

### Block form

```julia
cm = @model begin
    name₁ = model₁
    name₂ = model₂
    name₁ + name₂
end
```

Bindings define component names. The final expression is the composition tree. Every bound name must appear in the composition.

### Inline form

```julia
a = Gaussian1D(amplitude=1.0, mean=0.0, sigma=1.0)
b = Const1D(value=0.5)
cm = @model a + b
```

Names are taken from the in-scope variable symbols.

### Composing multiple components

```julia
cm = @model begin
    cont   = Linear1D(slope=0.0, intercept=1.0)
    line1  = Gaussian1D(amplitude=2.0, mean=6563.0, sigma=2.0)
    line2  = Gaussian1D(amplitude=0.7, mean=6548.0, sigma=2.0)
    cont + line1 + line2
end
```

n-ary `+` is left-folded at macro expansion time, so `a + b + c` becomes `(a + b) + c`, producing a binary tree.

### Nesting prefabs

A component can itself be a `CompiledModel` (a *prefab*). Its constraints travel into the parent model, namespaced under its binding name:

```julia
ha_line = @constrain Gaussian1D(amplitude=10.0, mean=6563.0, sigma=2.0) begin
    @bound amplitude in (0, Inf)
    @bound sigma     in (0.1, Inf)
end

spectrum = @model begin
    cont = Linear1D(slope=0.0, intercept=1.0)
    ha   = ha_line       # prefab: its constraints are inherited
    cont + ha
end

# spectrum.ha.amplitude is Bounded(0, Inf)  — inherited from ha_line
# spectrum.cont.slope is Free               — default for bare leaves
```

Prefab `Tied` master optics are automatically re-rooted under the new prefix. A tie that was `narrow.sigma` inside the prefab becomes `ha.narrow.sigma` in the parent.

### Pipe composition (redshift example)

```julia
Base.@kwdef struct Redshift1D{T<:Real} <: AbstractModel{1}
    z::T = 0.0
end
render(m::Redshift1D, λ) = λ / (1 + m.z)

z_shift = @constrain Redshift1D(z=0.05) begin
    @bound z in (0.0, 1.0)
end

observed = @model begin
    spectrum = rest_spectrum   # another CompiledModel
    z_shift  = z_shift
    spectrum ∘ z_shift         # evaluates as: spectrum(z_shift(λ_obs))
end
```

---

## Adding Constraints with `@constrain`

`@constrain` takes a model (or `CompiledModel`) and a block of constraint directives. It returns a new `CompiledModel` with the constraints applied.

```julia
cm = @constrain model begin
    @fix   component.param = value    # lock to value
    @fix   component.param            # lock at current value
    @bound component.param in (lo, hi)
    @tie   component.param = expr(other.param, ...)
    @free  component.param            # release a constraint (back to Free)
end
```

### `@fix` — locking a parameter

```julia
@constrain cm begin
    @fix narrow.mean = 6562.8    # set and lock at 6562.8
    @fix broad.mean              # lock at the current value in cm
end
```

Fixed parameters are removed from the free parameter vector entirely.

### `@bound` — bounding a parameter

```julia
@constrain cm begin
    @bound narrow.amplitude in (0, Inf)
    @bound narrow.sigma     in (0.5, 20.0)
end
```

Bounded parameters remain in the free parameter vector. If the current value already violates the bounds, `@constrain` throws immediately — there is no silent clamping.

### `@tie` — dependent parameters

```julia
@constrain cm begin
    @tie broad.mean  = narrow.mean                  # single master
    @tie broad.sigma = narrow.sigma                 # same
    @tie blue.mean   = (4958.9 / 5006.8) * red.mean # expression with masters
    @tie blue.amp    = red.amp / 2.98               # ratio constraint
end
```

The RHS is an arbitrary Julia expression. Every `component.param` reference in the RHS is auto-detected as a master. The tie is stored as a closure `f(masters...) -> value` and re-evaluated every time masters change.

Constraints on `Tied` targets are rejected (`@bound`, `@fix`, `@prior` on a tied param all throw). Tie chains (master is itself tied) are also rejected.

### `@free` — releasing a constraint

```julia
@constrain cm begin
    @free narrow.sigma    # previously Fixed or Tied; now free again
end
```

### Constraint merging

Calling `@constrain` on an existing `CompiledModel` merges the new constraints with the existing ones. Within the same block, the last entry for a given parameter wins. New entries override old ones:

```julia
# ha_line already has @bound amplitude in (0, Inf)
spectrum = @constrain spectrum begin
    @bound ha.amplitude in (0, 100)   # overrides the prefab bound
end
```

### Cross-component ties

Ties can reference parameters in sibling components, including across nested prefabs:

```julia
rest_spec = @constrain rest_spec begin
    @tie hbeta.amplitude = ha_nii.ha.amplitude / 2.86  # Balmer decrement
    @tie hbeta.sigma     = oiii.blue.sigma              # shared kinematics
end
```

The master paths are resolved against the full registry at the time `@constrain` runs.

---

## Working with Parameters

### Extracting the parameter vector

```julia
p = paramvector(cm)    # Vector{Float64} of all free parameter values
```

The vector contains only `Free` and `Bounded` parameters, in the order they appear in the spec (which reflects the order of `@model` bindings and operator tree structure). `Fixed` and `Tied` parameters are excluded.

```julia
nfree(cm)              # Int: number of free parameters
freevals(cm)           # NTuple of free parameter values (allocation-free)
```

### Bounds

```julia
lo, hi = bounds_vectors(cm.spec)
# lo, hi are Vector{Float64} aligned with paramvector(cm)
# Free params → (-Inf, Inf); Bounded params → (lo, hi)
```

### Rebuilding from a vector: `withparams`

`withparams` is the hot path for the fitting loop:

```julia
cm_new = withparams(cm, p)
```

It scatters `p` into the free parameter positions, then re-resolves all `Tied` parameters. The `CompiledModel` type (including `.spec` and `.names`) is unchanged; only `.model` is updated.

`withparams` allocates only the new model struct fields — no intermediate arrays.

### Reading individual parameters

```julia
cm.component.param         # value or ComponentRef
cm.component               # ComponentRef (cursor into the subtree)
cm[:component].param       # explicit bracket form (avoids reserved name conflicts)
```

Parameter access is purely read-path: it applies the stored optic to `.model`.

### Updating a single parameter

```julia
using AstroFit  # re-exports @set from Accessors
cm2 = @set cm.narrow.amplitude = 3.5
```

`@set` validates bounds and rejects writes to `Tied` or reserved fields. It returns a new `CompiledModel` with the updated value and all ties re-resolved.

---

## Fitting Loop

### Gradient-free optimizer

```julia
using Optim

cm = @constrain ... end
lo, hi = bounds_vectors(cm.spec)
p0     = paramvector(cm)

result = optimize(
    p -> sum(abs2, render(withparams(cm, p), x) .- y),
    lo, hi, p0,
    Fminbox(NelderMead()),
)
cm_fit = withparams(cm, result.minimizer)
```

### Gradient-based optimizer with ForwardDiff

```julia
using Optim, ForwardDiff

# Define a stable loss functor (not a closure) to avoid type instability
struct SpectralLoss{CM, X, Y}
    cm::CM
    x::X
    y::Y
end
(l::SpectralLoss)(p) = sum(abs2, render(withparams(l.cm, p), l.x) .- l.y)

loss = SpectralLoss(cm, wavelengths, observed_flux)
p0   = paramvector(cm)

result = optimize(loss, p0, LBFGS(); autodiff=:forward)
cm_fit = withparams(cm, result.minimizer)
```

> **ForwardDiff note:** use a named functor or a `struct` callable rather than an anonymous closure. ForwardDiff tags the function type into `Dual{Tag{F,V},V,N}`. An anonymous closure created freshly each call has a different (anonymous) type each time, triggering recompilation. A named functor has a stable type, so the generated code is compiled exactly once per `(model_type, chunk_size)` combination.

---

## Extending AstroFit

Any Julia struct that subtypes `AbstractModel{N}` is a first-class AstroFit model. You only need to:

1. Subtype `AbstractModel{N}` (N = number of input dimensions)
2. Define `render(m::YourModel, x::Number...)` (scalar evaluation)

Everything else — composition, naming, constraints, `withparams`, property access — works automatically.

### Example: redshift operator

```julia
Base.@kwdef struct Redshift1D{T<:Real} <: AbstractModel{1}
    z::T = 0.0
end

# Constructor for non-keyword call
Redshift1D(z::Real) = Redshift1D{typeof(float(z))}(float(z))

# Scalar evaluation: maps observed wavelength to rest-frame
render(m::Redshift1D, λ) = λ / (1 + m.z)
```

This model can then be used everywhere:

```julia
z_shift = @constrain Redshift1D(z=0.05) begin
    @bound z in (0.0, 1.0)
end

observed = @model begin
    spectrum = rest_spectrum
    z_shift  = z_shift
    spectrum ∘ z_shift      # pipe: evaluates spectrum(z_shift(λ_obs))
end

nfree(observed)             # counts z as free
paramvector(observed)       # includes z
```

### Rules for custom models

- All fields must be the same type `T <: Real` (use `Base.@kwdef` + `promote` for mixed literals).
- `render` must accept `Number` arguments (not just `Float64`) so ForwardDiff Dual values flow through.
- Do not implement `broadcastable` — the default `Ref(m)` inherited from `AbstractModel` is correct.
- Compound models (`Sum`, `Pipe`, etc.) are constructed automatically by operators; you never define them directly.

---

## Internal Design

This section describes the implementation in detail. It is not required for using the library, but is useful for contributors and for understanding performance characteristics.

### Architecture overview

AstroFit is built on three principles:

1. **Value semantics everywhere.** Models, specs, and `CompiledModel` are all immutable. Every operation (constraining, updating parameters) returns a new object. There is no shared mutable state.

2. **Type-encoded specification.** The constraint spec is a `Tuple` (not a `Vector`). Each entry's *type* encodes which constraint variant it holds (`Free`, `Bounded`, etc.). This means Julia's type inference can inspect the spec at compile time and generate specialized code — in particular, `@generated` functions that emit optimal code without runtime branching.

3. **Single invariant (I1).** The model tree stored in `.model` is always tie-resolved. This is enforced by routing every constructor through `_compiled`, which calls `resolve` before storing the tree. Code that reads `.model` never needs to re-resolve.

---

### Type hierarchy

```
AbstractModel{N}                        # abstract base; N = input dims
│
├── Gaussian1D{T<:Real}                 fields: amplitude, mean, sigma
├── Const1D{T<:Real}                    fields: value
├── Linear1D{T<:Real}                   fields: slope, intercept
├── Gaussian2D{T<:Real}                 fields: amplitude, x0, y0, sigma_x, sigma_y, theta
├── ExponentialDisk2D{T<:Real}          fields: amplitude, x0, y0, r_eff, ellip, theta
│
└── compound wrappers (binary; users don't construct these directly)
    ├── Sum{N, L<:AbstractModel{N}, R<:AbstractModel{N}}
    ├── Difference{N, L<:AbstractModel{N}, R<:AbstractModel{N}}
    ├── Product{N, L<:AbstractModel{N}, R<:AbstractModel{N}}
    ├── Quotient{N, L<:AbstractModel{N}, R<:AbstractModel{N}}
    └── Pipe{N, L<:AbstractModel{N}, R<:AbstractModel{1}}
        # inner: N-dim input → scalar
        # outer: scalar → scalar
        # combined: N-dim → scalar

CompiledModel{M, S<:Tuple, P<:Tuple, R<:NamedTuple}
    .model  :: M   — the resolved model tree
    .spec   :: S   — tuple of (optic, constraint) pairs
    .priors :: P   — tuple of (optic, distribution) pairs
    .names  :: R   — component name → optic or Registry

ComponentRef{CM<:CompiledModel, O, R<:NamedTuple}
    — a cursor into a named component; returned by property access on CompiledModel

Registry{O, R<:NamedTuple}
    — entry in .names when the component is itself a CompiledModel (prefab)
    — holds the optic to the subtree root + the prefab's own sub-registry
```

Compound wrappers hold their children in `.left` and `.right` fields, mirroring the operator argument order (or reversed for `∘`, since `a ∘ b` evaluates `a(b(x))`).

---

### Constraint types

```julia
struct Free end                        # no data; singleton type

struct Fixed{T}
    value::T                           # the locked value (T=Nothing means "lock at current")
end

struct Bounded{T}
    lower::T
    upper::T
end

struct Tied{F, Ms<:Tuple}
    f::F         # closure: f(master1_val, master2_val, ...) -> dependent_value
    masters::Ms  # tuple of optics pointing to each master parameter
end
```

Constraint types are deliberately minimal. The type tag (`Free`, `Fixed`, `Bounded`, `Tied`) is what `@generated` functions inspect at compile time — the variant is in the type, not a runtime tag field.

---

### `CompiledModel` internals

The four type parameters of `CompiledModel{M, S, P, R}` are all inferred from the arguments:

- **`M`**: the concrete model tree type (e.g. `Sum{1, Gaussian1D{Float64}, Gaussian1D{Float64}}`). This changes every time a parameter changes type (e.g. when ForwardDiff substitutes `Dual` for `Float64`).
- **`S`**: the spec tuple type. This encodes the number of constrained parameters, their optic types, and their constraint variant types. It is fixed for the lifetime of a compiled model — `@constrain` produces a new `S`, but `withparams` preserves it.
- **`P`**: the prior tuple type. Same structure as `S`.
- **`R`**: the names `NamedTuple` type. Encodes the component names as type-level symbols. Fixed after `@model`.

**Invariant I1** is maintained by the private constructor `_compiled`:

```julia
_compiled(model, spec, names, priors=()) =
    CompiledModel(resolve(model, spec), spec, priors, names)
```

`resolve(model, spec)` calls `_resolve(model, tied_entries(spec))`, which iterates over all `Tied` entries and writes the computed values into the model tree. Because every public-facing operation (constraint application, parameter update, `@set`) routes through `_compiled`, I1 holds everywhere.

---

### `@model` macro pipeline

The `@model` macro transforms a composition expression into a `CompiledModel`. Here is the full pipeline:

```
Source code                   Macro expansion (compile time)
─────────────────────────────────────────────────────────────

@model begin                  1. Parse block: extract bindings + composition expr
  a = M1                         bindings = [a => M1, b => M2]
  b = M2                         comp_expr = :(a + b)
  a + b
end                           2. _walk_optics(comp_expr)
                                 Walks the expression tree recursively.
                                 a + b  → Sum has .left=a, .right=b
                                 Returns: [a => [:left], b => [:right]]

                              3. _path_to_optic_expr([:left])
                                 → PropertyLens{:left}()
                                 _path_to_optic_expr([:right])
                                 → PropertyLens{:right}()
                                 (deeper paths: lens1 ⨟ lens2 ⨟ ...)

                              4. Build the composition closure:
                                 (ga, gb) -> ga + gb
                                 (leaf symbols replaced with gensym args)

                              5. Emit call to _build_model:
                                 _build_model(closure, (:a, :b),
                                              (PropertyLens{:left}(), PropertyLens{:right}()),
                                              (M1, M2))

Runtime (_build_model)
──────────────────────

_build_model(f, names, optics, values):
  1. bare = map(_strip_leaf, values)
       CompiledModel → its .model field
       AbstractModel → as-is
       anything else → error

  2. tree = f(bare...)
       Calls the closure: Sum(M1_bare, M2_bare)
       This builds the actual model tree.

  3. _identity_check(tree, names, optics, bare)
       For each (name, optic, bare_leaf):
         optic(tree) === bare_leaf   (identity check, not ==)
       Guards against bugs in _walk_optics.

  4. spec = _collect_spec(optics, values)
       For a bare AbstractModel at optic o:
         map each field f to (o ⨟ PropertyLens{f}(), Free())
       For a CompiledModel (prefab) at optic o:
         map each (t, c) in prefab.spec to (_compose(o, t), _reroot(o, c))
         _reroot rewrites Tied masters: each master optic m → _compose(o, m)

  5. priors = _collect_priors(optics, values)
       Same logic for prior entries.

  6. registry = NamedTuple{names}(map(_registry_entry, optics, values))
       For AbstractModel at optic o:   entry = o  (the optic itself)
       For CompiledModel at optic o:   entry = Registry(o, prefab.names)

  7. _compiled(tree, spec, registry, priors)
       → CompiledModel (with I1 established via resolve)
```

**`_walk_optics` for `∘` and `|>`:**

`a ∘ b` creates `Pipe(b, a)` (inner=b, outer=a), so `a` lives at `.right` and `b` at `.left`. The walker reflects this:
- `a ∘ b`: `a → [:right]`, `b → [:left]`
- `a |> b`: `a → [:left]`, `b → [:right]`

---

### `@constrain` macro pipeline

`@constrain` generates runtime code that resolves component names against the registry, builds constraint entries, and calls `_constrain`.

```
Source code                   Macro expansion (compile time)
─────────────────────────────────────────────────────────────

@constrain cm begin            1. Parse block: iterate over directives
  @fix   a.x = 1.0                Each directive produces one entry expression.
  @bound a.y in (0, Inf)
  @tie   b.z = a.x * 2.0      2. For each path (e.g. a.x):
end                                 _extract_path(:(a.x))  →  (:a, :x)
                                    optic_expr = :(_resolve_path(cm_var, (:a, :x)))

                              3. For @tie b.z = a.x * 2.0:
                                 _extract_and_replace_masters(:(a.x * 2.0))
                                   → replaced_rhs:  :(m_gensym1 * 2.0)
                                   → masters: [(:a,:x) => :m_gensym1]
                                 closure:  (m_gensym1,) -> m_gensym1 * 2.0
                                 master_optics: [_resolve_path(cm_var, (:a,:x))]
                                 entry:  (optic_for_b_z, Tied(closure, (master_optic,)))

                              4. Emit:
                                 let cm_var = _as_compiled(cm)
                                   _constrain(cm_var,
                                     (optic_for_a_x, Fixed(1.0)),
                                     (optic_for_a_y, Bounded(0, Inf)),
                                     (optic_for_b_z, Tied(closure, (master_optic,))))
                                 end

Runtime (_resolve_path)
───────────────────────

_resolve_path(cm, (:a, :x)):
  1. Look up :a in cm.names  → entry (optic or Registry)
  2. If Registry: compose registry.optic with descent into registry.names for rest of path
  3. If optic + remaining path symbols: compose with PropertyLens for each

Runtime (_constrain)
────────────────────

_constrain(cm, entries, prior_entries):
  1. Validate targets: each target optic must point to a scalar, not a subtree.
  2. _dedupe_last(entries): within the new block, last entry per target wins.
  3. Merge with existing spec:
       merged = (filter old entries not overridden by new) ++ new entries
  4. _validate_spec(merged):
       V1: no tie chains (a master cannot itself be Tied)
       V2: no self-ties (target is its own master)
       V3: sane bounds (lo < hi)
  5. model = _apply_fixed(model, new entries)
       Writes Fixed values into the tree (only entries from THIS block).
       Older Fixed entries are already baked into .model.
  6. _check_bounds(model, merged):  V4: current values inside bounds
  7. _compiled(model, merged, names, priors)  ← I1 re-established
```

---

### Parameter engine

The parameter engine is the hot path called inside every iteration of the fitting loop. Its goal: scatter a `Vector{Float64}` (or `Vector{Dual}`) into the model tree with zero dynamic dispatch and zero heap allocation beyond the new model struct.

Three `@generated` functions make this possible.

#### `free_lenses` — compile-time parameter selection

```julia
@generated function free_lenses(spec::Tuple)
    idx = [k for (k, T) in enumerate(spec.parameters)
           if _constraint_type(T) <: Union{Free, Bounded}]
    Expr(:tuple, (:(spec[$k][1]) for k in idx)...)
end
```

`spec` has type `Tuple{Tuple{Optic1,Free}, Tuple{Optic2,Fixed{Float64}}, Tuple{Optic3,Bounded{Float64}}, ...}`. The `@generated` body runs at *compile time* (when specializing on the spec type) and inspects `spec.parameters` — the type-level description of each tuple element.

For a spec where entries 1 and 3 are `Free`/`Bounded` and entry 2 is `Fixed`, the emitted code is literally:

```julia
(spec[1][1], spec[3][1])
```

A constant-structure tuple expression. No loops, no runtime type checks. Calling `free_lenses` on a fixed spec type is as cheap as reading two fields.

#### `_set` — straight-line lens application

```julia
@generated function _set(obj, lens, v)
    prims = try
        _primitive_lenses(lens)
    catch
        nothing
    end
    prims === nothing && return :(set(obj, lens, v))
    _set_expr(:obj, :lens, :v, prims)
end
```

`_primitive_lenses` decomposes a `ComposedFunction{O,I}` (an Accessors composed lens) into a flat vector of primitive lenses at compile time:

```
PropertyLens{:a}() ⨟ PropertyLens{:b}() ⨟ PropertyLens{:c}()
→ [PropertyLens{:a}(), PropertyLens{:b}(), PropertyLens{:c}()]
```

`_set_expr` then emits a straight-line get-down / set-up block. For a 3-level lens:

```julia
begin
    @inline
    o1 = obj             # root
    o2 = lens_a(o1)      # descend with lens a
    o3 = lens_b(o2)      # descend with lens b
    s1 = set(o3, lens_c, v)   # set the leaf
    s2 = set(o2, lens_b, s1)  # rebuild level 2
    s3 = set(o1, lens_a, s2)  # rebuild level 1
    s3
end
```

**Why not use recursive `Accessors.set` directly?** The standard `Accessors.set` on a `ComposedFunction` is recursive. Julia's type inference has a recursion limit (roughly 100 frames by default). For deeply nested models — or when the value type changes (e.g., `Float64` → `Dual{Tag,Float64,N}` during ForwardDiff differentiation) — the recursive form hits this limit, falls back to dynamic dispatch, and allocates on every call.

The straight-line form produced by `_set_expr` has a bounded, constant depth from the compiler's perspective. Each individual `set(o, primitive_lens, v)` call is a simple struct update, fully inferable regardless of value type.

#### `_scatter` — unrolled parameter write

```julia
@generated function _scatter(model, lenses::Tuple, vals)
    N = length(lenses.parameters)
    out  = Expr(:block, Expr(:meta, :inline), :(m0 = model))
    prev = :m0
    for k in 1:N
        mk = Symbol(:m, k)
        push!(out.args, :($mk = _set($prev, lenses[$k], @inbounds(vals[$k]))))
        prev = mk
    end
    push!(out.args, prev)
    out
end
```

At compile time, `N` = the number of free parameters (known from the lenses tuple type). The emitted code for N=3:

```julia
begin
    @inline
    m0 = model
    m1 = _set(m0, lenses[1], @inbounds(vals[1]))
    m2 = _set(m1, lenses[2], @inbounds(vals[2]))
    m3 = _set(m2, lenses[3], @inbounds(vals[3]))
    m3
end
```

Each `_set` call changes the model's type (because it creates a new immutable struct with a potentially different field type). The sequential unrolling allows the compiler to track the type chain `m0 → m1 → m2 → m3` without inference ambiguity.

#### `withparams` — the full hot path

```julia
withparams(cm::CompiledModel, p) =
    _compiled(
        _scatter(getfield(cm, :model), free_lenses(getfield(cm, :spec)), p),
        getfield(cm, :spec), getfield(cm, :names), getfield(cm, :priors)
    )
```

Call chain for a model with 3 free parameters:

```
withparams(cm, p)
│
├─ free_lenses(cm.spec)          [compile-time: emits (spec[1][1], spec[3][1])]
│    → (optic_1, optic_3)        [runtime: field reads, O(1)]
│
├─ _scatter(cm.model, lenses, p) [compile-time: unrolls 3 _set calls]
│    m0 = cm.model
│    m1 = _set(m0, optic_1, p[1])   [straight-line, no dispatch]
│    m2 = _set(m1, optic_3, p[2])
│    → m2                            [updated model tree, no heap alloc]
│
└─ _compiled(m2, cm.spec, cm.names, cm.priors)
     └─ resolve(m2, cm.spec)    [re-evaluates all Tied parameters]
          → CompiledModel(m2_resolved, cm.spec, cm.priors, cm.names)
```

The only allocations are the new model structs (unavoidable for immutable value semantics) and the final `CompiledModel` wrapper. No intermediate arrays or closures are created.

---

### Optic composition helpers

Two small utilities are used throughout to manage optic equality and composition:

```julia
_compose(::typeof(identity), ::typeof(identity)) = identity
_compose(::typeof(identity), o) = o
_compose(o, ::typeof(identity)) = o
_compose(a, b) = a ⨟ b
```

`_compose` avoids wrapping optics in unnecessary `ComposedFunction` layers when one side is `identity`. This matters because `_primitive_lenses` must decompose lenses at compile time — a spurious `identity` wrapper would appear as a primitive and generate a no-op `set` call.

```julia
_optic_leaves(o::ComposedFunction) = (_optic_leaves(o.inner)..., _optic_leaves(o.outer)...)
_optic_leaves(::typeof(identity)) = ()
_optic_leaves(o) = (o,)
_same_optic(a, b) = _optic_leaves(a) == _optic_leaves(b)
```

`_same_optic` flattens both optics to their primitive sequence before comparing. This makes structural equality robust to composition associativity: `(a ⨟ b) ⨟ c` and `a ⨟ (b ⨟ c)` both flatten to `(a, b, c)` and compare equal.

`_same_optic` is used in `_constrain` to detect which existing spec entries are overridden by new ones, and in `_validate_spec` to detect self-ties and tie chains.

---

### ForwardDiff compatibility

AstroFit is fully compatible with ForwardDiff. The fitting loop can differentiate through `withparams` and `render` with no extra effort.

ForwardDiff substitutes `Float64` with `Dual{Tag, Float64, N}` in the parameter vector. This changes the type of `p` (and subsequently of the model tree after `_scatter`). The generated code handles this transparently because:

1. `_set` produces straight-line code where each `set(o, primitive_lens, v)` is a struct constructor call. Julia can infer the output type of each call regardless of whether `v` is `Float64` or `Dual`.

2. `_scatter` sequentially chains `_set` calls. Since each output type is inferable, the compiler tracks `m0::Model{Float64}` → `m1::Model{Dual}` → ... without hitting inference limits.

3. `render` on leaf models is a scalar arithmetic expression (`m.amplitude * exp(...)` etc.) which ForwardDiff differentiates through automatically.

**When does recompilation occur?**

Julia specializes on types, not values. Recompilation happens when the *type* of an argument changes:

| Scenario | Recompiles? |
|----------|-------------|
| Same model, same optimizer, different parameter values | No |
| First call with ForwardDiff (Float64 → Dual) | Yes, once |
| Different chunk size `N` in `Dual{Tag,V,N}` | Yes, once per `N` |
| Different function type in ForwardDiff tag | Yes, once per function type |

The last point is the most common pitfall. ForwardDiff's tag includes the function type: `Tag{F, V}`. If `F` is an anonymous closure created freshly each call (e.g., `p -> loss(cm, p)` written inside a loop), its type changes each iteration, triggering recompilation every time. The fix is to use a named struct callable (shown in the [Fitting Loop](#fitting-loop) section).

Once compiled, subsequent calls with the same type combination hit the cache and pay no compilation overhead.
