# AstroFit.jl by example

A tour of the API, from a single model component to a constrained, fittable
emission-line spectrum. Every code block below is runnable as-is; later blocks
assume `using AstroFit` and build on the models defined earlier.

```julia
using AstroFit
```

## 1. Model components

A *model* is an immutable struct that maps a coordinate to a value through
`render`. The built-in 1-D components:

```julia
g = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.5)   # keyword form
g = Gaussian1D(2.0, 0.0, 1.5)                              # positional form
c = Const1D(3.0)                                           # flat: value
l = Linear1D(slope = 2.0, intercept = 1.0)                # slope·x + intercept
```

`render` evaluates a component — at a point, or broadcast over a grid:

```julia
render(g, 0.0)                 # 2.0   (the peak)
render(l, 4.0)                 # 9.0
render(g, [-1.5, 0.0, 1.5])    # 3-element vector
```

## 2. Composing components

Components combine with ordinary operators into bigger models — no macro needed:

| Operator | Node | `render` |
|----------|------|----------|
| `a + b`  | sum        | `a(x) + b(x)` |
| `a - b`  | difference | `a(x) - b(x)` |
| `a * b`  | product    | `a(x) * b(x)` |
| `a / b`  | quotient   | `a(x) / b(x)` |
| `a \|> b` | pipe       | `b(a(x))` — apply `a`, then `b` |
| `a ∘ b`  | pipe       | `a(b(x))` — math composition (reverse of `\|>`) |

```julia
# An emission line on a flat continuum:
m = Const1D(0.5) + Gaussian1D(1.0, 0.0, 1.0)
render(m, 0.0)        # 1.5

# An absorption line (continuum minus a Gaussian):
absorption = Const1D(1.0) - Gaussian1D(0.4, 0.0, 1.0)
render(absorption, 0.0)   # 0.6

# Pipe: remap the coordinate, then evaluate a profile (here: centre a line at 6563):
profile = Linear1D(1.0, -6563.0) |> Gaussian1D(1.0, 0.0, 2.0)
render(profile, 6563.0)   # 1.0
```

## 3. `@model`: named, compiled models

`@model` takes a `begin … end` block of `name = component` bindings plus a final
composition. It returns a `CompiledModel`: the same tree, but with each named leaf
ready to carry constraints. Every parameter starts **free**.

```julia
spec = @model begin
    cont  = Linear1D(0.0, 1.0)             # local continuum
    ha    = Gaussian1D(5.0, 6563.0, 2.0)   # Hα 6563 Å
    n6548 = Gaussian1D(1.0, 6548.0, 2.0)   # [NII] 6548 Å
    n6583 = Gaussian1D(3.0, 6583.0, 2.0)   # [NII] 6583 Å
    cont + ha + n6548 + n6583
end
```

## 4. Navigating a compiled model

Each name is a property returning that leaf — resolved from the type, so it is
exact and allocation-free:

```julia
spec.ha                # the Leaf tagged :ha
spec.ha.model          # Gaussian1D(5.0, 6563.0, 2.0)
spec.ha.constraints    # (Free(), Free(), Free())  — positional to the model's fields
```

## 5. Inspecting parameters

The free parameters are everything not pinned by a constraint. These accessors
all enumerate them in the **same order**, so they line up slot-for-slot:

```julia
nfree(spec)        # 11   (Linear1D has 2 fields, each Gaussian1D has 3)
params(spec)       # the current free values — your optimizer's starting point p₀
paramnames(spec)   # [:cont_slope, :cont_intercept, :ha_amplitude, …]  — slot labels
bounds(spec)       # (lower, upper) vectors; (-Inf, Inf) where unbounded
```

`withparams` is the bridge from a flat parameter vector back to a concrete model
you can `render` — this is the hot path, called once per optimizer step:

```julia
λ = collect(6540.0:2.0:6590.0)
model = withparams(spec, params(spec))   # rebuild the tree with these values
flux  = render(model, λ)                 # evaluate on the grid
```

## 6. Constraints

Each leaf field carries one of four constraint kinds:

| Kind | Meaning | Free slot? |
|------|---------|-----------|
| `Free()`               | a free parameter                              | yes |
| `Bounded(lo, hi)`      | free, but confined to `[lo, hi]`              | yes |
| `Fixed(v)`             | pinned to a constant                          | no  |
| `Tied(paths, f)`       | `value = f(master₁, …)` of other free params  | no  |

You rarely write these by hand — the verbs in §7 do — but the shape is:

```julia
Free()
Bounded(0.0, 10.0)
Fixed(6563.0)
Tied(((:ha, :sigma),), identity)              # value = ha.sigma  (one master)
Tied(((:n6548, :amplitude),), x -> 2.96x)     # value = 2.96 · n6548.amplitude
```

A `Tied` references one or more **free** masters by `(leaf, field)` path and writes
one slot; its masters must themselves be free (`Free`/`Bounded`) — no chaining.

## 7. Constraint verbs (standalone)

`@fix`, `@bound`, `@free`, and `@tie` each edit one parameter and return a **new**
`CompiledModel` (the original is untouched — rebind to keep the result):

```julia
s = @fix   spec.ha.mean = 6563.0          # pin to a constant
s = @bound s.ha.amplitude 0 Inf           # confine to [0, ∞)  (emission ⇒ ≥ 0)
s = @free  s.cont.slope                   # release back to free
s = @tie   s.n6583.amplitude = 2.96 * s.n6548.amplitude   # fixed line ratio
```

`@tie` reads its master parameters from the right-hand side; any other value in
the expression (like the literal `2.96`) is an ordinary coefficient. Note: a
coefficient that is a *variable* is captured **live**, not frozen — use a literal
or a `const`, since a reassigned captured variable both tracks later changes and
deoptimizes the hot path.

## 8. `@constrain`: a block of edits

For more than one or two edits, `@constrain cm begin … end` is the ergonomic form:
leaf names are bare (the model is implicit), edits are threaded for you, a
parameter constrained twice is a **compile-time error**, and the whole model is
validated at the end (so a tie broken by a later edit is caught):

```julia
fit = @constrain spec begin
    @fix   ha.mean    = 6563.0            # known rest wavelengths
    @fix   n6548.mean = 6548.0
    @fix   n6583.mean = 6583.0
    @bound ha.amplitude    0 Inf          # emission lines are non-negative
    @bound n6548.amplitude 0 Inf
    @tie   n6548.sigma     = ha.sigma     # all lines share one velocity width
    @tie   n6583.sigma     = ha.sigma
    @tie   n6583.amplitude = 2.96 * n6548.amplitude   # fixed [NII] atomic ratio
end

nfree(fit)        # 5
paramnames(fit)   # [:cont_slope, :cont_intercept, :ha_amplitude, :ha_sigma, :n6548_amplitude]
```

The doublet ratio and shared width now hold automatically in the rebuilt model:

```julia
m = withparams(fit, params(fit))
m.right.amplitude ≈ 2.96 * m.left.right.amplitude   # [NII] 6583 = 2.96 × 6548
```

## 9. Fitting: the bridge to an optimizer

AstroFit doesn't impose an optimizer — your objective is just a function of
`withparams` + `render`, and `params`/`bounds`/`paramnames` give you everything an
optimizer needs. Least-squares against data `(λ, y)`:

```julia
y = render(withparams(fit, params(fit)), λ)          # synthetic "observed" spectrum
loss(p) = sum(abs2, render(withparams(fit, p), λ) .- y)

loss(params(fit))     # 0.0 at the truth
```

The objective is fully differentiable, so gradient-based optimizers and AD work
out of the box:

```julia
using ForwardDiff
ForwardDiff.gradient(loss, params(fit))   # 5-element gradient
```

Hand it to the optimizer of your choice — `bounds(fit)` supplies the box, and
`paramnames(fit)` labels the result. For example, with Optim.jl added to your
environment:

```julia
# using Optim
# lo, hi = bounds(fit)
# res = optimize(loss, lo, hi, params(fit), Fminbox(LBFGS()))
# best = withparams(fit, Optim.minimizer(res))
```

## 10. The low-level engine

The verbs lower to one function, useful directly when you build constraints
programmatically:

```julia
s = setconstraint(spec, :ha, :sigma, Bounded(0.5, 10.0))     # leaf, field, constraint
s = setconstraint(s, :n6548, :sigma, Tied(((:ha, :sigma),), identity))
```

`validate` checks every tie points at a free master, returning the model so it
chains — and throwing a clear error otherwise:

```julia
validate(s)        # returns s

bad = setconstraint(s, :ha, :sigma, Fixed(2.0))   # ha.sigma is now fixed…
validate(bad)      # ArgumentError: tie on `n6548.sigma` references `ha.sigma`,
                   #                which is not a free parameter
```
