"""
    AbstractModel{I, O}

Supertype of every model: `I` numbers in and `O` numbers out per point. To define a
model, subtype it and extend `AstroFit.evaluate` (not exported):

```julia
struct ExpDisk2D{A <: Real, X <: Real, Y <: Real, H <: Real} <: AbstractModel{2, 1}
    amplitude::A
    x0::X
    y0::Y
    h::H
end
AstroFit.evaluate(m::ExpDisk2D, (x, y)::NTuple{2, Number}) =
    m.amplitude * exp(-hypot(x - m.x0, y - m.y0) / m.h)
```

- A point is a `Number` when `I = 1` and an `NTuple{I, Number}` otherwise; `evaluate`
  returns a `Number` when `O = 1` and an `NTuple{O, Number}` otherwise. [`render`](@ref)
  checks both and names the model when either is wrong.
- Give each fittable field its own type parameter `<: Real`, so a fixed field can sit
  beside a ForwardDiff dual (ADR-0005).
- Constants that depend only on the parameters go in a last field named `_cache_`,
  computed by an inner positional constructor that takes the other fields in order.
  `withparams` calls that constructor; `_cache_` gets no slot and no constraint.
- Models compose by signature: `+`, `-`, `*`, `/` join two `I => 1` models, and
  `a |> b` feeds the `M` outputs of an `I => M` model to an `M => O` one.
"""
abstract type AbstractModel{I, O} end

# A model broadcasts as a scalar. `Ref`, not a 1-tuple: a tuple would broadcast a
# 0-dimensional input to a 1-element vector.
Base.broadcastable(m::AbstractModel) = Ref(m)
