abstract type AbstractModel end

Base.broadcastable(m::AbstractModel) = (m,)

# ---------------------------------------------------------------------------
# The field contract
# ---------------------------------------------------------------------------

"""
    AstroFit._isparamfield(F) -> Bool

Whether a field of declared type `F` is a **parameter field**.

This is the contract AstroFit makes with a model author, and it is decided by
the field's type alone:

- **Floating-point fields are parameter fields.** They may be fitted, they get a
  slot in the flat parameter vector when `Free`, and — the part that constrains
  the implementation — they must be able to hold a dual number, because that is
  how ForwardDiff propagates derivatives through `withparams`. Fields of a model
  are promoted to a common type on reconstruction so that a dual in one slot
  lifts the others; declare them with the model's type parameter (`::T` where
  `T <: Real`), never as a concrete `Float64`.
- **Everything else is an internal value.** Integers, `Bool`s, `Symbol`s,
  arrays, strings, functions, tuples: carried through reconstruction untouched
  and never fitted. A measured PSF's sampled kernel, a half-width in samples, an
  edge policy, a normalization flag — all internal.

`Bool` and `Int` are `Number`s in Julia but are *not* parameter fields here.
That is deliberate: a gradient-based optimizer cannot perturb a discrete value,
and an `Int` field cannot hold a dual number, so treating them as parameters
could only fail. It also means the natural way to write a structural field —
`halfwidth::Int`, `normalize::Bool` — is the correct one.

Dual numbers are `Real` but not `AbstractFloat`, so the predicate is written to
include them: re-parameterizing a model that already carries duals still works.

See also: [`withparams`](@ref), [`AbstractKernel`](@ref)
"""
@inline _isparamfield(::Type{F}) where {F} = F <: Real && !(F <: Integer)

# ---------------------------------------------------------------------------
# Evaluation style — how a model turns coordinates into values.
#
# Pointwise:  defines render(m, x::Number...); arrays come from broadcasting it,
#             so a whole pointwise subtree fuses into ONE traversal.
# Domainwise: defines render(m, xs::AbstractArray) directly, because the value at
#             one point depends on neighbouring values (a PSF convolution).
#
# The trait is computed from the model TYPE, so it folds to a compile-time
# constant: a kernel-free tree takes exactly the broadcast path it took before
# kernels existed, with no runtime test.
# ---------------------------------------------------------------------------

struct Pointwise end
struct Domainwise end

"""
    evalstyle(m) -> Pointwise() | Domainwise()

How `m` is evaluated over an array of coordinates. Pointwise models are
broadcast (and fuse with their pointwise neighbours); domainwise models — see
[`AbstractKernel`](@ref) — are handed the whole array at once.

Compound nodes combine their children's styles: a node is pointwise only if both
sides are.
"""
evalstyle(m) = evalstyle(typeof(m))
evalstyle(::Type{<:AbstractModel}) = Pointwise()

@inline _combine(::Pointwise, ::Pointwise) = Pointwise()
@inline _combine(_, _) = Domainwise()

# Array render: one entry point, style-dispatched.
render(m::AbstractModel, xs::AbstractArray...) = _render(evalstyle(m), m, xs...)

# Pointwise — the pre-kernel path, unchanged on purpose: this is the fused
# single-pass broadcast the ≤1.0x-vs-handwritten benchmark rests on.
@inline _render(::Pointwise, m, xs::AbstractArray...) = render.(m, xs...)

# Domainwise — structural recursion (_arender, compound.jl). Pointwise subtrees
# inside still take the fused path above, so fusion breaks only at kernels.
@inline _render(::Domainwise, m, xs::AbstractArray...) = _arender(m, xs...)

function render!(out::AbstractArray, m::AbstractModel, xs...)
    return _render!(evalstyle(m), out, m, xs...)
end
_render!(::Pointwise, out, m, xs...) = (out .= render.(m, xs...); out)
_render!(::Domainwise, out, m, xs...) = (out .= render(m, xs...); out)
