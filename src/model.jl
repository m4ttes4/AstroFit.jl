abstract type AbstractModel end

Base.broadcastable(m::AbstractModel) = (m,)

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
