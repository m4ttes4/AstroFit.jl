# Everything about turning a model tree into values: the evaluation-style trait, the
# lazy evaluator behind `render`/`render!`, and the structural recursion over the tree.
# The types themselves stay with their own files (model.jl, compound.jl, compiled.jl,
# kernel.jl); the per-model formulas stay in the zoo next to their structs.

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
evalstyle(::Type{<:AbstractKernel}) = Domainwise()

@inline _combine(::Pointwise, ::Pointwise) = Pointwise()
@inline _combine(_, _) = Domainwise()

# A compound node is pointwise only if both sides are: one kernel anywhere makes
# the whole node domainwise. Dispatches on the type, so it folds at compile time.
evalstyle(::Type{N}) where {N <: _COMPOUND} =
    _combine(evalstyle(N.parameters[1]), evalstyle(N.parameters[2]))

# A leaf is evaluated exactly like the model it wraps; the wrapper is transparent
# to the array path too.
evalstyle(::Type{Leaf{name, M, C}}) where {name, M, C} = evalstyle(M)

evalstyle(::Type{CompiledModel{T, P}}) where {T, P} = evalstyle(T)

# ---------------------------------------------------------------------------
# Rendering — one lazy evaluator behind both entry points.
#
# `_eval` returns something you can either materialize (`render`) or write into
# (`render!`), so there is no separate in-place plumbing:
#
#   Pointwise  → an un-materialized `Broadcasted`. The whole pointwise subtree
#                still fuses into ONE traversal — this is what the
#                ≤1.0x-vs-handwritten benchmark rests on — and `out .= bc` fills
#                a buffer without allocating.
#   Domainwise → whatever the structural recursion (`_arender`) built: an array
#                where a kernel forced one, still a `Broadcasted` for the
#                compound nodes above it. `materialize` is the identity on the
#                array case, so nothing is copied.
#
# `evalstyle` is computed from the TYPE, so the branch folds at compile time.
# ---------------------------------------------------------------------------

@inline _eval(m, xs...) = _eval(evalstyle(m), m, xs...)
@inline _eval(::Pointwise, m, xs...) =
    Base.Broadcast.instantiate(Base.Broadcast.broadcasted(render, (m,), xs...))
@inline _eval(::Domainwise, m, xs...) = _arender(m, xs...)

# A lone matrix means different things on the two branches, so the rule lives
# here rather than at the entry point. To a pointwise model it is a *template*:
# its index space IS the coordinate grid
# ([ADR-0006](docs/adr/0006-grid-form-coordinates.md)), and reshaping a range is
# lazy, so this costs no coordinate memory. To a domainwise one it is
# intensities, which `_arender` already routes to the kernel — including when the
# kernel sits behind a `Leaf`, where its own `render(k, ::AbstractMatrix)` is out
# of reach.
@inline _gridaxes(a::AbstractMatrix) = (axes(a, 1), reshape(axes(a, 2), 1, :))
@inline _eval(::Pointwise, m, image::AbstractMatrix) = _eval(Pointwise(), m, _gridaxes(image)...)

render(m::AbstractModel, xs::AbstractArray...) = Base.Broadcast.materialize(_eval(m, xs...))

render!(out::AbstractArray, m::AbstractModel, xs...) = (out .= _eval(m, xs...); out)
render!(out::AbstractMatrix, m::AbstractModel) = render!(out, m, _gridaxes(out)...)

# ---------------------------------------------------------------------------
# Scalar rendering down the tree — the pointwise path, one coordinate at a time.
# ---------------------------------------------------------------------------

# NOTE if not inlined performance are not good because of tree recursion
@inline render(m::Sum, x::Number...) = render(m.left, x...) + render(m.right, x...)
@inline render(m::Difference, x::Number...) = render(m.left, x...) - render(m.right, x...)
@inline render(m::Product, x::Number...) = render(m.left, x...) * render(m.right, x...)
@inline render(m::Quotient, x::Number...) = render(m.left, x...) / render(m.right, x...)
@inline render(m::Pipe, x::Number...) = render(m.right, render(m.left, x...))

@inline render(l::Leaf, x::Number...) = render(l.model, x...)

# Unwrap so a hand-written `render!` in the zoo is reachable through the tree.
# At least one coordinate is required on purpose: the no-coordinate call takes
# the grid from `out`, and that belongs to the generic method above — a
# `x...` here would only be ambiguous with it.
render!(out::AbstractArray, l::Leaf, x, xs...) = render!(out, l.model, x, xs...)

# ---------------------------------------------------------------------------
# Array rendering down the tree — entered only when a node is domainwise.
# ---------------------------------------------------------------------------

# Each side goes back through `_eval`, and the node itself stays lazy: a pointwise
# subtree remains an un-materialized `Broadcasted`, so fusion breaks only where a
# kernel actually forces an array. Materializing here instead would allocate one
# array per node — the caller is the one that knows whether an array is wanted at
# all, and `chi2` never wants one.
@inline _lazyop(op, l, r) = Base.Broadcast.instantiate(Base.Broadcast.broadcasted(op, l, r))

@inline _arender(m::Sum, xs::AbstractArray...) = _lazyop(+, _eval(m.left, xs...), _eval(m.right, xs...))
@inline _arender(m::Difference, xs::AbstractArray...) = _lazyop(-, _eval(m.left, xs...), _eval(m.right, xs...))
@inline _arender(m::Product, xs::AbstractArray...) = _lazyop(*, _eval(m.left, xs...), _eval(m.right, xs...))
@inline _arender(m::Quotient, xs::AbstractArray...) = _lazyop(/, _eval(m.left, xs...), _eval(m.right, xs...))

# Pipe keeps its scalar meaning — feed the left output into the right — with the
# whole array as the unit. When the right side is a kernel this is convolution;
# when it is pointwise it is the usual broadcast composition.
@inline _arender(m::Pipe, xs::AbstractArray...) = render(m.right, render(m.left, xs...))

@inline _arender(l::Leaf, xs::AbstractArray...) = render(l.model, xs...)

# Reached only when a kernel has no render method for the array it was handed
# (e.g. a vector-only kernel applied to a matrix). Beats a bare MethodError on
# the internal _arender.
function _arender(k::AbstractKernel, xs::AbstractArray...)
    return throw(
        ArgumentError(
            "$(nameof(typeof(k))) has no render method for $(join(map(x -> string(typeof(x)), xs), ", ")) — " *
                "a kernel must define `render(k, ys::AbstractVector)` (intensities in, same-size array out)"
        )
    )
end

# ---------------------------------------------------------------------------
# CompiledModel — renders exactly like the tree it wraps.
# ---------------------------------------------------------------------------

render(cm::CompiledModel, x...) = render(getfield(cm, :tree), x...)
render!(out::AbstractArray, cm::CompiledModel, x...) = render!(out, getfield(cm, :tree), x...)

# The lazy path forwards too: `chi2` calls `_eval` on the CompiledModel directly,
# and on the domainwise branch that lands here rather than on `render`.
@inline _arender(cm::CompiledModel, xs::AbstractArray...) = _arender(getfield(cm, :tree), xs...)
