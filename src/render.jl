struct Pointwise end
struct Domainwise end

"""
    evalstyle(m) -> Pointwise() | Domainwise()

Declare how a model consumes array inputs. The default, `Pointwise()`, broadcasts
the model's scalar `render(m, x::Number...)` method. An array-native model defines
`render(m, xs::AbstractArray...)` and opts in with
`AstroFit.evalstyle(::Type{<:MyModel}) = Domainwise()`.

[`AbstractKernel`](@ref) already declares `Domainwise()`. Its single array is
upstream intensities, and its output must preserve the input axes. Other
array-native models receive coordinates at the root, or values inside a pipe.

Wrappers preserve the style; a compound is pointwise only if both children are.
"""
evalstyle(m) = evalstyle(typeof(m))
evalstyle(::Type{<:AbstractModel}) = Pointwise()
evalstyle(::Type{<:AbstractKernel}) = Domainwise()
@inline _combine(::Pointwise, ::Pointwise) = Pointwise()
@inline _combine(_, _) = Domainwise()
evalstyle(::Type{N}) where {N <: _COMPOUND} =
    _combine(evalstyle(fieldtype(N, :left)), evalstyle(fieldtype(N, :right)))
evalstyle(::Type{<:Leaf{name, I, O, M}}) where {name, I, O, M} = evalstyle(M)
evalstyle(::Type{CompiledModel{T, P}}) where {T, P} = evalstyle(T)

# `template` is true for external coordinates and false for upstream values.
# Only external lone matrices can invoke the legacy index-grid shorthand.
# The flag is constant through each traversal and is not an authoring protocol.
@inline _eval(m, xs...) = _evaluate(true, m, xs...)
@inline _evaluate(template::Bool, m, xs...) = _evaluate(evalstyle(m), template, m, xs...)
@inline _evaluate(template::Bool, l::Leaf, xs...) = _evaluate(template, l.model, xs...)
@inline _evaluate(template::Bool, cm::CompiledModel, xs...) = _evaluate(template, getfield(cm, :tree), xs...)

@inline _broadcast(m, xs...) = Base.Broadcast.instantiate(Base.Broadcast.broadcasted(render, (m,), xs...))
@inline _gridaxes(a::AbstractMatrix) = (axes(a, 1), reshape(axes(a, 2), 1, :))
@inline _evaluate(::Pointwise, ::Bool, m, xs...) = _broadcast(m, xs...)
@inline _evaluate(::Pointwise, template::Bool, m, image::AbstractMatrix) =
    template ? _broadcast(m, _gridaxes(image)...) : _broadcast(m, image)

# Concrete array methods are the domainwise primitive. The public fallback below
# throws for a domainwise leaf, so an unsupported signature cannot recurse here.
@inline _evaluate(::Domainwise, ::Bool, m, xs...) = render(m, map(Base.Broadcast.materialize, xs)...)
@inline function _evaluate(::Domainwise, ::Bool, k::AbstractKernel, xs...)
    inputs = map(Base.Broadcast.materialize, xs)
    length(inputs) == 1 && inputs[1] isa AbstractArray || throw(
        ArgumentError("$(nameof(typeof(k))) requires one intensity array")
    )
    values = render(k, inputs...)
    values isa AbstractArray && axes(values) == axes(inputs[1]) || throw(
        DimensionMismatch("$(nameof(typeof(k))) must return an array with the same axes as its input")
    )
    return values
end

@inline _lazyop(op, l, r) = Base.Broadcast.instantiate(Base.Broadcast.broadcasted(op, l, r))
@inline _evaluate(::Domainwise, t::Bool, m::Sum, xs...) =
    _lazyop(+, _evaluate(t, m.left, xs...), _evaluate(t, m.right, xs...))
@inline _evaluate(::Domainwise, t::Bool, m::Difference, xs...) =
    _lazyop(-, _evaluate(t, m.left, xs...), _evaluate(t, m.right, xs...))
@inline _evaluate(::Domainwise, t::Bool, m::Product, xs...) =
    _lazyop(*, _evaluate(t, m.left, xs...), _evaluate(t, m.right, xs...))
@inline _evaluate(::Domainwise, t::Bool, m::Quotient, xs...) =
    _lazyop(/, _evaluate(t, m.left, xs...), _evaluate(t, m.right, xs...))
@inline _evaluate(::Domainwise, t::Bool, m::Pipe, xs...) =
    _evaluate(false, m.right, _evaluate(t, m.left, xs...))

# The generic allocating entry point is also the terminal fallback for missing
# domainwise primitives. Structural nodes and wrappers have their own entry.
const _RenderInput = Union{Number, AbstractArray}
@inline render(m::AbstractModel, xs::_RenderInput...) = _renderfallback(evalstyle(m), m, xs...)
@inline _renderfallback(::Pointwise, m, xs...) = Base.Broadcast.materialize(_eval(m, xs...))
function _renderfallback(::Domainwise, m, xs...)
    throw(ArgumentError("$(nameof(typeof(m))) has no array render method for $(map(typeof, xs))"))
end
# Missing scalar formulas must terminate instead of broadcasting themselves.
render(m::AbstractModel, xs::Number...) = throw(MethodError(render, (m, xs...)))
render(m::Union{_COMPOUND, Leaf}, xs::_RenderInput...) = Base.Broadcast.materialize(_eval(m, xs...))

@inline function _checkrenderaxes(out, expected)
    axes(out) == expected || throw(DimensionMismatch("render destination axes $(axes(out)) do not match prediction axes $expected"))
    return nothing
end
@inline function _copyrender!(out, values)
    _checkrenderaxes(out, axes(values))
    out .= values
    return out
end

"""
    render!(out, model, coordinates...)
    render!(out::AbstractMatrix, model)

Write the prediction into `out` and return it. Its axes must match the prediction
exactly, and its element type must hold the result (including dual numbers when
differentiating). Whole-array models may allocate intermediate arrays.

With no coordinates, a matrix destination supplies its own index grid. A lone
matrix coordinate is likewise a template for pointwise models, but is array data
for domainwise leaves. Values produced inside a pipe are never templates.
"""
render!(out::AbstractArray, m::AbstractModel, xs...) = _copyrender!(out, _eval(m, xs...))
render!(out::AbstractMatrix, m::AbstractModel) = render!(out, m, _gridaxes(out)...)
# Preserve the zoo's parameter-hoisted in-place methods through named leaves.
render!(out::AbstractArray, l::Leaf, x, xs...) = render!(out, l.model, x, xs...)

# Scalar recursion also provides the fused pointwise-subtree broadcast primitive.
@inline render(m::Sum, x::Number...) = render(m.left, x...) + render(m.right, x...)
@inline render(m::Difference, x::Number...) = render(m.left, x...) - render(m.right, x...)
@inline render(m::Product, x::Number...) = render(m.left, x...) * render(m.right, x...)
@inline render(m::Quotient, x::Number...) = render(m.left, x...) / render(m.right, x...)
@inline render(m::Pipe, x::Number...) = render(m.right, render(m.left, x...))
@inline render(l::Leaf, x::Number...) = render(l.model, x...)

render(cm::CompiledModel, x...) = render(getfield(cm, :tree), x...)
render!(out::AbstractArray, cm::CompiledModel, x...) = render!(out, getfield(cm, :tree), x...)
