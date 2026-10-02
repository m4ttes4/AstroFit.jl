"""
    evaluate(m, x::Number)
    evaluate(m, p::NTuple{I, Number})

The method a model author implements: the output of `m` at one point, a `Number`
for a 1-input model and an `NTuple{I, Number}` otherwise. It returns a `Number`
(`O = 1`) or an `NTuple{O, Number}`. Not exported: extend it as
`AstroFit.evaluate(m::MyModel, x::Number) = …`. Users call [`render`](@ref).
"""
function evaluate end

"""
    render(m, x)
    render(m, A::AbstractArray)

Evaluate model `m` at one point `x` (a `Number` for a 1-input model, an
`NTuple{I, Number}` or a `CartesianIndex{I}` otherwise), or at every point of an
array: a vector of numbers (1D), `CartesianIndices(img)`, `Coords(x, y, …)`, or any
array of points. `out .= render.(m, A)` is the in-place form.
"""
@inline function render(m::AbstractModel{1, O}, x::Number) where {O}
    y = evaluate(m, x)
    y isa (O == 1 ? Number : NTuple{O, Number}) ||
        throw(ArgumentError("$(_label(m)) declares $O output(s) per point, but evaluate returned a $(typeof(y))"))
    return y
end
@inline function render(m::AbstractModel{N, O}, p::NTuple{N, Number}) where {N, O}
    y = evaluate(m, p)
    y isa (O == 1 ? Number : NTuple{O, Number}) ||
        throw(ArgumentError("$(_label(m)) declares $O output(s) per point, but evaluate returned a $(typeof(y))"))
    return y
end
render(m::AbstractModel{1}, p::Tuple{Number}) =
    throw(ArgumentError("$(_label(m)) takes 1 number per point: pass the number, not a 1-tuple"))
@inline render(m::AbstractModel{1}, i::CartesianIndex{1}) = render(m, i[1])
@inline render(m::AbstractModel{N}, i::CartesianIndex{N}) where {N} = render(m, Tuple(i))

# Arrays of points: one method per point type, the array's N tied to the model's I.
render(m::AbstractModel{1}, v::AbstractVector{<:Number}) = render.(m, v)
render(m::AbstractModel{N}, A::AbstractArray{CartesianIndex{N}}) where {N} = render.(m, A)
render(m::AbstractModel{N}, A::AbstractArray{<:NTuple{N, Number}}) where {N} = render.(m, A)

# Wrong inputs, from narrow to broad; each strictly less specific than the valid methods.
render(m::AbstractModel{I}, p) where {I} =
    throw(ArgumentError("$(_label(m)) takes $I number(s) per point, got $(typeof(p))"))
render(m::AbstractModel{I}, v::AbstractVector{<:Number}) where {I} =
    throw(ArgumentError("$(_label(m)) takes $I numbers per point, got a vector of numbers: pass Coords(x, y, …), CartesianIndices, or a vector of $I-tuples"))
render(m::AbstractModel, A::AbstractArray{<:Number}) =
    throw(ArgumentError("an array of numbers with $(ndims(A)) dimension(s) is not a set of points: use CartesianIndices(A) for its pixels, or Coords(x, y, …) for physical axes"))
render(m::AbstractModel{I}, A::AbstractArray) where {I} =
    throw(ArgumentError("$(_label(m)) takes $I number(s) per point, got an array of $(eltype(A))"))

# Compound nodes forward to render on their children, so every child is checked.
# `p` is untyped on purpose: render already accepted it as a point.
@inline evaluate(m::Sum, p) = render(m.left, p) + render(m.right, p)
@inline evaluate(m::Difference, p) = render(m.left, p) - render(m.right, p)
@inline evaluate(m::Product, p) = render(m.left, p) * render(m.right, p)
@inline evaluate(m::Quotient, p) = render(m.left, p) / render(m.right, p)
@inline evaluate(m::Pipe, p) = render(m.right, render(m.left, p))
@inline evaluate(l::Leaf, p) = render(l.model, p)

@inline render(cm::CompiledModel, x) = render(getfield(cm, :tree), x)  # forwarding: x untyped on purpose
Base.broadcastable(cm::CompiledModel) = Ref(cm)
