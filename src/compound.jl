struct Sum{L <: AbstractModel, R <: AbstractModel} <: AbstractModel
    left::L
    right::R
end

struct Difference{L <: AbstractModel, R <: AbstractModel} <: AbstractModel
    left::L
    right::R
end

struct Product{L <: AbstractModel, R <: AbstractModel} <: AbstractModel
    left::L
    right::R
end

struct Quotient{L <: AbstractModel, R <: AbstractModel} <: AbstractModel
    left::L
    right::R
end

struct Pipe{L <: AbstractModel, R <: AbstractModel} <: AbstractModel
    left::L
    right::R
end

# NOTE if not inlined performance are not good because of tree recursion
@inline render(m::Sum, x::Number...) = render(m.left, x...) + render(m.right, x...)
@inline render(m::Difference, x::Number...) = render(m.left, x...) - render(m.right, x...)
@inline render(m::Product, x::Number...) = render(m.left, x...) * render(m.right, x...)
@inline render(m::Quotient, x::Number...) = render(m.left, x...) / render(m.right, x...)
@inline render(m::Pipe, x::Number...) = render(m.right, render(m.left, x...))

const _COMPOUND = Union{Sum, Difference, Product, Quotient, Pipe}

# A compound node is pointwise only if both sides are: one kernel anywhere makes
# the whole node domainwise. Dispatches on the type, so it folds at compile time.
evalstyle(::Type{N}) where {N <: _COMPOUND} =
    _combine(evalstyle(N.parameters[1]), evalstyle(N.parameters[2]))

# Structural recursion, entered only when the node is domainwise. Each side goes
# back through `render`, so a pointwise subtree renders as ONE fused broadcast
# and fusion breaks only where a kernel actually sits.
@inline _arender(m::Sum, xs::AbstractArray...) = render(m.left, xs...) .+ render(m.right, xs...)
@inline _arender(m::Difference, xs::AbstractArray...) = render(m.left, xs...) .- render(m.right, xs...)
@inline _arender(m::Product, xs::AbstractArray...) = render(m.left, xs...) .* render(m.right, xs...)
@inline _arender(m::Quotient, xs::AbstractArray...) = render(m.left, xs...) ./ render(m.right, xs...)

# Pipe keeps its scalar meaning — feed the left output into the right — with the
# whole array as the unit. When the right side is a kernel this is convolution;
# when it is pointwise it is the usual broadcast composition.
@inline _arender(m::Pipe, xs::AbstractArray...) = render(m.right, render(m.left, xs...))

Base.:+(a::AbstractModel, b::AbstractModel) = Sum(a, b)
Base.:-(a::AbstractModel, b::AbstractModel) = Difference(a, b)
Base.:*(a::AbstractModel, b::AbstractModel) = Product(a, b)
Base.:/(a::AbstractModel, b::AbstractModel) = Quotient(a, b)
Base.:∘(a::AbstractModel, b::AbstractModel) = Pipe(b, a)
Base.:|>(a::AbstractModel, b::AbstractModel) = Pipe(a, b)
