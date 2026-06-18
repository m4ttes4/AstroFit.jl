struct Sum{L<:AbstractModel, R<:AbstractModel} <: AbstractModel
    left::L
    right::R
end

struct Difference{L<:AbstractModel, R<:AbstractModel} <: AbstractModel
    left::L
    right::R
end

struct Product{L<:AbstractModel, R<:AbstractModel} <: AbstractModel
    left::L
    right::R
end

struct Quotient{L<:AbstractModel, R<:AbstractModel} <: AbstractModel
    left::L
    right::R
end

struct Pipe{L<:AbstractModel, R<:AbstractModel} <: AbstractModel
    left::L
    right::R
end

render(m::Sum, x::Number...)        = render(m.left, x...) + render(m.right, x...)
render(m::Difference, x::Number...) = render(m.left, x...) - render(m.right, x...)
render(m::Product, x::Number...)    = render(m.left, x...) * render(m.right, x...)
render(m::Quotient, x::Number...)   = render(m.left, x...) / render(m.right, x...)
render(m::Pipe, x::Number...)       = render(m.right, render(m.left, x...))

Base.:+(a::AbstractModel, b::AbstractModel) = Sum(a, b)
Base.:-(a::AbstractModel, b::AbstractModel) = Difference(a, b)
Base.:*(a::AbstractModel, b::AbstractModel) = Product(a, b)
Base.:/(a::AbstractModel, b::AbstractModel) = Quotient(a, b)
Base.:∘(a::AbstractModel, b::AbstractModel) = Pipe(b, a)
Base.:|>(a::AbstractModel, b::AbstractModel) = Pipe(a, b)
