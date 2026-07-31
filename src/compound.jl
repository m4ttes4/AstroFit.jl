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

const _COMPOUND = Union{Sum, Difference, Product, Quotient, Pipe}

Base.:+(a::AbstractModel, b::AbstractModel) = Sum(a, b)
Base.:-(a::AbstractModel, b::AbstractModel) = Difference(a, b)
Base.:*(a::AbstractModel, b::AbstractModel) = Product(a, b)
Base.:/(a::AbstractModel, b::AbstractModel) = Quotient(a, b)
Base.:∘(a::AbstractModel, b::AbstractModel) = Pipe(b, a)
Base.:|>(a::AbstractModel, b::AbstractModel) = Pipe(a, b)
