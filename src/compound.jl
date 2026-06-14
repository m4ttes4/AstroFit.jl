struct Sum{N, L<:AbstractModel{N}, R<:AbstractModel{N}} <: AbstractModel{N}
    left::L
    right::R
end

struct Difference{N, L<:AbstractModel{N}, R<:AbstractModel{N}} <: AbstractModel{N}
    left::L
    right::R
end

struct Product{N, L<:AbstractModel{N}, R<:AbstractModel{N}} <: AbstractModel{N}
    left::L
    right::R
end

struct Quotient{N, L<:AbstractModel{N}, R<:AbstractModel{N}} <: AbstractModel{N}
    left::L
    right::R
end

# inner: N-dim input → scalar; outer: scalar → scalar
struct Pipe{N, L<:AbstractModel{N}, R<:AbstractModel{1}} <: AbstractModel{N}
    left::L
    right::R
end

render(m::Sum, x::Number...)        = render(m.left, x...) + render(m.right, x...)
render(m::Difference, x::Number...) = render(m.left, x...) - render(m.right, x...)
render(m::Product, x::Number...)    = render(m.left, x...) * render(m.right, x...)
render(m::Quotient, x::Number...)   = render(m.left, x...) / render(m.right, x...)
render(m::Pipe, x::Number...)       = render(m.right, render(m.left, x...))

Base.:+(a::AbstractModel{N}, b::AbstractModel{N}) where {N} = Sum(a, b)
Base.:-(a::AbstractModel{N}, b::AbstractModel{N}) where {N} = Difference(a, b)
Base.:*(a::AbstractModel{N}, b::AbstractModel{N}) where {N} = Product(a, b)
Base.:/(a::AbstractModel{N}, b::AbstractModel{N}) where {N} = Quotient(a, b)

# a ∘ b  →  a(b(x)):  a is outer (must be 1D), b is inner
Base.:∘(a::AbstractModel{1}, b::AbstractModel{N}) where {N} = Pipe(b, a)

# a |> b  →  b(a(x)):  b is outer (must be 1D), a is inner
Base.:|>(a::AbstractModel{N}, b::AbstractModel{1}) where {N} = Pipe(a, b)

# Model algebra is model⊗model: bare scalars are an informative error, never
# silent — use a named Const1D so the constant stays fittable and addressable.
const _SCALAR_HINT = "model algebra is model⊗model: no bare scalars. " *
                     "Use a named Const1D (e.g. `c = Const1D(value=2.0)`) " *
                     "so the constant stays fittable, constrainable and addressable."

for op in (:+, :-, :*, :/)
    @eval Base.$op(::Number, ::AbstractModel) = throw(ArgumentError(_SCALAR_HINT))
    @eval Base.$op(::AbstractModel, ::Number) = throw(ArgumentError(_SCALAR_HINT))
end
Base.:-(::AbstractModel) = throw(ArgumentError(_SCALAR_HINT))
