struct Sum{L<:AbstractModel, R<:AbstractModel} <: AbstractModel
    left::L
    right::R
end

render(m::Sum, x::Number...) = render(m.left, x...) + render(m.right, x...)

Base.:+(a::AbstractModel, b::AbstractModel) = Sum(a, b)
