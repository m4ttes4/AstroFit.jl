struct Sum{I, L <: AbstractModel, R <: AbstractModel} <: AbstractModel{I, 1}
    left::L
    right::R
end
Sum(l::AbstractModel{I}, r::AbstractModel) where {I} = Sum{I, typeof(l), typeof(r)}(l, r)

struct Difference{I, L <: AbstractModel, R <: AbstractModel} <: AbstractModel{I, 1}
    left::L
    right::R
end
Difference(l::AbstractModel{I}, r::AbstractModel) where {I} = Difference{I, typeof(l), typeof(r)}(l, r)

struct Product{I, L <: AbstractModel, R <: AbstractModel} <: AbstractModel{I, 1}
    left::L
    right::R
end
Product(l::AbstractModel{I}, r::AbstractModel) where {I} = Product{I, typeof(l), typeof(r)}(l, r)

struct Quotient{I, L <: AbstractModel, R <: AbstractModel} <: AbstractModel{I, 1}
    left::L
    right::R
end
Quotient(l::AbstractModel{I}, r::AbstractModel) where {I} = Quotient{I, typeof(l), typeof(r)}(l, r)

struct Pipe{I, O, L <: AbstractModel, R <: AbstractModel} <: AbstractModel{I, O}
    left::L
    right::R
end
Pipe(l::AbstractModel{I}, r::AbstractModel{<:Any, O}) where {I, O} = Pipe{I, O, typeof(l), typeof(r)}(l, r)

const _COMPOUND = Union{Sum, Difference, Product, Quotient, Pipe}

# Composition rules are the operator signatures. The outer constructors above do no
# checks: rebuilds (withparams, _prefix, _setleaf) reach them through constructorof
# with children that already passed these rules.
Base.:+(a::AbstractModel{I, 1}, b::AbstractModel{I, 1}) where {I} = Sum(a, b)
Base.:-(a::AbstractModel{I, 1}, b::AbstractModel{I, 1}) where {I} = Difference(a, b)
Base.:*(a::AbstractModel{I, 1}, b::AbstractModel{I, 1}) where {I} = Product(a, b)
Base.:/(a::AbstractModel{I, 1}, b::AbstractModel{I, 1}) where {I} = Quotient(a, b)
Base.:|>(a::AbstractModel{I, M}, b::AbstractModel{M, O}) where {I, M, O} = Pipe(a, b)
# `r ∘ l` is function-composition order: apply l, then r. A deliberate synonym of `|>`.
Base.:∘(r::AbstractModel, l::AbstractModel) = l |> r

Base.:+(a::AbstractModel{I, O}, b::AbstractModel{J, P}) where {I, O, J, P} =
    throw(ArgumentError("cannot add $(_label(a)) ($I => $O) and $(_label(b)) ($J => $P)"))
Base.:-(a::AbstractModel{I, O}, b::AbstractModel{J, P}) where {I, O, J, P} =
    throw(ArgumentError("cannot subtract $(_label(a)) ($I => $O) and $(_label(b)) ($J => $P)"))
Base.:*(a::AbstractModel{I, O}, b::AbstractModel{J, P}) where {I, O, J, P} =
    throw(ArgumentError("cannot multiply $(_label(a)) ($I => $O) and $(_label(b)) ($J => $P)"))
Base.:/(a::AbstractModel{I, O}, b::AbstractModel{J, P}) where {I, O, J, P} =
    throw(ArgumentError("cannot divide $(_label(a)) ($I => $O) and $(_label(b)) ($J => $P)"))
Base.:|>(a::AbstractModel{I, M}, b::AbstractModel{J, O}) where {I, M, J, O} =
    throw(ArgumentError("$(_label(a)) produces $M value(s) per point, $(_label(b)) expects $J"))
