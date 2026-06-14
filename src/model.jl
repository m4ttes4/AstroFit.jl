using Accessors

abstract type AbstractModel{N} end
Base.ndims(::AbstractModel{N}) where {N} = N
Base.ndims(::Type{<:AbstractModel{N}}) where {N} = N

Base.broadcastable(m::AbstractModel) = Ref(m)

render(m::AbstractModel, xs::AbstractArray...) = render.(m, xs...)
render!(out, m::AbstractModel, xs::AbstractArray...) = (out .= render.(m, xs...); out)


Base.@kwdef struct Gaussian1D{T<:Real} <: AbstractModel{1}
    amplitude::T = 1.0
    mean::T      = 0.0
    sigma::T     = 1.0
end

Gaussian1D(amplitude::Real, mean::Real, sigma::Real) =
    Gaussian1D(promote(amplitude, mean, sigma)...)

render(m::Gaussian1D, x::Number) = m.amplitude * exp(-((x - m.mean) / m.sigma)^2 / 2)


Base.@kwdef struct Const1D{T<:Real} <: AbstractModel{1}
    value::T = 0.0
end

render(m::Const1D, x::Number) = m.value


Base.@kwdef struct Linear1D{T<:Real} <: AbstractModel{1}
    slope::T     = 1.0
    intercept::T = 0.0
end

Linear1D(slope::Real, intercept::Real) = Linear1D(promote(slope, intercept)...)

render(m::Linear1D, x::Number) = m.slope * x + m.intercept
