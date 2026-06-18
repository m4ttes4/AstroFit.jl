abstract type AbstractModel end

Base.broadcastable(m::AbstractModel) = Ref(m)

render(m::AbstractModel, xs::AbstractArray...) = render.(m, xs...)

Base.@kwdef struct Gaussian1D{T<:Real} <: AbstractModel
    amplitude::T = 1.0
    mean::T      = 0.0
    sigma::T     = 1.0
end

Gaussian1D(amplitude::Real, mean::Real, sigma::Real) =
    Gaussian1D(promote(amplitude, mean, sigma)...)

render(m::Gaussian1D, x::Number) = m.amplitude * exp(-((x - m.mean) / m.sigma)^2 / 2)


Base.@kwdef struct Const1D{T<:Real} <: AbstractModel
    value::T = 0.0
end

render(m::Const1D, x::Number) = m.value


Base.@kwdef struct Linear1D{T<:Real} <: AbstractModel
    slope::T     = 1.0
    intercept::T = 0.0
end

Linear1D(slope::Real, intercept::Real) = Linear1D(promote(slope, intercept)...)

render(m::Linear1D, x::Number) = m.slope * x + m.intercept
