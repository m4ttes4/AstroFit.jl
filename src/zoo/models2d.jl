# --- 2D model library ---

Base.@kwdef struct Gaussian2D{T<:Real} <: AbstractModel
    amplitude::T = 1.0
    x0::T        = 0.0
    y0::T        = 0.0
    sigma_x::T   = 1.0
    sigma_y::T   = 1.0
end

Gaussian2D(amplitude::Real, x0::Real, y0::Real, sigma_x::Real, sigma_y::Real) =
    Gaussian2D(promote(amplitude, x0, y0, sigma_x, sigma_y)...)

render(m::Gaussian2D, x::Number, y::Number) =
    m.amplitude * exp(-0.5 * (((x - m.x0) / m.sigma_x)^2 + ((y - m.y0) / m.sigma_y)^2))

function render!(out::AbstractArray, m::Gaussian2D, xs::AbstractArray, ys::AbstractArray)
    @inbounds for i in eachindex(out, xs, ys)
        out[i] = m.amplitude * exp(-0.5 * (((xs[i] - m.x0) / m.sigma_x)^2 +
                                            ((ys[i] - m.y0) / m.sigma_y)^2))
    end
    out
end


# ponytail: b_n via Ciotti & Bertin 1999 approximation, SpecialFunctions.jl if sub-percent needed
Base.@kwdef struct Sersic2D{T<:Real} <: AbstractModel
    amplitude::T = 1.0
    x0::T        = 0.0
    y0::T        = 0.0
    r_eff::T     = 1.0
    n::T         = 1.0
end

Sersic2D(amplitude::Real, x0::Real, y0::Real, r_eff::Real, n::Real) =
    Sersic2D(promote(amplitude, x0, y0, r_eff, n)...)

function render(m::Sersic2D, x::Number, y::Number)
    bn = 2 * m.n - 1/3 + 4 / (405 * m.n)
    r = sqrt((x - m.x0)^2 + (y - m.y0)^2)
    m.amplitude * exp(-bn * ((r / m.r_eff)^(1 / m.n) - 1))
end

function render!(out::AbstractArray, m::Sersic2D, xs::AbstractArray, ys::AbstractArray)
    bn = 2 * m.n - 1/3 + 4 / (405 * m.n)
    inv_n = 1 / m.n
    @inbounds for i in eachindex(out, xs, ys)
        r = sqrt((xs[i] - m.x0)^2 + (ys[i] - m.y0)^2)
        out[i] = m.amplitude * exp(-bn * ((r / m.r_eff)^inv_n - 1))
    end
    out
end


Base.@kwdef struct Moffat2D{T<:Real} <: AbstractModel
    amplitude::T = 1.0
    x0::T        = 0.0
    y0::T        = 0.0
    alpha::T     = 1.0
    beta::T      = 1.0
end

Moffat2D(amplitude::Real, x0::Real, y0::Real, alpha::Real, beta::Real) =
    Moffat2D(promote(amplitude, x0, y0, alpha, beta)...)

render(m::Moffat2D, x::Number, y::Number) =
    m.amplitude * (1 + ((x - m.x0)^2 + (y - m.y0)^2) / m.alpha^2)^(-m.beta)

function render!(out::AbstractArray, m::Moffat2D, xs::AbstractArray, ys::AbstractArray)
    a2 = m.alpha^2
    @inbounds for i in eachindex(out, xs, ys)
        out[i] = m.amplitude * (1 + ((xs[i] - m.x0)^2 + (ys[i] - m.y0)^2) / a2)^(-m.beta)
    end
    out
end


Base.@kwdef struct Beta2D{T<:Real} <: AbstractModel
    amplitude::T = 1.0
    x0::T        = 0.0
    y0::T        = 0.0
    r_core::T    = 1.0
    beta::T      = 0.67
end

Beta2D(amplitude::Real, x0::Real, y0::Real, r_core::Real, beta::Real) =
    Beta2D(promote(amplitude, x0, y0, r_core, beta)...)

render(m::Beta2D, x::Number, y::Number) =
    m.amplitude * (1 + ((x - m.x0)^2 + (y - m.y0)^2) / m.r_core^2)^(-3 * m.beta + 0.5)

function render!(out::AbstractArray, m::Beta2D, xs::AbstractArray, ys::AbstractArray)
    rc2 = m.r_core^2
    exp_val = -3 * m.beta + 0.5
    @inbounds for i in eachindex(out, xs, ys)
        out[i] = m.amplitude * (1 + ((xs[i] - m.x0)^2 + (ys[i] - m.y0)^2) / rc2)^exp_val
    end
    out
end
