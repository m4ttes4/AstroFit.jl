# --- 2D model library ---

Base.@kwdef struct Gaussian2D{T<:Real} <: AbstractModel
    amplitude::T = 1.0
    x0::T        = 0.0
    y0::T        = 0.0
    sigma::T     = 1.0
    q::T         = 1.0
    theta::T     = 0.0
end

Gaussian2D(amplitude::Real, x0::Real, y0::Real, sigma::Real, q::Real, theta::Real) =
    Gaussian2D(promote(amplitude, x0, y0, sigma, q, theta)...)

function render(m::Gaussian2D, x::Number, y::Number)
    dx, dy = x - m.x0, y - m.y0
    cost, sint = cos(m.theta), sin(m.theta)
    xr =  cost * dx + sint * dy
    yr = -sint * dx + cost * dy
    m.amplitude * exp(-0.5 * (xr^2 + (yr / m.q)^2) / m.sigma^2)
end

function render!(out::AbstractArray, m::Gaussian2D, xs::AbstractArray, ys::AbstractArray)
    cost, sint = cos(m.theta), sin(m.theta)
    inv_s2 = 1 / m.sigma^2
    inv_q2 = 1 / m.q^2
    @inbounds for i in eachindex(out, xs, ys)
        dx, dy = xs[i] - m.x0, ys[i] - m.y0
        xr =  cost * dx + sint * dy
        yr = -sint * dx + cost * dy
        out[i] = m.amplitude * exp(-0.5 * (xr^2 + yr^2 * inv_q2) * inv_s2)
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
    q::T         = 1.0
    theta::T     = 0.0
end

Sersic2D(amplitude::Real, x0::Real, y0::Real, r_eff::Real, n::Real, q::Real, theta::Real) =
    Sersic2D(promote(amplitude, x0, y0, r_eff, n, q, theta)...)

function render(m::Sersic2D, x::Number, y::Number)
    bn = 2 * m.n - 1/3 + 4 / (405 * m.n)
    dx, dy = x - m.x0, y - m.y0
    cost, sint = cos(m.theta), sin(m.theta)
    xr =  cost * dx + sint * dy
    yr = -sint * dx + cost * dy
    r = sqrt(xr^2 + (yr / m.q)^2)
    m.amplitude * exp(-bn * ((r / m.r_eff)^(1 / m.n) - 1))
end

function render!(out::AbstractArray, m::Sersic2D, xs::AbstractArray, ys::AbstractArray)
    bn = 2 * m.n - 1/3 + 4 / (405 * m.n)
    inv_n = 1 / m.n
    cost, sint = cos(m.theta), sin(m.theta)
    inv_q2 = 1 / m.q^2
    @inbounds for i in eachindex(out, xs, ys)
        dx, dy = xs[i] - m.x0, ys[i] - m.y0
        xr =  cost * dx + sint * dy
        yr = -sint * dx + cost * dy
        r = sqrt(xr^2 + yr^2 * inv_q2)
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
    q::T         = 1.0
    theta::T     = 0.0
end

Moffat2D(amplitude::Real, x0::Real, y0::Real, alpha::Real, beta::Real, q::Real, theta::Real) =
    Moffat2D(promote(amplitude, x0, y0, alpha, beta, q, theta)...)

function render(m::Moffat2D, x::Number, y::Number)
    dx, dy = x - m.x0, y - m.y0
    cost, sint = cos(m.theta), sin(m.theta)
    xr =  cost * dx + sint * dy
    yr = -sint * dx + cost * dy
    m.amplitude * (1 + (xr^2 + (yr / m.q)^2) / m.alpha^2)^(-m.beta)
end

function render!(out::AbstractArray, m::Moffat2D, xs::AbstractArray, ys::AbstractArray)
    a2 = m.alpha^2
    cost, sint = cos(m.theta), sin(m.theta)
    inv_q2 = 1 / m.q^2
    @inbounds for i in eachindex(out, xs, ys)
        dx, dy = xs[i] - m.x0, ys[i] - m.y0
        xr =  cost * dx + sint * dy
        yr = -sint * dx + cost * dy
        out[i] = m.amplitude * (1 + (xr^2 + yr^2 * inv_q2) / a2)^(-m.beta)
    end
    out
end


Base.@kwdef struct Beta2D{T<:Real} <: AbstractModel
    amplitude::T = 1.0
    x0::T        = 0.0
    y0::T        = 0.0
    r_core::T    = 1.0
    beta::T      = 0.67
    q::T         = 1.0
    theta::T     = 0.0
end

Beta2D(amplitude::Real, x0::Real, y0::Real, r_core::Real, beta::Real, q::Real, theta::Real) =
    Beta2D(promote(amplitude, x0, y0, r_core, beta, q, theta)...)

function render(m::Beta2D, x::Number, y::Number)
    dx, dy = x - m.x0, y - m.y0
    cost, sint = cos(m.theta), sin(m.theta)
    xr =  cost * dx + sint * dy
    yr = -sint * dx + cost * dy
    m.amplitude * (1 + (xr^2 + (yr / m.q)^2) / m.r_core^2)^(-3 * m.beta + 0.5)
end

function render!(out::AbstractArray, m::Beta2D, xs::AbstractArray, ys::AbstractArray)
    rc2 = m.r_core^2
    exp_val = -3 * m.beta + 0.5
    cost, sint = cos(m.theta), sin(m.theta)
    inv_q2 = 1 / m.q^2
    @inbounds for i in eachindex(out, xs, ys)
        dx, dy = xs[i] - m.x0, ys[i] - m.y0
        xr =  cost * dx + sint * dy
        yr = -sint * dx + cost * dy
        out[i] = m.amplitude * (1 + (xr^2 + yr^2 * inv_q2) / rc2)^exp_val
    end
    out
end
