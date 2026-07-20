# --- 2D model library ---

Base.@kwdef struct Gaussian2D{
        T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real, T5 <: Real, T6 <: Real,
    } <: AbstractModel
    amplitude::T1 = 1.0
    x0::T2 = 0.0
    y0::T3 = 0.0
    sigma::T4 = 1.0
    q::T5 = 1.0
    theta::T6 = 0.0
end

function render(m::Gaussian2D, x::Number, y::Number)
    dx, dy = x - m.x0, y - m.y0
    cost, sint = cos(m.theta), sin(m.theta)
    xr = cost * dx + sint * dy
    yr = -sint * dx + cost * dy
    return m.amplitude * exp(-0.5 * (xr^2 + (yr / m.q)^2) / m.sigma^2)
end

function render!(out::AbstractArray, m::Gaussian2D, xs::AbstractArray, ys::AbstractArray)
    cost, sint = cos(m.theta), sin(m.theta)
    inv_s2 = 1 / m.sigma^2
    inv_q2 = 1 / m.q^2
    @inbounds for i in eachindex(out, xs, ys)
        dx, dy = xs[i] - m.x0, ys[i] - m.y0
        xr = cost * dx + sint * dy
        yr = -sint * dx + cost * dy
        out[i] = m.amplitude * exp(-0.5 * (xr^2 + yr^2 * inv_q2) * inv_s2)
    end
    return out
end


# ponytail: b_n via Ciotti & Bertin 1999 approximation, SpecialFunctions.jl if sub-percent needed
Base.@kwdef struct Sersic2D{
        T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real, T5 <: Real, T6 <: Real, T7 <: Real,
    } <: AbstractModel
    amplitude::T1 = 1.0
    x0::T2 = 0.0
    y0::T3 = 0.0
    r_eff::T4 = 1.0
    n::T5 = 1.0
    q::T6 = 1.0
    theta::T7 = 0.0
end

function render(m::Sersic2D, x::Number, y::Number)
    bn = 2 * m.n - 1 / 3 + 4 / (405 * m.n)
    dx, dy = x - m.x0, y - m.y0
    cost, sint = cos(m.theta), sin(m.theta)
    xr = cost * dx + sint * dy
    yr = -sint * dx + cost * dy
    r = sqrt(xr^2 + (yr / m.q)^2)
    return m.amplitude * exp(-bn * ((r / m.r_eff)^(1 / m.n) - 1))
end

function render!(out::AbstractArray, m::Sersic2D, xs::AbstractArray, ys::AbstractArray)
    bn = 2 * m.n - 1 / 3 + 4 / (405 * m.n)
    inv_n = 1 / m.n
    cost, sint = cos(m.theta), sin(m.theta)
    inv_q2 = 1 / m.q^2
    @inbounds for i in eachindex(out, xs, ys)
        dx, dy = xs[i] - m.x0, ys[i] - m.y0
        xr = cost * dx + sint * dy
        yr = -sint * dx + cost * dy
        r = sqrt(xr^2 + yr^2 * inv_q2)
        out[i] = m.amplitude * exp(-bn * ((r / m.r_eff)^inv_n - 1))
    end
    return out
end


Base.@kwdef struct Moffat2D{
        T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real, T5 <: Real, T6 <: Real, T7 <: Real,
    } <: AbstractModel
    amplitude::T1 = 1.0
    x0::T2 = 0.0
    y0::T3 = 0.0
    alpha::T4 = 1.0
    beta::T5 = 1.0
    q::T6 = 1.0
    theta::T7 = 0.0
end

function render(m::Moffat2D, x::Number, y::Number)
    dx, dy = x - m.x0, y - m.y0
    cost, sint = cos(m.theta), sin(m.theta)
    xr = cost * dx + sint * dy
    yr = -sint * dx + cost * dy
    return m.amplitude * (1 + (xr^2 + (yr / m.q)^2) / m.alpha^2)^(-m.beta)
end

function render!(out::AbstractArray, m::Moffat2D, xs::AbstractArray, ys::AbstractArray)
    a2 = m.alpha^2
    cost, sint = cos(m.theta), sin(m.theta)
    inv_q2 = 1 / m.q^2
    @inbounds for i in eachindex(out, xs, ys)
        dx, dy = xs[i] - m.x0, ys[i] - m.y0
        xr = cost * dx + sint * dy
        yr = -sint * dx + cost * dy
        out[i] = m.amplitude * (1 + (xr^2 + yr^2 * inv_q2) / a2)^(-m.beta)
    end
    return out
end


Base.@kwdef struct Beta2D{
        T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real, T5 <: Real, T6 <: Real, T7 <: Real,
    } <: AbstractModel
    amplitude::T1 = 1.0
    x0::T2 = 0.0
    y0::T3 = 0.0
    r_core::T4 = 1.0
    beta::T5 = 0.67
    q::T6 = 1.0
    theta::T7 = 0.0
end

function render(m::Beta2D, x::Number, y::Number)
    dx, dy = x - m.x0, y - m.y0
    cost, sint = cos(m.theta), sin(m.theta)
    xr = cost * dx + sint * dy
    yr = -sint * dx + cost * dy
    return m.amplitude * (1 + (xr^2 + (yr / m.q)^2) / m.r_core^2)^(-3 * m.beta + 0.5)
end

function render!(out::AbstractArray, m::Beta2D, xs::AbstractArray, ys::AbstractArray)
    rc2 = m.r_core^2
    exp_val = -3 * m.beta + 0.5
    cost, sint = cos(m.theta), sin(m.theta)
    inv_q2 = 1 / m.q^2
    @inbounds for i in eachindex(out, xs, ys)
        dx, dy = xs[i] - m.x0, ys[i] - m.y0
        xr = cost * dx + sint * dy
        yr = -sint * dx + cost * dy
        out[i] = m.amplitude * (1 + (xr^2 + yr^2 * inv_q2) / rc2)^exp_val
    end
    return out
end
