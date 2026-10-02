# --- 2D model library ---
#
# Every model keeps its parameter-only constants (the rotation, the reciprocals) in
# `_cache_`, computed once by its positional constructor, so `evaluate` does only
# the per-point work. The evaluates are `@inline`: without it the image broadcast
# calls them per pixel and measured 1.25x slower (Gaussian2D, 256²).
#
# All four models start from the same rotated, flattened radius, so it lives in
# one helper; each model differs only in the profile applied to it.
@inline function _rot2d(x, y, x0, y0, cost, sint, inv_q2)
    dx, dy = x - x0, y - y0
    xr = cost * dx + sint * dy
    yr = -sint * dx + cost * dy
    return xr^2 + yr^2 * inv_q2
end

struct Gaussian2D{
        T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real, T5 <: Real, T6 <: Real, C,
    } <: AbstractModel{2, 1}
    amplitude::T1
    x0::T2
    y0::T3
    sigma::T4
    q::T5
    theta::T6
    _cache_::C
    function Gaussian2D(
            amplitude::T1, x0::T2, y0::T3, sigma::T4, q::T5, theta::T6,
        ) where {T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real, T5 <: Real, T6 <: Real}
        sint, cost = sincos(theta)
        c = (cost = cost, sint = sint, inv_q2 = 1 / q^2, inv_s2 = 1 / sigma^2)
        return new{T1, T2, T3, T4, T5, T6, typeof(c)}(amplitude, x0, y0, sigma, q, theta, c)
    end
end
Gaussian2D(; amplitude = 1.0, x0 = 0.0, y0 = 0.0, sigma = 1.0, q = 1.0, theta = 0.0) =
    Gaussian2D(amplitude, x0, y0, sigma, q, theta)

@inline function evaluate(m::Gaussian2D, (x, y)::NTuple{2, Number})
    (; cost, sint, inv_q2, inv_s2) = m._cache_
    return m.amplitude * exp(-0.5 * _rot2d(x, y, m.x0, m.y0, cost, sint, inv_q2) * inv_s2)
end


# ponytail: b_n via Ciotti & Bertin 1999 approximation, SpecialFunctions.jl if sub-percent needed
struct Sersic2D{
        T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real, T5 <: Real, T6 <: Real, T7 <: Real, C,
    } <: AbstractModel{2, 1}
    amplitude::T1
    x0::T2
    y0::T3
    r_eff::T4
    n::T5
    q::T6
    theta::T7
    _cache_::C
    function Sersic2D(
            amplitude::T1, x0::T2, y0::T3, r_eff::T4, n::T5, q::T6, theta::T7,
        ) where {T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real, T5 <: Real, T6 <: Real, T7 <: Real}
        sint, cost = sincos(theta)
        c = (
            bn = 2 * n - 1 / 3 + 4 / (405 * n), inv_n = 1 / n, inv_r = 1 / r_eff,
            cost = cost, sint = sint, inv_q2 = 1 / q^2,
        )
        return new{T1, T2, T3, T4, T5, T6, T7, typeof(c)}(amplitude, x0, y0, r_eff, n, q, theta, c)
    end
end
Sersic2D(; amplitude = 1.0, x0 = 0.0, y0 = 0.0, r_eff = 1.0, n = 1.0, q = 1.0, theta = 0.0) =
    Sersic2D(amplitude, x0, y0, r_eff, n, q, theta)

@inline function evaluate(m::Sersic2D, (x, y)::NTuple{2, Number})
    (; bn, inv_n, inv_r, cost, sint, inv_q2) = m._cache_
    r = sqrt(_rot2d(x, y, m.x0, m.y0, cost, sint, inv_q2))
    return m.amplitude * exp(-bn * ((r * inv_r)^inv_n - 1))
end


struct Moffat2D{
        T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real, T5 <: Real, T6 <: Real, T7 <: Real, C,
    } <: AbstractModel{2, 1}
    amplitude::T1
    x0::T2
    y0::T3
    alpha::T4
    beta::T5
    q::T6
    theta::T7
    _cache_::C
    function Moffat2D(
            amplitude::T1, x0::T2, y0::T3, alpha::T4, beta::T5, q::T6, theta::T7,
        ) where {T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real, T5 <: Real, T6 <: Real, T7 <: Real}
        sint, cost = sincos(theta)
        c = (inv_a2 = 1 / alpha^2, nbeta = -beta, cost = cost, sint = sint, inv_q2 = 1 / q^2)
        return new{T1, T2, T3, T4, T5, T6, T7, typeof(c)}(amplitude, x0, y0, alpha, beta, q, theta, c)
    end
end
Moffat2D(; amplitude = 1.0, x0 = 0.0, y0 = 0.0, alpha = 1.0, beta = 1.0, q = 1.0, theta = 0.0) =
    Moffat2D(amplitude, x0, y0, alpha, beta, q, theta)

@inline function evaluate(m::Moffat2D, (x, y)::NTuple{2, Number})
    (; inv_a2, nbeta, cost, sint, inv_q2) = m._cache_
    return m.amplitude * (1 + _rot2d(x, y, m.x0, m.y0, cost, sint, inv_q2) * inv_a2)^nbeta
end


struct Beta2D{
        T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real, T5 <: Real, T6 <: Real, T7 <: Real, C,
    } <: AbstractModel{2, 1}
    amplitude::T1
    x0::T2
    y0::T3
    r_core::T4
    beta::T5
    q::T6
    theta::T7
    _cache_::C
    function Beta2D(
            amplitude::T1, x0::T2, y0::T3, r_core::T4, beta::T5, q::T6, theta::T7,
        ) where {T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real, T5 <: Real, T6 <: Real, T7 <: Real}
        sint, cost = sincos(theta)
        c = (inv_rc2 = 1 / r_core^2, exp_val = -3 * beta + 0.5, cost = cost, sint = sint, inv_q2 = 1 / q^2)
        return new{T1, T2, T3, T4, T5, T6, T7, typeof(c)}(amplitude, x0, y0, r_core, beta, q, theta, c)
    end
end
Beta2D(; amplitude = 1.0, x0 = 0.0, y0 = 0.0, r_core = 1.0, beta = 0.67, q = 1.0, theta = 0.0) =
    Beta2D(amplitude, x0, y0, r_core, beta, q, theta)

@inline function evaluate(m::Beta2D, (x, y)::NTuple{2, Number})
    (; inv_rc2, exp_val, cost, sint, inv_q2) = m._cache_
    return m.amplitude * (1 + _rot2d(x, y, m.x0, m.y0, cost, sint, inv_q2) * inv_rc2)^exp_val
end
