# --- 1D model library ---
#
# Voigt1D keeps its parameter-only constants in `_cache_`, computed once by its
# positional constructor; the other models have nothing to share between points.
#
# The non-trivial evaluates are `@inline`: with ForwardDiff duals their bodies exceed
# the inlining threshold, and the call left in the χ² loop measured the Hα + [NII]
# gradient at 1.17x or 1.38x of handwritten depending on the run (0.79x inlined).

Base.@kwdef struct Gaussian1D{A <: Real, M <: Real, S <: Real} <: AbstractModel{1, 1}
    amplitude::A = 1.0
    mean::M = 0.0
    sigma::S = 1.0
end

@inline evaluate(m::Gaussian1D, x::Number) = m.amplitude * exp(-((x - m.mean) / m.sigma)^2 / 2)


Base.@kwdef struct Const1D{T <: Real} <: AbstractModel{1, 1}
    value::T = 0.0
end

evaluate(m::Const1D, ::Number) = m.value


Base.@kwdef struct Linear1D{S <: Real, I <: Real} <: AbstractModel{1, 1}
    slope::S = 1.0
    intercept::I = 0.0
end

evaluate(m::Linear1D, x::Number) = m.slope * x + m.intercept


Base.@kwdef struct Lorentzian1D{A <: Real, M <: Real, G <: Real} <: AbstractModel{1, 1}
    amplitude::A = 1.0
    mean::M = 0.0
    gamma::G = 1.0
end

@inline evaluate(m::Lorentzian1D, x::Number) = m.amplitude / (1 + ((x - m.mean) / m.gamma)^2)


# ponytail: Thompson et al. 1987 pseudo-Voigt, no SpecialFunctions dep
struct Voigt1D{A <: Real, M <: Real, S <: Real, G <: Real, C} <: AbstractModel{1, 1}
    amplitude::A
    mean::M
    sigma::S
    gamma::G
    _cache_::C
    # The profile width `f` and the mixing `η` depend only on the parameters.
    function Voigt1D(amplitude::A, mean::M, sigma::S, gamma::G) where {A <: Real, M <: Real, S <: Real, G <: Real}
        fg = 2 * sigma * sqrt(2 * log(2))
        fl = 2 * gamma
        f = (
            fg^5 + 2.69269fg^4 * fl + 2.42843fg^3 * fl^2 +
                4.47163fg^2 * fl^3 + 0.07842fg * fl^4 + fl^5
        )^0.2
        r = fl / f
        c = (f = f, η = 1.36603r - 0.47719r^2 + 0.11116r^3)
        return new{A, M, S, G, typeof(c)}(amplitude, mean, sigma, gamma, c)
    end
end
Voigt1D(; amplitude = 1.0, mean = 0.0, sigma = 1.0, gamma = 1.0) = Voigt1D(amplitude, mean, sigma, gamma)

@inline function evaluate(m::Voigt1D, x::Number)
    (; f, η) = m._cache_
    u = 2(x - m.mean) / f
    return m.amplitude * (η / (1 + u^2) + (1 - η) * exp(-log(2) * u^2))
end


Base.@kwdef struct PowerLaw1D{N <: Real, X <: Real, I <: Real} <: AbstractModel{1, 1}
    norm::N = 1.0
    x_ref::X = 1.0
    index::I = 1.0
end

@inline evaluate(m::PowerLaw1D, x::Number) = m.norm * (x / m.x_ref)^(-m.index)


Base.@kwdef struct BlackBody1D{A <: Real, T <: Real} <: AbstractModel{1, 1}
    amplitude::A = 1.0
    temperature::T = 1.0
end

@inline evaluate(m::BlackBody1D, x::Number) = m.amplitude * x^3 / (exp(x / m.temperature) - 1)


Base.@kwdef struct BrokenPowerLaw1D{N <: Real, X <: Real, I1 <: Real, I2 <: Real} <: AbstractModel{1, 1}
    norm::N = 1.0
    x_break::X = 1.0
    index1::I1 = 1.0
    index2::I2 = 2.0
end

@inline evaluate(m::BrokenPowerLaw1D, x::Number) =
    m.norm * (x / m.x_break)^(x <= m.x_break ? -m.index1 : -m.index2)


Base.@kwdef struct Exponential1D{A <: Real, T <: Real} <: AbstractModel{1, 1}
    amplitude::A = 1.0
    tau::T = 1.0
end

@inline evaluate(m::Exponential1D, x::Number) = m.amplitude * exp(-x / m.tau)


# Coordinate-only transform: no amplitude, just warps x before an inner model
# renders it. Compose via Pipe (`z |> line`), not by embedding it inside
# another leaf's constructor — see zoo_tests.jl for the single-leaf-per-`@model`-line rule.
Base.@kwdef struct Redshift1D{T <: Real} <: AbstractModel{1, 1}
    z::T = 0.0
end

evaluate(m::Redshift1D, x::Number) = x / (1 + m.z)
