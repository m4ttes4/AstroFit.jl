# --- 1D model library ---

Base.@kwdef struct Gaussian1D{A <: Real, M <: Real, S <: Real} <: AbstractModel
    amplitude::A = 1.0
    mean::M = 0.0
    sigma::S = 1.0
end

render(m::Gaussian1D, x::Number) = m.amplitude * exp(-((x - m.mean) / m.sigma)^2 / 2)

function render!(out::AbstractArray, m::Gaussian1D, xs::AbstractArray)
    @inbounds for i in eachindex(out, xs)
        x = xs[i]
        out[i] = m.amplitude * exp(-((x - m.mean) / m.sigma)^2 / 2)
    end
    return out
end


Base.@kwdef struct Const1D{T <: Real} <: AbstractModel
    value::T = 0.0
end

render(m::Const1D, ::Number) = m.value

function render!(out::AbstractArray, m::Const1D, xs::AbstractArray)
    @inbounds for i in eachindex(out, xs)
        out[i] = m.value
    end
    return out
end


Base.@kwdef struct Linear1D{S <: Real, I <: Real} <: AbstractModel
    slope::S = 1.0
    intercept::I = 0.0
end

render(m::Linear1D, x::Number) = m.slope * x + m.intercept

function render!(out::AbstractArray, m::Linear1D, xs::AbstractArray)
    @inbounds for i in eachindex(out, xs)
        out[i] = m.slope * xs[i] + m.intercept
    end
    return out
end


Base.@kwdef struct Lorentzian1D{A <: Real, M <: Real, G <: Real} <: AbstractModel
    amplitude::A = 1.0
    mean::M = 0.0
    gamma::G = 1.0
end

render(m::Lorentzian1D, x::Number) = m.amplitude / (1 + ((x - m.mean) / m.gamma)^2)

function render!(out::AbstractArray, m::Lorentzian1D, xs::AbstractArray)
    @inbounds for i in eachindex(out, xs)
        out[i] = m.amplitude / (1 + ((xs[i] - m.mean) / m.gamma)^2)
    end
    return out
end


# ponytail: Thompson et al. 1987 pseudo-Voigt, no SpecialFunctions dep
Base.@kwdef struct Voigt1D{A <: Real, M <: Real, S <: Real, G <: Real} <: AbstractModel
    amplitude::A = 1.0
    mean::M = 0.0
    sigma::S = 1.0
    gamma::G = 1.0
end

function render(m::Voigt1D, x::Number)
    fg = 2 * m.sigma * sqrt(2 * log(2))
    fl = 2 * m.gamma
    f = (
        fg^5 + 2.69269fg^4 * fl + 2.42843fg^3 * fl^2 +
            4.47163fg^2 * fl^3 + 0.07842fg * fl^4 + fl^5
    )^0.2
    r = fl / f
    η = 1.36603r - 0.47719r^2 + 0.11116r^3
    u = 2(x - m.mean) / f
    return m.amplitude * (η / (1 + u^2) + (1 - η) * exp(-log(2) * u^2))
end

function render!(out::AbstractArray, m::Voigt1D, xs::AbstractArray)
    fg = 2 * m.sigma * sqrt(2 * log(2))
    fl = 2 * m.gamma
    f = (
        fg^5 + 2.69269fg^4 * fl + 2.42843fg^3 * fl^2 +
            4.47163fg^2 * fl^3 + 0.07842fg * fl^4 + fl^5
    )^0.2
    r = fl / f
    η = 1.36603r - 0.47719r^2 + 0.11116r^3
    ln2 = log(2)
    @inbounds for i in eachindex(out, xs)
        u = 2(xs[i] - m.mean) / f
        out[i] = m.amplitude * (η / (1 + u^2) + (1 - η) * exp(-ln2 * u^2))
    end
    return out
end


Base.@kwdef struct PowerLaw1D{N <: Real, X <: Real, I <: Real} <: AbstractModel
    norm::N = 1.0
    x_ref::X = 1.0
    index::I = 1.0
end

render(m::PowerLaw1D, x::Number) = m.norm * (x / m.x_ref)^(-m.index)

function render!(out::AbstractArray, m::PowerLaw1D, xs::AbstractArray)
    @inbounds for i in eachindex(out, xs)
        out[i] = m.norm * (xs[i] / m.x_ref)^(-m.index)
    end
    return out
end


Base.@kwdef struct BlackBody1D{A <: Real, T <: Real} <: AbstractModel
    amplitude::A = 1.0
    temperature::T = 1.0
end

render(m::BlackBody1D, x::Number) = m.amplitude * x^3 / (exp(x / m.temperature) - 1)

function render!(out::AbstractArray, m::BlackBody1D, xs::AbstractArray)
    @inbounds for i in eachindex(out, xs)
        x = xs[i]
        out[i] = m.amplitude * x^3 / (exp(x / m.temperature) - 1)
    end
    return out
end


Base.@kwdef struct BrokenPowerLaw1D{N <: Real, X <: Real, I1 <: Real, I2 <: Real} <: AbstractModel
    norm::N = 1.0
    x_break::X = 1.0
    index1::I1 = 1.0
    index2::I2 = 2.0
end

render(m::BrokenPowerLaw1D, x::Number) =
    m.norm * (x / m.x_break)^(x <= m.x_break ? -m.index1 : -m.index2)

function render!(out::AbstractArray, m::BrokenPowerLaw1D, xs::AbstractArray)
    @inbounds for i in eachindex(out, xs)
        x = xs[i]
        out[i] = m.norm * (x / m.x_break)^(x <= m.x_break ? -m.index1 : -m.index2)
    end
    return out
end


Base.@kwdef struct Exponential1D{A <: Real, T <: Real} <: AbstractModel
    amplitude::A = 1.0
    tau::T = 1.0
end

render(m::Exponential1D, x::Number) = m.amplitude * exp(-x / m.tau)

function render!(out::AbstractArray, m::Exponential1D, xs::AbstractArray)
    @inbounds for i in eachindex(out, xs)
        out[i] = m.amplitude * exp(-xs[i] / m.tau)
    end
    return out
end


# Coordinate-only transform: no amplitude, just warps x before an inner model
# renders it. Compose via Pipe (`z |> line`), not by embedding it inside
# another leaf's constructor — see zoo_tests.jl for the single-leaf-per-`@model`-line rule.
Base.@kwdef struct Redshift1D{T <: Real} <: AbstractModel
    z::T = 0.0
end

render(m::Redshift1D, x::Number) = x / (1 + m.z)

function render!(out::AbstractArray, m::Redshift1D, xs::AbstractArray)
    @inbounds for i in eachindex(out, xs)
        out[i] = xs[i] / (1 + m.z)
    end
    return out
end
