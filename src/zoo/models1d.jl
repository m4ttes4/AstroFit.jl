# --- 1D model library ---

Base.@kwdef struct Gaussian1D{T <: Real} <: AbstractModel
    amplitude::T = 1.0
    mean::T = 0.0
    sigma::T = 1.0
end

Gaussian1D(amplitude::Real, mean::Real, sigma::Real) =
    Gaussian1D(promote(amplitude, mean, sigma)...)

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

render(m::Const1D, x::Number) = m.value

function render!(out::AbstractArray, m::Const1D, xs::AbstractArray)
    @inbounds for i in eachindex(out, xs)
        out[i] = m.value
    end
    return out
end


Base.@kwdef struct Linear1D{T <: Real} <: AbstractModel
    slope::T = 1.0
    intercept::T = 0.0
end

Linear1D(slope::Real, intercept::Real) = Linear1D(promote(slope, intercept)...)

render(m::Linear1D, x::Number) = m.slope * x + m.intercept

function render!(out::AbstractArray, m::Linear1D, xs::AbstractArray)
    @inbounds for i in eachindex(out, xs)
        out[i] = m.slope * xs[i] + m.intercept
    end
    return out
end


Base.@kwdef struct Lorentzian1D{T <: Real} <: AbstractModel
    amplitude::T = 1.0
    mean::T = 0.0
    gamma::T = 1.0
end

Lorentzian1D(amplitude::Real, mean::Real, gamma::Real) =
    Lorentzian1D(promote(amplitude, mean, gamma)...)

render(m::Lorentzian1D, x::Number) = m.amplitude / (1 + ((x - m.mean) / m.gamma)^2)

function render!(out::AbstractArray, m::Lorentzian1D, xs::AbstractArray)
    @inbounds for i in eachindex(out, xs)
        out[i] = m.amplitude / (1 + ((xs[i] - m.mean) / m.gamma)^2)
    end
    return out
end


# ponytail: Thompson et al. 1987 pseudo-Voigt, no SpecialFunctions dep
Base.@kwdef struct Voigt1D{T <: Real} <: AbstractModel
    amplitude::T = 1.0
    mean::T = 0.0
    sigma::T = 1.0
    gamma::T = 1.0
end

Voigt1D(amplitude::Real, mean::Real, sigma::Real, gamma::Real) =
    Voigt1D(promote(amplitude, mean, sigma, gamma)...)

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


Base.@kwdef struct PowerLaw1D{T <: Real} <: AbstractModel
    norm::T = 1.0
    x_ref::T = 1.0
    index::T = 1.0
end

PowerLaw1D(norm::Real, x_ref::Real, index::Real) =
    PowerLaw1D(promote(norm, x_ref, index)...)

render(m::PowerLaw1D, x::Number) = m.norm * (x / m.x_ref)^(-m.index)

function render!(out::AbstractArray, m::PowerLaw1D, xs::AbstractArray)
    @inbounds for i in eachindex(out, xs)
        out[i] = m.norm * (xs[i] / m.x_ref)^(-m.index)
    end
    return out
end


Base.@kwdef struct BlackBody1D{T <: Real} <: AbstractModel
    amplitude::T = 1.0
    temperature::T = 1.0
end

BlackBody1D(amplitude::Real, temperature::Real) =
    BlackBody1D(promote(amplitude, temperature)...)

render(m::BlackBody1D, x::Number) = m.amplitude * x^3 / (exp(x / m.temperature) - 1)

function render!(out::AbstractArray, m::BlackBody1D, xs::AbstractArray)
    @inbounds for i in eachindex(out, xs)
        x = xs[i]
        out[i] = m.amplitude * x^3 / (exp(x / m.temperature) - 1)
    end
    return out
end


Base.@kwdef struct BrokenPowerLaw1D{T <: Real} <: AbstractModel
    norm::T = 1.0
    x_break::T = 1.0
    index1::T = 1.0
    index2::T = 2.0
end

BrokenPowerLaw1D(norm::Real, x_break::Real, index1::Real, index2::Real) =
    BrokenPowerLaw1D(promote(norm, x_break, index1, index2)...)

render(m::BrokenPowerLaw1D, x::Number) =
    m.norm * (x / m.x_break)^(x <= m.x_break ? -m.index1 : -m.index2)

function render!(out::AbstractArray, m::BrokenPowerLaw1D, xs::AbstractArray)
    @inbounds for i in eachindex(out, xs)
        x = xs[i]
        out[i] = m.norm * (x / m.x_break)^(x <= m.x_break ? -m.index1 : -m.index2)
    end
    return out
end


Base.@kwdef struct Exponential1D{T <: Real} <: AbstractModel
    amplitude::T = 1.0
    tau::T = 1.0
end

Exponential1D(amplitude::Real, tau::Real) =
    Exponential1D(promote(amplitude, tau)...)

render(m::Exponential1D, x::Number) = m.amplitude * exp(-x / m.tau)

function render!(out::AbstractArray, m::Exponential1D, xs::AbstractArray)
    @inbounds for i in eachindex(out, xs)
        out[i] = m.amplitude * exp(-xs[i] / m.tau)
    end
    return out
end
