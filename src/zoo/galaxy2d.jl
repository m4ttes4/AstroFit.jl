"""
    Gaussian2D(; amplitude=1.0, x0=0.0, y0=0.0,
                sigma_x=1.0, sigma_y=1.0, theta=0.0)

Rotated elliptical 2D Gaussian surface-brightness profile.
"""
Base.@kwdef struct Gaussian2D{T<:Real} <: AbstractModel{2}
    amplitude::T = 1.0
    x0::T        = 0.0
    y0::T        = 0.0
    sigma_x::T   = 1.0
    sigma_y::T   = 1.0
    theta::T     = 0.0
end

Gaussian2D(amplitude::Real, x0::Real, y0::Real,
           sigma_x::Real, sigma_y::Real, theta::Real) =
    Gaussian2D(promote(amplitude, x0, y0, sigma_x, sigma_y, theta)...)

function render(m::Gaussian2D, x::Number, y::Number)
    c = cos(m.theta)
    s = sin(m.theta)
    dx = x - m.x0
    dy = y - m.y0
    xp =  c * dx + s * dy
    yp = -s * dx + c * dy
    m.amplitude * exp(-0.5 * ((xp / m.sigma_x)^2 + (yp / m.sigma_y)^2))
end

"""
    ExponentialDisk2D(; amplitude=1.0, x0=0.0, y0=0.0,
                       scale_radius=1.0, axis_ratio=1.0, theta=0.0)

Rotated elliptical exponential disk profile, useful as a simple projected
galaxy line-emission surface-brightness model.
"""
Base.@kwdef struct ExponentialDisk2D{T<:Real} <: AbstractModel{2}
    amplitude::T    = 1.0
    x0::T           = 0.0
    y0::T           = 0.0
    scale_radius::T = 1.0
    axis_ratio::T   = 1.0
    theta::T        = 0.0
end

ExponentialDisk2D(amplitude::Real, x0::Real, y0::Real,
                  scale_radius::Real, axis_ratio::Real, theta::Real) =
    ExponentialDisk2D(promote(amplitude, x0, y0,
                              scale_radius, axis_ratio, theta)...)

function render(m::ExponentialDisk2D, x::Number, y::Number)
    c = cos(m.theta)
    s = sin(m.theta)
    dx = x - m.x0
    dy = y - m.y0
    xp =  c * dx + s * dy
    yp = -s * dx + c * dy
    r = sqrt(xp^2 + (yp / m.axis_ratio)^2)
    m.amplitude * exp(-r / m.scale_radius)
end

function _check_positive(name, value)
    value > zero(value) || throw(ArgumentError(
        "$name must be positive, got $value"))
    value
end

function _check_axis_ratio(axis_ratio)
    zero(axis_ratio) < axis_ratio <= one(axis_ratio) || throw(ArgumentError(
        "axis_ratio must be in (0, 1], got $axis_ratio"))
    axis_ratio
end

"""
    GalaxyGaussianLineProfile2D(; amplitude=1.0, x0=0.0, y0=0.0,
                                 sigma_x=1.0, sigma_y=sigma_x, theta=0.0)

Constrained rotated Gaussian 2D line-emission profile. The amplitude and
widths are positive; centroid and angle are free.
"""
function GalaxyGaussianLineProfile2D(; amplitude=1.0, x0=0.0, y0=0.0,
                                      sigma_x=1.0, sigma_y=sigma_x,
                                      theta=0.0)
    _check_positive("amplitude", amplitude)
    _check_positive("sigma_x", sigma_x)
    _check_positive("sigma_y", sigma_y)

    profile = Gaussian2D(amplitude=amplitude, x0=x0, y0=y0,
                         sigma_x=sigma_x, sigma_y=sigma_y, theta=theta)
    @constrain profile begin
        @bound amplitude in (0, Inf)
        @bound sigma_x in (0, Inf)
        @bound sigma_y in (0, Inf)
    end
end

"""
    GalaxyExponentialLineProfile2D(; amplitude=1.0, x0=0.0, y0=0.0,
                                    scale_radius=1.0, axis_ratio=1.0,
                                    theta=0.0)

Constrained elliptical exponential disk line-emission profile. The amplitude,
scale radius, and axis ratio are physically bounded; centroid and angle are free.
"""
function GalaxyExponentialLineProfile2D(; amplitude=1.0, x0=0.0, y0=0.0,
                                         scale_radius=1.0, axis_ratio=1.0,
                                         theta=0.0)
    _check_positive("amplitude", amplitude)
    _check_positive("scale_radius", scale_radius)
    _check_axis_ratio(axis_ratio)

    profile = ExponentialDisk2D(amplitude=amplitude, x0=x0, y0=y0,
                                scale_radius=scale_radius,
                                axis_ratio=axis_ratio, theta=theta)
    @constrain profile begin
        @bound amplitude in (0, Inf)
        @bound scale_radius in (0, Inf)
        @bound axis_ratio in (eps(Float64), 1)
    end
end
