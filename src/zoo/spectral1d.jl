"""
    ConstantBackground1D(; value=0.0)

Constant non-negative spectral background.
"""
function ConstantBackground1D(; value=0.0)
    @constrain Const1D(value=value) begin
        @bound value in (0, Inf)
    end
end

"""
    LinearContinuum1D(; slope=0.0, intercept=1.0)

Linear continuum with non-negative normalization at the reference point.
The slope is left free.
"""
function LinearContinuum1D(; slope=0.0, intercept=1.0)
    @constrain Linear1D(slope=slope, intercept=intercept) begin
        @bound intercept in (0, Inf)
    end
end

function _check_center_window(center_window)
    center_window < zero(center_window) && throw(ArgumentError(
        "center_window must be non-negative, got $center_window"))
    center_window
end

function _constrain_emission_line(line, center, center_window)
    center_window = _check_center_window(center_window)
    if iszero(center_window)
        return @constrain line begin
            @bound amplitude in (0, Inf)
            @fix mean = center
            @bound sigma in (0, Inf)
        end
    elseif isfinite(center_window)
        lo = center - center_window
        hi = center + center_window
        return @constrain line begin
            @bound amplitude in (0, Inf)
            @bound mean in (lo, hi)
            @bound sigma in (0, Inf)
        end
    end
    @constrain line begin
        @bound amplitude in (0, Inf)
        @bound sigma in (0, Inf)
    end
end

function _constrain_absorption_line(line, center, center_window)
    center_window = _check_center_window(center_window)
    if iszero(center_window)
        return @constrain line begin
            @bound amplitude in (-Inf, 0)
            @fix mean = center
            @bound sigma in (0, Inf)
        end
    elseif isfinite(center_window)
        lo = center - center_window
        hi = center + center_window
        return @constrain line begin
            @bound amplitude in (-Inf, 0)
            @bound mean in (lo, hi)
            @bound sigma in (0, Inf)
        end
    end
    @constrain line begin
        @bound amplitude in (-Inf, 0)
        @bound sigma in (0, Inf)
    end
end

"""
    EmissionLine1D(; center, amplitude=1.0, sigma=1.0, center_window=Inf)

Gaussian emission line with non-negative amplitude and non-negative width.
Set `center_window=0` to fix the line center, a finite positive value to bound
it around `center`, or leave it at `Inf` to fit the center freely.
"""
function EmissionLine1D(; center, amplitude=1.0, sigma=1.0, center_window=Inf)
    line = Gaussian1D(amplitude=amplitude, mean=center, sigma=sigma)
    _constrain_emission_line(line, center, center_window)
end

"""
    AbsorptionLine1D(; center, depth=0.1, sigma=1.0, center_window=Inf)

Gaussian absorption line represented as a negative-amplitude Gaussian.
`depth` is positive in the public constructor and stored as `amplitude=-depth`.
"""
function AbsorptionLine1D(; center, depth=0.1, sigma=1.0, center_window=Inf)
    line = Gaussian1D(amplitude=-depth, mean=center, sigma=sigma)
    _constrain_absorption_line(line, center, center_window)
end

"""
    EmissionDoublet1D(; blue_center, red_center, amplitude=1.0,
                       sigma=1.0, ratio=1.0, center_window=Inf)

Two Gaussian emission lines with shared velocity shift, shared width, and fixed
red/blue amplitude ratio. The free masters are the blue amplitude, blue center,
and blue width; the red component is tied to those masters.
"""
function EmissionDoublet1D(; blue_center, red_center, amplitude=1.0,
                            sigma=1.0, ratio=1.0, center_window=Inf)
    blue_center <= zero(blue_center) && throw(ArgumentError(
        "blue_center must be positive, got $blue_center"))
    red_center <= zero(red_center) && throw(ArgumentError(
        "red_center must be positive, got $red_center"))
    ratio <= zero(ratio) && throw(ArgumentError(
        "ratio must be positive, got $ratio"))

    center_window = _check_center_window(center_window)
    wavelength_ratio = red_center / blue_center

    doublet = @model begin
        blue = Gaussian1D(amplitude=amplitude, mean=blue_center, sigma=sigma)
        red  = Gaussian1D(amplitude=ratio * amplitude,
                          mean=red_center,
                          sigma=sigma)
        blue + red
    end

    if iszero(center_window)
        return @constrain doublet begin
            @bound blue.amplitude in (0, Inf)
            @fix blue.mean = blue_center
            @bound blue.sigma in (0, Inf)
            @tie red.amplitude = ratio * blue.amplitude
            @tie red.mean = wavelength_ratio * blue.mean
            @tie red.sigma = blue.sigma
        end
    elseif isfinite(center_window)
        lo = blue_center - center_window
        hi = blue_center + center_window
        return @constrain doublet begin
            @bound blue.amplitude in (0, Inf)
            @bound blue.mean in (lo, hi)
            @bound blue.sigma in (0, Inf)
            @tie red.amplitude = ratio * blue.amplitude
            @tie red.mean = wavelength_ratio * blue.mean
            @tie red.sigma = blue.sigma
        end
    end

    @constrain doublet begin
        @bound blue.amplitude in (0, Inf)
        @bound blue.sigma in (0, Inf)
        @tie red.amplitude = ratio * blue.amplitude
        @tie red.mean = wavelength_ratio * blue.mean
        @tie red.sigma = blue.sigma
    end
end

"""
    EmissionLineSpectrum1D(; center, amplitude=1.0, sigma=1.0,
                            slope=0.0, intercept=1.0, center_window=Inf)

Linear continuum plus one constrained Gaussian emission line.
"""
function EmissionLineSpectrum1D(; center, amplitude=1.0, sigma=1.0,
                                  slope=0.0, intercept=1.0,
                                  center_window=Inf)
    @model begin
        continuum = LinearContinuum1D(slope=slope, intercept=intercept)
        line = EmissionLine1D(center=center, amplitude=amplitude,
                              sigma=sigma, center_window=center_window)
        continuum + line
    end
end

"""
    AbsorptionLineSpectrum1D(; center, depth=0.1, sigma=1.0,
                              slope=0.0, intercept=1.0, center_window=Inf)

Linear continuum plus one constrained Gaussian absorption line.
"""
function AbsorptionLineSpectrum1D(; center, depth=0.1, sigma=1.0,
                                    slope=0.0, intercept=1.0,
                                    center_window=Inf)
    @model begin
        continuum = LinearContinuum1D(slope=slope, intercept=intercept)
        line = AbsorptionLine1D(center=center, depth=depth,
                                sigma=sigma, center_window=center_window)
        continuum + line
    end
end
