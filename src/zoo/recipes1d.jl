# Pre-constrained model factories for common astrophysics patterns.

function emission_line(; center, amplitude = 1.0, sigma = 1.0, center_window = 2.0)
    cm = @model begin
        line = Gaussian1D(amplitude = amplitude, mean = center, sigma = sigma)
        line
    end
    @constrain cm begin
        line.amplitude in (0.0, Inf)
        line.sigma in (0.0, Inf)
        line.mean in (center - center_window, center + center_window)
    end
    return cm
end

function absorption_line(; center, amplitude = 1.0, sigma = 1.0, center_window = 2.0)
    cm = @model begin
        line = Gaussian1D(amplitude = -abs(amplitude), mean = center, sigma = sigma)
        line
    end
    @constrain cm begin
        line.amplitude in (-Inf, 0.0)
        line.sigma in (0.0, Inf)
        line.mean in (center - center_window, center + center_window)
    end
    return cm
end

function doublet(;
        blue_center, red_center, ratio = 2.98,
        amplitude = 1.0, sigma = 2.0, center_window = 2.0
    )
    cm = @model begin
        blue = Gaussian1D(amplitude = amplitude, mean = blue_center, sigma = sigma)
        red = Gaussian1D(amplitude = ratio * amplitude, mean = red_center, sigma = sigma)
        blue + red
    end
    @constrain cm begin
        blue.amplitude in (0.0, Inf)
        blue.sigma in (0.0, Inf)
        blue.mean in (blue_center - center_window, blue_center + center_window)
        red.amplitude -> ratio * blue.amplitude
        red.mean -> (red_center / blue_center) * blue.mean
        red.sigma -> blue.sigma
    end
    return cm
end

function powerlaw_continuum(; norm = 1.0, x_ref = 1.0, index = 1.0)
    cm = @model begin
        pl = PowerLaw1D(norm = norm, x_ref = x_ref, index = index)
        pl
    end
    @constrain cm begin
        pl.norm in (0.0, Inf)
        pl.x_ref = x_ref
    end
    return cm
end

function blackbody_continuum(; amplitude = 1.0, temperature = 1.0)
    cm = @model begin
        bb = BlackBody1D(amplitude = amplitude, temperature = temperature)
        bb
    end
    @constrain cm begin
        bb.amplitude in (0.0, Inf)
        bb.temperature in (0.0, Inf)
    end
    return cm
end
