@testitem "zoo-style spectral line factory uses current constraints" tags = [:zoo] begin
    using AstroFit

    function emission_line(; center, amplitude = 2.0, sigma = 1.5, center_window = 2.0)
        cm = @model begin
            line = Gaussian1D(amplitude = amplitude, mean = center, sigma = sigma)
            line
        end
        @constrain cm begin
            line.amplitude in (0.0, Inf)
            line.sigma in (0.0, Inf)
            line.mean in (center - center_window, center + center_window)
        end
        cm
    end

    emission = emission_line(center = 6563.0)
    @test emission isa CompiledModel
    @test nfree(emission) == 3

    lo, hi = bounds(emission)
    @test 0.0 in lo
    @test 6561.0 in lo
    @test 6565.0 in hi
    @test render(emission, 6563.0) ≈ 2.0
end

@testitem "zoo-style doublet ties physical ratios" tags = [:zoo, :tied] begin
    using AstroFit

    blue_center = 4959.0
    red_center = 5007.0
    ratio = 2.98

    doublet = @model begin
        blue = Gaussian1D(amplitude = 1.0, mean = blue_center, sigma = 2.0)
        red = Gaussian1D(amplitude = ratio, mean = red_center, sigma = 2.0)
        blue + red
    end

    @constrain doublet begin
        blue.amplitude in (0.0, Inf)
        blue.sigma in (0.0, Inf)
        red.amplitude -> ratio * blue.amplitude
        red.mean -> (red_center / blue_center) * blue.mean
        red.sigma -> blue.sigma
    end

    @test nfree(doublet) == 3
    rebuilt = withparams(doublet, [2.0, 4960.0, 3.0])
    @test rebuilt.red.model.amplitude ≈ ratio * 2.0
    @test rebuilt.red.model.mean ≈ (red_center / blue_center) * 4960.0
    @test rebuilt.red.model.sigma == 3.0
    @test :red_amplitude ∉ paramnames(doublet)
    @test :red_sigma ∉ paramnames(doublet)
end

@testitem "zoo-style spectrum composition namespaces flat leaves" tags = [:zoo] begin
    using AstroFit

    spectrum = @model begin
        continuum = Linear1D(slope = 1.0e-4, intercept = 1.0)
        line = Gaussian1D(amplitude = 2.0, mean = 6563.0, sigma = 1.5)
        continuum + line
    end

    @constrain spectrum begin
        continuum.intercept in (0.0, Inf)
        line.amplitude in (0.0, Inf)
        line.sigma in (0.0, Inf)
    end

    @test nfree(spectrum) == 5
    @test spectrum.continuum.model.intercept == 1.0
    @test spectrum.line.model.amplitude == 2.0
    continuum_at_center = render(spectrum.continuum.model, 6563.0)
    @test render(spectrum, 6563.0) > continuum_at_center
end

@testitem "zoo-style 2D line profiles render and constrain physical params" tags = [:zoo, :twod] begin
    using AstroFit

    struct Gaussian2D{T <: Real} <: AbstractModel
        amplitude::T
        x0::T
        y0::T
        sigma_x::T
        sigma_y::T
    end

    AstroFit.render(m::Gaussian2D, x::Number, y::Number) =
        m.amplitude * exp(
        -0.5 * (
            ((x - m.x0) / m.sigma_x)^2 +
                ((y - m.y0) / m.sigma_y)^2
        )
    )

    scene = @model begin
        bulge = Gaussian2D(3.0, 1.0, -2.0, 2.0, 1.0)
        disk = Gaussian2D(1.0, 1.0, -2.0, 5.0, 3.0)
        bulge + disk
    end

    @constrain scene begin
        bulge.amplitude in (0.0, Inf)
        bulge.sigma_x in (0.0, Inf)
        bulge.sigma_y in (0.0, Inf)
        disk.amplitude in (0.0, Inf)
        disk.sigma_x in (0.0, Inf)
        disk.sigma_y in (0.0, Inf)
    end

    @test nfree(scene) == 10
    @test render(scene, 1.0, -2.0) ≈ 4.0
    @test render(scene, 4.0, -2.0) < render(scene, 1.0, -2.0)

    x = [1.0 2.0; 3.0 4.0]
    y = fill(-2.0, size(x))
    out = similar(x)
    @test render!(out, scene, x, y) === out
    @test out ≈ render(scene, x, y)
end
