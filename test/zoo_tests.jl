@testitem "zoo: single-component spectral prefabs have physical constraints" tags=[:zoo] begin
    using AstroFit
    using Accessors: @set

    emission = EmissionLine1D(center=6563.0, amplitude=2.0,
                              sigma=1.5, center_window=2.0)
    @test emission isa CompiledModel
    @test nfree(emission) == 3
    lo, hi = bounds_vectors(emission.spec)
    @test 0.0 in lo
    @test 6561.0 in lo
    @test 6565.0 in hi
    @test_throws ArgumentError @set emission.amplitude = -1.0
    @test_throws ArgumentError @set emission.sigma = -0.1

    fixed_center = EmissionLine1D(center=5007.0, center_window=0.0)
    @test nfree(fixed_center) == 2
    @test fixed_center.mean == 5007.0

    absorption = AbsorptionLine1D(center=5890.0, depth=0.3, sigma=2.0)
    @test absorption.amplitude == -0.3
    @test_throws ArgumentError @set absorption.amplitude = 0.1
    @test (@set absorption.amplitude = -0.5).amplitude == -0.5

    continuum = LinearContinuum1D(slope=1e-4, intercept=1.0)
    @test nfree(continuum) == 2
    @test_throws ArgumentError @set continuum.intercept = -1.0

    background = ConstantBackground1D(value=0.2)
    @test nfree(background) == 1
    @test_throws ArgumentError @set background.value = -0.1
end

@testitem "zoo: emission doublet ties physical ratios" tags=[:zoo, :tied] begin
    using AstroFit
    using Accessors: @set

    doublet = EmissionDoublet1D(blue_center=4959.0, red_center=5007.0,
                                amplitude=1.0, sigma=2.0,
                                ratio=2.98, center_window=5.0)

    @test nfree(doublet) == 3
    @test doublet.red.amplitude ≈ 2.98 * doublet.blue.amplitude
    @test doublet.red.mean ≈ (5007.0 / 4959.0) * doublet.blue.mean
    @test doublet.red.sigma == doublet.blue.sigma
    @test_throws ArgumentError @set doublet.red.sigma = 3.0

    rebuilt = withparams(doublet, [2.0, 4960.0, 3.0])
    @test rebuilt.red.amplitude ≈ 5.96
    @test rebuilt.red.mean ≈ (5007.0 / 4959.0) * 4960.0
    @test rebuilt.red.sigma == 3.0

    fixed_center = EmissionDoublet1D(blue_center=3726.0, red_center=3729.0,
                                     ratio=1.5, center_window=0.0)
    @test nfree(fixed_center) == 2
    @test fixed_center.blue.mean == 3726.0
    @test fixed_center.red.mean ≈ 3729.0
end

@testitem "zoo: spectrum prefabs compose and namespace constraints" tags=[:zoo, :prefab] begin
    using AstroFit
    using Accessors: @set

    emission_spectrum = EmissionLineSpectrum1D(center=6563.0, amplitude=2.0,
                                               sigma=1.5, intercept=1.0,
                                               center_window=1.0)
    @test nfree(emission_spectrum) == 5
    @test emission_spectrum.continuum.intercept == 1.0
    @test emission_spectrum.line.amplitude == 2.0
    @test_throws ArgumentError @set emission_spectrum.continuum.intercept = -1.0
    @test_throws ArgumentError @set emission_spectrum.line.sigma = -1.0

    absorption_spectrum = AbsorptionLineSpectrum1D(center=5890.0, depth=0.2,
                                                   sigma=1.0, intercept=1.0)
    @test nfree(absorption_spectrum) == 5
    @test absorption_spectrum.line.amplitude == -0.2
    continuum_at_center = absorption_spectrum.continuum.slope * 5890.0 +
                          absorption_spectrum.continuum.intercept
    @test render(absorption_spectrum, 5890.0) < continuum_at_center
end

@testitem "zoo: 2D galaxy line profiles evaluate and constrain physical params" tags=[:zoo, :twod] begin
    using AstroFit
    using Accessors: @set

    g = GalaxyGaussianLineProfile2D(amplitude=3.0, x0=1.0, y0=-2.0,
                                    sigma_x=2.0, sigma_y=1.0,
                                    theta=π / 4)
    @test g isa CompiledModel
    @test ndims(getfield(g, :model)) == 2
    @test nfree(g) == 6
    @test render(g, 1.0, -2.0) ≈ 3.0
    @test render(g, 3.0, -2.0) < render(g, 1.0, -2.0)
    @test_throws ArgumentError @set g.amplitude = -1.0
    @test_throws ArgumentError @set g.sigma_x = -0.1
    @test_throws ArgumentError GalaxyGaussianLineProfile2D(sigma_y=0.0)

    d = GalaxyExponentialLineProfile2D(amplitude=5.0, x0=0.0, y0=0.0,
                                       scale_radius=2.0, axis_ratio=0.5)
    @test d isa CompiledModel
    @test ndims(getfield(d, :model)) == 2
    @test nfree(d) == 6
    @test render(d, 0.0, 0.0) ≈ 5.0
    @test render(d, 2.0, 0.0) ≈ 5.0 * exp(-1)
    @test render(d, 0.0, 1.0) ≈ 5.0 * exp(-1)
    @test_throws ArgumentError @set d.axis_ratio = 0.0
    @test_throws ArgumentError @set d.axis_ratio = 1.5
    @test_throws ArgumentError GalaxyExponentialLineProfile2D(axis_ratio=0.0)
end

@testitem "zoo: 2D line profiles compose inside @model" tags=[:zoo, :twod, :prefab] begin
    using AstroFit

    scene = @model begin
        bulge = GalaxyGaussianLineProfile2D(amplitude=2.0, sigma_x=1.0,
                                            sigma_y=1.0)
        disk = GalaxyExponentialLineProfile2D(amplitude=1.0,
                                              scale_radius=3.0,
                                              axis_ratio=0.8)
        bulge + disk
    end

    @test scene.bulge.amplitude == 2.0
    @test scene.disk.axis_ratio == 0.8
    @test render(scene, 0.0, 0.0) ≈ 3.0
    @test nfree(scene) == 12

    lo, hi = bounds_vectors(scene.spec)
    @test count(==(0.0), lo) == 5      # amplitudes, Gaussian widths, disk scale
    @test eps(Float64) in lo
    @test 1.0 in hi                    # disk axis-ratio upper bound
end
