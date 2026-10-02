@testitem "recipes come with physical bounds and ties" tags = [:zoo, :prefab, :tied] begin
    using AstroFit

    em = emission_line(center = 6563.0)
    @test paramnames(em) == [:line_amplitude, :line_mean, :line_sigma]
    @test bounds(em) == ([0.0, 6561.0, 0.0], [Inf, 6565.0, Inf])

    ab = absorption_line(center = 5890.0)
    @test bounds(ab) == ([-Inf, 5888.0, 0.0], [0.0, 5892.0, Inf])
    @test absorption_line(center = 5890.0, amplitude = 0.5).line.model.amplitude == -0.5
    @test absorption_line(center = 5890.0, amplitude = -0.5).line.model.amplitude == -0.5  # a dip, whatever the sign

    # [O III]: one free line; the red one follows at 2.98× the flux, the same velocity
    # (λ scales with the rest-wavelength ratio) and the same width.
    oiii = doublet(blue_center = 4959.0, red_center = 5007.0)
    @test paramnames(oiii) == [:blue_amplitude, :blue_mean, :blue_sigma]
    red = withparams(oiii, [2.0, 4960.0, 3.0]).red.model
    @test red.amplitude ≈ 2.98 * 2.0
    @test red.mean ≈ 5007 / 4959 * 4960.0
    @test red.sigma == 3.0

    pl = powerlaw_continuum(x_ref = 5000.0)
    @test paramnames(pl) == [:pl_norm, :pl_index]  # x_ref only sets the units
    @test bounds(pl) == ([0.0, -Inf], [Inf, Inf])

    bb = blackbody_continuum()
    @test bounds(bb) == ([0.0, 0.0], [Inf, Inf])
end
