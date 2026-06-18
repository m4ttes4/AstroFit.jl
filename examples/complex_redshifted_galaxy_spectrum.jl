# Complex galaxy-spectrum example with redshifted wavelength and flux density.
#
# Model:
#   continuum + Hbeta + [OIII] doublet + Halpha + [NII] doublet
#
# For flux density per unit wavelength:
#   lambda_rest = lambda_obs / (1 + z)
#   F_lambda_obs(lambda_obs) = F_lambda_rest(lambda_rest) / (1 + z)
#
# Run with:
#   julia --project=/home/matteo/.julia/environments/v1.12 examples/complex_redshifted_galaxy_spectrum.jl

using AstroFit
using Plots

const L_HB = 4861.3
const L_OIII_B = 4958.9
const L_OIII_R = 5006.8
const L_HA = 6562.8
const L_NII_B = 6548.1
const L_NII_R = 6583.4

Base.@kwdef struct RedshiftAxis{T<:Real} <: AbstractModel{1}
    z::T = 0.0
end
RedshiftAxis(z::Real) = RedshiftAxis{typeof(float(z))}(float(z))
AstroFit.render(m::RedshiftAxis, lambda::Number) = lambda / (1 + m.z)

Base.@kwdef struct RedshiftFlux{T<:Real} <: AbstractModel{1}
    z::T = 0.0
end
RedshiftFlux(z::Real) = RedshiftFlux{typeof(float(z))}(float(z))
AstroFit.render(m::RedshiftFlux, lambda::Number) = inv(1 + m.z)

# Rest-frame galaxy spectrum. The tied parameters encode common physical
# assumptions: shared narrow-line kinematics and fixed atomic doublet ratios.
rest_spec = @model begin
    cont = Linear1D(slope = -2.0e-5, intercept = 1.15)

    hbeta = Gaussian1D(amplitude = 2.4, mean = L_HB, sigma = 2.8)

    oiii_b = Gaussian1D(amplitude = 1.6, mean = L_OIII_B, sigma = 2.8)
    oiii_r = Gaussian1D(amplitude = 4.8, mean = L_OIII_R, sigma = 2.8)

    ha = Gaussian1D(amplitude = 8.5, mean = L_HA, sigma = 3.2)
    nii_b = Gaussian1D(amplitude = 1.0, mean = L_NII_B, sigma = 3.2)
    nii_r = Gaussian1D(amplitude = 3.1, mean = L_NII_R, sigma = 3.2)

    cont + hbeta + oiii_b + oiii_r + ha + nii_b + nii_r
end

rest_spec = @constrain rest_spec begin
    @bound cont.intercept in (0.0, Inf)

    @bound ha.amplitude in (0.0, Inf)
    @bound ha.mean      in (L_HA - 20.0, L_HA + 20.0)
    @bound ha.sigma     in (0.3, 20.0)

    @tie nii_b.amplitude = ha.amplitude / 3.0
    @tie nii_r.amplitude = (3.06 / 3.0) * ha.amplitude
    @tie nii_b.mean      = (L_NII_B / L_HA) * ha.mean
    @tie nii_r.mean      = (L_NII_R / L_HA) * ha.mean
    @tie nii_b.sigma     = ha.sigma
    @tie nii_r.sigma     = ha.sigma

    @tie hbeta.amplitude = ha.amplitude / 2.86
    @tie hbeta.mean      = (L_HB / L_HA) * ha.mean
    @tie hbeta.sigma     = ha.sigma

    @bound oiii_r.amplitude in (0.0, Inf)
    @bound oiii_r.mean      in (L_OIII_R - 20.0, L_OIII_R + 20.0)
    @bound oiii_r.sigma     in (0.3, 20.0)

    @tie oiii_b.amplitude = oiii_r.amplitude / 2.98
    @tie oiii_b.mean      = (L_OIII_B / L_OIII_R) * oiii_r.mean
    @tie oiii_b.sigma     = oiii_r.sigma
end

# Observed-frame wrapper: one redshift controls both the wavelength transform
# and the F_lambda scaling.
observed_template = @model begin
    rest = rest_spec
    wavelength_shift = RedshiftAxis(z = 0.0)
    flux_scale = RedshiftFlux(z = 0.0)

    (rest ∘ wavelength_shift) * flux_scale
end

observed_template = @constrain observed_template begin
    @bound wavelength_shift.z in (0.0, 1.0)
    @tie   flux_scale.z = wavelength_shift.z
end

lambda_rest = collect(range(4700.0, 6750.0; length = 5000))
rest_flux = render(rest_spec, lambda_rest)

z_values = [0.0, 0.1, 0.2, 0.5]
palette = cgrad(:viridis, length(z_values), categorical = true)

plt = plot(
    size = (1250, 620),
    legend = :topright,
    background_color = :white,
    foreground_color = :gray15,
    gridcolor = :gray88,
    framestyle = :box,
    left_margin = 9Plots.mm,
    bottom_margin = 7Plots.mm,
    right_margin = 5Plots.mm,
    top_margin = 5Plots.mm,
)

plot!(
    plt,
    lambda_rest,
    rest_flux;
    label = "rest frame",
    color = :black,
    linewidth = 3.0,
    xlabel = "observed wavelength [Angstrom]",
    ylabel = "observed F_lambda",
    guidefontsize = 12,
    tickfontsize = 10,
    legendfontsize = 10,
)

for (i, z) in enumerate(z_values)
    observed = @set observed_template.wavelength_shift.z = z
    lambda_obs = lambda_rest .* (1 + z)
    flux_obs = render(observed, lambda_obs)

    # Exact checks for the redshift wrapper and tie.
    @assert observed.flux_scale.z == observed.wavelength_shift.z
    @assert isapprox(
        render(observed, L_HA * (1 + z)),
        render(rest_spec, L_HA) / (1 + z);
        rtol = 1e-12,
    )

    label = "z = $(z)"

    plot!(
        plt,
        lambda_obs,
        flux_obs;
        label,
        color = palette[i],
        linewidth = 2.4,
        alpha = 0.95,
    )
end

for line in (L_HB, L_OIII_R, L_HA)
    vline!(plt, [line]; color = :gray35, alpha = 0.16, linewidth = 1.0, label = false)
end

outpath = joinpath(@__DIR__, "complex_redshifted_galaxy_spectrum.png")
savefig(plt, outpath)

println("saved plot -> ", outpath)
println("free parameters in rest-frame constrained spectrum: ", nfree(rest_spec))
println("free parameters in observed model: ", nfree(observed_template))
println("tie check at z=0.12: flux_scale.z = ",
        (@set observed_template.wavelength_shift.z = 0.12).flux_scale.z)
