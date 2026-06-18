# Demonstrate a redshift model that scales both wavelength and flux.
#
# For flux density per unit wavelength:
#   lambda_rest = lambda_obs / (1 + z)
#   F_lambda_obs(lambda_obs) = F_lambda_rest(lambda_rest) / (1 + z)
#
# Run with:
#   julia --project=/home/matteo/.julia/environments/v1.12 examples/redshift_axis_flux_scaling.jl

using AstroFit
using Plots

Base.@kwdef struct RedshiftAxis{T<:Real} <: AbstractModel{1}
    z::T = 0.0
end
RedshiftAxis(z::Real) = RedshiftAxis{typeof(float(z))}(float(z))
AstroFit.render(m::RedshiftAxis, λ::Number) = λ / (1 + m.z)

Base.@kwdef struct RedshiftFlux{T<:Real} <: AbstractModel{1}
    z::T = 0.0
end
RedshiftFlux(z::Real) = RedshiftFlux{typeof(float(z))}(float(z))
AstroFit.render(m::RedshiftFlux, λ::Number) = inv(1 + m.z)

rest_spec = @model begin
    cont = Const1D(value = 0.15)
    line = Gaussian1D(amplitude = 1.0, mean = 5000.0, sigma = 35.0)
    wing = Gaussian1D(amplitude = 0.35, mean = 5150.0, sigma = 90.0)
    cont + line + wing
end

observed_template = @model begin
    rest             = rest_spec
    wavelength_shift = RedshiftAxis(z = 0.0)
    flux_scale       = RedshiftFlux(z = 0.0)

    (rest ∘ wavelength_shift) * flux_scale
end

observed_template = @constrain observed_template begin
    @fix   rest.cont.value
    @fix   rest.line.amplitude
    @fix   rest.line.mean
    @fix   rest.line.sigma
    @fix   rest.wing.amplitude
    @fix   rest.wing.mean
    @fix   rest.wing.sigma
    @bound wavelength_shift.z in (0.0, 0.5)
    @tie   flux_scale.z = wavelength_shift.z
end

λ_rest = collect(range(4800.0, 5450.0; length = 1000))
z_values = [0.0, 0.05, 0.10, 0.20]

rest_flux = render(rest_spec, λ_rest)
λ_peak_rest = 5000.0
F_peak_rest = render(rest_spec, λ_peak_rest)

plt = plot(
    layout = (2, 1),
    size = (1050, 760),
    legend = :topright,
    left_margin = 8Plots.mm,
    bottom_margin = 6Plots.mm,
)

plot!(
    plt[1],
    λ_rest,
    rest_flux;
    label = "rest frame",
    color = :black,
    linewidth = 3,
    xlabel = "wavelength [Angstrom]",
    ylabel = "F_lambda",
    title = "Redshifted spectrum: axis stretch and flux-density scaling",
)

palette = [:steelblue, :seagreen, :darkorange, :crimson]
for (i, z) in enumerate(z_values)
    observed = @set observed_template.wavelength_shift.z = z

    λ_obs = λ_rest .* (1 + z)
    flux_obs = render(observed, λ_obs)

    expected_peak_λ = λ_peak_rest * (1 + z)
    expected_peak_F = F_peak_rest / (1 + z)
    measured_peak_F = render(observed, expected_peak_λ)

    @assert observed.flux_scale.z == observed.wavelength_shift.z
    @assert isapprox(measured_peak_F, expected_peak_F; rtol = 1e-12)

    label = "z=$(z), peak -> $(round(expected_peak_λ; digits=1)) A, F/(1+z)"

    plot!(
        plt[1],
        λ_obs,
        flux_obs;
        label,
        color = palette[i],
        linewidth = 2.5,
    )

    scatter!(
        plt[1],
        [expected_peak_λ],
        [expected_peak_F];
        label = false,
        color = palette[i],
        markersize = 4,
    )

    plot!(
        plt[2],
        λ_rest,
        flux_obs .* (1 + z);
        label = "z=$(z) corrected",
        color = palette[i],
        linewidth = 2,
        linestyle = i == 1 ? :solid : :dash,
        xlabel = "lambda_obs / (1+z) [Angstrom]",
        ylabel = "(1+z) F_lambda_obs",
        title = "Undoing the transform recovers the rest-frame flux density",
    )
end

plot!(
    plt[2],
    λ_rest,
    rest_flux;
    label = "rest frame",
    color = :black,
    linewidth = 3,
    alpha = 0.75,
)

outpath = joinpath(@__DIR__, "redshift_axis_flux_scaling.png")
savefig(plt, outpath)

println("saved plot -> ", outpath)
println("free parameters in observed template: ", nfree(observed_template))
println("tie check at z=0.20: flux_scale.z = ",
        (@set observed_template.wavelength_shift.z = 0.20).flux_scale.z)
