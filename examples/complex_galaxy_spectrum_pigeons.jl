# Bayesian fit of a redshifted AGN host-galaxy spectrum with Pigeons.jl.
#
# Same physical model as complex_galaxy_spectrum_fit.jl (tied emission-line
# physics, redshift, 43 raw fields → ~15 free parameters), but sampled with
# parallel tempering instead of gradient-based optimisation.
#
# Run with:  julia --project=. examples/complex_galaxy_spectrum_pigeons.jl

using AstroFit
using Distributions
using Pigeons
using CairoMakie
using Random
using Statistics

const L_HB = 4861.33
const L_OIII_B = 4958.91
const L_OIII_R = 5006.84
const L_NAD_D2 = 5889.95
const L_NAD_D1 = 5895.92
const L_HA = 6562.8
const L_NII_B = 6548.05
const L_NII_R = 6583.45
const L_SII_B = 6716.44
const L_SII_R = 6730.82
const L_REF = 5500.0

Base.@kwdef struct RedshiftAxis1D{T <: Real} <: AbstractModel
    z::T = 0.0
end

AstroFit.render(m::RedshiftAxis1D, lambda::Number) = lambda / (1 + m.z)

Base.@kwdef struct RedshiftFluxScale1D{T <: Real} <: AbstractModel
    z::T = 0.0
end

AstroFit.render(m::RedshiftFluxScale1D, lambda::Number) = inv(1 + m.z)

function galaxy_spectrum_model(;
        z = 0.036,
        cont_slope = -1.0e-4,
        cont_intercept = 1.2,
        pl_norm = 0.75,
        pl_index = 1.1,
        ha_amplitude = 8.0,
        narrow_sigma = 4.0,
        broad_ha_amplitude = 3.5,
        broad_sigma = 24.0,
        oiii_blue_amplitude = 1.9,
        nii_blue_amplitude = 0.75,
        sii_blue_amplitude = 0.9,
        sii_red_amplitude = 0.75,
        nad_d2_amplitude = -0.22,
        nad_sigma = 1.1
    )

    cm = @model begin
        cont = Linear1D(slope = cont_slope, intercept = cont_intercept)
        stellar = PowerLaw1D(norm = pl_norm, x_ref = L_REF, index = pl_index)

        hbeta = Gaussian1D(amplitude = ha_amplitude / 2.86, mean = L_HB, sigma = narrow_sigma)
        broad_hbeta = Gaussian1D(amplitude = broad_ha_amplitude / 3.1, mean = L_HB, sigma = broad_sigma)
        oiii_b = Gaussian1D(amplitude = oiii_blue_amplitude, mean = L_OIII_B, sigma = narrow_sigma)
        oiii_r = Gaussian1D(amplitude = 2.98 * oiii_blue_amplitude, mean = L_OIII_R, sigma = narrow_sigma)

        ha = Gaussian1D(amplitude = ha_amplitude, mean = L_HA, sigma = narrow_sigma)
        broad_ha = Gaussian1D(amplitude = broad_ha_amplitude, mean = L_HA, sigma = broad_sigma)
        nii_b = Gaussian1D(amplitude = nii_blue_amplitude, mean = L_NII_B, sigma = narrow_sigma)
        nii_r = Gaussian1D(amplitude = 3.06 * nii_blue_amplitude, mean = L_NII_R, sigma = narrow_sigma)
        sii_b = Gaussian1D(amplitude = sii_blue_amplitude, mean = L_SII_B, sigma = narrow_sigma)
        sii_r = Gaussian1D(amplitude = sii_red_amplitude, mean = L_SII_R, sigma = narrow_sigma)

        nad_d2 = Gaussian1D(amplitude = nad_d2_amplitude, mean = L_NAD_D2, sigma = nad_sigma)
        nad_d1 = Gaussian1D(amplitude = 0.65 * nad_d2_amplitude, mean = L_NAD_D1, sigma = nad_sigma)

        redshift = RedshiftAxis1D(z = z)
        flux_scale = RedshiftFluxScale1D(z = z)

        (
            (
                cont + stellar + hbeta + broad_hbeta + oiii_b + oiii_r + ha +
                    broad_ha + nii_b + nii_r + sii_b + sii_r + nad_d2 + nad_d1
            ) ∘ redshift
        ) * flux_scale
    end

    @constrain cm begin
        cont.slope in (-5.0e-4, 2.0e-4)
        cont.intercept in (0.0, 4.0)
        stellar.norm in (0.0, 6.0)
        stellar.x_ref
        stellar.index in (0.0, 5.0)

        hbeta.amplitude -> ha.amplitude / 2.86
        hbeta.mean
        hbeta.sigma -> ha.sigma
        broad_hbeta.amplitude -> broad_ha.amplitude / 3.1
        broad_hbeta.mean
        broad_hbeta.sigma -> broad_ha.sigma

        oiii_b.amplitude in (0.0, 20.0)
        oiii_b.mean
        oiii_b.sigma -> ha.sigma
        oiii_r.amplitude -> 2.98 * oiii_b.amplitude
        oiii_r.mean
        oiii_r.sigma -> ha.sigma

        ha.amplitude in (0.0, 40.0)
        ha.mean
        ha.sigma in (1.0, 12.0)
        broad_ha.amplitude in (0.0, 20.0)
        broad_ha.mean
        broad_ha.sigma in (10.0, 80.0)

        nii_b.amplitude in (0.0, 15.0)
        nii_b.mean
        nii_b.sigma -> ha.sigma
        nii_r.amplitude -> 3.06 * nii_b.amplitude
        nii_r.mean
        nii_r.sigma -> ha.sigma

        sii_b.amplitude in (0.0, 15.0)
        sii_b.mean
        sii_b.sigma -> ha.sigma
        sii_r.amplitude in (0.0, 15.0)
        sii_r.mean
        sii_r.sigma -> ha.sigma

        nad_d2.amplitude in (-5.0, 0.0)
        nad_d2.mean
        nad_d2.sigma in (0.2, 5.0)
        nad_d1.amplitude -> 0.65 * nad_d2.amplitude
        nad_d1.mean
        nad_d1.sigma -> nad_d2.sigma

        redshift.z in (0.03, 0.055)
        flux_scale.z -> redshift.z
    end

    return cm
end

function slot(pnames, p, name)
    i = findfirst(==(name), pnames)
    i === nothing && error("no fitted parameter named $name")
    return p[i]
end

function fitted_components(cm, p, lambda_obs)
    names = paramnames(cm)
    get(name) = slot(names, p, name)

    z = get(:redshift_z)
    lambda_rest = lambda_obs ./ (1 + z)
    scale = inv(1 + z)

    cont = Linear1D(slope = get(:cont_slope), intercept = get(:cont_intercept))
    stellar = PowerLaw1D(norm = get(:stellar_norm), x_ref = L_REF, index = get(:stellar_index))

    ha_amp = get(:ha_amplitude)
    sigma = get(:ha_sigma)
    broad_ha_amp = get(:broad_ha_amplitude)
    broad_sigma = get(:broad_ha_sigma)
    oiii_b_amp = get(:oiii_b_amplitude)
    nii_b_amp = get(:nii_b_amplitude)
    sii_b_amp = get(:sii_b_amplitude)
    sii_r_amp = get(:sii_r_amplitude)
    nad_d2_amp = get(:nad_d2_amplitude)
    nad_sigma = get(:nad_d2_sigma)

    continuum = scale .* render(cont + stellar, lambda_rest)
    balmer = scale .* render(
        Gaussian1D(amplitude = ha_amp / 2.86, mean = L_HB, sigma = sigma) +
            Gaussian1D(amplitude = ha_amp, mean = L_HA, sigma = sigma),
        lambda_rest
    )
    broad_balmer = scale .* render(
        Gaussian1D(amplitude = broad_ha_amp / 3.1, mean = L_HB, sigma = broad_sigma) +
            Gaussian1D(amplitude = broad_ha_amp, mean = L_HA, sigma = broad_sigma),
        lambda_rest
    )
    forbidden = scale .* render(
        Gaussian1D(amplitude = oiii_b_amp, mean = L_OIII_B, sigma = sigma) +
            Gaussian1D(amplitude = 2.98 * oiii_b_amp, mean = L_OIII_R, sigma = sigma) +
            Gaussian1D(amplitude = nii_b_amp, mean = L_NII_B, sigma = sigma) +
            Gaussian1D(amplitude = 3.06 * nii_b_amp, mean = L_NII_R, sigma = sigma) +
            Gaussian1D(amplitude = sii_b_amp, mean = L_SII_B, sigma = sigma) +
            Gaussian1D(amplitude = sii_r_amp, mean = L_SII_R, sigma = sigma),
        lambda_rest
    )
    absorption = scale .* render(
        Gaussian1D(amplitude = nad_d2_amp, mean = L_NAD_D2, sigma = nad_sigma) +
            Gaussian1D(amplitude = 0.65 * nad_d2_amp, mean = L_NAD_D1, sigma = nad_sigma),
        lambda_rest
    )

    return (; z, continuum, balmer, broad_balmer, forbidden, absorption)
end

# ---------------------------------------------------------------------------
# Synthetic data (same truth as the optimisation example)
# ---------------------------------------------------------------------------
Random.seed!(31415)

truth_cm = galaxy_spectrum_model(
    z = 0.041,
    cont_slope = -1.0e-5,
    cont_intercept = 0.55,
    pl_norm = 2.25,
    pl_index = 2.35,
    ha_amplitude = 10.5,
    narrow_sigma = 4.3,
    broad_ha_amplitude = 4.8,
    broad_sigma = 31.0,
    oiii_blue_amplitude = 2.2,
    nii_blue_amplitude = 0.95,
    sii_blue_amplitude = 1.05,
    sii_red_amplitude = 0.82,
    nad_d2_amplitude = -0.32,
    nad_sigma = 0.9,
)

cm = galaxy_spectrum_model(
    z = 0.04,
    cont_slope = -6.0e-5,
    cont_intercept = 0.8,
    pl_norm = 1.8,
    pl_index = 1.8,
    ha_amplitude = 8.0,
    narrow_sigma = 5.2,
    broad_ha_amplitude = 3.8,
    broad_sigma = 24.0,
    oiii_blue_amplitude = 1.6,
    nii_blue_amplitude = 0.7,
    sii_blue_amplitude = 0.7,
    sii_red_amplitude = 0.7,
    nad_d2_amplitude = -0.2,
    nad_sigma = 1.3,
)

lambda = collect(range(4900.0, 7100.0; length = 1400))
truth = withparams(truth_cm, AstroFit.params(truth_cm))
flux_true = render(truth, lambda)
err = 0.055 .+ 0.018 .* sqrt.(clamp.(flux_true, 0.0, Inf))
flux = flux_true .+ err .* randn(length(lambda))

# ---------------------------------------------------------------------------
# Pigeons sampling
# ---------------------------------------------------------------------------
target = ObjectiveFunction(cm, lambda, flux, err; statistic = logposterior)

println("free parameters : ", nfree(cm))
println("parameter names : ", paramnames(cm))
println("\nsampling with Pigeons...")

pt = pigeons(
    target = target,
    n_rounds = 8,
    n_chains = 2,
    # seed = 123,
    record = [traces; record_default()],
)

# ---------------------------------------------------------------------------
# Extract posterior
# ---------------------------------------------------------------------------
raw_samples = sample_array(pt)
param_samples = raw_samples[:, 1:nfree(cm), :]
sample_matrix = reshape(permutedims(param_samples, (1, 3, 2)), :, nfree(cm))
posterior_median = vec(median(sample_matrix; dims = 1))

fit = withparams(cm, posterior_median)
flux_fit = render(fit, lambda)
resid_sigma = (flux .- flux_fit) ./ err

println("\nraw parameters  : 43")
println("free parameters : ", nfree(cm))
println("posterior log density: ", round(target(posterior_median); digits = 3))
println()

names = paramnames(cm)
report(name, truth_value, fit_value) =
    println(
    rpad(String(name), 22), "truth = ", rpad(round(truth_value; digits = 5), 12),
    "median = ", round(fit_value; digits = 5)
)

report(
    :redshift_z, slot(paramnames(truth_cm), params(truth_cm), :redshift_z),
    slot(names, posterior_median, :redshift_z)
)
report(
    :ha_amplitude, slot(paramnames(truth_cm), params(truth_cm), :ha_amplitude),
    slot(names, posterior_median, :ha_amplitude)
)
report(
    :ha_sigma, slot(paramnames(truth_cm), params(truth_cm), :ha_sigma),
    slot(names, posterior_median, :ha_sigma)
)
report(
    :broad_ha_amplitude, slot(paramnames(truth_cm), params(truth_cm), :broad_ha_amplitude),
    slot(names, posterior_median, :broad_ha_amplitude)
)
report(
    :broad_ha_sigma, slot(paramnames(truth_cm), params(truth_cm), :broad_ha_sigma),
    slot(names, posterior_median, :broad_ha_sigma)
)
report(
    :oiii_b_amplitude, slot(paramnames(truth_cm), params(truth_cm), :oiii_b_amplitude),
    slot(names, posterior_median, :oiii_b_amplitude)
)
println("tie check [OIII] 5007/4959 : ", 2.98)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
lambda_plot = collect(range(first(lambda), last(lambda); length = 2200))
fit_plot = render(fit, lambda_plot)
truth_plot = render(truth, lambda_plot)
parts = fitted_components(cm, posterior_median, lambda_plot)

fig = Figure(size = (1550, 760))
ax = Axis(
    fig[1, 1];
    xlabel = "",
    ylabel = "flux density",
    title = "Redshifted galaxy spectrum — Pigeons posterior ($(nfree(cm)) free params)",
)
zax = Axis(
    fig[1, 2];
    xlabel = "",
    ylabel = "",
    title = "Halpha + [NII] zoom",
)

scatter!(ax, lambda, flux; color = (:gray45, 0.45), markersize = 3, label = "data")
lines!(ax, lambda_plot, truth_plot; color = (:black, 0.35), linestyle = :dash, linewidth = 1.6, label = "truth")

n_draw = min(80, size(sample_matrix, 1))
draw_ids = unique(round.(Int, range(1, size(sample_matrix, 1); length = n_draw)))
for i in draw_ids
    y_draw = render(withparams(cm, vec(sample_matrix[i, :])), lambda_plot)
    lines!(ax, lambda_plot, y_draw; color = (:red, 0.04))
end

lines!(ax, lambda_plot, fit_plot; color = :black, linewidth = 2.2, label = "posterior median")
lines!(ax, lambda_plot, parts.continuum; color = :gray35, linestyle = :dot, linewidth = 1.8, label = "continuum")
lines!(ax, lambda_plot, parts.continuum .+ parts.balmer; color = :crimson, linewidth = 1.5, label = "narrow Balmer")
lines!(ax, lambda_plot, parts.continuum .+ parts.broad_balmer; color = :purple3, linewidth = 1.7, label = "broad AGN Balmer")
lines!(ax, lambda_plot, parts.continuum .+ parts.forbidden; color = :dodgerblue3, linewidth = 1.5, label = "forbidden lines")
lines!(ax, lambda_plot, parts.continuum .+ parts.absorption; color = :darkorange3, linewidth = 1.5, label = "Na D absorption")

ha_center_obs = L_HA * (1 + parts.z)
zoom_lo = ha_center_obs - 95.0
zoom_hi = ha_center_obs + 95.0
zoom_data = (zoom_lo .<= lambda) .& (lambda .<= zoom_hi)
zoom_plot = (zoom_lo .<= lambda_plot) .& (lambda_plot .<= zoom_hi)

scatter!(
    zax, lambda[zoom_data], flux[zoom_data]; color = (:gray35, 0.75),
    markersize = 5, label = "data"
)
lines!(
    zax, lambda_plot[zoom_plot], truth_plot[zoom_plot]; color = (:black, 0.35),
    linestyle = :dash, linewidth = 1.8, label = "truth"
)

zoom_draw_ids = unique(round.(Int, range(1, size(sample_matrix, 1); length = min(40, size(sample_matrix, 1)))))
for i in zoom_draw_ids
    y_draw = render(withparams(cm, vec(sample_matrix[i, :])), lambda_plot)
    lines!(zax, lambda_plot[zoom_plot], y_draw[zoom_plot]; color = (:red, 0.06))
end

lines!(
    zax, lambda_plot[zoom_plot], fit_plot[zoom_plot]; color = :black,
    linewidth = 2.8, label = "posterior median"
)
lines!(
    zax, lambda_plot[zoom_plot], parts.continuum[zoom_plot]; color = :gray35,
    linestyle = :dot, linewidth = 2.0, label = "continuum"
)
lines!(
    zax, lambda_plot[zoom_plot], parts.continuum[zoom_plot] .+ parts.balmer[zoom_plot];
    color = :crimson, linewidth = 2.2, label = "narrow Halpha"
)
lines!(
    zax, lambda_plot[zoom_plot], parts.continuum[zoom_plot] .+ parts.broad_balmer[zoom_plot];
    color = :purple3, linewidth = 2.4, label = "broad Halpha"
)
lines!(
    zax, lambda_plot[zoom_plot], parts.continuum[zoom_plot] .+ parts.forbidden[zoom_plot];
    color = :dodgerblue3, linewidth = 2.0, label = "[NII]"
)

line_labels = [
    ("Hbeta", L_HB),
    ("[OIII]", L_OIII_B),
    ("[OIII]", L_OIII_R),
    ("Na D", L_NAD_D2),
    ("Na D", L_NAD_D1),
    ("[NII]", L_NII_B),
    ("Halpha", L_HA),
    ("[NII]", L_NII_R),
    ("[SII]", L_SII_B),
    ("[SII]", L_SII_R),
]

ymax = maximum(fit_plot)
for (i, (label, lambda0)) in enumerate(line_labels)
    lambda_shifted = lambda0 * (1 + parts.z)
    vlines!(ax, [lambda_shifted]; color = (:gray20, 0.25), linewidth = 0.8)
    y_label = ymax * (1.02 + 0.035 * ((i - 1) % 3))
    text!(
        ax, lambda_shifted, y_label; text = label, fontsize = 8,
        align = (:center, :bottom), rotation = pi / 5, color = :gray20
    )
end

axislegend(ax; position = :lt, framevisible = false)
ylims!(ax, nothing, ymax * 1.16)

for (i, (label, lambda0)) in enumerate((("[NII]", L_NII_B), ("Halpha", L_HA), ("[NII]", L_NII_R)))
    lambda_shifted = lambda0 * (1 + parts.z)
    vlines!(zax, [lambda_shifted]; color = (:gray20, 0.28), linewidth = 1.0)
    text!(
        zax, lambda_shifted, maximum(fit_plot[zoom_plot]) * (1.02 + 0.045 * (i - 1));
        text = label, fontsize = 10, align = (:center, :bottom),
        rotation = pi / 5, color = :gray20
    )
end

axislegend(zax; position = :lt, framevisible = false)
xlims!(zax, zoom_lo, zoom_hi)
ylims!(zax, minimum(parts.continuum[zoom_plot]) * 0.92, maximum(fit_plot[zoom_plot]) * 1.22)
colsize!(fig.layout, 1, Relative(0.68))

rax = Axis(
    fig[2, 1:2];
    xlabel = "observed wavelength [Angstrom]",
    ylabel = "residual / sigma",
    height = 160,
)
scatter!(rax, lambda, resid_sigma; color = (:gray35, 0.7), markersize = 2)
hlines!(rax, [0.0]; color = :black, linewidth = 1)
hlines!(rax, [-3.0, 3.0]; color = (:red, 0.35), linestyle = :dash, linewidth = 1)
linkxaxes!(ax, rax)
rowsize!(fig.layout, 2, Relative(0.24))
display(fig)
outpath = joinpath(@__DIR__, "complex_galaxy_spectrum_pigeons.png")
save(outpath, fig; px_per_unit = 2)
println("saved -> ", outpath)
