# Fit a toy, response-folded AGN X-ray spectrum with an XSPEC-like model.
#
# The model is the usual obscured type-2 AGN construction:
#
#     phabs * (zpcfabs * zcutoffpl + Fe Kα + Fe Kβ)
#
# XSPEC definitions:
# - phabs/zphabs: https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/XSmodelPhabs.html
# - zpcfabs: https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/XSmodelPcfabs.html
# - zcutoffpl: https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/XSmodelCutoffpl.html
# - zgauss: https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/node182.html
# - fit statistics: https://heasarc.gsfc.nasa.gov/docs/software/xspec/manual/XSappendixStatistics.html
#
# The important distinction in this example is that the model tree renders a
# photon spectrum (photons keV^-1 cm^-2 s^-1). Counts are produced only after
# folding that spectrum through a toy effective area, exposure time, and RMF.
# The response is deliberately small and analytic, not a substitute for a
# calibrated instrument response or a real pexmon/atable model.
#
# Run with: julia --project=examples examples/other/agn_xray_fit.jl

using AstroFit
using Optimization, OptimizationOptimJL, ForwardDiff
using Distributions: Poisson
using LinearAlgebra
using CairoMakie
using Random

# ---------------------------------------------------------------------------
# 0. XSPEC-like model components
# ---------------------------------------------------------------------------
#
# XSPEC defines phabs as (see the phabs/zphabs reference above)
#
#     M(E) = exp[-N_H sigma(E)]
#
# and zphabs as
#
#     M(E) = exp[-N_H sigma(E (1 + z))].
#
# Here nH is N_H in units of 10^22 cm^-2. XSPEC's sigma(E) is a tabulated,
# abundance-dependent cross-section selected by xsect/abund; it is not one
# universal closed-form E^-3 law. We keep the example dependency-free and use
# a smooth, correctly-scaled approximation with sigma(1 keV) ≈ 3e-22 cm^2/H.
# It gets the broad turnover right, but intentionally does not claim to reproduce
# the detailed absorption edges of real phabs/tbabs.
const NH_UNIT = 1.0e22       # cm^-2
const SIGMA_1KEV = 3.0e-22   # cm^2 per H atom; smooth educational approximation

photoelectric_cross_section(E) = SIGMA_1KEV * (E / 1.0)^(-3)

Base.@kwdef struct PhAbs1D{N <: Real, Z <: Real} <: AbstractModel
    nH::N = 1.0              # equivalent hydrogen column, 10^22 cm^-2
    z::Z = 0.0               # absorber redshift
end

AstroFit.render(m::PhAbs1D, E::Number) = begin
    E_rest = E * (1 + m.z)
    exp(-m.nH * NH_UNIT * photoelectric_cross_section(E_rest))
end

# XSPEC zcutoffpl is a photon spectrum, not a count spectrum (see the official
# zcutoffpl definition above):
#
#     A(E) = K [E (1 + z) / 1 keV]^(-alpha) exp[-E(1+z)/Ecut].
#
# K is the photon flux density at 1 keV in the source-frame convention used by
# this example (photons keV^-1 cm^-2 s^-1).
Base.@kwdef struct ZCutoffPowerLaw1D{N <: Real, I <: Real, C <: Real, Z <: Real} <: AbstractModel
    norm::N = 1.0
    index::I = 1.0           # photon index Gamma
    cutoff::C = 80.0         # source-frame e-folding energy, keV
    z::Z = 0.0
end

AstroFit.render(m::ZCutoffPowerLaw1D, E::Number) = begin
    E_rest = E * (1 + m.z)
    m.norm * E_rest^(-m.index) * exp(-E_rest / m.cutoff)
end

# Partial covering, following XSPEC zpcfabs:
# M(E) = (1-f) + f exp[-N_H sigma(E(1+z))].
Base.@kwdef struct ZPartialCovering1D{N <: Real, F <: Real, Z <: Real} <: AbstractModel
    nH::N = 1.0
    covering::F = 0.9
    z::Z = 0.0
end

AstroFit.render(m::ZPartialCovering1D, E::Number) = begin
    E_rest = E * (1 + m.z)
    transmission = exp(-m.nH * NH_UNIT * photoelectric_cross_section(E_rest))
    (1 - m.covering) + m.covering * transmission
end

# XSPEC zgauss (with the positive-energy truncation omitted here because the Fe
# line
# is many sigma above E=0 here. `norm` is the integrated observed line flux
# (photons cm^-2 s^-1), and sigma is in the source frame. We use the physically
# normalized observed-energy profile here, so its integral is `norm`:
#
#     A(E) ≈ K (1+z) / [sigma sqrt(2pi)]
#            exp[-(E(1+z) - E_l)^2 / (2 sigma^2)].
Base.@kwdef struct ZGaussianLine1D{K <: Real, E <: Real, S <: Real, Z <: Real} <: AbstractModel
    norm::K = 1.0e-5        # observed integrated photons cm^-2 s^-1
    line_energy::E = 6.4    # source-frame keV
    sigma::S = 0.1          # source-frame keV
    z::Z = 0.0
end

AstroFit.render(m::ZGaussianLine1D, E::Number) = begin
    u = (E * (1 + m.z) - m.line_energy) / m.sigma
    m.norm * (1 + m.z) / (m.sigma * sqrt(2pi)) * exp(-u^2 / 2)
end

# ---------------------------------------------------------------------------
# 1. Toy detector response and deterministic grouping
# ---------------------------------------------------------------------------
#
# A real XSPEC fit evaluates the photon model through an ARF (effective area)
# and RMF (energy redistribution). The RMF below is a Gaussian energy response;
# it is enough to make the line visibly instrument-broadened without a FITS
# response dependency.
const EXPOSURE = 50_000.0   # s

effective_area(E) = begin
    low_energy_cutoff = 1 - exp(-(E / 0.45)^4)
    50.0 + 900.0 * low_energy_cutoff * exp(-0.045E)
end

native_edges = collect(range(0.3, 30.0; step = 0.05))
native_centers = (native_edges[1:end-1] .+ native_edges[2:end]) ./ 2
native_widths = diff(native_edges)

function toy_rmf(energies)
    response = zeros(length(energies), length(energies))
    for j in eachindex(energies)
        fwhm = 0.08 + 0.015 * sqrt(energies[j])
        sigma = fwhm / 2.355
        weights = exp.(-0.5 .* ((energies .- energies[j]) ./ sigma) .^ 2)
        response[:, j] .= weights ./ sum(weights)
    end
    return response
end

const RMF = toy_rmf(native_centers)

# Grouping sums adjacent detector channels. A sum of independent Poisson
# counts is still Poisson, so grouping does NOT by itself turn the likelihood
# into a Gaussian one. The expected counts must be summed over the same native
# channels; evaluating the model only at the grouped midpoint is not equivalent
# around a sharp absorption turnover or line.
group_size = 4
groups = [
    first:min(first + group_size - 1, length(native_centers)) for
    first in 1:group_size:length(native_centers)
]
group_centers = [
    (native_edges[first(g)] + native_edges[last(g) + 1]) / 2 for g in groups
]

function folded_group_counts(model)
    # photons keV^-1 cm^-2 s^-1 × cm^2 × s × keV = expected incident counts
    incident = render(model, native_centers) .* effective_area.(native_centers) .* EXPOSURE .* native_widths
    detected = RMF * incident
    return [sum(@view detected[g]) for g in groups]
end

# ---------------------------------------------------------------------------
# 2. True model and synthetic grouped counts
# ---------------------------------------------------------------------------
z_src = 0.05

true_model = @model begin
    gal = PhAbs1D(nH = 0.02, z = 0.0)                  # phabs
    partial = ZPartialCovering1D(nH = 1.5, covering = 0.92, z = z_src)
    direct = ZCutoffPowerLaw1D(norm = 4.0e-3, index = 1.8, cutoff = 80.0, z = z_src)
    fe_kalpha = ZGaussianLine1D(norm = 1.5e-5, line_energy = 6.4, sigma = 0.08, z = z_src)
    fe_kbeta = ZGaussianLine1D(norm = 0.113 * 1.5e-5, line_energy = 7.06, sigma = 0.08, z = z_src)
    gal * (partial * direct + fe_kalpha + fe_kbeta)
end

background_native = 0.12 .+ 0.01 .* (native_centers ./ 10) .^ 2
background = [sum(@view background_native[g]) for g in groups]
λ_true = folded_group_counts(true_model) .+ background
Random.seed!(7)
grouped_counts = [rand(Poisson(λ)) for λ in λ_true]

# ---------------------------------------------------------------------------
# 3. Fitting model — same physics, intentionally displaced initial values
# ---------------------------------------------------------------------------
cm = @model begin
    gal = PhAbs1D(nH = 0.02, z = 0.0)
    partial = ZPartialCovering1D(nH = 0.8, covering = 0.75, z = z_src)
    direct = ZCutoffPowerLaw1D(norm = 2.0e-3, index = 1.5, cutoff = 50.0, z = z_src)
    fe_kalpha = ZGaussianLine1D(norm = 5.0e-6, line_energy = 6.4, sigma = 0.18, z = z_src)
    fe_kbeta = ZGaussianLine1D(norm = 0.113 * 5.0e-6, line_energy = 7.06, sigma = 0.18, z = z_src)
    gal * (partial * direct + fe_kalpha + fe_kbeta)
end

@fix cm.gal.nH = 0.02
@fix cm.gal.z = 0.0
@fix cm.partial.z = z_src
@fix cm.direct.z = z_src
@fix cm.direct.cutoff = 80.0
@fix cm.fe_kalpha.line_energy = 6.4
@fix cm.fe_kalpha.z = z_src
@fix cm.fe_kbeta.line_energy = 7.06
@fix cm.fe_kbeta.z = z_src
@fix cm.fe_kbeta.sigma = 0.08

@tie cm.fe_kbeta.norm -> 0.113 * cm.fe_kalpha.norm

@bound cm.partial.nH in (0.01, 20.0)
@bound cm.partial.covering in (0.01, 1.0)
@bound cm.direct.norm in (1.0e-5, 0.1)
@bound cm.direct.index in (0.5, 3.5)
@bound cm.fe_kalpha.norm in (1.0e-8, 1.0e-3)
@fix cm.fe_kalpha.sigma = 0.08

# ---------------------------------------------------------------------------
# 4. Cash/C statistic on grouped Poisson counts
# ---------------------------------------------------------------------------
#
# XSPEC's cstat is twice the negative Poisson log-likelihood, up to terms that
# depend only on the observed data:
#
#     C = 2 sum [ mu - n + n log(n / mu) ],
#
# with the n=0 contribution defined as 2mu. The grouped data above are still
# integer counts. The model expectations mu are floats, as they should be.
# Here `background` is a known detector background, so this is the pstat-like
# case: the observed source-region counts are Poisson with mean source + bg.
# A measured background spectrum with its own uncertainty needs a joint/profile
# likelihood (XSPEC W/pgstat), which is deliberately not added to this example.
cash_term(mu, n) = n == 0 ? 2mu : 2 * (mu - n + n * (log(n) - log(mu)))

cash_statistic(model, y) = begin
    mu = folded_group_counts(model) .+ background
    sum(i -> cash_term(mu[i], y[i]), eachindex(y))
end

cash_statistic(f::ObjectiveFunction, p) = cash_statistic(withparams(f.cm, p), f.y)

# Optimization.jl minimizes, so the statistic is already the quantity to pass
# directly: C is not a log-likelihood with a sign that needs another negation.
initial_cstat = cash_statistic(cm, grouped_counts)
@assert isfinite(initial_cstat)
prob = OptimizationProblem(cm, group_centers, grouped_counts; statistic = cash_statistic)
sol = solve(prob, LBFGS())
fit_tree = withparams(cm, sol.u)
fit_cstat = cash_statistic(fit_tree, grouped_counts)
@assert fit_cstat < initial_cstat "optimizer did not improve the initial C-stat"

println("retcode         : ", sol.retcode)
println("free parameters : ", nfree(cm))
println("parameter names : ", paramnames(cm))
println("best fit values : ", round.(sol.u; digits = 6))
println("C-stat initial  : ", round(initial_cstat; digits = 3))
println("C-stat at fit   : ", round(fit_cstat; digits = 3))
println("grouped bins    : ", length(grouped_counts), " (", group_size, " native channels/group)")
println()

# ---------------------------------------------------------------------------
# 5. Plot folded counts, not the un-folded photon spectrum
# ---------------------------------------------------------------------------
# Gehrels intervals are only display intervals. They are not supplied to the
# C-stat objective, which uses the exact Poisson likelihood for the counts.
gehrels_upper(n) = n + 1 + sqrt(n + 0.75)
gehrels_lower(n) = n == 0 ? 0.0 : n * (1 - 1 / (9n) - 1 / (3 * sqrt(n)))^3
err_hi = [gehrels_upper(n) - n for n in grouped_counts]
err_lo = [n - gehrels_lower(n) for n in grouped_counts]

lambda_init = folded_group_counts(cm) .+ background
lambda_fit = folded_group_counts(fit_tree) .+ background

floor_for_log = 0.5
detected = grouped_counts .> 0

fig = Figure(size = (900, 500))
ax = Axis(
    fig[1, 1]; xlabel = "Observed energy (keV)", ylabel = "Counts / grouped channel",
    xscale = log10, yscale = log10,
    limits = (nothing, nothing, floor_for_log, nothing),
    title = "partial-covering AGN + cutoff + Fe K complex, grouped C-stat fit"
)

errorbars!(
    ax, group_centers[detected], grouped_counts[detected],
    err_lo[detected], err_hi[detected]; color = :grey60, whiskerwidth = 3
)
scatter!(
    ax, group_centers[detected], grouped_counts[detected]; color = :grey60,
    markersize = 4, label = "data (source + known background, Gehrels 1σ)"
)
lines!(ax, group_centers, max.(λ_true, floor_for_log); color = :black, linestyle = :dash, label = "truth")
lines!(ax, group_centers, max.(lambda_init, floor_for_log); color = :dodgerblue, linestyle = :dot, label = "initial guess")
lines!(ax, group_centers, max.(lambda_fit, floor_for_log); color = :red, linewidth = 2, label = "best fit")

axislegend(ax; position = :lb)

display(fig)
save("examples/other/agn_xray_fit.png", fig; px_per_unit = 2)
println("saved → examples/other/agn_xray_fit.png")
