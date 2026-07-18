# Fit an absorbed AGN X-ray spectrum — Galactic + intrinsic photoelectric
# absorption on a power-law continuum plus a Fe Kα line — to synthetic
# photon-counting data, using a custom Poisson statistic. Mirrors the
# classic XSPEC combination `phabs * zphabs * powerlaw` used for obscured
# (type 2) AGN.
#
# Run with:  julia --project=. examples/agn_xray_fit.jl

using AstroFit
using Optimization, OptimizationOptimJL, ForwardDiff
using Distributions: Poisson, logpdf
using CairoMakie
using Random

# ---------------------------------------------------------------------------
# 0. Photoelectric absorption — a user-defined AbstractModel, no changes to
#    AstroFit itself needed (only a @kwdef struct + scalar `render`, exactly
#    how zoo models are written). Simplified E^-3 cross-section — real XSPEC
#    `phabs` uses tabulated Wisconsin/Verner cross-sections; this toy form
#    reproduces the qualitative turnover shape, not the precise edges.
#    `z` shifts the absorber into its own rest frame: z=0 is plain `phabs`
#    (Galactic, foreground); z>0 is `zphabs` (intrinsic, at the AGN itself).
# ---------------------------------------------------------------------------
Base.@kwdef struct PhAbs1D{T <: Real} <: AbstractModel
    nH::T = 1.0     # column density, units of 1e22 cm^-2
    z::T = 0.0      # redshift of the absorbing material
    e0::T = 1.0     # keV, cross-section reference energy
end

# Qualified as AstroFit.render (not bare render): `using AstroFit` brings the
# name into scope for calling, but a bare `render(...) = ...` here would define
# a new Main.render shadowing it, invisible to the Sum/Product dispatch chain.
AstroFit.render(m::PhAbs1D, E::Number) = exp(-m.nH * ((E * (1 + m.z)) / m.e0)^(-3))

# ---------------------------------------------------------------------------
# 1. True model — phabs (Galactic) * zphabs (intrinsic) * (powerlaw + Fe Kα)
# ---------------------------------------------------------------------------
# Energy in keV, model output = expected photon counts per bin.
z_src = 0.05                        # source redshift, known from optical spectroscopy
fe_energy_obs = 6.4 / (1 + z_src)   # rest-frame 6.4 keV, redshifted into the observed frame

true_model = @model begin
    gal = PhAbs1D(nH = 0.02, z = 0.0)          # Galactic foreground, thin
    intrinsic = PhAbs1D(nH = 5.0, z = z_src)   # obscured AGN (Seyfert 2-like)
    cont = PowerLaw1D(norm = 8.0, x_ref = 1.0, index = 1.8)   # Γ=1.8
    fe_line = Gaussian1D(amplitude = 3.0, mean = fe_energy_obs, sigma = 0.15)
    gal * intrinsic * (cont + fe_line)
end

# ---------------------------------------------------------------------------
# 2. Synthetic data — Poisson counts, not Gaussian noise
# ---------------------------------------------------------------------------
Random.seed!(7)
E = collect(0.3:0.05:10.0)
λ_true = render(true_model, E)
counts = [rand(Poisson(λ)) for λ in λ_true]

# ---------------------------------------------------------------------------
# 3. Fitting model — off initial guess. Galactic nH/z and the reference
#    energies are known independently (HI maps, optical redshift) and stay
#    fixed; the intrinsic column density is the actual fit target.
# ---------------------------------------------------------------------------
cm = @model begin
    gal = PhAbs1D(nH = 0.02, z = 0.0)
    intrinsic = PhAbs1D(nH = 2.0, z = z_src)
    cont = PowerLaw1D(norm = 4.0, x_ref = 1.0, index = 1.5)
    fe_line = Gaussian1D(amplitude = 1.0, mean = 5.8, sigma = 0.3)
    gal * intrinsic * (cont + fe_line)
end

@fix cm.gal.nH = 0.02
@fix cm.gal.z = 0.0
@fix cm.gal.e0 = 1.0
@fix cm.intrinsic.z = z_src
@fix cm.intrinsic.e0 = 1.0
@fix cm.cont.x_ref = 1.0
@bound cm.intrinsic.nH in (0.01, 50)
@bound cm.cont.norm in (0.01, 50)
@bound cm.cont.index in (0.1, 5)
@bound cm.fe_line.amplitude in (0.01, 20)
@bound cm.fe_line.mean in (5.5, 6.5)
@bound cm.fe_line.sigma in (0.02, 1.0)

# ---------------------------------------------------------------------------
# 4. Poisson log-likelihood as a custom `statistic`, then fit
# ---------------------------------------------------------------------------
# logpdf(Poisson(λ), 0) is correctly 0 at λ→0, but its ForwardDiff derivative
# is NaN there (the k*log(λ) chain rule still evaluates log(λ)=-Inf before
# the k=0 factor zeroes it out). At k=0 the exact log-likelihood collapses to
# -λ anyway (log and loggamma(1) terms both vanish), so branch around it —
# this is a numerically-safe rewrite of the same formula, not a domain guard.
# This IS the XSPEC "cstat"/Cash (1979) statistic: fit directly on
# unrebinned counts, no minimum-counts-per-bin grouping needed, unlike the
# chi²-on-grouped-data alternative XSPEC also offers.
poisson_ll_term(λ, k) = k == 0 ? -λ : logpdf(Poisson(λ), k)

# Same two-method shape as AstroFit's own `chi2(model, coords, y, err)` /
# `chi2(f::ObjectiveFunction, p)` (src/fit/loss.jl) — keeps the tree-walk
# out of the per-point term and makes the statistic reusable outside fitting
# (e.g. to score `λ_true` below without touching `ObjectiveFunction`).
poisson_loglike(model, coords, y) = begin
    Es = coords[1]
    sum(i -> poisson_ll_term(render(model, Es[i]), y[i]), eachindex(y))
end
poisson_loglike(f::ObjectiveFunction, p) = poisson_loglike(withparams(f.cm, p), f.coords, f.y)

# Optimization.jl minimizes — negate the log-likelihood to fit.
prob = OptimizationProblem(cm, E, counts; statistic = (f, p) -> -poisson_loglike(f, p))
sol = solve(prob, LBFGS())

fit_tree = withparams(cm, sol.u)

println("retcode         : ", sol.retcode)
println("free parameters : ", nfree(cm))
println("parameter names : ", paramnames(cm))
println("best fit values : ", round.(sol.u; digits = 4))
println()

# ---------------------------------------------------------------------------
# 5. Plot: initial guess vs best fit vs data
# ---------------------------------------------------------------------------
# √N is undefined/misleading at N=0 (and biased for small N in general), so
# use the Gehrels (1986) approximate 1σ Poisson confidence limits instead —
# the standard asymmetric error astronomers quote on low-count spectra, and
# well-defined at N=0 (upper limit only, lower error clamps to 0).
gehrels_upper(n) = n + 1 + sqrt(n + 0.75)
gehrels_lower(n) = n == 0 ? 0.0 : n * (1 - 1 / (9n) - 1 / (3 * sqrt(n)))^3
err_hi = [gehrels_upper(n) - n for n in counts]
err_lo = [n - gehrels_lower(n) for n in counts]

λ_init = render(cm, E)
λ_fit = render(fit_tree, E)

# log-log, as XSPEC always plots folded spectra — linear axes hide the
# absorption turnover (E^-3 cross-section spans orders of magnitude below
# ~1 keV) and the Fe Kα bump under the continuum peak. log(0) is undefined,
# so: model curves get a floor matched to the axis's own lower limit (the
# fit itself never sees this — `counts`, `sol`, `fit_tree` are all
# untouched), and zero-count bins are dropped from the data points (nothing
# physically wrong with them, see poisson_ll_term — they just can't sit on a
# log y-axis). A floor many decades below the plotted range (e.g. 1e-6)
# would draw a flat shelf across most of the panel instead of letting the
# curve run off the bottom edge like a real folded-spectrum plot.
floor_for_log = 1e-2
λ_true_plot = max.(λ_true, floor_for_log)
λ_init_plot = max.(λ_init, floor_for_log)
λ_fit_plot = max.(λ_fit, floor_for_log)
detected = counts .> 0

fig = Figure(size = (900, 500))
ax = Axis(
    fig[1, 1]; xlabel = "Energy (keV)", ylabel = "Counts",
    xscale = log10, yscale = log10, limits = (nothing, nothing, floor_for_log, nothing),
    title = "Absorbed AGN Fit: phabs × zphabs × (powerlaw + Fe Kα), Poisson statistic"
)

errorbars!(
    ax, E[detected], counts[detected], err_lo[detected], err_hi[detected];
    color = :grey60, whiskerwidth = 3
)
scatter!(
    ax, E[detected], counts[detected]; color = :grey60, markersize = 4,
    label = "data (counts, Gehrels 1σ)"
)
lines!(ax, E, λ_true_plot; color = :black, linestyle = :dash, label = "truth")
lines!(ax, E, λ_init_plot; color = :dodgerblue, linestyle = :dot, label = "initial guess")
lines!(ax, E, λ_fit_plot; color = :red, linewidth = 2, label = "best fit")

axislegend(ax; position = :lb)

display(fig)
# save("examples/agn_xray_fit.png", fig; px_per_unit = 2)
# println("saved → examples/agn_xray_fit.png")
