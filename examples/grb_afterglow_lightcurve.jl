# Fit a GRB X-ray afterglow light curve — broken power-law decay in time plus
# a constant background — to synthetic Poisson counting data, using the same
# custom Poisson statistic pattern as agn_xray_fit.jl. Mirrors the canonical
# Swift-XRT light curve shape (Nousek et al. 2006, Zhang et al. 2006): a
# shallow "plateau" segment (α₁ ≈ 0.5) steepening at a break time into the
# standard afterglow decay (α₂ ≈ 1.5), on top of a detector background,
# binned counts per time interval, plotted log-log as light curves always
# are (time since trigger spans orders of magnitude). A single unbroken
# power law would fit the late-time decay but miss the plateau entirely.
#
# Unlike agn_xray_fit.jl this uses only built-in zoo models (BrokenPowerLaw1D
# + Const1D) — no custom AbstractModel needed — and the background floor is
# picked high enough that Poisson draws essentially never hit zero, so every
# point carries a well-defined symmetric-ish error without special-casing.
#
# Run with:  julia --project=. examples/grb_afterglow_lightcurve.jl

using AstroFit
using Optimization, OptimizationOptimJL, ForwardDiff
using Distributions: Poisson, logpdf
using CairoMakie
using Random

# ---------------------------------------------------------------------------
# 1. True model — broken power-law afterglow decay + constant background
# ---------------------------------------------------------------------------
# t in seconds since trigger, model output = expected counts per time bin.
true_model = @model begin
    afterglow = BrokenPowerLaw1D(norm = 50.0, x_break = 300.0, index1 = 0.5, index2 = 1.5)
    bg = Const1D(value = 8.0)   # detector background
    afterglow + bg
end

# ---------------------------------------------------------------------------
# 2. Synthetic data — Poisson counts, log-spaced time bins
# ---------------------------------------------------------------------------
# Background (8 cts/bin) is the floor: even at the latest, faintest bin the
# afterglow has decayed into the background, so λ never drops much below 8
# and P(Poisson(λ)=0) stays negligible (e^-8 ≈ 3e-4) — no zero-count bins
# by construction, not by post-hoc filtering.
t = exp10.(range(log10(10.0), log10(1.0e4), length = 45))
λ_true = render(true_model, t)
Random.seed!(7)
counts = [rand(Poisson(λ)) for λ in λ_true]
@assert !any(iszero, counts) "background too low for this seed/grid — raise bg or amplitude"

# ---------------------------------------------------------------------------
# 3. Fitting model — off initial guess. norm, x_break, index1, index2, bg all
#    free: the break time is itself a fit target, same as in real XRT
#    light-curve fitting (it marks the end of the plateau phase).
# ---------------------------------------------------------------------------
cm = @model begin
    afterglow = BrokenPowerLaw1D(norm = 20.0, x_break = 100.0, index1 = 0.1, index2 = 0.8)
    bg = Const1D(value = 3.0)
    afterglow + bg
end

@bound cm.afterglow.norm in (0.1, 500)
@bound cm.afterglow.x_break in (10, 5000)
@bound cm.afterglow.index1 in (0.0, 3)
@bound cm.afterglow.index2 in (0.0, 4)
@bound cm.bg.value in (0.1, 50)

# ---------------------------------------------------------------------------
# 4. Poisson log-likelihood as a custom `statistic`, then fit
# ---------------------------------------------------------------------------
# logpdf(Poisson(λ), 0) is correctly 0 at λ→0, but its ForwardDiff derivative
# is NaN there (the k*log(λ) chain rule still evaluates log(λ)=-Inf before
# the k=0 factor zeroes it out) — this dataset never hits k=0 by design, but
# the branch is kept since the optimizer can walk through low-λ regions
# mid-fit even when no observed bin lands exactly there.
poisson_ll_term(λ, k) = k == 0 ? -λ : logpdf(Poisson(λ), k)

# Same two-method shape as AstroFit's own `chi2(model, coords, y, err)` /
# `chi2(f::ObjectiveFunction, p)` (src/fit/loss.jl).
poisson_loglike(model, coords, y) = begin
    ts = coords[1]
    sum(i -> poisson_ll_term(render(model, ts[i]), y[i]), eachindex(y))
end
poisson_loglike(f::ObjectiveFunction, p) = poisson_loglike(withparams(f.cm, p), f.coords, f.y)

# Optimization.jl minimizes — negate the log-likelihood to fit.
prob = OptimizationProblem(cm, t, counts; statistic = (f, p) -> -poisson_loglike(f, p))
sol = solve(prob, LBFGS())

fit_tree = withparams(cm, sol.u)

println("retcode         : ", sol.retcode)
println("free parameters : ", nfree(cm))
println("parameter names : ", paramnames(cm))
println("best fit values : ", round.(sol.u; digits = 4))
println()

# ---------------------------------------------------------------------------
# 5. Plot: initial guess vs best fit vs data, log-log as light curves
#    spanning orders of magnitude in time always are
# ---------------------------------------------------------------------------
# Gehrels (1986) approximate 1σ Poisson confidence limits — asymmetric,
# well-defined even near the background floor where √N would understate the
# upper tail. No zero-count bins here, so nothing needs floor-clamping or
# filtering for the log axis (contrast with agn_xray_fit.jl).
gehrels_upper(n) = n + 1 + sqrt(n + 0.75)
gehrels_lower(n) = n * (1 - 1 / (9n) - 1 / (3 * sqrt(n)))^3
err_hi = [gehrels_upper(n) - n for n in counts]
err_lo = [n - gehrels_lower(n) for n in counts]

λ_init = render(cm, t)
λ_fit = render(fit_tree, t)

fig = Figure(size = (900, 500))
ax = Axis(
    fig[1, 1]; xlabel = "Time since trigger (s)", ylabel = "Counts / bin",
    xscale = log10, yscale = log10,
    title = "GRB X-ray Afterglow: broken power-law decay + background, Poisson statistic"
)

errorbars!(ax, t, counts, err_lo, err_hi; color = :grey60, whiskerwidth = 3)
scatter!(ax, t, counts; color = :grey60, markersize = 5, label = "data (counts, Gehrels 1σ)")
lines!(ax, t, λ_true; color = :black, linestyle = :dash, label = "truth")
lines!(ax, t, λ_init; color = :dodgerblue, linestyle = :dot, label = "initial guess")
lines!(ax, t, λ_fit; color = :red, linewidth = 2, label = "best fit")

axislegend(ax; position = :lb)

display(fig)
# save("examples/grb_afterglow_lightcurve.png", fig; px_per_unit = 2)
# println("saved → examples/grb_afterglow_lightcurve.png")
