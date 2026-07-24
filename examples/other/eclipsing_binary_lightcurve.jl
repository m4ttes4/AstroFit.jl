# Fit an eclipsing binary light curve — two stars mutually occulting each
# other once per orbit — to synthetic Gaussian-noise photometry. Showcases
# AstroFit's compositional power: the SAME custom occultation model is
# reused twice (once per star) inside one Sum-of-Products tree, and nearly
# every parameter of the second occultation is `@tie`d to the first — same
# period, phase-locked at half a period, reciprocal radius ratio,
# complementary light fraction — leaving only 4 truly free parameters for
# a whole two-star system.
#
# Run with:  julia --project=. examples/eclipsing_binary_lightcurve.jl

using AstroFit
using Optimization, OptimizationOptimJL, ForwardDiff
using CairoMakie
using Random

# ---------------------------------------------------------------------------
# 0. Circle-circle occultation — a user-defined AbstractModel (same closed-
#    form geometry as a planetary transit, Mandel & Agol 2002 uniform-disk
#    case), no changes to AstroFit itself needed.
#
#    sin(φ)/cos(φ) both vanish twice per orbit (φ=0 and φ=π) — i.e. BOTH
#    conjunctions look like a potential eclipse to the raw projected
#    separation ρ(φ). Which star is actually in front is decided by the
#    sign of cos(φ) (front near this leaf's own t0, back near the far
#    conjunction) — without that gate, each star's occultation would
#    incorrectly also dip at its *partner's* eclipse.
# ---------------------------------------------------------------------------
Base.@kwdef struct Occult1D{T <: Real} <: AbstractModel
    rp_rs::T = 0.1    # occulter/occulted radius ratio
    a_rs::T = 10.0    # scaled semi-major axis, a/R(this star)
    b::T = 0.3        # impact parameter, in units of R(this star)
    period::T = 3.5
    t0::T = 0.0       # this star's own mid-eclipse time
end

function AstroFit.render(m::Occult1D, t::Number)
    φ = 2π * (t - m.t0) / m.period
    cos(φ) < 0 && return one(φ)   # far conjunction — the partner is in front, not this star
    x = m.a_rs * sin(φ)
    y = m.b * cos(φ)
    z = sqrt(x^2 + y^2)
    k = m.rp_rs
    if z >= 1 + k
        one(z)
    elseif z <= abs(1 - k)
        k <= 1 ? 1 - k^2 : zero(z)   # occulter fully inside disk vs. disk fully hidden
    else
        κ0 = acos(clamp((k^2 + z^2 - 1) / (2k * z), -1, 1))
        κ1 = acos(clamp((1 - k^2 + z^2) / (2z), -1, 1))
        area = k^2 * κ0 + κ1 - 0.5 * sqrt(max(4z^2 - (1 + z^2 - k^2)^2, 0))
        1 - area / π
    end
end

# ---------------------------------------------------------------------------
# 1. True model — F(t) = L_A · Occult_A(t) + L_B · Occult_B(t)
# ---------------------------------------------------------------------------
# t in days. dipB's a_rs/b are dipA's rescaled by 1/rp_rs (different star,
# same physical orbit) — consistent by construction, not fit-derived here
# since both stay fixed (known independently, e.g. from radial velocities).
true_model = @model begin
    dipA = Occult1D(rp_rs = 0.4, a_rs = 8.0, b = 0.2, period = 2.5, t0 = 0.2)
    dipB = Occult1D(rp_rs = 2.5, a_rs = 20.0, b = 0.5, period = 2.5, t0 = 0.2 + 2.5 / 2)
    LA = Const1D(value = 0.7)
    LB = Const1D(value = 0.3)
    LA * dipA + LB * dipB
end

# ---------------------------------------------------------------------------
# 2. Synthetic data — Gaussian photometric noise, constant per-point error
# ---------------------------------------------------------------------------
t = collect(range(-0.2, 2 * 2.5 + 0.2, length = 400))
flux_true = render(true_model, t)
σ = 0.004
Random.seed!(3)
flux = flux_true .+ σ .* randn(length(t))
err = fill(σ, length(t))

# ---------------------------------------------------------------------------
# 3. Fitting model — off initial guess on the free parameters only.
#    a_rs/b stay fixed at their true (externally known) values; everything
#    about dipB beyond that is tied to dipA in one @constrain block.
#
#    period/t0 bounds are deliberately tight (±0.2 d / ±0.3 d around the
#    guess) — eclipses cover ~5% of the orbit, so chi2's gradient is
#    essentially flat everywhere except right at a dip. A local optimizer
#    given a wide net (e.g. period ∈ (1,5)) has no signal to follow and
#    drifts into a basin with zero eclipses overlapping data at all. Real
#    pipelines resolve this the same way: a periodogram/box-least-squares
#    search fixes the period to within a narrow window *before* the
#    detailed physical model is fit — that coarse search isn't repeated
#    here, just its outcome (a tight bound).
# ---------------------------------------------------------------------------
cm = @model begin
    dipA = Occult1D(rp_rs = 0.25, a_rs = 8.0, b = 0.2, period = 2.4, t0 = 0.1)
    dipB = Occult1D(rp_rs = 2.0, a_rs = 20.0, b = 0.5, period = 2.4, t0 = 1.35)
    LA = Const1D(value = 0.5)
    LB = Const1D(value = 0.5)
    LA * dipA + LB * dipB
end

@constrain cm begin
    dipA.a_rs = 8.0
    dipA.b = 0.2
    dipB.a_rs = 20.0
    dipB.b = 0.5
    dipA.rp_rs in (0.05, 1.0)
    dipA.period in (2.2, 2.8)
    dipA.t0 in (-0.2, 0.4)
    LA.value in (0.01, 0.99)
    dipB.period -> dipA.period
    dipB.t0 -> dipA.t0 + dipA.period / 2
    dipB.rp_rs -> 1 / dipA.rp_rs
    LB.value -> 1 - LA.value
end

# ---------------------------------------------------------------------------
# 4. Fit — native chi2 statistic, no custom likelihood needed this time
# ---------------------------------------------------------------------------
prob = OptimizationProblem(cm, t, flux, err)
sol = solve(prob, LBFGS())

fit_tree = withparams(cm, sol.u)

println("retcode         : ", sol.retcode)
println("free parameters : ", nfree(cm))
println("parameter names : ", paramnames(cm))
println("best fit values : ", round.(sol.u; digits = 4))
println()

# ---------------------------------------------------------------------------
# 5. Plot: data vs truth vs initial guess vs best fit
# ---------------------------------------------------------------------------
flux_init = render(cm, t)
flux_fit = render(fit_tree, t)

fig = Figure(size = (900, 500))

ax = Axis(
    fig[1, 1]; xlabel = "Time (days)", ylabel = "Relative flux",
    title = "Eclipsing Binary: LA·dipA + LB·dipB, shared orbit via @tie"
)
errorbars!(ax, t, flux, err; color = :grey70, whiskerwidth = 2)
scatter!(ax, t, flux; color = :grey60, markersize = 4, label = "data")
lines!(ax, t, flux_true; color = :black, linestyle = :dash, label = "truth")
lines!(ax, t, flux_init; color = :dodgerblue, linestyle = :dot, label = "initial guess")
lines!(ax, t, flux_fit; color = :red, linewidth = 2, label = "best fit")
axislegend(ax; position = :rb)

display(fig)
# save("examples/eclipsing_binary_lightcurve.png", fig; px_per_unit = 2)
# println("saved → examples/eclipsing_binary_lightcurve.png")
