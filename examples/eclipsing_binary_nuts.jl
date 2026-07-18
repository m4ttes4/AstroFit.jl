# Bayesian fit of the eclipsing binary light curve with NUTS (AdvancedHMC.jl).
# Same scene/model as eclipsing_binary_lightcurve.jl (see that file for the
# Occult1D geometry and the @tie reasoning); here the point estimate is
# replaced by a full posterior over the 4 free parameters, with priors
# attached in the same @constrain block as the bounds/ties/fixes.
#
# Run with:  julia --project=examples/ examples/eclipsing_binary_nuts.jl

using AstroFit
using Distributions
using LogDensityProblems, LogDensityProblemsAD
using AdvancedHMC, AbstractMCMC
using ForwardDiff
using CairoMakie, PairPlots
using MCMCChains
using Random
using Statistics

# ---------------------------------------------------------------------------
# 0. Circle-circle occultation model — identical to eclipsing_binary_lightcurve.jl
# ---------------------------------------------------------------------------
Base.@kwdef struct Occult1D{T <: Real} <: AbstractModel
    rp_rs::T = 0.1
    a_rs::T = 10.0
    b::T = 0.3
    period::T = 3.5
    t0::T = 0.0
end

function AstroFit.render(m::Occult1D, t::Number)
    φ = 2π * (t - m.t0) / m.period
    cos(φ) < 0 && return one(φ)
    x = m.a_rs * sin(φ)
    y = m.b * cos(φ)
    z = sqrt(x^2 + y^2)
    k = m.rp_rs
    if z >= 1 + k
        one(z)
    elseif z <= abs(1 - k)
        k <= 1 ? 1 - k^2 : zero(z)
    else
        κ0 = acos(clamp((k^2 + z^2 - 1) / (2k * z), -1, 1))
        κ1 = acos(clamp((1 - k^2 + z^2) / (2z), -1, 1))
        area = k^2 * κ0 + κ1 - 0.5 * sqrt(max(4z^2 - (1 + z^2 - k^2)^2, 0))
        1 - area / π
    end
end

# ---------------------------------------------------------------------------
# 1. True model + synthetic data — identical to eclipsing_binary_lightcurve.jl
# ---------------------------------------------------------------------------
true_model = @model begin
    dipA = Occult1D(rp_rs = 0.4, a_rs = 8.0, b = 0.2, period = 2.5, t0 = 0.2)
    dipB = Occult1D(rp_rs = 2.5, a_rs = 20.0, b = 0.5, period = 2.5, t0 = 0.2 + 2.5 / 2)
    LA = Const1D(value = 0.7)
    LB = Const1D(value = 0.3)
    LA * dipA + LB * dipB
end

t = collect(range(-0.2, 2 * 2.5 + 0.2, length = 400))
flux_true = render(true_model, t)
σ = 0.004
Random.seed!(3)
flux = flux_true .+ σ .* randn(length(t))
err = fill(σ, length(t))

# ---------------------------------------------------------------------------
# 2. Model with constraints AND priors, all in one @constrain block
# ---------------------------------------------------------------------------
# Priors sit on top of the same bounds used in the deterministic example —
# NUTS still needs the walls (rp_rs > 0, period/t0 within the periodogram-
# resolved window) but the priors are what keep the sampler from wasting
# time exploring flat, eclipse-free regions of that window.
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
    dipA.rp_rs ~ LogNormal(log(0.3), 0.3)
    dipA.period ~ Normal(2.4, 0.1)
    dipA.t0 ~ Normal(0.1, 0.15)
    LA.value ~ Beta(2, 2)
end

# ---------------------------------------------------------------------------
# 3. Log-posterior target + AD gradient wrapper
# ---------------------------------------------------------------------------
target = ObjectiveFunction(cm, t, flux, err; statistic = logposterior)
ℓ = ADgradient(:ForwardDiff, target)

println("free parameters : ", nfree(cm))
println("parameter names : ", paramnames(cm))

# ---------------------------------------------------------------------------
# 4. NUTS sampling
# ---------------------------------------------------------------------------
θ_init = AstroFit.params(cm)
n_adapts = 500
n_samples = 1000

Random.seed!(42)
chain = AbstractMCMC.sample(
    Random.default_rng(),
    AbstractMCMC.LogDensityModel(ℓ),
    NUTS(0.8; max_depth = 10),
    n_adapts + n_samples;
    n_adapts = n_adapts,
    initial_params = θ_init,
    discard_initial = n_adapts,
    progress = true,
    chain_type = Chains,
    param_names = string.(paramnames(cm)),
)

# ---------------------------------------------------------------------------
# 5. Diagnostics
# ---------------------------------------------------------------------------
println("\n", chain)

posterior_median = vec(median(Array(chain); dims = 1))
fit = withparams(cm, posterior_median)
flux_fit = render(fit, t)

# ---------------------------------------------------------------------------
# 6. Plot: data vs truth vs posterior median, with a posterior-draw band
# ---------------------------------------------------------------------------
draws = Array(chain)
n_band = 200
idx = rand(Random.default_rng(), axes(draws, 1), n_band)
band = reduce(hcat, (render(withparams(cm, draws[i, :]), t) for i in idx))
lo = vec(mapslices(x -> quantile(x, 0.16), band; dims = 2))
hi = vec(mapslices(x -> quantile(x, 0.84), band; dims = 2))

fig = Figure(size = (900, 500))
ax = Axis(
    fig[1, 1]; xlabel = "Time (days)", ylabel = "Relative flux",
    title = "Eclipsing Binary — NUTS posterior ($(nfree(cm)) free params)"
)
band!(ax, t, lo, hi; color = (:red, 0.2), label = "posterior 16–84%")
errorbars!(ax, t, flux, err; color = :grey70, whiskerwidth = 2)
scatter!(ax, t, flux; color = :grey60, markersize = 4, label = "data")
lines!(ax, t, flux_true; color = :black, linestyle = :dash, label = "truth")
lines!(ax, t, flux_fit; color = :red, linewidth = 2, label = "posterior median")
axislegend(ax; position = :rb)

display(fig)
save("examples/eclipsing_binary_nuts_fit.png", fig; px_per_unit = 2)

# ---------------------------------------------------------------------------
# 7. Pair plot of the 4 free parameters
# ---------------------------------------------------------------------------
free_names = string.(paramnames(cm))
true_vals = [0.7, 0.4, 2.5, 0.2]   # LA_value, dipA_rp_rs, dipA_period, dipA_t0

pp = pairplot(chain[free_names], PairPlots.Truth(
    Dict(n => v for (n, v) in zip(free_names, true_vals))))
save("examples/eclipsing_binary_nuts_pairs.png", pp; px_per_unit = 2)

println("saved → examples/eclipsing_binary_nuts_fit.png")
println("saved → examples/eclipsing_binary_nuts_pairs.png")
