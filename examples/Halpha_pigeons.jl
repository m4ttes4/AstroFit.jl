# Bayesian fit of the Hα + [NII] triplet with Pigeons.jl.
# Compares AstroFit CompiledModel vs a handwritten log-posterior.
#
# Run with:  julia --project=. examples/Halpha_pigeons.jl

using AstroFit
using Distributions
using Pigeons
using CairoMakie
using Random
using Statistics

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
const λ_Ha = 6562.8
const λ_NII_b = 6548.05
const λ_NII_r = 6583.45

# ---------------------------------------------------------------------------
# 1. AstroFit model: Hα + [NII] doublet on linear continuum
# ---------------------------------------------------------------------------
# Ties: NII amplitudes fixed to Hα via atomic ratios, NII means locked to Hα
# via wavelength ratios, all three lines share sigma.
# Free: slope, intercept, ha.A, ha.μ, ha.σ → 5 free params.

true_cm = @model begin
    cont = Linear1D(slope = 0.002, intercept = 1.0)
    ha   = Gaussian1D(amplitude = 8.0, mean = λ_Ha, sigma = 4.0)
    nii_r = Gaussian1D(amplitude = 3.06 / 3.0 * 8.0, mean = (λ_NII_r / λ_Ha) * λ_Ha, sigma = 4.0)
    nii_b = Gaussian1D(amplitude = 8.0 / 3.0, mean = (λ_NII_b / λ_Ha) * λ_Ha, sigma = 4.0)
    cont + ha + nii_r + nii_b
end

@constrain true_cm begin
    cont.slope in (-0.05, 0.05)
    cont.intercept in (0.0, 5.0)
    ha.amplitude in (0.1, 30.0)
    ha.mean in (6540.0, 6590.0)
    ha.sigma in (1.0, 12.0)
    nii_r.amplitude -> (3.06 / 3.0) * ha.amplitude
    nii_r.mean -> (λ_NII_r / λ_Ha) * ha.mean
    nii_r.sigma -> ha.sigma
    nii_b.amplitude -> ha.amplitude / 3.0
    nii_b.mean -> (λ_NII_b / λ_Ha) * ha.mean
    nii_b.sigma -> ha.sigma
end

# Fitting model — offset initial guess
cm = @model begin
    cont = Linear1D(slope = 0.0, intercept = 0.5)
    ha   = Gaussian1D(amplitude = 5.0, mean = 6560.0, sigma = 5.0)
    nii_r = Gaussian1D(amplitude = 3.06 / 3.0 * 5.0, mean = (λ_NII_r / λ_Ha) * 6560.0, sigma = 5.0)
    nii_b = Gaussian1D(amplitude = 5.0 / 3.0, mean = (λ_NII_b / λ_Ha) * 6560.0, sigma = 5.0)
    cont + ha + nii_r + nii_b
end

@constrain cm begin
    cont.slope in (-0.05, 0.05)
    cont.intercept in (0.0, 5.0)
    ha.amplitude in (0.1, 30.0)
    ha.mean in (6540.0, 6590.0)
    ha.sigma in (1.0, 12.0)
    nii_r.amplitude -> (3.06 / 3.0) * ha.amplitude
    nii_r.mean -> (λ_NII_r / λ_Ha) * ha.mean
    nii_r.sigma -> ha.sigma
    nii_b.amplitude -> ha.amplitude / 3.0
    nii_b.mean -> (λ_NII_b / λ_Ha) * ha.mean
    nii_b.sigma -> ha.sigma
    # logposterior no longer auto-rejects out-of-bounds points; these Uniform
    # priors reproduce hand_logposterior's hard_lower/hand_upper walls below
    # so af_target and hand stay the same distribution, not just equal at p0.
    cont.slope ~ Uniform(-0.05, 0.05)
    cont.intercept ~ Uniform(0.0, 5.0)
    ha.amplitude ~ Uniform(0.1, 30.0)
    ha.mean ~ Uniform(6540.0, 6590.0)
    ha.sigma ~ Uniform(1.0, 12.0)
end

# ---------------------------------------------------------------------------
# 2. Synthetic data
# ---------------------------------------------------------------------------
Random.seed!(42)
x = collect(6500.0:1.0:6650.0)
y_true = render(withparams(true_cm, AstroFit.params(true_cm)), x)
σ_noise = 0.3
y = y_true .+ σ_noise .* randn(length(x))
err = fill(σ_noise, length(x))

# ---------------------------------------------------------------------------
# 3. Handwritten log-posterior (same physics, no AstroFit overhead)
# ---------------------------------------------------------------------------
const hand_lower = [-0.05, 0.0, 0.1, 6540.0, 1.0]
const hand_upper = [ 0.05, 5.0, 30.0, 6590.0, 12.0]

function hand_logposterior(p, x, y, err)
    s, ic, A, μ, σ = p
    for i in eachindex(p)
        (hand_lower[i] <= p[i] <= hand_upper[i]) || return -Inf
    end
    rA = (3.06 / 3.0) * A
    bA = A / 3.0
    rμ = (λ_NII_r / λ_Ha) * μ
    bμ = (λ_NII_b / λ_Ha) * μ
    χ2 = 0.0
    @inbounds @fastmath for i in eachindex(y)
        m = s * x[i] + ic +
            A * exp(-((x[i] - μ) / σ)^2 / 2) +
            rA * exp(-((x[i] - rμ) / σ)^2 / 2) +
            bA * exp(-((x[i] - bμ) / σ)^2 / 2)
        χ2 += abs2((m - y[i]) / err[i])
    end
    n = length(y)
    return -0.5 * χ2 - sum(log, err) - n / 2 * log(2π)
end

struct HandTarget
    x::Vector{Float64}
    y::Vector{Float64}
    err::Vector{Float64}
end

(t::HandTarget)(p) = hand_logposterior(p, t.x, t.y, t.err)

Pigeons.initialization(t::HandTarget, ::Random.AbstractRNG, ::Int) =
    [0.0, 0.5, 5.0, 6560.0, 5.0]

function Pigeons.sample_names(::Array, p::Pigeons.InterpolatedLogPotential)
    target = p.path.target
    if target isa AstroFit.ObjectiveFunction
        return [Symbol.(target.names); :log_density]
    elseif target isa HandTarget
        return [:slope, :intercept, :ha_amplitude, :ha_mean, :ha_sigma, :log_density]
    end
    n = length(Pigeons.initialization(target, Random.default_rng(), 1))
    return [map(i -> Symbol("param_$i"), 1:n); :log_density]
end

function Pigeons.default_reference(t::HandTarget)
    dists = map(eachindex(hand_lower)) do i
        Uniform(hand_lower[i], hand_upper[i])
    end
    return Pigeons.DistributionLogPotential(product_distribution(dists))
end

# ---------------------------------------------------------------------------
# 4. Verify equivalence
# ---------------------------------------------------------------------------
p0 = AstroFit.params(cm)
af_target = ObjectiveFunction(cm, x, y, err; statistic = logposterior)
hand = HandTarget(x, y, err)

# af_target's Uniform priors add logpdf's normalization -log(hi-lo) per
# parameter, which hand_logposterior's raw box-check doesn't — subtract it
# so this checks the posterior shape, not an incidental constant offset.
uniform_norm_const = -sum(log, hand_upper .- hand_lower)
@assert af_target(p0) - uniform_norm_const ≈ hand(p0) "AstroFit vs handwritten log-posterior diverge: $(af_target(p0)) vs $(hand(p0))"
println("log-posterior equivalence check: ✓")

# ---------------------------------------------------------------------------
# 5. Warmup (JIT compilation, not timed)
# ---------------------------------------------------------------------------
println("\nwarming up JIT...")
pigeons(target = af_target, n_rounds = 2, n_chains = 2, seed = 1)
pigeons(target = hand, n_rounds = 2, n_chains = 2, seed = 1)
println("warmup done")

# ---------------------------------------------------------------------------
# 6. Sample with Pigeons — AstroFit
# ---------------------------------------------------------------------------
println("\n--- AstroFit Pigeons ---")
t_af = @elapsed begin
    pt_af = pigeons(
        target = af_target,
        n_rounds = 10,
        n_chains = 8,
        seed = 123,
        record = [traces; record_default()],
    )
end
println("  time: $(round(t_af; digits=2))s")

# ---------------------------------------------------------------------------
# 7. Sample with Pigeons — Handwritten
# ---------------------------------------------------------------------------
println("\n--- Handwritten Pigeons ---")
t_hand = @elapsed begin
    pt_hand = pigeons(
        target = hand,
        n_rounds = 10,
        n_chains = 8,
        seed = 123,
        record = [traces; record_default()],
    )
end
println("  time: $(round(t_hand; digits=2))s")

println("\n  speedup: $(round(t_af / t_hand; digits=2))x (AstroFit / handwritten)")

# ---------------------------------------------------------------------------
# 7. Extract posteriors
# ---------------------------------------------------------------------------
function extract_samples(pt, nparams)
    raw = sample_array(pt)
    ps = raw[:, 1:nparams, :]
    return reshape(permutedims(ps, (1, 3, 2)), :, nparams)
end

samples_af = extract_samples(pt_af, nfree(cm))
samples_hand = extract_samples(pt_hand, 5)

med_af = vec(median(samples_af; dims=1))
med_hand = vec(median(samples_hand; dims=1))

pnames = paramnames(cm)
println("\nPosterior medians:")
println("  ", rpad("param", 18), rpad("AstroFit", 14), "handwritten")
for i in 1:nfree(cm)
    println("  ", rpad(string(pnames[i]), 18),
            rpad(round(med_af[i]; digits=4), 14),
            round(med_hand[i]; digits=4))
end

# ---------------------------------------------------------------------------
# 8. Plot
# ---------------------------------------------------------------------------
fit_af = withparams(cm, med_af)
fit_hand_tree = withparams(cm, med_hand)

fig = Figure(size=(1100, 800))

# Top: data + fits + posterior draws
ax1 = Axis(fig[1, 1]; ylabel="flux", title="Hα + [NII] — AstroFit Pigeons ($(round(t_af; digits=1))s)")
ax2 = Axis(fig[1, 2]; title="Hα + [NII] — Handwritten Pigeons ($(round(t_hand; digits=1))s)")

for (ax, samples, med_tree) in ((ax1, samples_af, fit_af), (ax2, samples_hand, fit_hand_tree))
    scatter!(ax, x, y; color=:grey60, markersize=4, label="data")
    lines!(ax, x, y_true; color=:black, linestyle=:dash, linewidth=1.5, label="truth")

    n_draw = min(80, size(samples, 1))
    ids = unique(round.(Int, range(1, size(samples, 1); length=n_draw)))
    for i in ids
        y_draw = render(withparams(cm, vec(samples[i, :])), x)
        lines!(ax, x, y_draw; color=(:red, 0.04))
    end

    lines!(ax, x, render(med_tree, x); color=:red, linewidth=2, label="posterior median")
    axislegend(ax; position=:lt, framevisible=false)
end

# Bottom: corner-style marginals comparison
ax_hist = [Axis(fig[2, i]; xlabel=string(pnames[i]), ylabel="density") for i in 1:nfree(cm)]

for i in 1:nfree(cm)
    hist!(ax_hist[i], samples_af[:, i]; bins=40, normalization=:pdf,
          color=(:red, 0.4), label=i == 1 ? "AstroFit" : "")
    hist!(ax_hist[i], samples_hand[:, i]; bins=40, normalization=:pdf,
          color=(:dodgerblue, 0.4), label=i == 1 ? "handwritten" : "")
end
axislegend(ax_hist[1]; position=:lt, framevisible=false)

rowsize!(fig.layout, 2, Relative(0.35))



# outpath = joinpath(@__DIR__, "Halpha_pigeons.png")
# save(outpath, fig; px_per_unit=2)
# println("\nsaved -> ", outpath)
