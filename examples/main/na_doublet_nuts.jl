# Bayesian version of na_doublet_fit.jl: the Na I D doublet + He I emission,
# seen through a fixed instrumental PSF, with NUTS (AdvancedHMC.jl) instead of
# the LBFGS point estimate. Same scene and the same physical ties — the depth
# ratio, atomic separation, shared velocity/dispersion — now the posterior
# gives credible intervals on the gas velocity and the intrinsic width.
#
# Prior rationale (all Truncated to their physical bound, since logposterior no
# longer auto-rejects out-of-bounds points — the Truncated wall is what NUTS
# feels):
#   d2.amplitude → Normal, absorption only (negative)
#   d2.mean      → Normal on the velocity window (drives the velocity posterior)
#   d2.sigma     → LogNormal, strictly positive intrinsic width
#   hei.amplitude→ LogNormal, emission only (positive)
#   hei.sigma    → LogNormal, own width
#   cont.slope/intercept → Normal on ℝ (continuum is unbounded, no truncation)
#
# Run with:  julia --project=examples/ examples/main/na_doublet_nuts.jl

using AstroFit
using Distributions
using LogDensityProblems, LogDensityProblemsAD
using AdvancedHMC, AbstractMCMC
using ForwardDiff
using CairoMakie, PairPlots
using MCMCChains
using Random
using Statistics

const L_HEI = 5875.62
const L_NAD_D2 = 5889.95
const L_NAD_D1 = 5895.92
const C_KMS = 2.998e5

const STEP = 0.1             # grid step [A/sample] — kernels work in samples
const SIGMA_INST = 1.6       # instrumental resolution [A]: partially blends the doublet

# ---------------------------------------------------------------------------
# 1. True model — doublet at +45 km/s, ratio 2:1, intrinsic width 0.45 A,
#    plus He I emission; the instrument smears every line to ~2.25 A
# ---------------------------------------------------------------------------
v_true = 45.0
shift = L_NAD_D2 * v_true / C_KMS
sigma_true = 0.45

true_model = @model begin
    cont = Linear1D(slope = -0.0015, intercept = 9.835)
    d2 = Gaussian1D(amplitude = -0.85, mean = L_NAD_D2 + shift, sigma = sigma_true)
    d1 = Gaussian1D(amplitude = -0.425, mean = L_NAD_D1 + shift, sigma = sigma_true)
    hei = Gaussian1D(amplitude = 0.55, mean = L_HEI + shift, sigma = 0.9)
    psf = GaussianPSF(sigma = SIGMA_INST / STEP)
    (cont + d2 + d1 + hei) |> psf
end

# ---------------------------------------------------------------------------
# 2. Synthetic data
# ---------------------------------------------------------------------------
Random.seed!(123)
λ = collect(5860.0:STEP:5925.0)
σ_noise = 0.02
y_true = render(true_model, λ)
y = y_true .+ σ_noise .* randn(length(λ))
err = fill(σ_noise, length(λ))

# ---------------------------------------------------------------------------
# 3. Fitting model — start at rest wavelength, D1 fully derived from D2
# ---------------------------------------------------------------------------
cm = @model begin
    cont = Linear1D(slope = 0.0, intercept = 1.0)
    d2 = Gaussian1D(amplitude = -0.4, mean = L_NAD_D2, sigma = 0.8)
    d1 = Gaussian1D(amplitude = -0.2, mean = L_NAD_D1, sigma = 0.8)
    hei = Gaussian1D(amplitude = 0.3, mean = L_HEI, sigma = 1.2)
    psf = GaussianPSF(sigma = SIGMA_INST / STEP)
    (cont + d2 + d1 + hei) |> psf
end

# Same bounds/ties/fixes as the MAP example, now with a prior on every free
# parameter — Bayesian inference needs one per free parameter (the continuum
# included). Priors sit in the same @constrain block as the constraints.
@constrain cm begin
    # --- bounds ---
    d2.amplitude in (-5.0, 0.0)                    # absorption only
    d2.mean in (5885.0, 5895.0)                    # velocity window
    d2.sigma in (0.1, 3.0)
    hei.amplitude in (0.0, 5.0)                    # emission only
    hei.sigma in (0.1, 5.0)                        # different gas, own width

    # --- ties / fixes ---
    d1.amplitude -> 0.5 * d2.amplitude             # optically thin 2:1
    d1.mean -> d2.mean + (L_NAD_D1 - L_NAD_D2)     # atomic separation
    d1.sigma -> d2.sigma                           # same gas
    hei.mean -> d2.mean + (L_HEI - L_NAD_D2)       # same systemic velocity
    psf.sigma                                      # known calibration, fixed
    cont.slope 
    # --- priors ---
    # continuum lives on ℝ: near the data the level is ~1, but slope and
    # intercept are strongly degenerate over this window, so keep slope tight
    # (a 0.001 change swings the continuum by ~6) and the intercept wide.
    # cont.slope ~ Normal(0.0, 0.002)
    cont.intercept ~ Normal(1.0, 10.0)
    d2.amplitude ~ Normal(-0.5, 0.5)
    d2.mean ~ Normal(5890.0, 2.0)
    d2.sigma ~ LogNormal(log(0.5), 0.6)
    hei.amplitude ~ LogNormal(log(0.5), 0.8)
    hei.sigma ~ LogNormal(log(0.9), 0.5)
end

# ---------------------------------------------------------------------------
# 4. Log-posterior target + AD gradient wrapper
# ---------------------------------------------------------------------------
target = ObjectiveFunction(cm, λ, y, err; statistic = logposterior)
ℓ = ADgradient(:ForwardDiff, target)

println("free parameters : ", nfree(cm))
println("parameter names : ", paramnames(cm))

# ---------------------------------------------------------------------------
# 5. NUTS sampling
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
# 6. Diagnostics — velocity and intrinsic width as posterior credible intervals
# ---------------------------------------------------------------------------
println("\n", chain)

draws = Array(chain)
names = string.(paramnames(cm))
col(name) = draws[:, findfirst(==(name), names)]

v_chain = (col("d2_mean") .- L_NAD_D2) ./ L_NAD_D2 .* C_KMS
sigma_chain = col("d2_sigma")

pctl(x) = quantile(x, (0.16, 0.5, 0.84))
vq = pctl(v_chain)
sq = pctl(sigma_chain)

println()
println("gas velocity   : ", round(vq[2]; digits = 1), " km/s  (+",
    round(vq[3] - vq[2]; digits = 1), " / -", round(vq[2] - vq[1]; digits = 1),
    ")   truth: ", v_true)
println("intrinsic sigma: ", round(sq[2]; digits = 3), " A     (+",
    round(sq[3] - sq[2]; digits = 3), " / -", round(sq[2] - sq[1]; digits = 3),
    ")   truth: ", sigma_true)
println()

# ---------------------------------------------------------------------------
# 7. Plot: data, truth, posterior-median fit, and a 16–84% posterior band
# ---------------------------------------------------------------------------
posterior_median = vec(median(draws; dims = 1))
fit_tree = withparams(cm, posterior_median)
y_fit = render(fit_tree, λ)

n_band = 200
idx = rand(Random.default_rng(), axes(draws, 1), n_band)
band = reduce(hcat, (render(withparams(cm, draws[i, :]), λ) for i in idx))
lo = vec(mapslices(x -> quantile(x, 0.16), band; dims = 2))
hi = vec(mapslices(x -> quantile(x, 0.84), band; dims = 2))

fig = Figure(size = (900, 500))
ax = Axis(
    fig[1, 1]; xlabel = "wavelength [Angstrom]", ylabel = "normalized flux",
    title = "Na I D doublet + He I — NUTS posterior ($(nfree(cm)) free parameters)"
)

vlines!(ax, [L_HEI, L_NAD_D2, L_NAD_D1]; color = (:gray30, 0.4), linestyle = :dash, linewidth = 1, label = "rest wavelength")
band!(ax, λ, lo, hi; color = (:red, 0.2), label = "posterior 16–84%")
scatter!(ax, λ, y; color = :grey60, markersize = 4, label = "data")
lines!(ax, λ, y_true; color = :black, linestyle = :dash, label = "truth")
lines!(ax, λ, y_fit; color = :red, linewidth = 2, label = "posterior median")

axislegend(ax; position = :rb)

display(fig)
save("examples/na_doublet_nuts_fit.png", fig; px_per_unit = 2)

# ---------------------------------------------------------------------------
# 8. Pair plot of the physically interesting parameters
# ---------------------------------------------------------------------------
phys_names = ["d2_mean", "d2_sigma", "d2_amplitude", "hei_amplitude"]
true_vals = [L_NAD_D2 + shift, sigma_true, -0.85, 0.55]

pp = pairplot(chain[phys_names], PairPlots.Truth(
    Dict(n => v for (n, v) in zip(phys_names, true_vals))))
save("examples/na_doublet_nuts_pairs.png", pp; px_per_unit = 2)

println("saved → examples/na_doublet_nuts_fit.png")
println("saved → examples/na_doublet_nuts_pairs.png")
