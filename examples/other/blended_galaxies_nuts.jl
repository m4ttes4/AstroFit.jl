# Bayesian bulge+disk decomposition of two blended galaxies with NUTS (AdvancedHMC.jl).
# Same scene as blended_galaxies_fit.jl; Gaussian likelihood + physically motivated priors.
#
# Prior rationale:
#   amplitudes  → LogNormal  (strictly positive, factor-2 uncertainty)
#   positions   → Normal(μ, 2 px)  (loosely centred on rough location)
#   sizes       → LogNormal  (strictly positive, factor-1.6 uncertainty)
#   axis ratio  → Beta(4,2)  (mean 0.67, slightly prefers rounder galaxies)
#   theta       → Uniform on bounds  (no a-priori preferred orientation)
#
# NOTE: logposterior no longer auto-rejects out-of-bounds points, so every
# prior below is Truncated to its bound — that's what gives -Inf outside the
# box. NUTS handles this well when priors keep the posterior away from walls.
# For wall-free sampling use Bijectors.jl to transform to ℝⁿ first.
#
# Run with:  julia --project=examples/ examples/blended_galaxies_nuts.jl

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
# 1. True scene
# ---------------------------------------------------------------------------
true_scene = @model begin
    bulge1 = Gaussian2D(amplitude = 60.0, x0 = -2.0, y0 = -1.0, sigma = 1.0, q = 0.7, theta = 0.8)
    disk1  = Sersic2D(amplitude = 20.0, x0 = -2.0, y0 = -1.0, r_eff = 3.5, n = 1.0, q = 0.5, theta = 0.8)
    bulge2 = Gaussian2D(amplitude = 35.0, x0 = 3.0, y0 = 1.5, sigma = 0.7, q = 0.8, theta = -0.5)
    disk2  = Sersic2D(amplitude = 12.0, x0 = 3.0, y0 = 1.5, r_eff = 2.8, n = 1.0, q = 0.6, theta = -0.5)
    bulge1 + disk1 + bulge2 + disk2
end

# ---------------------------------------------------------------------------
# 2. Synthetic image
# ---------------------------------------------------------------------------
Random.seed!(42)
npix    = 100
coord   = range(-8.0, 8.0; length = npix)
X       = [x for x in coord, _ in coord]
Y       = [y for _ in coord, y in coord]
σ_noise = 0.4
img_data = render(true_scene, X, Y) .+ σ_noise .* randn(size(X))
err      = fill(σ_noise, size(X))

# ---------------------------------------------------------------------------
# 3. Model with constraints and priors
# ---------------------------------------------------------------------------
cm = @model begin
    bulge1 = Gaussian2D(amplitude = 40.0, x0 = -3.5, y0 = 0.5, sigma = 1.8, q = 0.9, theta = 0.3)
    disk1  = Sersic2D(amplitude = 12.0, x0 = -3.5, y0 = 0.5, r_eff = 5.0, n = 1.0, q = 0.7, theta = 0.3)
    bulge2 = Gaussian2D(amplitude = 20.0, x0 = 4.5, y0 = 0.0, sigma = 1.2, q = 0.9, theta = -0.2)
    disk2  = Sersic2D(amplitude = 8.0, x0 = 4.5, y0 = 0.0, r_eff = 4.0, n = 1.0, q = 0.8, theta = -0.2)
    bulge1 + disk1 + bulge2 + disk2
end

@constrain cm begin
    # --- structural constraints (same as MAP example) ---
    disk1.n
    disk2.n
    bulge1.x0    -> disk1.x0
    bulge1.y0    -> disk1.y0
    bulge1.theta -> disk1.theta
    bulge2.x0    -> disk2.x0
    bulge2.y0    -> disk2.y0
    bulge2.theta -> disk2.theta

    # --- bounds ---
    bulge1.amplitude in (0.1, 300.0)
    bulge1.sigma     in (0.1, 5.0)
    bulge1.q         in (0.1, 1.0)
    disk1.amplitude  in (0.1, 200.0)
    disk1.x0         in (-6.0, 2.0)
    disk1.y0         in (-5.0, 3.0)
    disk1.r_eff      in (0.3, 8.0)
    disk1.q          in (0.1, 1.0)
    disk1.theta      in (-1.6, 1.6)
    bulge2.amplitude in (0.1, 300.0)
    bulge2.sigma     in (0.1, 5.0)
    bulge2.q         in (0.1, 1.0)
    disk2.amplitude  in (0.1, 200.0)
    disk2.x0         in (-1.0, 7.0)
    disk2.y0         in (-3.0, 5.0)
    disk2.r_eff      in (0.3, 8.0)
    disk2.q          in (0.1, 1.0)
    disk2.theta      in (-1.6, 1.6)

    # --- priors (Truncated to the bounds above wherever support is wider) ---
    bulge1.amplitude ~ Truncated(LogNormal(log(50.0), 0.8), 0.1, 300.0)
    bulge1.sigma     ~ Truncated(LogNormal(log(1.5),  0.5), 0.1, 5.0)
    bulge1.q         ~ Truncated(Beta(4, 2), 0.1, 1.0)
    disk1.amplitude  ~ Truncated(LogNormal(log(15.0), 0.8), 0.1, 200.0)
    disk1.x0         ~ Truncated(Normal(-2.0, 2.0), -6.0, 2.0)
    disk1.y0         ~ Truncated(Normal(-0.5, 2.0), -5.0, 3.0)
    disk1.r_eff      ~ Truncated(LogNormal(log(4.0),  0.5), 0.3, 8.0)
    disk1.q          ~ Truncated(Beta(4, 2), 0.1, 1.0)
    disk1.theta      ~ Uniform(-1.6, 1.6)
    bulge2.amplitude ~ Truncated(LogNormal(log(30.0), 0.8), 0.1, 300.0)
    bulge2.sigma     ~ Truncated(LogNormal(log(1.0),  0.5), 0.1, 5.0)
    bulge2.q         ~ Truncated(Beta(4, 2), 0.1, 1.0)
    disk2.amplitude  ~ Truncated(LogNormal(log(10.0), 0.8), 0.1, 200.0)
    disk2.x0         ~ Truncated(Normal(3.0, 2.0), -1.0, 7.0)
    disk2.y0         ~ Truncated(Normal(1.5, 2.0), -3.0, 5.0)
    disk2.r_eff      ~ Truncated(LogNormal(log(3.5),  0.5), 0.3, 8.0)
    disk2.q          ~ Truncated(Beta(4, 2), 0.1, 1.0)
    disk2.theta      ~ Uniform(-1.6, 1.6)
end

# ---------------------------------------------------------------------------
# 4. Log-posterior target + AD gradient wrapper
# ---------------------------------------------------------------------------
target = ObjectiveFunction(cm, (X, Y), img_data, err; statistic = logposterior)
ℓ = ADgradient(:ForwardDiff, target)

println("free parameters : ", nfree(cm))
println("parameter names : ", paramnames(cm))

# ---------------------------------------------------------------------------
# 5. NUTS sampling
# ---------------------------------------------------------------------------
θ_init    = AstroFit.params(cm)
n_adapts  = 500
n_samples = 1000

Random.seed!(42)
chain = AbstractMCMC.sample(
    Random.default_rng(),
    AbstractMCMC.LogDensityModel(ℓ),
    NUTS(0.8; max_depth = 10),
    n_adapts + n_samples;
    n_adapts        = n_adapts,
    initial_params  = θ_init,
    discard_initial = n_adapts,
    progress        = true,
    chain_type      = Chains,
    param_names     = string.(paramnames(cm)),
)

# ---------------------------------------------------------------------------
# 6. Diagnostics
# ---------------------------------------------------------------------------
println("\n", chain)

posterior_median = vec(median(Array(chain); dims = 1))

# ---------------------------------------------------------------------------
# 7. Best-fit image from posterior median
# ---------------------------------------------------------------------------
fit = withparams(cm, posterior_median)
img_fit   = render(fit, X, Y)
img_resid = img_data .- img_fit

logstretch(img) = log10.(clamp.(img, 0.1, Inf))

fig = Figure(size = (1600, 450))
crange_log = extrema(logstretch(img_data))
titles = ["Data", "Posterior median fit", "Residual"]
images = [logstretch(img_data), logstretch(img_fit), img_resid]
cmaps  = [:inferno, :inferno, :balance]

for (i, (ttl, img, cmap)) in enumerate(zip(titles, images, cmaps))
    ax = Axis(fig[1, 2i-1]; title = ttl, aspect = DataAspect(), xlabel = "x",
              ylabel = i == 1 ? "y" : "")
    cr = cmap === :balance ? ((-1, 1) .* maximum(abs, img_resid)) : crange_log
    hm = heatmap!(ax, coord, coord, img; colormap = cmap, colorrange = cr)
    Colorbar(fig[1, 2i], hm; width = 12)
end
Label(fig[0, :], "NUTS posterior — blended galaxies ($(nfree(cm)) free params, log₁₀ stretch)";
      fontsize = 16, font = :bold)
display(fig)
save("examples/blended_galaxies_nuts_fit.png", fig; px_per_unit = 2)

# ---------------------------------------------------------------------------
# 8. Pair plot of galaxy-1 disk parameters (position + morphology)
# ---------------------------------------------------------------------------
g1_names = ["disk1_x0", "disk1_y0", "disk1_r_eff", "disk1_q", "disk1_theta",
            "bulge1_amplitude", "bulge1_sigma", "bulge1_q"]
true_vals = [-2.0, -1.0, 3.5, 0.5, 0.8, 60.0, 1.0, 0.7]

pp = pairplot(chain[g1_names], PairPlots.Truth(
    Dict(n => v for (n, v) in zip(g1_names, true_vals))))
save("examples/blended_galaxies_nuts_pairs_g1.png", pp; px_per_unit = 2)
println("saved → examples/blended_galaxies_nuts_fit.png")
println("saved → examples/blended_galaxies_nuts_pairs_g1.png")
