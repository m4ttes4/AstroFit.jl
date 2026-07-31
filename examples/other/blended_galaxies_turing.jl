# Bayesian bulge+disk decomposition of two blended galaxies with Turing.jl + NUTS.
# Self-contained: the Gaussian2D / Sérsic2D renderers are reimplemented as plain
# helper functions (matching AstroFit's formulas) so this example has no dependency
# on AstroFit — only Turing and the plotting stack.
#
# Why Turing:
#   Hard −Inf bounds (to keep r_eff > 0, q ∈ (0,1), …) have undefined gradients and
#   crash some AD backends (e.g. ReverseDiff: `log` of a negative base inside `^`).
#   Turing reparametrizes each bounded prior (LogNormal/Beta/Uniform) to an
#   unconstrained space and adds the log-Jacobian, so the sampler never sees an
#   invalid value — gradients are defined everywhere and any AD backend works.
#
# Structural assumptions (baked into the parametrization, not sampled):
#   * Sérsic index n fixed to 1 (exponential disk).
#   * Each bulge shares its centre (x0,y0) and orientation (theta) with its disk.
#   ⇒ 18 free parameters total.
#
# Run with:  julia --project=examples/ examples/blended_galaxies_turing.jl

using Distributions
using Turing
using ADTypes, ReverseDiff
using CairoMakie, PairPlots
using MCMCChains
using Random
using Statistics

# ---------------------------------------------------------------------------
# 0. Renderers (match AstroFit's Gaussian2D / Sersic2D, single-pixel form)
# ---------------------------------------------------------------------------
function gaussian2d(amp, x0, y0, sigma, q, theta, x, y)
    dx, dy = x - x0, y - y0
    c, s = cos(theta), sin(theta)
    xr =  c * dx + s * dy
    yr = -s * dx + c * dy
    return amp * exp(-0.5 * (xr^2 + (yr / q)^2) / sigma^2)
end

function sersic2d(amp, x0, y0, r_eff, n, q, theta, x, y)
    bn = 2n - 1 / 3 + 4 / (405n)            # Ciotti & Bertin 1999 approx
    dx, dy = x - x0, y - y0
    c, s = cos(theta), sin(theta)
    xr =  c * dx + s * dy
    yr = -s * dx + c * dy
    r = sqrt(xr^2 + (yr / q)^2)
    return amp * exp(-bn * ((r / r_eff)^(1 / n) - 1))
end

# Full scene from the 18-vector θ (constraints applied: n=1, bulge centre/theta
# tied to its disk). Order of θ matches `paramnames` below and the Turing model.
function scene_image(θ, X, Y)
    b1_amp, b1_sig, b1_q,
    d1_amp, d1_x0, d1_y0, d1_reff, d1_q, d1_theta,
    b2_amp, b2_sig, b2_q,
    d2_amp, d2_x0, d2_y0, d2_reff, d2_q, d2_theta = θ
    n = 1.0
    return gaussian2d.(b1_amp, d1_x0, d1_y0, b1_sig, b1_q, d1_theta, X, Y) .+
           sersic2d.(d1_amp, d1_x0, d1_y0, d1_reff, n, d1_q, d1_theta, X, Y) .+
           gaussian2d.(b2_amp, d2_x0, d2_y0, b2_sig, b2_q, d2_theta, X, Y) .+
           sersic2d.(d2_amp, d2_x0, d2_y0, d2_reff, n, d2_q, d2_theta, X, Y)
end

paramnames = [:bulge1_amplitude, :bulge1_sigma, :bulge1_q,
              :disk1_amplitude, :disk1_x0, :disk1_y0, :disk1_r_eff, :disk1_q, :disk1_theta,
              :bulge2_amplitude, :bulge2_sigma, :bulge2_q,
              :disk2_amplitude, :disk2_x0, :disk2_y0, :disk2_r_eff, :disk2_q, :disk2_theta]

# ---------------------------------------------------------------------------
# 1. Synthetic image from the true scene
# ---------------------------------------------------------------------------
Random.seed!(42)
npix    = 100
coord   = range(-8.0, 8.0; length = npix)
X       = [x for x in coord, _ in coord]
Y       = [y for _ in coord, y in coord]
σ_noise = 0.4

true_img =
    gaussian2d.(60.0, -2.0, -1.0, 1.0, 0.7,  0.8, X, Y) .+
    sersic2d.(  20.0, -2.0, -1.0, 3.5, 1.0, 0.5,  0.8, X, Y) .+
    gaussian2d.(35.0,  3.0,  1.5, 0.7, 0.8, -0.5, X, Y) .+
    sersic2d.(  12.0,  3.0,  1.5, 2.8, 1.0, 0.6, -0.5, X, Y)

img_data = true_img .+ σ_noise .* randn(size(X))

# ---------------------------------------------------------------------------
# 2. Turing model — priors (in paramnames order) + Gaussian likelihood
# ---------------------------------------------------------------------------
@model function blended_model(X, Y, img_data, σ)
    bulge1_amplitude ~ LogNormal(log(50.0), 0.8)
    bulge1_sigma     ~ LogNormal(log(1.5),  0.5)
    bulge1_q         ~ Beta(4, 2)
    disk1_amplitude  ~ LogNormal(log(15.0), 0.8)
    disk1_x0         ~ Normal(-2.0, 2.0)
    disk1_y0         ~ Normal(-0.5, 2.0)
    disk1_r_eff      ~ LogNormal(log(4.0),  0.5)
    disk1_q          ~ Beta(4, 2)
    disk1_theta      ~ Uniform(-1.6, 1.6)
    bulge2_amplitude ~ LogNormal(log(30.0), 0.8)
    bulge2_sigma     ~ LogNormal(log(1.0),  0.5)
    bulge2_q         ~ Beta(4, 2)
    disk2_amplitude  ~ LogNormal(log(10.0), 0.8)
    disk2_x0         ~ Normal(3.0, 2.0)
    disk2_y0         ~ Normal(1.5, 2.0)
    disk2_r_eff      ~ LogNormal(log(3.5),  0.5)
    disk2_q          ~ Beta(4, 2)
    disk2_theta      ~ Uniform(-1.6, 1.6)

    θ = [bulge1_amplitude, bulge1_sigma, bulge1_q,
         disk1_amplitude, disk1_x0, disk1_y0, disk1_r_eff, disk1_q, disk1_theta,
         bulge2_amplitude, bulge2_sigma, bulge2_q,
         disk2_amplitude, disk2_x0, disk2_y0, disk2_r_eff, disk2_q, disk2_theta]

    μ = scene_image(θ, X, Y)
    # isotropic Gaussian likelihood, written out to stay light + AD-friendly
    Turing.@addlogprob! -sum(abs2, (img_data .- μ) ./ σ) / 2 -
                        length(img_data) * log(σ * sqrt(2π))
end

# ---------------------------------------------------------------------------
# 3. NUTS sampling (ReverseDiff backend — safe here, no walls)
# ---------------------------------------------------------------------------
n_adapts  = 500
n_samples = 1000

model = blended_model(X, Y, img_data, σ_noise)

Random.seed!(42)
chain = sample(
    model,
    NUTS(n_adapts, 0.8; max_depth = 10, adtype = AutoReverseDiff()),
    n_samples;
    progress = true,
)

# ---------------------------------------------------------------------------
# 4. Diagnostics
# ---------------------------------------------------------------------------
println("\n", chain)

# Array(chain) drops Turing's internal columns and keeps the parameters in
# declaration order == paramnames order.
posterior_median = vec(median(Array(chain); dims = 1))

# ---------------------------------------------------------------------------
# 5. Best-fit image from posterior median
# ---------------------------------------------------------------------------
img_fit   = scene_image(posterior_median, X, Y)
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
Label(fig[0, :], "Turing/NUTS posterior — blended galaxies ($(length(paramnames)) free params, log₁₀ stretch)";
      fontsize = 16, font = :bold)
display(fig)
save("examples/blended_galaxies_turing_fit.png", fig; px_per_unit = 2)

# ---------------------------------------------------------------------------
# 6. Pair plot of galaxy-1 disk parameters (position + morphology)
# ---------------------------------------------------------------------------
g1_names = [:disk1_x0, :disk1_y0, :disk1_r_eff, :disk1_q, :disk1_theta,
            :bulge1_amplitude, :bulge1_sigma, :bulge1_q]
true_vals = [-2.0, -1.0, 3.5, 0.5, 0.8, 60.0, 1.0, 0.7]

pp = pairplot(chain[g1_names], PairPlots.Truth(
    Dict(n => v for (n, v) in zip(g1_names, true_vals))))
save("examples/blended_galaxies_turing_pairs_g1.png", pp; px_per_unit = 2)
println("saved → examples/blended_galaxies_turing_fit.png")
println("saved → examples/blended_galaxies_turing_pairs_g1.png")
