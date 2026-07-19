# Fit a double Gaussian + linear continuum to synthetic noisy data with Pigeons.jl.
#
# Run with:  julia --project=. examples/double_gaussian_pigeons.jl

using AstroFit
using Distributions
using Pigeons
using CairoMakie
using Random
using Statistics

# ---------------------------------------------------------------------------
# 1. True model — two emission lines on a sloped continuum
# ---------------------------------------------------------------------------
true_model = @model begin
    cont = Linear1D(slope = 0.02, intercept = 1.0)
    g1 = Gaussian1D(amplitude = 8.0, mean = 5.0, sigma = 0.6)
    g2 = Gaussian1D(amplitude = 4.0, mean = 7.5, sigma = 0.9)
    cont + g1 + g2
end

# ---------------------------------------------------------------------------
# 2. Synthetic data
# ---------------------------------------------------------------------------
Random.seed!(123)
x = collect(0.0:0.05:12.0)
σ_noise = 0.3
y_true = render(true_model, x)
y = y_true .+ σ_noise .* randn(length(x))
err = fill(σ_noise, length(x))

# ---------------------------------------------------------------------------
# 3. Fitting model — deliberately off initial guess + bounded priors + ties
# ---------------------------------------------------------------------------
cm = @model begin
    cont = Linear1D(slope = 0.0, intercept = 0.5)
    g1 = Gaussian1D(amplitude = 5.0, mean = 4.5, sigma = 0.8)
    g2 = Gaussian1D(amplitude = 3.0, mean = 8.0, sigma = 0.8)
    cont + g1 + g2
end

@constrain cm begin
    cont.slope in (-0.1, 0.1)
    cont.intercept in (0.0, 2.0)
    g1.amplitude in (0.0, 15.0)
    g1.mean in (3.5, 6.5)
    g1.sigma in (0.2, 1.5)
    g2.mean in (6.0, 9.0)
    # g2 width tied to g1 width (same instrument resolution)
    g2.sigma -> g1.sigma
    # g2 amplitude tied to half of g1
    g2.amplitude -> 0.5 * g1.amplitude
    # logposterior no longer auto-rejects out-of-bounds points — priors below
    # are what give -Inf outside the box (Pigeons needs this hard boundary)
    cont.slope ~ Uniform(-0.1, 0.1)
    cont.intercept ~ Uniform(0.0, 2.0)
    g1.amplitude ~ Uniform(0.0, 15.0)
    g1.mean ~ Uniform(3.5, 6.5)
    g1.sigma ~ Uniform(0.2, 1.5)
    g2.mean ~ Uniform(6.0, 9.0)
end

# ---------------------------------------------------------------------------
# 4. Sample the posterior with Pigeons
# ---------------------------------------------------------------------------
target = ObjectiveFunction(cm, x, y, err; statistic = logposterior)
pt = pigeons(
    target = target,
    n_rounds = 10,
    n_chains = 8,
    # seed = 123,
    record = [traces; record_default()],
)

raw_samples = sample_array(pt)
param_samples = raw_samples[:, 1:nfree(cm), :]
sample_matrix = reshape(permutedims(param_samples, (1, 3, 2)), :, nfree(cm))
posterior_median = vec(median(sample_matrix; dims = 1))
fit_tree = withparams(cm, posterior_median)

println("free parameters     : ", nfree(cm))
println("parameter names     : ", paramnames(cm))
println("posterior medians   : ", round.(posterior_median; digits = 4))
println("posterior log density: ", round(target(posterior_median); digits = 4))
println()

# ---------------------------------------------------------------------------
# 5. Plot: initial guess vs posterior samples vs data
# ---------------------------------------------------------------------------
y_init = render(cm, x)
y_fit = render(fit_tree, x)

fig = Figure(size = (900, 500))
ax = Axis(
    fig[1, 1]; xlabel = "x", ylabel = "y",
    title = "Double Gaussian + Linear Continuum Posterior Fit"
)

scatter!(ax, x, y; color = :grey60, markersize = 4, label = "data")
lines!(ax, x, y_true; color = :black, linestyle = :dash, label = "truth")
lines!(ax, x, y_init; color = :dodgerblue, linestyle = :dot, label = "initial guess")

n_plot = min(100, size(sample_matrix, 1))
draw_ids = unique(round.(Int, range(1, size(sample_matrix, 1); length = n_plot)))
for i in draw_ids
    y_draw = render(withparams(cm, vec(sample_matrix[i, :])), x)
    lines!(ax, x, y_draw; color = (:red, 0.04))
end

lines!(ax, x, y_fit; color = :red, linewidth = 2, label = "posterior median")

axislegend(ax; position = :lt)

save("examples/double_gaussian_pigeons.png", fig; px_per_unit = 2)
println("saved -> examples/double_gaussian_pigeons.png")
