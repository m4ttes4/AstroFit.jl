# Bayesian sampling of a double Gaussian + linear continuum with Pigeons.jl
#
# Run with:  julia --project=. examples/double_gaussian_pigeons.jl

using AstroFit
using Distributions
using Pigeons
using CairoMakie
using Random
using MCMCChains

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
# 3. Fitting model — bounded parameters + tied widths
# ---------------------------------------------------------------------------
cm = @model begin
    cont = Linear1D(slope = 0.0, intercept = 0.5)
    g1 = Gaussian1D(amplitude = 5.0, mean = 4.5, sigma = 0.8)
    g2 = Gaussian1D(amplitude = 3.0, mean = 8.0, sigma = 0.8)
    cont + g1 + g2
end

@constrain cm begin
    cont.slope in (-1.0, 1.0)
    cont.intercept in (-5.0, 10.0)
    g1.amplitude in (0.1, 20.0)
    g1.mean in (3.0, 7.0)
    g1.sigma in (0.1, 3.0)
    g2.mean in (6.0, 10.0)
    # g2 width tied to g1 width (same instrument resolution)
    g2.sigma -> g1.sigma
    # g2 amplitude tied to half of g1
    g2.amplitude -> 0.5 * g1.amplitude
end

# ---------------------------------------------------------------------------
# 4. Sample with Pigeons
# ---------------------------------------------------------------------------
target = PosteriorTarget(cm, x, y, err)

# Reference: same model structure but with flat likelihood (huge errors)
ref = PosteriorTarget(cm, x, zeros(length(x)), fill(1e6, length(x)))

pt = pigeons(
    target = target,
    reference = ref,
    n_rounds = 10,
    n_chains = 10,
    record = [traces; record_default()],
)

samples = Chains(pt)
display(samples)

# ---------------------------------------------------------------------------
# 5. Plot: data + posterior samples
# ---------------------------------------------------------------------------
chain_matrix = Array(samples)
n_draws = size(chain_matrix, 1)

fig = Figure(size = (900, 500))
ax = Axis(
    fig[1, 1]; xlabel = "x", ylabel = "y",
    title = "Double Gaussian — Pigeons.jl Posterior Samples",
)

scatter!(ax, x, y; color = :grey60, markersize = 4, label = "data")
lines!(ax, x, y_true; color = :black, linestyle = :dash, label = "truth")

n_plot = min(200, n_draws)
for i in 1:n_plot
    p_i = Vector(chain_matrix[i, 1:nfree(cm)])
    y_i = render(withparams(cm, p_i), x)
    lines!(ax, x, y_i; color = (:red, 0.03))
end

p_med = vec(median(chain_matrix[:, 1:nfree(cm)]; dims = 1))
lines!(ax, x, render(withparams(cm, p_med), x);
    color = :red, linewidth = 2, label = "median posterior")

axislegend(ax; position = :lt)

fig 

save("examples/double_gaussian_pigeons.png", fig; px_per_unit = 2)
println("saved → examples/double_gaussian_pigeons.png")
