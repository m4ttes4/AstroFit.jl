# Fit a double Gaussian + linear continuum to synthetic noisy data.
#
# Run with:  julia --project=. examples/double_gaussian_fit.jl

using AstroFit
using Optimization, OptimizationOptimJL, ForwardDiff
using CairoMakie
using Random

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
# 3. Fitting model — deliberately off initial guess + ties
# ---------------------------------------------------------------------------
cm = @model begin
    cont = Linear1D(slope = 0.0, intercept = 0.5)
    g1 = Gaussian1D(amplitude = 5.0, mean = 4.5, sigma = 0.8)
    g2 = Gaussian1D(amplitude = 3.0, mean = 8.0, sigma = 0.8)
    cont + g1 + g2
end

@constrain cm begin
    # g2 width tied to g1 width (same instrument resolution)
    g2.sigma -> g1.sigma
    # g2 amplitude tied to half of g1
    g2.amplitude -> 0.5 * g1.amplitude
end

# ---------------------------------------------------------------------------
# 4. Fit
# ---------------------------------------------------------------------------
prob = OptimizationProblem(cm, x, y, err)
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
y_init = render(cm, x)
y_fit = render(fit_tree, x)

fig = Figure(size = (900, 500))
ax = Axis(
    fig[1, 1]; xlabel = "x", ylabel = "y",
    title = "Double Gaussian + Linear Continuum Fit"
)

scatter!(ax, x, y; color = :grey60, markersize = 4, label = "data")
lines!(ax, x, y_true; color = :black, linestyle = :dash, label = "truth")
lines!(ax, x, y_init; color = :dodgerblue, linestyle = :dot, label = "initial guess")
lines!(ax, x, y_fit; color = :red, linewidth = 2, label = "best fit")

axislegend(ax; position = :lt)

save("examples/double_gaussian_fit.png", fig; px_per_unit = 2)
println("saved → examples/double_gaussian_fit.png")
