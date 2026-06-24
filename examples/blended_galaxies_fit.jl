# Bulge+disk decomposition of two blended galaxies with ellipticity.
# Bulge: Gaussian2D, disk: Sersic2D (n=1, exponential).
# Within each galaxy, bulge and disk share the same center and position angle.
#
# Run with:  julia --project=. examples/blended_galaxies_fit.jl

using AstroFit
using Optimization, OptimizationOptimJL, ForwardDiff
using CairoMakie
using Random

# ---------------------------------------------------------------------------
# 1. True scene — two elliptical galaxies, partially overlapping
# ---------------------------------------------------------------------------
true_scene = @model begin
    bulge1 = Gaussian2D(amplitude = 60.0, x0 = -2.0, y0 = -1.0, sigma = 1.0, q = 0.7, theta = 0.8)
    disk1 = Sersic2D(amplitude = 20.0, x0 = -2.0, y0 = -1.0, r_eff = 3.5, n = 1.0, q = 0.5, theta = 0.8)
    bulge2 = Gaussian2D(amplitude = 35.0, x0 = 3.0, y0 = 1.5, sigma = 0.7, q = 0.8, theta = -0.5)
    disk2 = Sersic2D(amplitude = 12.0, x0 = 3.0, y0 = 1.5, r_eff = 2.8, n = 1.0, q = 0.6, theta = -0.5)
    bulge1 + disk1 + bulge2 + disk2
end

# ---------------------------------------------------------------------------
# 2. Synthetic image
# ---------------------------------------------------------------------------
Random.seed!(42)
npix = 100
coord = range(-8.0, 8.0; length = npix)
X = [x for x in coord, _ in coord]
Y = [y for _ in coord, y in coord]
σ_noise = 0.4
img_true = render(true_scene, X, Y)
img_data = img_true .+ σ_noise .* randn(size(X))
err = fill(σ_noise, size(X))

# ---------------------------------------------------------------------------
# 3. Fitting model — deliberately bad initial guess + constraints
# ---------------------------------------------------------------------------
cm = @model begin
    bulge1 = Gaussian2D(amplitude = 20.0, x0 = -3.5, y0 = 0.5, sigma = 2.5, q = 1.0, theta = 0.0)
    disk1 = Sersic2D(amplitude = 8.0, x0 = -3.5, y0 = 0.5, r_eff = 5.0, n = 1.0, q = 0.9, theta = 0.0)
    bulge2 = Gaussian2D(amplitude = 15.0, x0 = 4.5, y0 = 0.0, sigma = 1.5, q = 1.0, theta = 0.0)
    disk2 = Sersic2D(amplitude = 5.0, x0 = 4.5, y0 = 0.0, r_eff = 4.5, n = 1.0, q = 0.9, theta = 0.0)
    bulge1 + disk1 + bulge2 + disk2
end

@constrain cm begin
    # fix disk Sersic index
    disk1.n
    disk2.n
    # bulge center and PA tied to its disk
    bulge1.x0 -> disk1.x0
    bulge1.y0 -> disk1.y0
    bulge1.theta -> disk1.theta
    bulge2.x0 -> disk2.x0
    bulge2.y0 -> disk2.y0
    bulge2.theta -> disk2.theta
    # physical bounds
    bulge1.amplitude in (0.1, 300.0)
    bulge1.sigma in (0.1, 5.0)
    bulge1.q in (0.1, 1.0)
    disk1.amplitude in (0.1, 200.0)
    disk1.x0 in (-6.0, 2.0)
    disk1.y0 in (-5.0, 3.0)
    disk1.r_eff in (0.3, 8.0)
    disk1.q in (0.1, 1.0)
    disk1.theta in (-1.6, 1.6)
    bulge2.amplitude in (0.1, 300.0)
    bulge2.sigma in (0.1, 5.0)
    bulge2.q in (0.1, 1.0)
    disk2.amplitude in (0.1, 200.0)
    disk2.x0 in (-1.0, 7.0)
    disk2.y0 in (-3.0, 5.0)
    disk2.r_eff in (0.3, 8.0)
    disk2.q in (0.1, 1.0)
    disk2.theta in (-1.6, 1.6)
end

# ---------------------------------------------------------------------------
# 4. Fit
# ---------------------------------------------------------------------------
prob = OptimizationProblem(cm, (X, Y), img_data, err)
sol = solve(prob, Fminbox(LBFGS()))

fit = withparams(cm, sol.u) 

println("retcode         : ", sol.retcode)
println("free parameters : ", nfree(cm))
println("parameter names : ", paramnames(cm))
println("best fit values : ", round.(sol.u; digits = 3))
println()

# ---------------------------------------------------------------------------
# 5. Plot: data | initial guess | best fit | residual
# ---------------------------------------------------------------------------
img_init = render(cm, X, Y)
img_fit = render(fit, X, Y)
img_resid = img_data .- img_fit

logstretch(img) = log10.(clamp.(img, 0.1, Inf))

fig = Figure(size = (1600, 450))
crange_log = extrema(logstretch(img_data))

titles = ["Data", "Initial guess", "Best fit", "Residual"]
images = [logstretch(img_data), logstretch(img_init), logstretch(img_fit), img_resid]
cmaps = [:inferno, :inferno, :inferno, :balance]

for (i, (ttl, img, cmap)) in enumerate(zip(titles, images, cmaps))
    ax = Axis(
        fig[1, 2i - 1]; title = ttl, aspect = DataAspect(), xlabel = "x",
        ylabel = i == 1 ? "y" : ""
    )
    cr = cmap === :balance ? ((-1, 1) .* maximum(abs, img_resid)) : crange_log
    hm = heatmap!(ax, coord, coord, img; colormap = cmap, colorrange = cr)
    Colorbar(fig[1, 2i], hm; width = 12)
end

Label(
    fig[0, :],
    "Blended galaxies: Gaussian bulge + Sérsic disk — $(nfree(cm)) free parameters (log₁₀ stretch)";
    fontsize = 16, font = :bold
)
display(fig)
save("examples/blended_galaxies_fit.png", fig; px_per_unit = 2)
println("saved → examples/blended_galaxies_fit.png")
