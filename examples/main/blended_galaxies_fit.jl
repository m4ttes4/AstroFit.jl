# Bulge+disk decomposition of two blended galaxies with ellipticity.
# Galaxy 1: inclined disk-dominated spiral (exponential disk, n~1).
# Galaxy 2: bulge-dominated early type (concentrated spheroid, n~3.5).
# Within each galaxy, bulge and disk share the same center and position angle;
# the Sersic index n is a free parameter of the fit.
#
# Run with:  julia --project=examples examples/main/blended_galaxies_fit.jl

using AstroFit
using Optimization, OptimizationOptimJL, ForwardDiff
using CairoMakie
using Random

# ---------------------------------------------------------------------------
# 1. True scene — two elliptical galaxies, partially overlapping
# ---------------------------------------------------------------------------
true_scene = @model begin
    bulge1 = Gaussian2D(amplitude = 25.0, x0 = -2.0, y0 = -1.0, sigma = 0.8, q = 0.75, theta = 0.8)
    disk1 = Sersic2D(amplitude = 50.0, x0 = -2.0, y0 = -1.0, r_eff = 2.5, n = 1.0, q = 0.38, theta = 0.8)
    bulge2 = Gaussian2D(amplitude = 90.0, x0 = 3.0, y0 = 1.5, sigma = 1.4, q = 0.9, theta = -0.5)
    disk2 = Sersic2D(amplitude = 15.0, x0 = 3.0, y0 = 1.5, r_eff = 1.6, n = 3.5, q = 0.85, theta = -0.5)
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
    bulge1 = Gaussian2D(amplitude = 15.0, x0 = -3.5, y0 = 0.5, sigma = 1.5, q = 0.9, theta = 0.3)
    disk1 = Sersic2D(amplitude = 35.0, x0 = -3.5, y0 = 0.5, r_eff = 3.5, n = 1.5, q = 0.5, theta = 0.3)
    bulge2 = Gaussian2D(amplitude = 60.0, x0 = 4.5, y0 = 0.0, sigma = 1.0, q = 0.9, theta = -0.2)
    disk2 = Sersic2D(amplitude = 8.0, x0 = 4.5, y0 = 0.0, r_eff = 2.5, n = 2.0, q = 0.9, theta = -0.2)
    bulge1 + disk1 + bulge2 + disk2
end

@constrain cm begin
    # free Sersic index: n~1 (exponential) vs n~3.5 (concentrated) is what
    # morphologically separates the two galaxies
    disk1.n in (0.5, 6.0)
    disk2.n in (0.5, 6.0)
    # bulge center and PA tied to its disk
    bulge1.x0 -> disk1.x0
    bulge1.y0 -> disk1.y0
    bulge1.theta -> disk1.theta
    bulge2.x0 -> disk2.x0
    bulge2.y0 -> disk2.y0
    bulge2.theta -> disk2.theta
    # loose physical bounds: positivity and the image footprint, little more.
    # theta stays in (-pi/2, pi/2] territory: theta and theta+pi are the same
    # ellipse, so a wider window would only create duplicate minima.
    bulge1.amplitude in (0.0, 500.0)
    bulge1.sigma in (0.05, 10.0)
    bulge1.q in (0.05, 1.0)
    disk1.amplitude in (0.0, 500.0)
    disk1.x0 in (-8.0, 8.0)
    disk1.y0 in (-8.0, 8.0)
    disk1.r_eff in (0.1, 10.0)
    disk1.q in (0.05, 1.0)
    disk1.theta in (-1.6, 1.6)
    bulge2.amplitude in (0.0, 500.0)
    bulge2.sigma in (0.05, 10.0)
    bulge2.q in (0.05, 1.0)
    disk2.amplitude in (0.0, 500.0)
    disk2.x0 in (-8.0, 8.0)
    disk2.y0 in (-8.0, 8.0)
    disk2.r_eff in (0.1, 10.0)
    disk2.q in (0.05, 1.0)
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
save(joinpath(@__DIR__, "blended_galaxies_fit.png"), fig; px_per_unit = 2)
println("saved → ", joinpath(@__DIR__, "blended_galaxies_fit.png"))
