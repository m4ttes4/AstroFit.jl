# Morphological decomposition of a grand-design spiral galaxy.
#
#   bulge   : Sérsic n=4 (de Vaucouleurs)      — round, concentrated
#   disk    : Sérsic n=1 (exponential)         — inclined, sets the projected ellipticity
#   bar     : Sérsic n≈0.5, high ellipticity    — flat bar, misaligned from the disk PA
#   arms    : SpiralArm2D (defined below)       — logarithmic spiral in the deprojected disk plane
#
# Co-centred smooth profiles (bulge/disk/bar) make the central flux degenerate — a
# near-flat likelihood direction that stalls the optimizer. Two fixes here: the bar
# is misaligned from the disk (a distinct angular signature the data can pin), and a
# weak prior on the bulge amplitude regularizes the residual flux trade-off (MAP fit).
#
# The spiral component does the "spherical/polar" work internally: it takes image
# (x, y), rotates by the position angle, deprojects the foreshortened minor axis by
# 1/cos(inclination), then converts to disk-plane polar (r, φ) and lays down a
# logarithmic-spiral brightness ridge. Because `render` is variadic on coordinates,
# it drops straight into a normal (x, y) model tree alongside the Sérsic components.
#
# Run with:  julia --project=examples/ examples/spiral_galaxy_fit.jl

using AstroFit
import AstroFit: render          # extend render for the inline model
using Optimization, OptimizationOptimJL, ForwardDiff
using Distributions               # weak prior to regularize the degenerate flux direction
using CairoMakie
using Random

# ---------------------------------------------------------------------------
# 0. A logarithmic-spiral-arm model living in an inclined disk plane
# ---------------------------------------------------------------------------
# Brightness:  A · exp(-r/r_s) · exp(κ·(cos(arg) − 1))
#   where  arg = m·(φ − ln(r)/tan(pitch) − φ₀)
# The second exponential is a von-Mises-like ridge: it is in (0, 1], peaks at 1 on
# the arm crest, is strictly positive and branch-free (AD-safe, no abs/min/max).
# `r_in` softens the radius so √ and ln stay finite and differentiable at the centre.
Base.@kwdef struct SpiralArm2D{T <: Real} <: AbstractModel
    amplitude::T = 1.0    # peak brightness on the arm ridge
    x0::T = 0.0           # centre
    y0::T = 0.0
    r_s::T = 3.0          # radial e-folding scale of arm brightness
    incl::T = 0.0         # disk inclination (0 = face-on), radians
    theta::T = 0.0        # position angle of the major axis, radians
    pitch::T = 0.3        # pitch angle of the logarithmic spiral, radians
    phi0::T = 0.0         # arm phase (azimuth offset)
    m::T = 2.0            # number of arms — discrete, kept fixed during fitting
    kappa::T = 3.0        # arm sharpness / contrast
    r_in::T = 0.5         # inner softening radius
end

SpiralArm2D(a::Real, x0::Real, y0::Real, r_s::Real, incl::Real, theta::Real,
    pitch::Real, phi0::Real, m::Real, kappa::Real, r_in::Real) =
    SpiralArm2D(promote(a, x0, y0, r_s, incl, theta, pitch, phi0, m, kappa, r_in)...)

function render(s::SpiralArm2D, x::Number, y::Number)
    dx, dy = x - s.x0, y - s.y0
    cost, sint = cos(s.theta), sin(s.theta)
    xr = cost * dx + sint * dy          # along major axis
    yr = -sint * dx + cost * dy         # along (projected) minor axis
    yd = yr / cos(s.incl)               # deproject into the disk plane
    r = sqrt(xr^2 + yd^2 + s.r_in^2)    # softened radius (finite gradient at centre)
    phi = atan(yd, xr)
    arg = s.m * (phi - log(r) / tan(s.pitch) - s.phi0)
    arm = exp(s.kappa * (cos(arg) - 1)) # ridge in (0, 1]
    return s.amplitude * exp(-r / s.r_s) * arm
end

# ---------------------------------------------------------------------------
# 1. True scene
# ---------------------------------------------------------------------------
# Disk/spiral share geometry: a disk inclined by i has projected axis ratio q = cos(i).
incl_true = 0.92                          # ≈ 53°
q_disk = cos(incl_true)                   # ≈ 0.60
pa = 0.35                                 # disk/arms shared position angle
bar_pa = -0.40                            # bar deliberately MISALIGNED from the disk (≈ 43°)

# Why bar_pa ≠ pa: bulge, disk and bar are all co-centred and smooth, so their
# central fluxes trade off — a near-flat direction (cond ≈ 4e8) that makes the fit
# crawl. A misaligned bar carries a distinct angular signature the data can pin,
# which breaks that degeneracy. (Real bars are commonly offset from the disk PA.)
true_scene = @model begin
    bulge = Sersic2D(amplitude = 90.0, x0 = 0.0, y0 = 0.0, r_eff = 1.0, n = 4.0, q = 0.9, theta = pa)
    disk = Sersic2D(amplitude = 22.0, x0 = 0.0, y0 = 0.0, r_eff = 4.5, n = 1.0, q = q_disk, theta = pa)
    bar = Sersic2D(amplitude = 28.0, x0 = 0.0, y0 = 0.0, r_eff = 2.8, n = 0.5, q = 0.30, theta = bar_pa)
    arms = SpiralArm2D(amplitude = 32.0, x0 = 0.0, y0 = 0.0, r_s = 4.5, incl = incl_true,
        theta = pa, pitch = 0.38, phi0 = 0.0, m = 2.0, kappa = 2.0, r_in = 0.7)
    bulge + disk + bar + arms
end

# ---------------------------------------------------------------------------
# 2. Synthetic image
# ---------------------------------------------------------------------------
Random.seed!(42)
npix = 100
coord = range(-12.0, 12.0; length = npix)
X = [x for x in coord, _ in coord]
Y = [y for _ in coord, y in coord]
σ_noise = 0.5
img_true = render(true_scene, X, Y)
img_data = img_true .+ σ_noise .* randn(size(X))
err = fill(σ_noise, size(X))

# ---------------------------------------------------------------------------
# 3. Fitting model + constraints
# ---------------------------------------------------------------------------
# Teaching setup: start the *well-behaved* parameters (amplitudes, sizes, geometry)
# from a deliberately bad guess, but start the gradient-hostile spiral parameters
# (pitch, phase) near the truth and bound them tightly — periodic/multimodal terms
# will otherwise drag LBFGS into the wrong basin. Arm count `m` is discrete and fixed.
cm = @model begin
    bulge = Sersic2D(amplitude = 60.0, x0 = 0.5, y0 = -0.5, r_eff = 1.6, n = 4.0, q = 0.9, theta = 0.2)
    disk = Sersic2D(amplitude = 14.0, x0 = 0.5, y0 = -0.5, r_eff = 6.0, n = 1.0, q = 0.7, theta = 0.2)
    bar = Sersic2D(amplitude = 18.0, x0 = 0.5, y0 = -0.5, r_eff = 3.6, n = 0.5, q = 0.45, theta = 0.2)
    arms = SpiralArm2D(amplitude = 20.0, x0 = 0.5, y0 = -0.5, r_s = 6.0, incl = 0.7,
        theta = 0.2, pitch = 0.40, phi0 = 0.05, m = 2.0, kappa = 2.0, r_in = 0.7)
    bulge + disk + bar + arms
end

@constrain cm begin
    # fixed structural choices
    bulge.n                     # de Vaucouleurs bulge
    bar.n                       # flat bar profile
    arms.m                      # 2 arms (discrete)
    arms.r_in                   # softening core

    # one shared galaxy centre (bulge is master)
    disk.x0 -> bulge.x0
    disk.y0 -> bulge.y0
    bar.x0 -> bulge.x0
    bar.y0 -> bulge.y0
    arms.x0 -> bulge.x0
    arms.y0 -> bulge.y0

    # disk and arms share the same inclined plane (PA + inclination)
    arms.theta -> disk.theta
    arms.incl -> acos(disk.q)   # projected axis ratio ↔ inclination

    # physical bounds — well-behaved parameters explore widely
    bulge.amplitude in (1.0, 300.0)
    bulge.r_eff in (0.3, 4.0)
    bulge.q in (0.5, 1.0)
    bulge.theta in (-1.6, 1.6)
    bulge.x0 in (-3.0, 3.0)
    bulge.y0 in (-3.0, 3.0)
    disk.amplitude in (1.0, 200.0)
    disk.r_eff in (1.0, 10.0)
    disk.n in (0.5, 2.0)
    disk.q in (0.2, 0.99)
    disk.theta in (-1.6, 1.6)
    bar.amplitude in (1.0, 200.0)
    bar.r_eff in (0.5, 6.0)
    bar.q in (0.1, 0.7)
    bar.theta in (-1.6, 1.6)
    arms.amplitude in (1.0, 200.0)
    arms.r_s in (1.0, 10.0)
    arms.kappa in (0.5, 5.0)

    # gradient-hostile parameters: tight bounds around the truth
    arms.pitch in (0.25, 0.55)
    arms.phi0 in (-0.6, 0.6)

    # weak prior on the parameter that dominates the flat direction (bulge_amplitude,
    # eigenvector weight 0.89). Broad and off-truth (truth = 90) — it only damps the
    # residual flux wander, it does not pin the value. Turns the fit into a MAP estimate.
    bulge.amplitude ~ Normal(100.0, 30.0)
end

# ---------------------------------------------------------------------------
# 4. Fit
# ---------------------------------------------------------------------------
# :neglogposterior so the prior above actually regularizes the fit (MAP estimate);
# without this the χ² objective would ignore the prior entirely.
prob = OptimizationProblem(cm, (X, Y), img_data, err; statistic = :neglogposterior)
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
    "Spiral galaxy: Sérsic bulge + disk + bar + logarithmic arms — $(nfree(cm)) free parameters (log₁₀ stretch)";
    fontsize = 16, font = :bold
)
display(fig)
save("examples/spiral_galaxy_fit.png", fig; px_per_unit = 2)
println("saved → examples/spiral_galaxy_fit.png")
