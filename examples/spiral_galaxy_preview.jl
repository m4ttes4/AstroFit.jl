# Preview-only: render the true spiral-galaxy scene and its components. No fitting.
# Run with:  julia --project=examples/ examples/spiral_galaxy_preview.jl

using AstroFit
import AstroFit: render          # extend render for the inline model
using CairoMakie

Base.@kwdef struct SpiralArm2D{T <: Real} <: AbstractModel
    amplitude::T = 1.0
    x0::T = 0.0
    y0::T = 0.0
    r_s::T = 3.0
    incl::T = 0.0
    theta::T = 0.0
    pitch::T = 0.3
    phi0::T = 0.0
    m::T = 2.0
    kappa::T = 3.0
    r_in::T = 0.5
end

function render(s::SpiralArm2D, x::Number, y::Number)
    dx, dy = x - s.x0, y - s.y0
    cost, sint = cos(s.theta), sin(s.theta)
    xr = cost * dx + sint * dy
    yr = -sint * dx + cost * dy
    yd = yr / cos(s.incl)
    r = sqrt(xr^2 + yd^2 + s.r_in^2)
    phi = atan(yd, xr)
    arg = s.m * (phi - log(r) / tan(s.pitch) - s.phi0)
    arm = exp(s.kappa * (cos(arg) - 1))
    return s.amplitude * exp(-r / s.r_s) * arm
end

incl_true = 0.92
q_disk = cos(incl_true)
pa = 0.35

bulge = @model begin
    b = Sersic2D(amplitude = 90.0, x0 = 0.0, y0 = 0.0, r_eff = 1.0, n = 4.0, q = 0.9, theta = pa)
    b
end
disk = @model begin
    d = Sersic2D(amplitude = 22.0, x0 = 0.0, y0 = 0.0, r_eff = 4.5, n = 1.0, q = q_disk, theta = pa)
    d
end
bar = @model begin
    a = Sersic2D(amplitude = 28.0, x0 = 0.0, y0 = 0.0, r_eff = 2.8, n = 0.5, q = 0.30, theta = pa)
    a
end
arms = @model begin
    s = SpiralArm2D(amplitude = 32.0, x0 = 0.0, y0 = 0.0, r_s = 4.5, incl = incl_true,
        theta = pa, pitch = 0.38, phi0 = 0.0, m = 2.0, kappa = 2.0, r_in = 0.7)
    s
end
scene = @model begin
    bulge = Sersic2D(amplitude = 90.0, x0 = 0.0, y0 = 0.0, r_eff = 1.0, n = 4.0, q = 0.9, theta = pa)
    disk = Sersic2D(amplitude = 22.0, x0 = 0.0, y0 = 0.0, r_eff = 4.5, n = 1.0, q = q_disk, theta = pa)
    bar = Sersic2D(amplitude = 28.0, x0 = 0.0, y0 = 0.0, r_eff = 2.8, n = 0.5, q = 0.30, theta = pa)
    arms = SpiralArm2D(amplitude = 32.0, x0 = 0.0, y0 = 0.0, r_s = 4.5, incl = incl_true,
        theta = pa, pitch = 0.38, phi0 = 0.0, m = 2.0, kappa = 2.0, r_in = 0.7)
    bulge + disk + bar + arms
end

npix = 50
coord = range(-12.0, 12.0; length = npix)
X = [x for x in coord, _ in coord]
Y = [y for _ in coord, y in coord]

logstretch(img) = log10.(clamp.(img, 0.1, Inf))

panels = [
    ("Bulge (n=4)", render(bulge, X, Y)),
    ("Disk (n=1)", render(disk, X, Y)),
    ("Bar (n≈0.5)", render(bar, X, Y)),
    ("Spiral arms", render(arms, X, Y)),
    ("Full scene", render(scene, X, Y)),
]

fig = Figure(size = (2000, 420))
crange = extrema(logstretch(last(panels[end])))
for (i, (ttl, img)) in enumerate(panels)
    ax = Axis(fig[1, 2i - 1]; title = ttl, aspect = DataAspect(), xlabel = "x", ylabel = i == 1 ? "y" : "")
    hm = heatmap!(ax, coord, coord, logstretch(img); colormap = :inferno, colorrange = crange)
    Colorbar(fig[1, 2i], hm; width = 12)
end
Label(fig[0, :], "Spiral galaxy model preview (log₁₀ stretch)"; fontsize = 16, font = :bold)
save("examples/spiral_galaxy_preview.png", fig; px_per_unit = 2)
println("saved → examples/spiral_galaxy_preview.png")
