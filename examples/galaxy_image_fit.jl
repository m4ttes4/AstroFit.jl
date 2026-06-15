# Decomposizione di un'immagine 2D di galassia — fit di una scena complessa.
#
# La "scena" osservata è una cutout astronomica sintetica composta da:
#   - cielo con gradiente   (modello 2D custom Sky­Plane2D, definito qui sotto)
#   - bulge                 (GalaxyGaussianLineProfile2D, profilo gaussiano 2D)
#   - disco                 (GalaxyExponentialLineProfile2D, disco esponenziale)
#   - 3 stelle in primo piano (Gaussian2D puntiformi, PSF condivisa)
#
# Le componenti NON sono indipendenti — la fisica della scena impone dei tie:
#   - bulge e disco condividono il centro della galassia (bulge.x0/y0 = disk.x0/y0)
#   - le 3 stelle condividono la stessa PSF (stessa σ, e σ_y = σ_x → stelle circolari)
# Così una scena a 6 componenti ha "solo" 23 parametri liberi.
#
# Mostra anche il fit 2D: l'estensione Optimization accetta una tupla di coordinate
# (X, Y) come variabile indipendente (vedi `OptimizationProblem(cm, (X, Y), ...)`).
#
# Run con:  julia examples/galaxy_image_fit.jl
import Pkg
Pkg.activate(temp = true)
Pkg.develop(path = joinpath(@__DIR__, ".."); io = devnull)
Pkg.add(["CairoMakie", "Optimization", "OptimizationOptimJL"]; io = devnull)

using AstroFit
using Optimization, OptimizationOptimJL
using CairoMakie
using Random

# ── Sky­Plane2D — cielo con gradiente lineare, definito qui per mostrare l'estensibilità ──
# Qualunque AbstractModel{2} entra in @model/@constrain e si compone con gli altri.
# NB: il costruttore con `promote` è ciò che rende il modello AD-friendly — quando i
# Dual di ForwardDiff vengono scritti in UN campo via `set`, gli altri restano Float64
# e i tipi devono unificare (stesso pattern dei modelli built-in come Gaussian2D).
Base.@kwdef struct SkyPlane2D{T<:Real} <: AbstractModel{2}
    bg::T = 0.0    # livello costante
    gx::T = 0.0    # gradiente lungo x
    gy::T = 0.0    # gradiente lungo y
end
SkyPlane2D(bg::Real, gx::Real, gy::Real) = SkyPlane2D(promote(bg, gx, gy)...)
AstroFit.render(m::SkyPlane2D, x::Number, y::Number) = m.bg + m.gx * x + m.gy * y

# ── Template della scena ──────────────────────────────────────────────────────
# Una sola funzione costruisce il modello vincolato; la chiamiamo due volte (verità
# e guess iniziale) con valori diversi.
function galaxy_scene(; sky_bg, sky_gx, sky_gy,
        bulge_amp, bulge_sx, bulge_sy, bulge_theta,
        disk_amp, disk_x0, disk_y0, disk_r, disk_q, disk_theta,
        psf, s1_amp, s1_x, s1_y, s2_amp, s2_x, s2_y, s3_amp, s3_x, s3_y)

    scene = @model begin
        sky   = SkyPlane2D(bg = sky_bg, gx = sky_gx, gy = sky_gy)
        bulge = GalaxyGaussianLineProfile2D(amplitude = bulge_amp, x0 = disk_x0, y0 = disk_y0,
                                            sigma_x = bulge_sx, sigma_y = bulge_sy, theta = bulge_theta)
        disk  = GalaxyExponentialLineProfile2D(amplitude = disk_amp, x0 = disk_x0, y0 = disk_y0,
                                               scale_radius = disk_r, axis_ratio = disk_q, theta = disk_theta)
        star1 = Gaussian2D(amplitude = s1_amp, x0 = s1_x, y0 = s1_y, sigma_x = psf, sigma_y = psf, theta = 0.0)
        star2 = Gaussian2D(amplitude = s2_amp, x0 = s2_x, y0 = s2_y, sigma_x = psf, sigma_y = psf, theta = 0.0)
        star3 = Gaussian2D(amplitude = s3_amp, x0 = s3_x, y0 = s3_y, sigma_x = psf, sigma_y = psf, theta = 0.0)
        sky + bulge + disk + star1 + star2 + star3
    end

    # bulge/disk portano già i propri bound (ampiezza, larghezze, axis_ratio) dai
    # prefab; qui aggiungiamo solo i tie fisici e i vincoli sulle stelle.
    @constrain scene begin
        @tie   bulge.x0 = disk.x0                 # centro galassia condiviso
        @tie   bulge.y0 = disk.y0
        @bound star1.amplitude in (0.0, Inf)
        @bound star1.sigma_x   in (0.1, Inf)      # PSF master
        @tie   star1.sigma_y   = star1.sigma_x    # stella circolare
        @fix   star1.theta     = 0.0
        @bound star2.amplitude in (0.0, Inf)
        @tie   star2.sigma_x   = star1.sigma_x    # PSF condivisa
        @tie   star2.sigma_y   = star1.sigma_x
        @fix   star2.theta     = 0.0
        @bound star3.amplitude in (0.0, Inf)
        @tie   star3.sigma_x   = star1.sigma_x
        @tie   star3.sigma_y   = star1.sigma_x
        @fix   star3.theta     = 0.0
    end
end

# ── Verità e guess iniziale ───────────────────────────────────────────────────
truth_p = (sky_bg = 2.0, sky_gx = 0.03, sky_gy = -0.02,
    bulge_amp = 32.0, bulge_sx = 1.7, bulge_sy = 1.3, bulge_theta = 0.6,
    disk_amp = 20.0, disk_x0 = 0.6, disk_y0 = -0.4, disk_r = 3.4, disk_q = 0.55, disk_theta = 0.9,
    psf = 0.9, s1_amp = 45.0, s1_x = -4.5, s1_y = 3.6, s2_amp = 28.0, s2_x = 4.7, s2_y = -3.1,
    s3_amp = 18.0, s3_x = -3.2, s3_y = -4.6)

guess_p = (sky_bg = 1.0, sky_gx = 0.0, sky_gy = 0.0,
    bulge_amp = 22.0, bulge_sx = 1.2, bulge_sy = 1.2, bulge_theta = 0.2,
    disk_amp = 14.0, disk_x0 = 0.0, disk_y0 = 0.0, disk_r = 4.2, disk_q = 0.8, disk_theta = 0.6,
    psf = 1.2, s1_amp = 35.0, s1_x = -4.2, s1_y = 3.3, s2_amp = 22.0, s2_x = 4.4, s2_y = -2.8,
    s3_amp = 14.0, s3_x = -3.5, s3_y = -4.3)

truth = galaxy_scene(; truth_p...)
cm    = galaxy_scene(; guess_p...)

# ── Immagine "osservata" sintetica = verità + rumore gaussiano ────────────────
Random.seed!(2024)
coord = range(-8.0, 8.0; length = 80)
X = [x for x in coord, y in coord]
Y = [y for x in coord, y in coord]
σ_noise = 0.6
image = render(truth, X, Y) .+ σ_noise .* randn(size(X))
err   = fill(σ_noise, size(X))

# ── Fit: scena bounded → solve box-aware. err fornito → negative log-likelihood ──
# L'indipendente è la tupla (X, Y): l'estensione la passa a render come render(m, X, Y).
prob = OptimizationProblem(cm, (X, Y), image, err)   # default AutoForwardDiff
sol  = solve(prob, Fminbox(LBFGS()))
fit  = withparams(cm, sol.u)

model_img = render(fit, X, Y)
resid     = image .- model_img

# ── Report ────────────────────────────────────────────────────────────────────
println("retcode          : ", sol.retcode)
println("parametri liberi : ", nfree(cm), "  (6 componenti, centro + PSF condivisi)")
println("objective finale : ", round(sol.objective; digits = 2))
println()
row(name, t, f) = println(rpad(name, 20), "verità = ", rpad(round(t; digits = 3), 10),
                          "fit = ", round(f; digits = 3))
row("centro galassia x", truth.disk.x0,          fit.disk.x0)
row("centro galassia y", truth.disk.y0,          fit.disk.y0)
row("disco raggio",      truth.disk.scale_radius, fit.disk.scale_radius)
row("disco axis_ratio",  truth.disk.axis_ratio,   fit.disk.axis_ratio)
row("disco theta",       truth.disk.theta,        fit.disk.theta)
row("bulge ampiezza",    truth.bulge.amplitude,   fit.bulge.amplitude)
row("PSF (σ stelle)",    truth.star1.sigma_x,     fit.star1.sigma_x)
row("stella1 ampiezza",  truth.star1.amplitude,   fit.star1.amplitude)
row("cielo bg",          truth.sky.bg,            fit.sky.bg)
# i tie sono rispettati automaticamente:
println("\nbulge centro == disco centro : ",
        fit.bulge.x0 == fit.disk.x0 && fit.bulge.y0 == fit.disk.y0)
println("stelle 2/3 σ == PSF master    : ",
        fit.star2.sigma_x == fit.star1.sigma_x && fit.star3.sigma_x == fit.star1.sigma_x)

# ── Plot: trittico dati | modello | residuo ───────────────────────────────────
fig = Figure(size = (1550, 540))
crange = (minimum(image), maximum(image))

ax1 = Axis(fig[1, 1]; title = "Dati sintetici", aspect = DataAspect(),
           xlabel = "x", ylabel = "y")
hm1 = heatmap!(ax1, coord, coord, image; colormap = :inferno, colorrange = crange)
Colorbar(fig[1, 2], hm1)

ax2 = Axis(fig[1, 3]; title = "Modello best-fit", aspect = DataAspect(), xlabel = "x")
hm2 = heatmap!(ax2, coord, coord, model_img; colormap = :inferno, colorrange = crange)
scatter!(ax2, [fit.star1.x0, fit.star2.x0, fit.star3.x0],
              [fit.star1.y0, fit.star2.y0, fit.star3.y0];
         color = :cyan, marker = :xcross, markersize = 14)
Colorbar(fig[1, 4], hm2)

rmax = maximum(abs, resid)
ax3 = Axis(fig[1, 5]; title = "Residuo (dati − modello)", aspect = DataAspect(), xlabel = "x")
hm3 = heatmap!(ax3, coord, coord, resid; colormap = :balance, colorrange = (-rmax, rmax))
Colorbar(fig[1, 6], hm3)

Label(fig[0, :], "Decomposizione galassia 2D  —  $(nfree(cm)) parametri liberi";
      fontsize = 18, font = :bold)

outpath = joinpath(@__DIR__, "galaxy_image_fit.png")
save(outpath, fig)
println("\nPlot salvato in: $outpath")
