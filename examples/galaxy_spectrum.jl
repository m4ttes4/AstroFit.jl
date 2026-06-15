# Spettro galattico sintetico — stress-test del framework AstroFit
#
# Componenti:
#   - Continuo lineare
#   - Doppietto [OIII] 4959/5007  (ratio 2.98, cinema / sigma condivisi)
#   - Hβ 4861                     (ampiezza legata a Hα via decremento Balmer,
#                                   sigma legata a [OIII].blue)
#   - Complesso Hα + [NII] 6548/6583  (sigma condivisa, ratio [NII] 3.06,
#                                       mean [NII] segue Hα via ratio λ)
#   - Doppietto Na D 5890/5896     (assorbimento ISM, sigma condivisa)
#   - Redshift1D come Pipe         (λ_obs → λ_rest = λ_obs/(1+z) → spettro)
#
# Parametri liberi totali: 13
#   z(1) + continuum(2) + oiii(2) + hbeta_sigma(1) + ha_nii(4) + nad(3)

import Pkg
Pkg.activate(temp=true)
Pkg.develop(path=joinpath(@__DIR__, ".."); io=devnull)
Pkg.add("CairoMakie"; io=devnull)

using AstroFit
using CairoMakie

# ── Redshift1D — definita qui per mostrare extensibilità del framework ────────
# Qualsiasi AbstractModel{1} può entrare in @model, essere vincolato con
# @constrain e comporsi via ∘ in un Pipe senza toccare la libreria.

Base.@kwdef struct Redshift1D{T<:Real} <: AbstractModel{1}
    z::T = 0.0
end
Redshift1D(z::Real) = Redshift1D{typeof(float(z))}(float(z))
# Extend AstroFit.render (must be qualified: a bare `render(...)` would define a
# new Main.render that shadows the one the framework dispatches on).
AstroFit.render(m::Redshift1D, λ::Number) = λ / (1 + m.z)

# ── 1. Complesso Hα + [NII] ───────────────────────────────────────────────────
# Tre Gaussiane con cinematica condivisa (regione di linee strette, NLR).
# Tie fisiche:
#   [NII] ratio  6548/6583 = 1/3.06       (fisica atomica, fissato)
#   [NII] mean               seguono Hα   (stesso spostamento in velocità)
#   [NII] sigma              = Hα sigma   (stessa turbolenza NLR)

ha_nii = let
    m = @model begin
        ha    = Gaussian1D(amplitude=15.0, mean=6562.8, sigma=3.0)
        nii_r = Gaussian1D(amplitude=5.0,  mean=6583.4, sigma=3.0)
        nii_b = Gaussian1D(amplitude=1.6,  mean=6548.1, sigma=3.0)
        ha + nii_r + nii_b
    end
    @constrain m begin
        @bound ha.amplitude    in (0, Inf)
        @bound ha.mean         in (6555.0, 6570.0)
        @bound ha.sigma        in (0.5, Inf)
        @bound nii_r.amplitude in (0, Inf)
        @tie   nii_b.amplitude = nii_r.amplitude / 3.06
        @tie   nii_r.mean      = (6583.4 / 6562.8) * ha.mean
        @tie   nii_b.mean      = (6548.1 / 6562.8) * ha.mean
        @tie   nii_r.sigma     = ha.sigma
        @tie   nii_b.sigma     = ha.sigma
    end
end

# ── 2. Doppietto [OIII] ───────────────────────────────────────────────────────
# EmissionDoublet1D gestisce già ratio, mean tied, sigma condivisa.
# ratio = I(5007)/I(4959) = 2.98

oiii = EmissionDoublet1D(
    blue_center  = 4958.9,
    red_center   = 5006.8,
    amplitude    = 1.68,
    sigma        = 2.5,
    ratio        = 2.98,
    center_window = 0,
)

# ── 3. Doppietto Na D (assorbimento ISM) ─────────────────────────────────────
# Na I D2 5889.95, D1 5895.92 — stessa nube ISM → sigma condivisa

nad = let
    m = @model begin
        d2 = Gaussian1D(amplitude=-0.4, mean=5889.95, sigma=1.0)
        d1 = Gaussian1D(amplitude=-0.2, mean=5895.92, sigma=1.0)
        d2 + d1
    end
    @constrain m begin
        @bound d2.amplitude in (-Inf, 0)
        @fix   d2.mean      = 5889.95
        @bound d2.sigma     in (0.2, Inf)
        @bound d1.amplitude in (-Inf, 0)
        @fix   d1.mean      = 5895.92
        @tie   d1.sigma     = d2.sigma
    end
end

# ── 4. Spettro in frame a riposo ─────────────────────────────────────────────
# Cross-component ties nel modello completo:
#   hbeta.amplitude  = ha_nii.ha.amplitude / 2.86   (decremento Balmer Case B)
#   hbeta.sigma      = oiii.blue.sigma               (stessa componente stretta)

cont     = LinearContinuum1D(slope=0.0001, intercept=2.0)
hbeta_cm = EmissionLine1D(center=4861.3, amplitude=5.25, sigma=2.5, center_window=0)

rest_spec = @model begin
    continuum = cont
    oiii      = oiii
    hbeta     = hbeta_cm
    ha_nii    = ha_nii
    nad       = nad
    continuum + oiii + hbeta + ha_nii + nad
end

rest_spec = @constrain rest_spec begin
    @tie hbeta.amplitude = ha_nii.ha.amplitude / 2.86
    @tie hbeta.sigma     = oiii.blue.sigma
end

# ── 5. Pipe di redshift ───────────────────────────────────────────────────────
# spectrum ∘ z_shift  →  Pipe(z_shift, spectrum)
#   valuta: spectrum( z_shift(λ_obs) ) = spectrum( λ_obs / (1+z) )
# Le linee si spostano a λ_obs = λ_rest * (1+z).

z_shift = @constrain Redshift1D(z=0.05) begin
    @bound z in (0.0, 1.0)
end

observed = @model begin
    spectrum = rest_spec
    z_shift  = z_shift
    spectrum ∘ z_shift
end

# ── 6. Verifica ───────────────────────────────────────────────────────────────
z_val = observed.z_shift.z
ha_amp  = observed.spectrum.ha_nii.ha.amplitude
hb_amp  = observed.spectrum.hbeta.amplitude
nii_r_amp = observed.spectrum.ha_nii.nii_r.amplitude
nii_b_amp = observed.spectrum.ha_nii.nii_b.amplitude

println("Parametri liberi: ", nfree(observed))
println("  z                      = $z_val")
println("  Hα amplitude           = $ha_amp")
println("  Hβ amplitude (Hα/2.86) = $(round(hb_amp; digits=3))  check: $(round(ha_amp/2.86; digits=3))")
println("  [NII] 6583             = $nii_r_amp")
println("  [NII] 6548 (÷3.06)     = $(round(nii_b_amp; digits=3))  check: $(round(nii_r_amp/3.06; digits=3))")
println("  [OIII] 5007            = $(observed.spectrum.oiii.red.amplitude)")
println("  [OIII] 4959 (÷2.98)    = $(observed.spectrum.oiii.blue.amplitude)")

# ── 7. Valutazione su griglia ─────────────────────────────────────────────────
λ_obs  = collect(range(4800.0, 7100.0, length=4000))
λ_rest = λ_obs ./ (1 + z_val)

flux_total = render(observed, λ_obs)
flux_cont  = render(cont, λ_rest)
flux_oiii  = render(oiii, λ_rest)
flux_hanii = render(ha_nii, λ_rest)
flux_nad   = render(nad, λ_rest)

# Hβ con parametri tied dal modello completo
hbeta_eval = Gaussian1D(
    amplitude = observed.spectrum.hbeta.amplitude,
    mean      = 4861.3,
    sigma     = observed.spectrum.hbeta.sigma,
)
flux_hbeta = render(hbeta_eval, λ_rest)

# ── 8. Plot ───────────────────────────────────────────────────────────────────
fig = Figure(size=(1300, 620))
ax  = Axis(fig[1, 1];
    xlabel       = "Lunghezza d'onda osservata (Å)",
    ylabel       = "Flusso (unità arbitrarie)",
    title        = "Spettro galattico sintetico  —  z = $z_val  —  $(nfree(observed)) param. liberi",
    xgridvisible = false,
    ygridvisible = false,
)

# Fills delle componenti sopra/sotto il continuo
band!(ax, λ_obs, flux_cont, flux_cont .+ flux_oiii;   color=(:steelblue, 0.35), label="[OIII] 4959/5007")
band!(ax, λ_obs, flux_cont, flux_cont .+ flux_hbeta;  color=(:seagreen,  0.35), label="Hβ (tied Balmer)")
band!(ax, λ_obs, flux_cont, flux_cont .+ flux_hanii;  color=(:firebrick, 0.35), label="Hα + [NII] 6548/6583")
band!(ax, λ_obs, flux_cont .+ flux_nad, flux_cont;    color=(:darkorange, 0.40), label="Na D (ass. ISM)")

# Continuo e spettro totale
lines!(ax, λ_obs, flux_cont;  color=:gray,  linewidth=1.5, linestyle=:dash, label="Continuo")
lines!(ax, λ_obs, flux_total; color=:black, linewidth=2.0, label="Totale")

# Etichette delle righe in frame osservato
line_labels = [
    ("Hβ",    4861.3),
    ("[OIII]", 4958.9),
    ("[OIII]", 5006.8),
    ("Na D",   5889.95),
    ("Na D",   5895.92),
    ("[NII]",  6548.1),
    ("Hα",     6562.8),
    ("[NII]",  6583.4),
]

for (label, λ0) in line_labels
    λ_shifted = λ0 * (1 + z_val)
    vlines!(ax, [λ_shifted]; color=:gray, linewidth=0.8, linestyle=:dot)
    text!(ax, λ_shifted, maximum(flux_total) * 1.01;
          text=label, fontsize=9, align=(:center, :bottom), rotation=π/6)
end

axislegend(ax; position=:rt, framevisible=false)
ylims!(ax, minimum(flux_total) * 0.97, maximum(flux_total) * 1.12)

outpath = joinpath(@__DIR__, "galaxy_spectrum.png")
save(outpath, fig)
println("\nPlot salvato in: $outpath")
