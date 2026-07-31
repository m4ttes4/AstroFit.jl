# Fit the Na I D absorption doublet + He I emission through a simulated
# instrumental PSF broad enough to partially blend the lines.
#
# Every constraint has a physical story:
#   - the 5.97 A separation is atomic physics, but a common Doppler shift is free
#   - the depth ratio D2/D1 = 2:1 is the optically thin oscillator-strength ratio
#   - one velocity dispersion: both Na lines come from the same gas
#   - He I rides at the same systemic velocity, tied to the doublet position
#   - the PSF is a known instrument calibration, so it stays fixed and the
#     fitted sigma is the INTRINSIC width, deconvolved from the instrument
#
# Run with:  julia --project=examples examples/main/na_doublet_fit.jl

using AstroFit
using Optimization, OptimizationOptimJL, ForwardDiff
using CairoMakie
using Random

const L_HEI = 5875.62
const L_NAD_D2 = 5889.95
const L_NAD_D1 = 5895.92
const C_KMS = 2.998e5

const STEP = 0.1             # grid step [A/sample] — kernels work in samples
const SIGMA_INST = 1.6       # instrumental resolution [A]: partially blends the doublet

# ---------------------------------------------------------------------------
# 1. True model — doublet at +45 km/s, ratio 2:1, intrinsic width 0.45 A,
#    plus He I emission; the instrument smears every line to ~2.25 A
# ---------------------------------------------------------------------------
v_true = 45.0
shift = L_NAD_D2 * v_true / C_KMS
sigma_true = 0.45

true_model = @model begin
    cont = Linear1D(slope = -0.0015, intercept = 9.835)
    d2 = Gaussian1D(amplitude = -0.85, mean = L_NAD_D2 + shift, sigma = sigma_true)
    d1 = Gaussian1D(amplitude = -0.425, mean = L_NAD_D1 + shift, sigma = sigma_true)
    hei = Gaussian1D(amplitude = 0.55, mean = L_HEI + shift, sigma = 0.9)
    psf = GaussianPSF(sigma = SIGMA_INST / STEP)
    (cont + d2 + d1 + hei) |> psf
end

# ---------------------------------------------------------------------------
# 2. Synthetic data
# ---------------------------------------------------------------------------
Random.seed!(123)
λ = collect(5860.0:STEP:5925.0)
σ_noise = 0.02
y_true = render(true_model, λ)
y = y_true .+ σ_noise .* randn(length(λ))
err = fill(σ_noise, length(λ))

# ---------------------------------------------------------------------------
# 3. Fitting model — start at rest wavelength, D1 fully derived from D2
# ---------------------------------------------------------------------------
cm = @model begin
    cont = Linear1D(slope = 0.0, intercept = 1.0)
    d2 = Gaussian1D(amplitude = -0.4, mean = L_NAD_D2, sigma = 0.8)
    d1 = Gaussian1D(amplitude = -0.2, mean = L_NAD_D1, sigma = 0.8)
    hei = Gaussian1D(amplitude = 0.3, mean = L_HEI, sigma = 1.2)
    psf = GaussianPSF(sigma = SIGMA_INST / STEP)
    (cont + d2 + d1 + hei) |> psf
end

@constrain cm begin
    d2.amplitude in (-5.0, 0.0)                    # absorption only
    d2.mean in (5885.0, 5895.0)                    # velocity window
    d2.sigma in (0.1, 3.0)
    d1.amplitude -> 0.5 * d2.amplitude             # optically thin 2:1
    d1.mean -> d2.mean + (L_NAD_D1 - L_NAD_D2)     # atomic separation
    d1.sigma -> d2.sigma                           # same gas
    hei.amplitude in (0.0, 5.0)                    # emission only
    hei.mean -> d2.mean + (L_HEI - L_NAD_D2)       # same systemic velocity
    hei.sigma in (0.1, 5.0)                        # different gas, own width
    psf.sigma                                      # known calibration, fixed
end

# ---------------------------------------------------------------------------
# 4. Fit
# ---------------------------------------------------------------------------
prob = OptimizationProblem(cm, λ, y, err)
sol = solve(prob, Optim.Fminbox(Optim.LBFGS()))

fit_tree = withparams(cm, sol.u)

v_fit = (fit_tree.d2.model.mean - L_NAD_D2) / L_NAD_D2 * C_KMS
sigma_fit = fit_tree.d2.model.sigma
sigma_obs = sqrt(sigma_fit^2 + SIGMA_INST^2)

println("retcode          : ", sol.retcode)
println("free parameters  : ", nfree(cm))
println("parameter names  : ", paramnames(cm))
println("gas velocity     : ", round(v_fit; digits = 1), " km/s  (truth: ", v_true, ")")
println("intrinsic sigma  : ", round(sigma_fit; digits = 3), " A     (truth: ", sigma_true, ")")
println("observed sigma   : ", round(sigma_obs; digits = 3), " A     (instrument-dominated)")
println()

# ---------------------------------------------------------------------------
# 5. Plot: data, truth, best fit, and the intrinsic (unconvolved) model
# ---------------------------------------------------------------------------
y_init = render(cm, λ)
y_fit = render(fit_tree, λ)
# same fitted components, composed without the PSF: what the gas looks like
# before the instrument
intrinsic = fit_tree.cont.model + fit_tree.d2.model + fit_tree.d1.model + fit_tree.hei.model
y_intrinsic = render(intrinsic, λ)

fig = Figure(size = (900, 500))
ax = Axis(
    fig[1, 1]; xlabel = "wavelength [Angstrom]", ylabel = "normalized flux",
    title = "Na I D doublet + He I through an instrumental PSF ($(nfree(cm)) free parameters)"
)

vlines!(ax, [L_HEI, L_NAD_D2, L_NAD_D1]; color = (:gray30, 0.4), linestyle = :dash, linewidth = 1, label = "rest wavelength")
scatter!(ax, λ, y; color = :grey60, markersize = 4, label = "data")
lines!(ax, λ, y_true; color = :black, linestyle = :dash, label = "truth")
lines!(ax, λ, y_init; color = :dodgerblue, linestyle = :dot, label = "initial guess")
lines!(ax, λ, y_intrinsic; color = (:seagreen, 0.8), linewidth = 1.5, label = "best fit (intrinsic)")
lines!(ax, λ, y_fit; color = :red, linewidth = 2, label = "best fit (convolved)")

axislegend(ax; position = :rb)

display(fig)
# save("na_doublet_fit.png", fig; px_per_unit = 2)
# println("saved → examples/main/na_doublet_fit.png")
