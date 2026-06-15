# MWE: fit a galaxy Hα + [NII] emission-line complex to synthetic data.
#
# The model is a linear continuum plus three Gaussians (Hα and the [NII]
# 6548/6583 doublet). The [NII] lines are not free: atomic physics fixes their
# 6583/6548 amplitude ratio to 3.06, the common galaxy redshift ties all three
# centroids together through their rest-wavelength ratios, and the lines share a
# kinematic width with Hα. So a 4-Gaussian-looking spectrum has only 5 free
# parameters: continuum slope + intercept, and Hα amplitude + centroid + width.
#
# Run with:  julia --project=. examples/halpha_nii_fit.jl

using AstroFit
using Optimization, OptimizationOptimJL
using Random

# Rest-frame wavelengths (Å)
const λ_Ha     = 6562.8
const λ_NII_r  = 6583.4
const λ_NII_b  = 6548.1

# ---------------------------------------------------------------------------
# Model: continuum + Hα + [NII] doublet, with the doublet tied to Hα.
# ---------------------------------------------------------------------------
function halpha_complex(; z0 = 0.0)
    spectrum = @model begin
        cont  = Linear1D(slope = 0.0, intercept = 1.0)
        ha    = Gaussian1D(amplitude = 10.0, mean = λ_Ha * (1 + z0), sigma = 3.0)
        nii_r = Gaussian1D(amplitude = 3.0,  mean = λ_NII_r * (1 + z0), sigma = 3.0)
        nii_b = Gaussian1D(amplitude = 1.0,  mean = λ_NII_b * (1 + z0), sigma = 3.0)
        cont + ha + nii_r + nii_b
    end

    @constrain spectrum begin
        @bound cont.intercept  in (0.0, Inf)
        @bound ha.amplitude    in (0.0, Inf)
        @bound ha.mean         in (λ_Ha - 30, λ_Ha + 30)   # redshift search window
        @bound ha.sigma        in (0.5, 20.0)
        # [NII] doublet locked to Hα (masters must be free, so both tie to ha
        # directly — no tie chains): 6583/6548 = 3.06, and [NII]6548/Hα ≈ 1/3.
        @tie   nii_b.amplitude = ha.amplitude / 3.0
        @tie   nii_r.amplitude = (3.06 / 3.0) * ha.amplitude     # 3.06 × nii_b
        @tie   nii_r.mean      = (λ_NII_r / λ_Ha) * ha.mean      # same redshift
        @tie   nii_b.mean      = (λ_NII_b / λ_Ha) * ha.mean      # same redshift
        @tie   nii_r.sigma     = ha.sigma                        # same kinematics
        @tie   nii_b.sigma     = ha.sigma
    end
end

# ---------------------------------------------------------------------------
# Synthetic "observed" spectrum from a known truth + Gaussian noise.
# ---------------------------------------------------------------------------
Random.seed!(42)

truth = @set halpha_complex().cont.slope = 0.0          # start from the template...
truth = @set truth.cont.intercept = 2.0
truth = @set truth.ha.amplitude   = 18.0
truth = @set truth.ha.mean        = 6562.8 * (1 + 0.002)   # z ≈ 0.002
truth = @set truth.ha.sigma       = 4.0

λ      = collect(6500.0:1.0:6650.0)
σ_noise = 0.4
flux   = render(truth, λ) .+ σ_noise .* randn(length(λ))
err    = fill(σ_noise, length(λ))

# ---------------------------------------------------------------------------
# Fit: bad-ish initial guess → native Optimization.jl solve → fitted model.
# err is supplied, so the objective is the (Gaussian) negative log-likelihood.
# ---------------------------------------------------------------------------
cm   = halpha_complex()
prob = OptimizationProblem(cm, λ, flux, err)             # default AutoForwardDiff
sol  = solve(prob, Fminbox(LBFGS()))                     # box-aware (model is bounded)
fit  = withparams(cm, sol.u)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
println("retcode            : ", sol.retcode)
println("free parameters    : ", nfree(cm), "  (4 Gaussians, only Hα + continuum free)")
println("final objective    : ", round(sol.objective; digits = 3))
println()
row(name, t, f) =println(rpad(name, 18), "truth = ", rpad(round(t; digits=4), 12),
                          "fit = ", round(f; digits=4))
row("cont.slope",     truth.cont.slope,     fit.cont.slope)
row("cont.intercept", truth.cont.intercept, fit.cont.intercept)
row("ha.amplitude",   truth.ha.amplitude,   fit.ha.amplitude)
row("ha.mean",        truth.ha.mean,        fit.ha.mean)
row("ha.sigma",       truth.ha.sigma,       fit.ha.sigma)
println()
# Tied components recovered for free, consistent with Hα:
row("nii_r.mean",     truth.nii_r.mean,     fit.nii_r.mean)
row("nii_r.amplitude",truth.nii_r.amplitude,fit.nii_r.amplitude)
println("\nimplied redshift z : ", round(fit.ha.mean / λ_Ha - 1; digits = 5))

# ---------------------------------------------------------------------------
# Plot the solution: data + total fit + component decomposition.
# (Plots.jl is an example-only dependency, not required by AstroFit itself.)
# Rebuild each component as a bare renderable model from its fitted values.
# ---------------------------------------------------------------------------
using Plots

λ_plot   = collect(range(first(λ), last(λ); length = 600))
cont_fit  = Linear1D(slope = fit.cont.slope, intercept = fit.cont.intercept)
ha_fit    = Gaussian1D(amplitude = fit.ha.amplitude,    mean = fit.ha.mean,    sigma = fit.ha.sigma)
nii_r_fit = Gaussian1D(amplitude = fit.nii_r.amplitude, mean = fit.nii_r.mean, sigma = fit.nii_r.sigma)
nii_b_fit = Gaussian1D(amplitude = fit.nii_b.amplitude, mean = fit.nii_b.mean, sigma = fit.nii_b.sigma)

baseline = render(cont_fit, λ_plot)   # lines are drawn sitting on the continuum

plt = Plots.scatter(λ, flux; yerr = err, label = "data", ms = 2,
              mc = :gray, msc = :gray, alpha = 0.6, legend = :topright,
              xlabel = "wavelength [Å]", ylabel = "flux",
              title = "Hα + [NII] decomposition")
plot!(plt, λ_plot, render(fit, λ_plot);                 label = "total fit", lw = 2, c = :black)
plot!(plt, λ_plot, baseline;                            label = "continuum", ls = :dash, c = :gray)
plot!(plt, λ_plot, baseline .+ render(ha_fit, λ_plot);    label = "Hα",        c = :crimson)
plot!(plt, λ_plot, baseline .+ render(nii_r_fit, λ_plot); label = "[NII] 6583", c = :dodgerblue)
plot!(plt, λ_plot, baseline .+ render(nii_b_fit, λ_plot); label = "[NII] 6548", c = :seagreen)

savefig(plt, joinpath(@__DIR__, "halpha_nii_fit.png"))
println("saved plot → ", joinpath(@__DIR__, "halpha_nii_fit.png"))
