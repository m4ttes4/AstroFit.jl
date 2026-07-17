# Fit an absorbed AGN X-ray spectrum — Galactic + intrinsic photoelectric
# absorption on a power-law continuum plus a Fe Kα line — to synthetic
# photon-counting data, using a custom Poisson statistic. Mirrors the
# classic XSPEC combination `phabs * zphabs * powerlaw` used for obscured
# (type 2) AGN.
#
# Run with:  julia --project=. examples/agn_xray_fit.jl

using AstroFit
using Optimization, OptimizationOptimJL, ForwardDiff
using Distributions: Poisson, logpdf
using CairoMakie
using Random

# ---------------------------------------------------------------------------
# 0. Photoelectric absorption — a user-defined AbstractModel, no changes to
#    AstroFit itself needed (only a @kwdef struct + scalar `render`, exactly
#    how zoo models are written). Simplified E^-3 cross-section — real XSPEC
#    `phabs` uses tabulated Wisconsin/Verner cross-sections; this toy form
#    reproduces the qualitative turnover shape, not the precise edges.
#    `z` shifts the absorber into its own rest frame: z=0 is plain `phabs`
#    (Galactic, foreground); z>0 is `zphabs` (intrinsic, at the AGN itself).
# ---------------------------------------------------------------------------
Base.@kwdef struct PhAbs1D{T <: Real} <: AbstractModel
    nH::T = 1.0     # column density, units of 1e22 cm^-2
    z::T = 0.0      # redshift of the absorbing material
    e0::T = 1.0     # keV, cross-section reference energy
end

# Qualified as AstroFit.render (not bare render): `using AstroFit` brings the
# name into scope for calling, but a bare `render(...) = ...` here would define
# a new Main.render shadowing it, invisible to the Sum/Product dispatch chain.
AstroFit.render(m::PhAbs1D, E::Number) = exp(-m.nH * ((E * (1 + m.z)) / m.e0)^(-3))

# ---------------------------------------------------------------------------
# 1. True model — phabs (Galactic) * zphabs (intrinsic) * (powerlaw + Fe Kα)
# ---------------------------------------------------------------------------
# Energy in keV, model output = expected photon counts per bin.
z_src = 0.05                        # source redshift, known from optical spectroscopy
fe_energy_obs = 6.4 / (1 + z_src)   # rest-frame 6.4 keV, redshifted into the observed frame

true_model = @model begin
    gal = PhAbs1D(nH = 0.02, z = 0.0)          # Galactic foreground, thin
    intrinsic = PhAbs1D(nH = 5.0, z = z_src)   # obscured AGN (Seyfert 2-like)
    cont = PowerLaw1D(norm = 8.0, x_ref = 1.0, index = 1.8)   # Γ=1.8
    fe_line = Gaussian1D(amplitude = 3.0, mean = fe_energy_obs, sigma = 0.15)
    gal * intrinsic * (cont + fe_line)
end

# ---------------------------------------------------------------------------
# 2. Synthetic data — Poisson counts, not Gaussian noise
# ---------------------------------------------------------------------------
Random.seed!(7)
E = collect(0.3:0.05:10.0)
λ_true = render(true_model, E)
counts = [rand(Poisson(λ)) for λ in λ_true]

# ---------------------------------------------------------------------------
# 3. Fitting model — off initial guess. Galactic nH/z and the reference
#    energies are known independently (HI maps, optical redshift) and stay
#    fixed; the intrinsic column density is the actual fit target.
# ---------------------------------------------------------------------------
cm = @model begin
    gal = PhAbs1D(nH = 0.02, z = 0.0)
    intrinsic = PhAbs1D(nH = 2.0, z = z_src)
    cont = PowerLaw1D(norm = 4.0, x_ref = 1.0, index = 1.5)
    fe_line = Gaussian1D(amplitude = 1.0, mean = 5.8, sigma = 0.3)
    gal * intrinsic * (cont + fe_line)
end

@fix cm.gal.nH = 0.02
@fix cm.gal.z = 0.0
@fix cm.gal.e0 = 1.0
@fix cm.intrinsic.z = z_src
@fix cm.intrinsic.e0 = 1.0
@fix cm.cont.x_ref = 1.0
@bound cm.intrinsic.nH in (0.01, 50)
@bound cm.cont.norm in (0.01, 50)
@bound cm.cont.index in (0.1, 5)
@bound cm.fe_line.amplitude in (0.01, 20)
@bound cm.fe_line.mean in (5.5, 6.5)
@bound cm.fe_line.sigma in (0.02, 1.0)

# ---------------------------------------------------------------------------
# 4. Poisson log-likelihood as a custom `statistic`, then fit
# ---------------------------------------------------------------------------
# logpdf(Poisson(λ), 0) is correctly 0 at λ→0, but its ForwardDiff derivative
# is NaN there (the k*log(λ) chain rule still evaluates log(λ)=-Inf before
# the k=0 factor zeroes it out). At k=0 the exact log-likelihood collapses to
# -λ anyway (log and loggamma(1) terms both vanish), so branch around it —
# this is a numerically-safe rewrite of the same formula, not a domain guard.
poisson_ll_term(λ, k) = k == 0 ? -λ : logpdf(Poisson(λ), k)

poisson_loglike(f::ObjectiveFunction, p) = begin
    m = withparams(f.cm, p)
    Es = f.coords[1]
    sum(i -> poisson_ll_term(render(m, Es[i]), f.y[i]), eachindex(f.y))
end

# Optimization.jl minimizes — negate the log-likelihood to fit.
prob = OptimizationProblem(cm, E, counts; statistic = (f, p) -> -poisson_loglike(f, p))
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
λ_init = render(cm, E)
λ_fit = render(fit_tree, E)

fig = Figure(size = (900, 500))
ax = Axis(
    fig[1, 1]; xlabel = "Energy (keV)", ylabel = "Counts",
    title = "Absorbed AGN Fit: phabs × zphabs × (powerlaw + Fe Kα), Poisson statistic"
)

scatter!(ax, E, counts; color = :grey60, markersize = 4, label = "data (counts)")
lines!(ax, E, λ_true; color = :black, linestyle = :dash, label = "truth")
lines!(ax, E, λ_init; color = :dodgerblue, linestyle = :dot, label = "initial guess")
lines!(ax, E, λ_fit; color = :red, linewidth = 2, label = "best fit")

axislegend(ax; position = :rt)

display(fig)
# save("examples/agn_xray_fit.png", fig; px_per_unit = 2)
# println("saved → examples/agn_xray_fit.png")
