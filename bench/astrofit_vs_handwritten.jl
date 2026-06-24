# AstroFit vs Handwritten benchmark — realistic Hα + [NII] fit.
#
# A scientist fitting the Hα + [NII] triplet would write a single function
# with ties hardcoded and a loop over data points. This benchmark compares
# that against AstroFit's composable model on the same task.
#
# Model: Linear continuum + 3 Gaussians (Hα, [NII]λ6548, [NII]λ6583)
# Ties:  NII amplitudes and means locked to Hα, all sigmas shared → 5 free params
# Data:  1000 points, σ = 0.3, realistic spectral range
#
# Run:  julia --startup-file=no --project=bench bench/astrofit_vs_handwritten.jl

using AstroFit
using BenchmarkTools
using ForwardDiff
using Optimization, OptimizationOptimJL
using Random, Printf



const λ_Ha    = 6562.8
const λ_NII_r = 6583.45
const λ_NII_b = 6548.05

# ============================================================================
# 1. AstroFit model — 5 free params after ties
# ============================================================================

cm = @model begin
    cont  = Linear1D(slope = 0.002, intercept = 1.0)
    ha    = Gaussian1D(amplitude = 8.0, mean = λ_Ha, sigma = 4.0)
    nii_r = Gaussian1D(amplitude = (3.06/3.0)*8.0, mean = (λ_NII_r/λ_Ha)*λ_Ha, sigma = 4.0)
    nii_b = Gaussian1D(amplitude = 8.0/3.0, mean = (λ_NII_b/λ_Ha)*λ_Ha, sigma = 4.0)
    cont + ha + nii_r + nii_b
end

@constrain cm begin
    cont.slope in (-0.05, 0.05)
    cont.intercept in (0.0, 5.0)
    ha.amplitude in (0.1, 30.0)
    ha.mean in (6540.0, 6590.0)
    ha.sigma in (1.0, 12.0)
    nii_r.amplitude -> (3.06 / 3.0) * ha.amplitude
    nii_r.mean -> (λ_NII_r / λ_Ha) * ha.mean
    nii_r.sigma -> ha.sigma
    nii_b.amplitude -> ha.amplitude / 3.0
    nii_b.mean -> (λ_NII_b / λ_Ha) * ha.mean
    nii_b.sigma -> ha.sigma
end

p0 = AstroFit.params(cm)

# ============================================================================
# 2. Handwritten — what a scientist would actually write
# ============================================================================

# p = [slope, intercept, A_ha, μ_ha, σ]
function hand_render(p, xi)
    s, ic, A, μ, σ = p
    rA = (3.06 / 3.0) * A
    bA = A / 3.0
    rμ = (λ_NII_r / λ_Ha) * μ
    bμ = (λ_NII_b / λ_Ha) * μ
    @fastmath s * xi + ic +
        A  * exp(-((xi - μ)  / σ)^2 / 2) +
        rA * exp(-((xi - rμ) / σ)^2 / 2) +
        bA * exp(-((xi - bμ) / σ)^2 / 2)
end

function hand_chi2(p, x, y, err)
    acc = zero(eltype(p))
    @inbounds @fastmath for i in eachindex(y)
        r = hand_render(p, x[i]) - y[i]
        acc += abs2(r / err[i])
    end
    return acc
end

# ============================================================================
# 3. Synthetic data — 1000 points
# ============================================================================

Random.seed!(42)
x = collect(range(6500.0, 6650.0; length = 1000))
y_true = render(withparams(cm, p0), x)
err = fill(0.3, length(x))
y = y_true .+ 0.3 .* randn(length(x))

# ============================================================================
# 4. Verify equivalence
# ============================================================================

af_y = render(withparams(cm, p0), x)
hand_y = hand_render.(Ref(p0), x)
@assert af_y ≈ hand_y "render mismatch"

af_obj = ObjectiveFunction(cm, x, y, err; statistic = :chi2)
@assert af_obj(p0) ≈ hand_chi2(p0, x, y, err) "chi2 mismatch"
println("model: Hα + [NII], $(nfree(cm)) free params, $(length(x)) data points")
println("equivalence: ok\n")

# ============================================================================
# 5. Render
# ============================================================================

println("="^60)
println("RENDER")
println("="^60)

b_af_r = @benchmark render(withparams($cm, $p0), $x)
b_hand_r = @benchmark hand_render.(Ref($p0), $x)

t_af_r = median(b_af_r).time
t_hand_r = median(b_hand_r).time

@printf("  AstroFit:    %8.1f ns  (%d allocs)\n", t_af_r, b_af_r.allocs)
@printf("  Handwritten: %8.1f ns  (%d allocs)\n", t_hand_r, b_hand_r.allocs)
@printf("  Ratio:       %8.2fx\n", t_af_r / t_hand_r)

# ============================================================================
# 6. Chi2 (objective function evaluation)
# ============================================================================

println("\n", "="^60)
println("CHI2")
println("="^60)

hand_obj(p) = hand_chi2(p, x, y, err)

b_af_c = @benchmark $af_obj($p0)
b_hand_c = @benchmark $hand_obj($p0)

t_af_c = median(b_af_c).time
t_hand_c = median(b_hand_c).time

@printf("  AstroFit:    %8.1f ns  (%d allocs)\n", t_af_c, b_af_c.allocs)
@printf("  Handwritten: %8.1f ns  (%d allocs)\n", t_hand_c, b_hand_c.allocs)
@printf("  Ratio:       %8.2fx\n", t_af_c / t_hand_c)

# ============================================================================
# 7. Gradient
# ============================================================================

println("\n", "="^60)
println("GRADIENT  (ForwardDiff)")
println("="^60)

ForwardDiff.gradient(af_obj, p0)
ForwardDiff.gradient(hand_obj, p0)

b_af_g = @benchmark ForwardDiff.gradient($af_obj, $p0)
b_hand_g = @benchmark ForwardDiff.gradient($hand_obj, $p0)

t_af_g = median(b_af_g).time
t_hand_g = median(b_hand_g).time

@printf("  AstroFit:    %8.1f ns  (%d allocs)\n", t_af_g, b_af_g.allocs)
@printf("  Handwritten: %8.1f ns  (%d allocs)\n", t_hand_g, b_hand_g.allocs)
@printf("  Ratio:       %8.2fx\n", t_af_g / t_hand_g)

# ============================================================================
# 8. Optimization (LBFGS)
# ============================================================================

println("\n", "="^60)
println("OPTIMIZATION  (LBFGS)")
println("="^60)

p_start = [0.0, 0.5, 5.0, 6560.0, 5.0]
lb, ub = AstroFit.bounds(cm)

function run_af_optim(p_start)
    obj = ObjectiveFunction(cm, x, y, err; statistic = :chi2)
    optf = OptimizationFunction(obj, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optf, p_start; lb, ub)
    return solve(prob, LBFGS())
end

function run_hand_optim(p_start)
    optf = OptimizationFunction((p, _) -> hand_chi2(p, x, y, err), Optimization.AutoForwardDiff())
    prob = OptimizationProblem(optf, p_start; lb, ub)
    return solve(prob, LBFGS())
end

sol_af = run_af_optim(p_start)
sol_hand = run_hand_optim(p_start)
@printf("  converged: AstroFit chi2=%.2f  Handwritten chi2=%.2f\n", sol_af.objective, sol_hand.objective)

b_af_o = @benchmark run_af_optim($p_start)
b_hand_o = @benchmark run_hand_optim($p_start)

t_af_o = median(b_af_o).time
t_hand_o = median(b_hand_o).time

@printf("  AstroFit:    %10.1f ns  (%d allocs)\n", t_af_o, b_af_o.allocs)
@printf("  Handwritten: %10.1f ns  (%d allocs)\n", t_hand_o, b_hand_o.allocs)
@printf("  Ratio:       %10.2fx\n", t_af_o / t_hand_o)

# ============================================================================
# 9. Summary
# ============================================================================

println("\n", "="^60)
println("SUMMARY — Hα + [NII], 5 free params, $(length(x)) data points")
println("="^60)
@printf("  %-14s  %12s  %12s  %8s\n", "", "AstroFit", "Handwritten", "Ratio")
@printf("  %-14s  %10.1f ns  %10.1f ns  %6.2fx\n", "render", t_af_r, t_hand_r, t_af_r / t_hand_r)
@printf("  %-14s  %10.1f ns  %10.1f ns  %6.2fx\n", "chi2", t_af_c, t_hand_c, t_af_c / t_hand_c)
@printf("  %-14s  %10.1f ns  %10.1f ns  %6.2fx\n", "gradient", t_af_g, t_hand_g, t_af_g / t_hand_g)
@printf("  %-14s  %10.1f ns  %10.1f ns  %6.2fx\n", "optimization", t_af_o, t_hand_o, t_af_o / t_hand_o)
