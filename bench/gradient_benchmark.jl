# Gradient benchmark: is AstroFit's AD advantage real or an artifact?
#
# The astrofit_vs_handwritten benchmark showed AstroFit ~4x faster on
# gradients. But the handwritten baseline was monolithic: ties inside a
# @fastmath loop. Two hypotheses for the gap:
#
#   H1: ties are recomputed per point (hoisting them would close the gap)
#   H2: @fastmath in the loop hurts ForwardDiff's Dual arithmetic
#
# Four variants, same model (Hα + [NII], 5 free, 1000 pts):
#
#   monolithic   — ties inside @fastmath loop (original baseline)
#   split        — ties hoisted, @fastmath loop
#   no-fastmath  — ties hoisted, plain loop (no @fastmath)
#   AstroFit     — withparams + ObjectiveFunction
#
# Run:  julia --project=bench bench/gradient_benchmark.jl

using AstroFit
using BenchmarkTools
using ForwardDiff
using CairoMakie
using Random
using Printf

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.0

const λ_Ha    = 6562.8
const λ_NII_r = 6583.45
const λ_NII_b = 6548.05

# ============================================================================
# AstroFit model
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

# ============================================================================
# Handwritten variants — all compute the same chi2
# ============================================================================

# 1. Monolithic: ties recomputed per point, @fastmath
function monolithic_chi2(p, x, y, err)
    s, ic, A, μ, σ = p
    acc = zero(eltype(p))
    @inbounds @fastmath for i in eachindex(y)
        xi = x[i]
        rA = (3.06 / 3.0) * A;  bA = A / 3.0
        rμ = (λ_NII_r / λ_Ha) * μ;  bμ = (λ_NII_b / λ_Ha) * μ
        m = s * xi + ic +
            A  * exp(-((xi - μ)  / σ)^2 / 2) +
            rA * exp(-((xi - rμ) / σ)^2 / 2) +
            bA * exp(-((xi - bμ) / σ)^2 / 2)
        acc += abs2((m - y[i]) / err[i])
    end
    return acc
end

# 2. Split: ties hoisted, @fastmath loop
function split_chi2(p, x, y, err)
    s, ic, A, μ, σ = p
    rA = (3.06 / 3.0) * A;  bA = A / 3.0
    rμ = (λ_NII_r / λ_Ha) * μ;  bμ = (λ_NII_b / λ_Ha) * μ
    acc = zero(eltype(p))
    @inbounds @fastmath for i in eachindex(y)
        xi = x[i]
        m = s * xi + ic +
            A  * exp(-((xi - μ)  / σ)^2 / 2) +
            rA * exp(-((xi - rμ) / σ)^2 / 2) +
            bA * exp(-((xi - bμ) / σ)^2 / 2)
        acc += abs2((m - y[i]) / err[i])
    end
    return acc
end

# 3. No-fastmath: ties hoisted, plain loop
function nofm_chi2(p, x, y, err)
    s, ic, A, μ, σ = p
    rA = (3.06 / 3.0) * A;  bA = A / 3.0
    rμ = (λ_NII_r / λ_Ha) * μ;  bμ = (λ_NII_b / λ_Ha) * μ
    acc = zero(eltype(p))
    @inbounds for i in eachindex(y)
        xi = x[i]
        m = s * xi + ic +
            A  * exp(-((xi - μ)  / σ)^2 / 2) +
            rA * exp(-((xi - rμ) / σ)^2 / 2) +
            bA * exp(-((xi - bμ) / σ)^2 / 2)
        acc += abs2((m - y[i]) / err[i])
    end
    return acc
end

# ============================================================================
# Data
# ============================================================================

Random.seed!(42)
x = collect(range(6500.0, 6650.0; length = 1000))
p0 = AstroFit.params(cm)
y_true = render(withparams(cm, p0), x)
err = fill(0.3, length(x))
y = y_true .+ 0.3 .* randn(length(x))

af_obj   = ObjectiveFunction(cm, x, y, err; statistic = :chi2)
mono(p)  = monolithic_chi2(p, x, y, err)
split(p) = split_chi2(p, x, y, err)
nofm(p)  = nofm_chi2(p, x, y, err)

# ============================================================================
# Verify
# ============================================================================

@assert af_obj(p0) ≈ mono(p0) ≈ split(p0) ≈ nofm(p0) "chi2 mismatch"
g_af, g_mono, g_split, g_nofm = (ForwardDiff.gradient(f, p0) for f in (af_obj, mono, split, nofm))
@assert g_af ≈ g_mono ≈ g_split ≈ g_nofm "gradient mismatch"

println("Hα + [NII] gradient benchmark — $(nfree(cm)) free, $(length(x)) pts")
println("equivalence: ok\n")

# ============================================================================
# Benchmark
# ============================================================================

variants = [
    ("AstroFit",    af_obj),
    ("monolithic",  mono),
    ("split",       split),
    ("no-fastmath", nofm),
]

chi2_times = Float64[]
grad_times = Float64[]
chi2_allocs = Int[]
grad_allocs = Int[]

for (name, f) in variants
    bf = @benchmark $f($p0)
    bg = @benchmark ForwardDiff.gradient($f, $p0)
    push!(chi2_times, median(bf).time)
    push!(grad_times, median(bg).time)
    push!(chi2_allocs, bf.allocs)
    push!(grad_allocs, bg.allocs)
end

println("=" ^ 65)
@printf("  %-14s  %12s  %8s  %12s  %8s\n", "", "chi2", "allocs", "gradient", "allocs")
println("-" ^ 65)
for i in eachindex(variants)
    @printf("  %-14s  %10.1f ns  %6d  %10.1f ns  %6d\n",
        variants[i][1], chi2_times[i], chi2_allocs[i], grad_times[i], grad_allocs[i])
end
println("=" ^ 65)

# ratios relative to no-fastmath (fairest baseline)
nofm_grad = grad_times[4]
println()
println("Gradient ratios (vs no-fastmath baseline):")
for i in eachindex(variants)
    @printf("  %-14s  %.2fx\n", variants[i][1], grad_times[i] / nofm_grad)
end

println()
println("Conclusions:")
split_vs_mono = grad_times[3] / grad_times[2]
nofm_vs_split = grad_times[4] / grad_times[3]
af_vs_nofm = grad_times[1] / grad_times[4]
@printf("  hoisting ties:      split/monolithic   = %.2fx  (H1: %s)\n",
    split_vs_mono, abs(1 - split_vs_mono) < 0.1 ? "no effect" : "matters")
@printf("  dropping @fastmath: nofm/split         = %.2fx  (H2: %s)\n",
    nofm_vs_split, abs(1 - nofm_vs_split) < 0.1 ? "no effect" : "matters")
@printf("  AstroFit vs fair:   AstroFit/nofm      = %.2fx\n", af_vs_nofm)

# ============================================================================
# Plot
# ============================================================================

names = [v[1] for v in variants]
colors = [:dodgerblue, :tomato, :seagreen, :orange]

fig = Figure(size = (800, 420), fontsize = 14)

ax = Axis(fig[1, 1];
    title = "Gradient: Hα + [NII], 5 free params, $(length(x)) pts",
    ylabel = "Time (µs)",
    xticks = (eachindex(names), names),
)

barplot!(ax, eachindex(names), grad_times ./ 1e3;
    color = colors, strokewidth = 0.5, strokecolor = :grey50)

for i in eachindex(names)
    ratio = grad_times[i] / nofm_grad
    lbl = @sprintf("%.1fx", ratio)
    text!(ax, i, grad_times[i] / 1e3 * 1.05;
        text = lbl, align = (:center, :bottom), fontsize = 12)
end

Label(fig[2, 1],
    "monolithic: ties in @fastmath loop · split: ties hoisted · no-fastmath: plain loop · ratios vs no-fastmath";
    fontsize = 10, color = :grey50)

save("bench/gradient_benchmark.png", fig; px_per_unit = 2)
println("\nsaved -> bench/gradient_benchmark.png")
