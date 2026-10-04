# Gradient benchmark: AstroFit vs handwritten, ForwardDiff cost separated
#
# Same model as astrofit_vs_handwritten.jl (Hα + [NII], 5 free, 1000 pts).
# Two variants:
#
#   AstroFit     — ObjectiveFunction (withparams + render + chi2)
#   handwritten  — ties hoisted out of a plain loop (no @fastmath)
#
# Three metrics per variant:
#
#   chi2   — f(p)
#   grad   — ForwardDiff.gradient(f, p): allocates GradientConfig + Dual
#            buffers each call, as Optimization's AutoForwardDiff does
#   grad!  — ForwardDiff.gradient!(g, f, p, cfg) with a preallocated
#            GradientConfig (Chunk{5}): the gradient alone
#
# Measured in R rounds, variants alternated within each round; each cell
# prints the min–max of the round medians, so drift within one process
# shows up. If the grad! ratio is stable across processes but grad is not,
# the variance comes from config/buffer allocation, not from AstroFit.
#
# Run:  julia --project=bench bench/gradient_benchmark.jl

using AstroFit
using BenchmarkTools
using ForwardDiff
using Random
using Printf

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.0

const R = 5

const λ_Ha = 6562.8
const λ_NII_r = 6583.45
const λ_NII_b = 6548.05

cm = @model begin
    cont = Linear1D(slope = 0.002, intercept = 1.0)
    ha = Gaussian1D(amplitude = 8.0, mean = λ_Ha, sigma = 4.0)
    nii_r = Gaussian1D(amplitude = (3.06 / 3.0) * 8.0, mean = (λ_NII_r / λ_Ha) * λ_Ha, sigma = 4.0)
    nii_b = Gaussian1D(amplitude = 8.0 / 3.0, mean = (λ_NII_b / λ_Ha) * λ_Ha, sigma = 4.0)
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

# p = [slope, intercept, A_ha, μ_ha, σ]
function hand_chi2(p, x, y, err)
    s, ic, A, μ, σ = p
    rA = (3.06 / 3.0) * A
    bA = A / 3.0
    rμ = (λ_NII_r / λ_Ha) * μ
    bμ = (λ_NII_b / λ_Ha) * μ
    acc = zero(eltype(p))
    @inbounds for i in eachindex(y)
        xi = x[i]
        m = s * xi + ic +
            A * exp(-((xi - μ) / σ)^2 / 2) +
            rA * exp(-((xi - rμ) / σ)^2 / 2) +
            bA * exp(-((xi - bμ) / σ)^2 / 2)
        acc += abs2((m - y[i]) / err[i])
    end
    return acc
end

Random.seed!(42)
const x = collect(range(6500.0, 6650.0; length = 1000))
p0 = AstroFit.params(cm)
const err = fill(0.3, length(x))
const y = render(withparams(cm, p0), x) .+ 0.3 .* randn(length(x))

af_obj = ObjectiveFunction(cm, x, y, err; statistic = chi2)
hand_obj(p) = hand_chi2(p, x, y, err)

@assert af_obj(p0) ≈ hand_obj(p0) "chi2 mismatch"
@assert ForwardDiff.gradient(af_obj, p0) ≈ ForwardDiff.gradient(hand_obj, p0) "gradient mismatch"

rev = readchomp(`git -C $(pkgdir(AstroFit)) rev-parse --short HEAD`)
println("Hα + [NII] gradient — $(nfree(cm)) free, $(length(x)) pts — AstroFit @ $rev ($(pathof(AstroFit)))")
println("equivalence: ok, $R rounds\n")

variants = (AstroFit = af_obj, handwritten = hand_obj)
metrics = (:chi2, :grad, :grad!)

# times[variant][metric] = round medians (ns); allocs from the last round
times = Dict(v => Dict(m => Float64[] for m in metrics) for v in keys(variants))
allocs = Dict(v => Dict{Symbol, Int}() for v in keys(variants))

for _ in 1:R, (v, f) in pairs(variants)
    g = similar(p0)
    cfg = ForwardDiff.GradientConfig(f, p0, ForwardDiff.Chunk{5}())
    bs = (
        chi2 = @benchmark($f($p0)),
        grad = @benchmark(ForwardDiff.gradient($f, $p0)),
        grad! = @benchmark(ForwardDiff.gradient!($g, $f, $p0, $cfg)),
    )
    for m in metrics
        push!(times[v][m], median(bs[m]).time)
        allocs[v][m] = bs[m].allocs
    end
end

cell(ts) = @sprintf("%8.0f (%.0f–%.0f)", median(ts), minimum(ts), maximum(ts))

@printf("  %-12s  %-24s  %-24s  %-24s  %s\n", "", "chi2 [ns]", "grad [ns]", "grad! [ns]", "allocs (grad / grad!)")
for v in keys(variants)
    @printf("  %-12s  %-24s  %-24s  %-24s  %d / %d\n", v,
        cell(times[v][:chi2]), cell(times[v][:grad]), cell(times[v][:grad!]),
        allocs[v][:grad], allocs[v][:grad!])
end
ratio(m) = median(times[:AstroFit][m]) / median(times[:handwritten][m])
@printf("  %-12s  %-24s  %-24s  %-24s\n", "ratio",
    @sprintf("%.2fx", ratio(:chi2)), @sprintf("%.2fx", ratio(:grad)), @sprintf("%.2fx", ratio(:grad!)))
