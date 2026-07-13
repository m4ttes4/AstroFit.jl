# Benchmark: manual loglikelihood vs Distributions.jl logpdf loop
#
# Four variants:
#   manual        — current impl: -0.5*chi2 + precomputed_const
#   dist_naive    — Normal(μᵢ, errᵢ) constructed + logpdf each iteration
#   dist_cached   — Normal(0, errᵢ) pre-built; logpdf(d, yᵢ - μᵢ) per call
#   dist_split    — pre-built dists, logconst precomputed; only kernel per call
#                   (this is the "fair" Distributions-based version)
#
# logpdf(Normal(μ,σ), y) = -0.5*((y-μ)/σ)² - log(σ) - 0.5*log(2π)
#                        = kernel(r)         + lognorm(σ)
# dist_split precomputes Σ lognorm(σᵢ) once, evaluates only kernel per call.
#
# Run: julia --startup-file=no --project=bench bench/loglike_vs_distributions.jl

using AstroFit, BenchmarkTools, Distributions, Printf

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.0

# ── model setup ─────────────────────────────────────────────────────────────

cm = @model begin
    g1 = Gaussian1D(amplitude = 10.0, mean = 6562.8, sigma = 3.0)
    g2 = Gaussian1D(amplitude = 3.0,  mean = 6583.4, sigma = 3.0)
    g1 + g2
end

@constrain cm begin
    g1.amplitude in (0.0, Inf)
    g1.mean      in (6530.0, 6600.0)
    g1.sigma     in (0.5, 20.0)
end

p = AstroFit.params(cm)

# ── variant 1: naive — Normal(μᵢ, σᵢ) + logpdf each iteration ──────────────

function loglike_dist_naive(cm, x, y, err, p)
    m = withparams(cm, p)
    s = 0.0
    @inbounds for i in eachindex(y)
        s += logpdf(Normal(AstroFit.render(m, x[i]), err[i]), y[i])
    end
    return s
end

# ── variant 2: cached dists — Normal(0, σᵢ) pre-built, logpdf per call ──────

function loglike_dist_cached(dists, cm, x, y, p)
    m = withparams(cm, p)
    s = 0.0
    @inbounds for i in eachindex(y)
        s += logpdf(dists[i], y[i] - AstroFit.render(m, x[i]))
    end
    return s
end

# ── variant 3: split — dists pre-built, lognorm precomputed, kernel only ─────
#
# logpdf(Normal(0,σ), r) = -0.5*(r/σ)² - log(σ) - 0.5*log(2π)
#                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We precompute Σ[-log(σᵢ) - 0.5*log(2π)] = logconst once.
# Per call: only evaluate -0.5*(rᵢ/σᵢ)² accessing dists[i].σ.

struct SplitDistLike{D}
    dists::Vector{D}
    logconst::Float64
end

function SplitDistLike(dists::Vector{D}) where D
    lc = sum(d -> -log(d.σ) - 0.5 * log(2π), dists)
    SplitDistLike{D}(dists, lc)
end

function loglike_dist_split(sd::SplitDistLike, cm, x, y, p)
    m = withparams(cm, p)
    s = 0.0
    @inbounds for i in eachindex(y)
        r = (y[i] - AstroFit.render(m, x[i])) / sd.dists[i].σ
        s -= 0.5 * r * r
    end
    return s + sd.logconst
end

# ── benchmark ────────────────────────────────────────────────────────────────

SIZES = [128, 1_024, 8_192]

@printf("\n%-6s  %11s  %11s  %11s  %11s  %8s  %8s  %8s\n",
    "N", "manual(ns)", "naive(ns)", "cached(ns)", "split(ns)",
    "naive/m", "cached/m", "split/m")
println(repeat('-', 90))

for N in SIZES
    x     = collect(range(6500.0, 6650.0; length = N))
    y     = AstroFit.render(withparams(cm, p), x) .+ 0.1 .* randn(N)
    err   = fill(0.1, N)
    dists = [Normal(0.0, err[i]) for i in eachindex(err)]
    sd    = SplitDistLike(dists)

    f = ObjectiveFunction(cm, x, y, err)

    b_man    = @benchmark AstroFit.loglikelihood($f, $p)
    b_naive  = @benchmark loglike_dist_naive($cm, $x, $y, $err, $p)
    b_cached = @benchmark loglike_dist_cached($dists, $cm, $x, $y, $p)
    b_split  = @benchmark loglike_dist_split($sd, $cm, $x, $y, $p)

    t_m, t_n, t_c, t_s =
        median(b_man).time, median(b_naive).time,
        median(b_cached).time, median(b_split).time

    @printf("%-6d  %11.1f  %11.1f  %11.1f  %11.1f  %8.2f  %8.2f  %8.2f\n",
        N, t_m, t_n, t_c, t_s, t_n/t_m, t_c/t_m, t_s/t_m)
end

println()
println("manual      = -0.5*chi2 + _loglike_const         (precomputed at ObjectiveFunction construction)")
println("dist_naive  = Σ logpdf(Normal(μᵢ,σᵢ), yᵢ)       (Normal object + log(σ) per point)")
println("dist_cached = Σ logpdf(Normal(0,σᵢ), yᵢ-μᵢ)     (Normal pre-built, log(σ) still inside logpdf)")
println("dist_split  = Σ -0.5*(rᵢ/σᵢ)² + precomputed_lc  (kernel only per point, logconst precomputed)")
