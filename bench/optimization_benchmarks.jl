# Optimization.jl benchmarks: AstroFit constrained models vs handwritten losses.
#
# Companion to benchmarks.jl. That script isolates render overhead; this one sits
# one layer up — the fit. For each case it compares, AstroFit vs handwritten:
#
#   objective f(p)     ObjectiveFunction(cm, x, y, err)  vs  hand_chi2(p, ...)
#   gradient  ∇f(p)    ForwardDiff through withparams+render  vs  through hand_chi2
#   solve(LBFGS)       OptimizationProblem(cm, …)  vs  hand OptimizationFunction
#
# The gradient is the headline: AutoForwardDiff pushes Dual numbers through
# withparams (which rebuilds the model tree and resolves ties every call), so any
# abstraction regression shows up there first. The handwritten chi2 kernels are
# *scalar, non-allocating* on purpose — mirroring AstroFit's chi2 loop — so the
# baseline doesn't pour Dual-array allocations into the gradient and skew the ratio.
# Each case asserts f(p0) ≈ hand(p0) and ∇ equal before timing: the guard against
# the constraints being hardcoded differently in the hand kernel.
#
# The solve ratio is the noisy metric: every case has bounds → Fminbox(LBFGS),
# sensitive to ULP-level objective differences, so the two solves can land on
# different iteration counts. iters_A vs iters_H is printed beside it and the ratio
# is flagged when they diverge — obj+grad stay the clean regression signal.
#
# Run (needs AstroFit, Optimization, OptimizationOptimJL, ForwardDiff,
# BenchmarkTools — the bench/ environment provides them):
#
#   julia --project=bench bench/optimization_benchmarks.jl
#
# A CSV of medians lands in bench/results/<timestamp>/optimization.csv (or set
# ASTROFIT_BENCH_OUTDIR / pass a directory as ARGS[1]) for diffing across commits.

using AstroFit
using Optimization, OptimizationOptimJL, ForwardDiff
using BenchmarkTools
using Dates, Printf

include(joinpath(@__DIR__, "kernels.jl"))   # models + render kernels + λ constants

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.5

# ===========================================================================
# Handwritten chi2 losses. Scalar, non-allocating, eltype(p)-typed so Duals flow.
# Constraints hardcoded to match the corresponding @constrain block in kernels.jl;
# the equivalence asserts in bench_opt catch any drift.
# ===========================================================================

# slope fixed at 0.1 → 4 free: p = [intercept, A, μ, σ]
function hand_small_chi2(p, x, y, err)
    intercept, A, μ, σ = p
    acc = zero(eltype(p))
    @inbounds for i in eachindex(y)
        m = 0.1 * x[i] + intercept + A * exp(-((x[i] - μ) / σ)^2 / 2)
        acc += abs2((m - y[i]) / err[i])
    end
    return acc
end

# bounds + 6 ties → 5 free: p = [slope, intercept, ha.A, ha.μ, ha.σ]
function hand_complex_chi2(p, x, y, err)
    s, ic, A, μ, σ = p
    rA = (3.06 / 3.0) * A
    bA = A / 3.0
    rμ = (λ_NII_r / λ_Ha) * μ
    bμ = (λ_NII_b / λ_Ha) * μ
    acc = zero(eltype(p))
    @inbounds for i in eachindex(y)
        m = s * x[i] + ic +
            A * exp(-((x[i] - μ) / σ)^2 / 2) +
            rA * exp(-((x[i] - rμ) / σ)^2 / 2) +
            bA * exp(-((x[i] - bμ) / σ)^2 / 2)
        acc += abs2((m - y[i]) / err[i])
    end
    return acc
end

# amplitudes tied to g1 (gᵢ = 0.5·g1, i>1) → 2N+1 free:
# p = [g1.A, g1.μ, g1.σ, g2.μ, g2.σ, …, gN.μ, gN.σ]
function hand_nbump_chi2(p, x, y, err, N)
    A = p[1]
    acc = zero(eltype(p))
    @inbounds for j in eachindex(y)
        m = zero(eltype(p))
        for i in 1:N
            μ = p[2i]
            σ = p[2i + 1]
            Ai = i == 1 ? A : 0.5 * A
            m += Ai * exp(-((x[j] - μ) / σ)^2 / 2)
        end
        acc += abs2((m - y[j]) / err[j])
    end
    return acc
end

# ===========================================================================
# Fused chi2: withparams once, then scalar @inbounds loop with per-point render.
# Same tree dispatch as ObjectiveFunction, but avoids sum(eachindex) closure.
# ===========================================================================

function fused_chi2(cm, p, x, y, err)
    model = withparams(cm, p)
    acc = zero(eltype(p))
    @inbounds for i in eachindex(y)
        r = render(model, x[i]) - y[i]
        acc += abs2(r / err[i])
    end
    return acc
end

# ===========================================================================
# Driver
# ===========================================================================

stat(b) = (time = median(b).time, allocs = b.allocs, memory = b.memory)
ns_to_us(t) = t / 1.0e3

# Optimization.jl OptimizationStats; fall back if a solver omits it.
_iters(sol) = try
    sol.stats.iterations
catch
    -1
end

function _solve_prob(optf, u0, lb, ub)
    return (all(isinf, lb) && all(isinf, ub)) ?
        OptimizationProblem(optf, u0) :
        OptimizationProblem(optf, u0; lb, ub)
end

# con_cm: the constrained AstroFit model. hand_render(p,x)::Vector generates the
# data; hand_chi2(p,x,y,err)::scalar is the handwritten loss. Both take the SAME
# free-parameter vector AstroFit uses (params/withparams DFS order).
# bench_solve=false skips the solve timing (still does obj+grad). A large bounded
# problem runs many Fminbox outer iterations → a single solve dwarfs the 0.5 s
# budget, giving a slow 1-sample median; obj+grad stay the clean signal there.
function bench_opt(label, con_cm, x, hand_render, hand_chi2; bench_solve = true)
    p0 = AstroFit.params(con_cm)
    lb, ub = AstroFit.bounds(con_cm)
    p_true = clamp.(p0 .* 1.1, lb, ub)          # data off the start point → real solve
    y = hand_render(p_true, x)
    err = ones(length(x))

    af_render(p) = render(withparams(con_cm, p), x)
    f = ObjectiveFunction(con_cm, x, y, err)    # statistic = chi2
    hand(p) = hand_chi2(p, x, y, err)

    y_af = af_render(p0)
    y_hand = hand_render(p0, x)
    rerr = maximum(abs.(y_af .- y_hand))
    @assert y_af ≈ y_hand "$label: handwritten render diverges (max|Δ| = $rerr)"

    fused(p) = fused_chi2(con_cm, p, x, y, err)

    @assert f(p0) ≈ hand(p0) "$label: handwritten chi2 diverges from ObjectiveFunction"
    @assert fused(p0) ≈ hand(p0) "$label: fused chi2 diverges from handwritten"
    ga = ForwardDiff.gradient(f, p0)
    gf = ForwardDiff.gradient(fused, p0)
    gh = ForwardDiff.gradient(hand, p0)
    gerr = maximum(abs.(ga .- gh))
    @assert ga ≈ gh "$label: handwritten gradient diverges (max|Δ| = $gerr)"
    @assert gf ≈ gh "$label: fused gradient diverges"

    r_a = stat(@benchmark $af_render($p0))
    r_h = stat(@benchmark (p -> $hand_render(p, $x))($p0))
    o_a = stat(@benchmark $f($p0))
    o_f = stat(@benchmark $fused($p0))
    o_h = stat(@benchmark $hand($p0))
    g_a = stat(@benchmark ForwardDiff.gradient($f, $p0))
    g_f = stat(@benchmark ForwardDiff.gradient($fused, $p0))
    g_h = stat(@benchmark ForwardDiff.gradient($hand, $p0))

    @printf("\n== %s ==\n", label)
    @printf(
        "    data points: %d | free params: %d | max|Δrender|: %.2e | max|Δgrad|: %.2e\n",
        length(x), nfree(con_cm), rerr, gerr
    )
    @printf("    %-16s %14s %14s %14s %10s %12s\n", "", "AstroFit", "fused", "handwritten", "A/hand", "allocs A/H")
    _mrow3("render m(p)", r_a, nothing, r_h, false)
    _mrow3("objective f(p)", o_a, o_f, o_h, false)
    _mrow3("gradient ∇f(p)", g_a, g_f, g_h, false)

    s_a = s_h = nothing
    itA = itH = -1
    if bench_solve
        prob_a = OptimizationProblem(con_cm, x, y, err)
        optf_h = OptimizationFunction((p, _) -> hand_chi2(p, x, y, err), AutoForwardDiff())
        prob_h = _solve_prob(optf_h, p0, lb, ub)
        sol_a = solve(prob_a, LBFGS())
        sol_h = solve(prob_h, LBFGS())
        itA, itH = _iters(sol_a), _iters(sol_h)
        fminΔ = abs(sol_a.objective - sol_h.objective)
        uΔ = maximum(abs.(sol_a.u .- sol_h.u))
        s_a = stat(@benchmark solve($prob_a, LBFGS()))
        s_h = stat(@benchmark solve($prob_h, LBFGS()))
        _mrow3("solve(LBFGS)", s_a, nothing, s_h, true)
        flag = itA == itH ? "" : "  ⚠ iters differ → solve ratio not comparable"
        @printf(
            "    converge: iters %d vs %d · fmin Δ %.2e · u Δ %.2e%s\n",
            itA, itH, fminΔ, uΔ, flag
        )
    else
        println("    solve(LBFGS)      skipped (large problem, slow/noisy single sample)")
    end

    flush(stdout)
    return (
        label = label, nfree = nfree(con_cm),
        rend = (r_a, r_h), obj = (o_a, o_h), grad = (g_a, g_h), solve = (s_a, s_h),
        itA = itA, itH = itH,
    )
end

# one metric line with optional fused column; `us` picks µs vs ns formatting
function _mrow3(name, a, f, h, us)
    r = a.time / h.time
    fs = f === nothing ? "" : @sprintf("%11.1f ns", f.time)
    fa = f === nothing ? "" : @sprintf("%6d", f.allocs)
    if us
        fs = f === nothing ? "" : @sprintf("%11.3f µs", ns_to_us(f.time))
        @printf(
            "    %-16s %11.3f µs %14s %11.3f µs %9.2fx %6d/%s/%-6d\n",
            name, ns_to_us(a.time), fs, ns_to_us(h.time), r, a.allocs, fa, h.allocs
        )
    else
        @printf(
            "    %-16s %11.1f ns %14s %11.1f ns %9.2fx %6d/%s/%-6d\n",
            name, a.time, fs, h.time, r, a.allocs, fa, h.allocs
        )
    end
end

# ===========================================================================
# Run
# ===========================================================================

const OUTDIR = get(
    ENV, "ASTROFIT_BENCH_OUTDIR",
    get(ARGS, 1, joinpath(@__DIR__, "results", Dates.format(now(), "yyyymmdd_HHMMSS")))
)
mkpath(OUTDIR)

println("AstroFit Optimization benchmarks")
println("================================")
println("output directory: ", OUTDIR)

results = NamedTuple[]

_, con_s = small_models()
push!(results, bench_opt(
    "Small (Linear1D + Gaussian1D, slope fixed)",
    con_s, collect(-10.0:0.1:10.0), hand_small_con, hand_small_chi2,
))

_, con_c = complex_models()
push!(results, bench_opt(
    "Hα + [NII] (bounds + 6 ties)",
    con_c, collect(6500.0:1.0:6650.0), hand_complex_con, hand_complex_chi2,
))

for N in (1, 2, 4, 8, 16)
    _, con_n = nbump_models(N)
    x = collect(range(0.0, N + 1.0; length = 200))
    push!(results, bench_opt(
        "Scaling N=$N Gaussians (amps tied)",
        con_n, x,
        (p, xx) -> hand_nbump_con(p, xx, N),
        (p, xx, yy, ee) -> hand_nbump_chi2(p, xx, yy, ee, N);
        bench_solve = N <= 8,
    ))
end

# CSV of medians for cross-commit diffing.
csv = joinpath(OUTDIR, "optimization.csv")
open(csv, "w") do io
    println(io, "case,nfree,metric,astrofit_ns,handwritten_ns,ratio,iters_a,iters_h")
    for r in results
        for (metric, pair) in (("render", r.rend), ("objective", r.obj), ("gradient", r.grad), ("solve", r.solve))
            a, h = pair
            a === nothing && continue             # solve skipped for this case
            @printf(
                io, "%s,%d,%s,%.1f,%.1f,%.4f,%d,%d\n",
                r.label, r.nfree, metric, a.time, h.time, a.time / h.time, r.itA, r.itH
            )
        end
    end
end
println("\nsaved -> ", csv)
