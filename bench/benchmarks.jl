# Constraint-overhead benchmarks for AstroFit.
#
# Thesis: because the model and its constraints are unrolled into straight-line
# code at compile time (the @generated free_lenses / _scatter / _resolve in
# src/params.jl), adding constraints (bounds, fixes, ties) does not meaningfully
# change runtime. A constrained model evaluates / differentiates as fast as the
# same model with every parameter free, and both match a handwritten kernel with
# the constraints baked directly into the code.
#
# Ties are resolved inside `withparams` (the _resolve step); `render` walks the
# identical tree regardless of constraints. So `withparams` is exactly where any
# constraint overhead would show up — that is the headline measurement.
#
# Run:
#   cd /home/matteo/julia/AstroFit.jl
#   julia bench/benchmarks.jl
#
# Uses the global Julia env (~/.julia/environments/v1.12), which has AstroFit
# (dev-linked), BenchmarkTools, ForwardDiff and Plots. No bench-local Project.

using AstroFit
using BenchmarkTools
using ForwardDiff
using Printf

include(joinpath(@__DIR__, "kernels.jl"))

# Keep each measurement short — there are ~50 of them.
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.0

# least-squares loss closures for the gradient benchmarks
af_loss(cm, x, y)  = p -> sum(abs2, render(withparams(cm, p), x) .- y)
hand_loss(f, x, y) = p -> sum(abs2, f(p, x) .- y)

stat(b) = (median(b).time, b.allocs)           # (ns, allocations)
row(label, t, a) = @printf("    %-26s %10.1f ns   %6d allocs\n", label, t, a)

# ---------------------------------------------------------------------------
# Part A — fixed model, 4-way comparison
# ---------------------------------------------------------------------------
function run_fixed(title, free_cm, con_cm, hand_free, hand_con, x)
    pf = paramvector(free_cm)
    pc = paramvector(con_cm)
    y  = render(free_cm, x)

    # apples-to-apples check: kernels compute the same thing as the models
    @assert hand_free(pf, x) ≈ render(free_cm, x)
    @assert hand_con(pc, x)  ≈ render(con_cm, x)

    @printf("\n== %s  (%d data points;  free=%d params, constrained=%d) ==\n",
            title, length(x), nfree(free_cm), nfree(con_cm))

    println("  withparams (rebuild; ties resolve here):")
    (t, a) = stat(@benchmark withparams($free_cm, $pf)); row("AstroFit free", t, a)
    (t, a) = stat(@benchmark withparams($con_cm, $pc));  row("AstroFit constrained", t, a)

    println("  full evaluation  render(withparams(cm, p), x):")
    (t, a) = stat(@benchmark render(withparams($free_cm, $pf), $x))
    row("AstroFit free", t, a);                    r_aff = t
    (t, a) = stat(@benchmark render(withparams($con_cm, $pc), $x))
    row("AstroFit constrained", t, a);             r_acon = t
    (t, a) = stat(@benchmark $hand_free($pf, $x)); row("handwritten free", t, a); r_hf = t
    (t, a) = stat(@benchmark $hand_con($pc, $x));  row("handwritten constrained", t, a); r_hc = t

    println("  gradient  ForwardDiff.gradient(least-squares loss):")
    lf = af_loss(free_cm, x, y); lc = af_loss(con_cm, x, y)
    hf = hand_loss(hand_free, x, y); hc = hand_loss(hand_con, x, y)
    (t, a) = stat(@benchmark ForwardDiff.gradient($lf, $pf)); row("AstroFit free", t, a); g_aff = t
    (t, a) = stat(@benchmark ForwardDiff.gradient($lc, $pc)); row("AstroFit constrained", t, a); g_acon = t
    (t, a) = stat(@benchmark ForwardDiff.gradient($hf, $pf)); row("handwritten free", t, a); g_hf = t
    (t, a) = stat(@benchmark ForwardDiff.gradient($hc, $pc)); row("handwritten constrained", t, a); g_hc = t

    (render = (af_free = r_aff, af_con = r_acon, hand_free = r_hf, hand_con = r_hc),
     grad   = (af_free = g_aff, af_con = g_acon, hand_free = g_hf, hand_con = g_hc))
end

# ---------------------------------------------------------------------------
# Part B — scaling sweep, AstroFit free vs constrained across N Gaussians
# ---------------------------------------------------------------------------
function run_sweep(Ns)
    Nf = Float64[]
    wpf = Float64[]; wpc = Float64[]
    rdf = Float64[]; rdc = Float64[]
    grf = Float64[]; grc = Float64[]
    println("\n== Scaling sweep: sum of N Gaussians (AstroFit free vs constrained) ==")
    for N in Ns
        free_cm, con_cm = nbump_models(N)
        x  = collect(range(0.0, N + 1.0; length = 200))
        pf = paramvector(free_cm); pc = paramvector(con_cm)
        y  = render(free_cm, x)
        lf = af_loss(free_cm, x, y); lc = af_loss(con_cm, x, y)

        push!(Nf, N)
        push!(wpf, median(@benchmark withparams($free_cm, $pf)).time)
        push!(wpc, median(@benchmark withparams($con_cm, $pc)).time)
        push!(rdf, median(@benchmark render(withparams($free_cm, $pf), $x)).time)
        push!(rdc, median(@benchmark render(withparams($con_cm, $pc), $x)).time)
        push!(grf, median(@benchmark ForwardDiff.gradient($lf, $pf)).time)
        push!(grc, median(@benchmark ForwardDiff.gradient($lc, $pc)).time)

        @printf("  N=%2d  params free=%2d con=%2d | withparams %6.0f/%-6.0f  render %7.0f/%-7.0f  grad %8.0f/%-8.0f ns\n",
                N, nfree(free_cm), nfree(con_cm),
                wpf[end], wpc[end], rdf[end], rdc[end], grf[end], grc[end])
    end
    (; N = Nf, wpf, wpc, rdf, rdc, grf, grc)
end

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
println("AstroFit constraint-overhead benchmarks")
println("=======================================")

free_s, con_s = small_models()
small_res = run_fixed("Small  (Linear1D + Gaussian1D)", free_s, con_s,
                      hand_small_free, hand_small_con, collect(-10.0:0.1:10.0))

free_c, con_c = complex_models()
cplx_res = run_fixed("Hα + [NII] complex (4 Gaussians)", free_c, con_c,
                     hand_complex_free, hand_complex_con, collect(6500.0:1.0:6650.0))

sweep = run_sweep([1, 2, 4, 8, 16])

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
println("\nplotting…")
using Plots

# Figure 1 — scaling sweep: free vs constrained overlap as N grows.
ns_to_us(v) = v ./ 1e3
p1 = plot(layout = (1, 3), size = (1200, 380), legend = :topleft,
          left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)
plot!(p1[1], sweep.N, ns_to_us.([sweep.wpf sweep.wpc]); marker = :o,
      label = ["free" "constrained"], title = "withparams (rebuild)",
      xlabel = "N Gaussians", ylabel = "time [µs]")
plot!(p1[2], sweep.N, ns_to_us.([sweep.rdf sweep.rdc]); marker = :o,
      label = ["free" "constrained"], title = "render (200 pts)",
      xlabel = "N Gaussians", ylabel = "time [µs]")
plot!(p1[3], sweep.N, ns_to_us.([sweep.grf sweep.grc]); marker = :o,
      label = ["free" "constrained"], title = "ForwardDiff gradient",
      xlabel = "N Gaussians", ylabel = "time [µs]")
savefig(p1, joinpath(@__DIR__, "scaling.png"))

# Figure 2 — fixed Hα/[NII] complex: 4-way bars (zero-overhead in absolute terms).
labels = ["AF\nfree" "AF\nconstr" "hand\nfree" "hand\nconstr"]
rvals = ns_to_us.([cplx_res.render.af_free, cplx_res.render.af_con,
                   cplx_res.render.hand_free, cplx_res.render.hand_con])
gvals = ns_to_us.([cplx_res.grad.af_free, cplx_res.grad.af_con,
                   cplx_res.grad.hand_free, cplx_res.grad.hand_con])
barcols = [:steelblue, :steelblue, :indianred, :indianred]  # blue = AstroFit, red = handwritten
p2 = plot(layout = (1, 2), size = (900, 380), legend = false,
          bottom_margin = 6Plots.mm, left_margin = 6Plots.mm)
bar!(p2[1], vec(labels), rvals; title = "complex: render", ylabel = "time [µs]",
     fillcolor = barcols)
bar!(p2[2], vec(labels), gvals; title = "complex: gradient", ylabel = "time [µs]",
     fillcolor = barcols)
savefig(p2, joinpath(@__DIR__, "fixed_complex.png"))

println("saved → ", joinpath(@__DIR__, "scaling.png"))
println("saved → ", joinpath(@__DIR__, "fixed_complex.png"))
