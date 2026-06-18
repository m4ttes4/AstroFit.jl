# Benchmarks for AstroFit's `withparams` rendering overhead.
#
# Thesis: the fastest way to evaluate a constrained model is a handwritten
# function where bounds, fixed values, and ties are hardcoded. AstroFit aims to
# get close to that baseline while preserving reusable, composable models.
#
# The headline measurement is therefore:
#
#   render(withparams(constrained_cm, p), x)  vs  handwritten_constrained(p, x)
#
# `withparams` is timed separately because it is where AstroFit scatters the
# flat parameter vector into the model and resolves ties. Those operations are
# generated as straight-line code from the constraint spec.
#
# Run:
#   cd /home/matteo/julia/AstroFit.jl
#   julia --project=/home/matteo/.julia/environments/v1.12 bench/benchmarks.jl
#
# Uses the Julia environment in /home/matteo/.julia/environments/v1.12.
# It needs AstroFit, BenchmarkTools and Plots.

using AstroFit
using BenchmarkTools
using Printf

include(joinpath(@__DIR__, "kernels.jl"))

# Keep the script useful for quick local runs while still collecting medians.
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.75

stat(b) = (time = median(b).time, allocs = b.allocs, memory = b.memory)
ratio(a, b) = a / b
ns_to_us(x) = x / 1e3

function row(label, s)
    @printf("    %-34s %10.1f ns   %6d allocs   %8d bytes\n",
            label, s.time, s.allocs, s.memory)
end

function ratio_row(label, num, den)
    @printf("    %-34s %10.3f x\n", label, ratio(num.time, den.time))
end

function check_equivalent(label, astro_y, hand_y)
    @assert astro_y ≈ hand_y "$label: handwritten constrained kernel diverges from AstroFit"
    maximum(abs.(astro_y .- hand_y))
end

function benchmark_fixed(label, free_cm, con_cm, hand_con, x)
    pf = paramvector(free_cm)
    pc = paramvector(con_cm)

    astro_y = render(withparams(con_cm, pc), x)
    hand_y = hand_con(pc, x)
    maxerr = check_equivalent(label, astro_y, hand_y)

    @printf("\n== %s ==\n", label)
    @printf("    data points: %d | free params: %d | constrained params: %d | max |Δ|: %.3e\n",
            length(x), nfree(free_cm), nfree(con_cm), maxerr)

    b_wp_con = @benchmark withparams($con_cm, $pc)
    b_af_con = @benchmark render(withparams($con_cm, $pc), $x)
    b_hand_con = @benchmark $hand_con($pc, $x)

    s_wp_con = stat(b_wp_con)
    s_af_con = stat(b_af_con)
    s_hand_con = stat(b_hand_con)

    println("  primary constrained comparison:")
    row("AstroFit withparams only", s_wp_con)
    row("AstroFit render(withparams)", s_af_con)
    row("handwritten constrained render", s_hand_con)
    ratio_row("AstroFit / handwritten render", s_af_con, s_hand_con)
    @printf("    %-34s %10.3f %%\n",
            "withparams share of AstroFit render",
            100 * ratio(s_wp_con.time, s_af_con.time))

    b_wp_free = @benchmark withparams($free_cm, $pf)
    b_af_free = @benchmark render(withparams($free_cm, $pf), $x)
    s_wp_free = stat(b_wp_free)
    s_af_free = stat(b_af_free)

    println("  secondary AstroFit free vs constrained:")
    row("AstroFit free withparams", s_wp_free)
    row("AstroFit free render(withparams)", s_af_free)
    ratio_row("constrained/free withparams", s_wp_con, s_wp_free)
    ratio_row("constrained/free render", s_af_con, s_af_free)

    (label = label,
     nfree_free = nfree(free_cm),
     nfree_con = nfree(con_cm),
     withparams = s_wp_con,
     af_render = s_af_con,
     hand_render = s_hand_con,
     free_withparams = s_wp_free,
     free_render = s_af_free)
end

function make_sweep_cases(Ns)
    map(Ns) do N
        free_cm, con_cm = nbump_models(N)
        x = collect(range(0.0, N + 1.0; length = 200))
        pc = paramvector(con_cm)
        (N = N, free_cm = free_cm, con_cm = con_cm, x = x, pc = pc)
    end
end

function benchmark_sweep(cases)
    rows = NamedTuple[]
    println("\n== Scaling: constrained AstroFit vs handwritten constrained ==")
    println("    N    params   withparams ns   AstroFit us   handwritten us   ratio")

    for case in cases
        N = case.N
        con_cm = case.con_cm
        x = case.x
        pc = case.pc

        astro_y = render(withparams(con_cm, pc), x)
        hand_y = hand_nbump_con(pc, x, N)
        check_equivalent("N=$N sweep", astro_y, hand_y)

        b_wp = @benchmark withparams($con_cm, $pc)
        b_af = @benchmark render(withparams($con_cm, $pc), $x)
        b_hand = @benchmark hand_nbump_con($pc, $x, $N)

        s_wp = stat(b_wp)
        s_af = stat(b_af)
        s_hand = stat(b_hand)
        r = ratio(s_af.time, s_hand.time)

        @printf("    %2d      %3d      %10.1f    %10.3f      %10.3f   %6.3f x\n",
                N, nfree(con_cm), s_wp.time, ns_to_us(s_af.time),
                ns_to_us(s_hand.time), r)

        push!(rows, (N = N,
                     nfree = nfree(con_cm),
                     withparams = s_wp,
                     af_render = s_af,
                     hand_render = s_hand,
                     render_ratio = r))
    end

    rows
end

println("AstroFit withparams overhead benchmarks")
println("=======================================")

free_s, con_s = small_models()
small = benchmark_fixed("Small constrained line (Linear1D + Gaussian1D)",
                        free_s, con_s, hand_small_con, collect(-10.0:0.1:10.0))

free_c, con_c = complex_models()
complex = benchmark_fixed("Hα + [NII] constrained complex",
                          free_c, con_c, hand_complex_con, collect(6500.0:1.0:6650.0))

sweep_cases = make_sweep_cases([1, 2, 4, 8, 16])
sweep = benchmark_sweep(sweep_cases)

println("\nplotting...")
using Plots

fixed_labels = ["small", "Hα+[NII]"]
fixed_ratios = [ratio(small.af_render.time, small.hand_render.time),
                ratio(complex.af_render.time, complex.hand_render.time)]
fixed_withparams = ns_to_us.([small.withparams.time, complex.withparams.time])

p_fixed = plot(layout = (1, 2), size = (920, 360), legend = false,
               bottom_margin = 6Plots.mm, left_margin = 6Plots.mm)
bar!(p_fixed[1], fixed_labels, fixed_ratios;
     title = "AstroFit / handwritten", ylabel = "render ratio")
hline!(p_fixed[1], [1.0]; color = :black, linestyle = :dash)
bar!(p_fixed[2], fixed_labels, fixed_withparams;
     title = "withparams only", ylabel = "time [µs]")
savefig(p_fixed, joinpath(@__DIR__, "fixed_complex.png"))

Ns = [r.N for r in sweep]
sweep_ratios = [r.render_ratio for r in sweep]
sweep_withparams = ns_to_us.([r.withparams.time for r in sweep])
sweep_af = ns_to_us.([r.af_render.time for r in sweep])
sweep_hand = ns_to_us.([r.hand_render.time for r in sweep])

p_sweep = plot(layout = (1, 3), size = (1200, 380), legend = :topleft,
               bottom_margin = 6Plots.mm, left_margin = 6Plots.mm)
plot!(p_sweep[1], Ns, sweep_ratios; marker = :o, label = "ratio",
      title = "AstroFit / handwritten", xlabel = "N Gaussians",
      ylabel = "render ratio")
hline!(p_sweep[1], [1.0]; color = :black, linestyle = :dash, label = "1x")
plot!(p_sweep[2], Ns, sweep_withparams; marker = :o, label = "withparams",
      title = "withparams only", xlabel = "N Gaussians", ylabel = "time [µs]")
plot!(p_sweep[3], Ns, [sweep_af sweep_hand]; marker = :o,
      label = ["AstroFit" "handwritten"], title = "constrained render",
      xlabel = "N Gaussians", ylabel = "time [µs]")
savefig(p_sweep, joinpath(@__DIR__, "scaling.png"))

println("saved -> ", joinpath(@__DIR__, "fixed_complex.png"))
println("saved -> ", joinpath(@__DIR__, "scaling.png"))
