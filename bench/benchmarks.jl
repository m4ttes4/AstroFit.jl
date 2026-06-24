# AstroFit render overhead benchmark.
#
# Measures how close AstroFit's composable models get to a handwritten function
# where constraints are hardcoded. Two cases:
#
#   1. Fixed:  Hα + [NII] triplet (5 free params, 6 ties)
#   2. Scaling: N Gaussians with all amplitudes tied (2N+1 free params)
#
# Run:  julia --project=bench bench/benchmarks.jl

using AstroFit
using BenchmarkTools
using CairoMakie
using Printf

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.75

# ============================================================================
# 1. Hα + [NII] — fixed realistic case
# ============================================================================

const λ_Ha    = 6562.8
const λ_NII_r = 6583.4
const λ_NII_b = 6548.1

cm = @model begin
    cont  = Linear1D(slope = 0.0, intercept = 1.0)
    ha    = Gaussian1D(amplitude = 10.0, mean = λ_Ha, sigma = 3.0)
    nii_r = Gaussian1D(amplitude = 3.0, mean = λ_NII_r, sigma = 3.0)
    nii_b = Gaussian1D(amplitude = 1.0, mean = λ_NII_b, sigma = 3.0)
    cont + ha + nii_r + nii_b
end

@constrain cm begin
    cont.intercept in (0.0, Inf)
    ha.amplitude   in (0.0, Inf)
    ha.mean        in (λ_Ha - 30, λ_Ha + 30)
    ha.sigma       in (0.5, 20.0)
    nii_r.amplitude -> (3.06 / 3.0) * ha.amplitude
    nii_r.mean      -> (λ_NII_r / λ_Ha) * ha.mean
    nii_r.sigma     -> ha.sigma
    nii_b.amplitude -> ha.amplitude / 3.0
    nii_b.mean      -> (λ_NII_b / λ_Ha) * ha.mean
    nii_b.sigma     -> ha.sigma
end

# Handwritten equivalent: same ties baked in, same parameter order.
# p = [slope, intercept, ha.A, ha.μ, ha.σ]
function hand_render(p, x)
    s, ic, A, μ, σ = p
    rA = (3.06 / 3.0) * A
    bA = A / 3.0
    rμ = (λ_NII_r / λ_Ha) * μ
    bμ = (λ_NII_b / λ_Ha) * μ
    return @. s * x + ic +
        A  * exp(-((x - μ)  / σ)^2 / 2) +
        rA * exp(-((x - rμ) / σ)^2 / 2) +
        bA * exp(-((x - bμ) / σ)^2 / 2)
end

x_fixed = collect(6500.0:1.0:6650.0)
p_fixed = AstroFit.params(cm)

# sanity check
@assert render(withparams(cm, p_fixed), x_fixed) ≈ hand_render(p_fixed, x_fixed)

println("== Hα + [NII]:  $(nfree(cm)) free params, $(length(x_fixed)) points ==\n")

b_wp   = @benchmark withparams($cm, $p_fixed)
b_af   = @benchmark render(withparams($cm, $p_fixed), $x_fixed)
b_hand = @benchmark hand_render($p_fixed, $x_fixed)

t_wp   = median(b_wp).time
t_af   = median(b_af).time
t_hand = median(b_hand).time

@printf("  %-30s %10.1f ns  %d allocs\n", "withparams alone",          t_wp,   b_wp.allocs)
@printf("  %-30s %10.1f ns  %d allocs\n", "AstroFit render(withparams)", t_af,  b_af.allocs)
@printf("  %-30s %10.1f ns  %d allocs\n", "handwritten render",        t_hand, b_hand.allocs)
@printf("  %-30s %10.2f x\n",             "ratio AstroFit/handwritten", t_af / t_hand)

# ============================================================================
# 2. Scaling sweep: N Gaussians, all amplitudes tied to g1
# ============================================================================

function make_models(N)
    decls = [:($(Symbol("g", i)) = Gaussian1D(amplitude=1.0, mean=$(Float64(i)), sigma=1.0)) for i in 1:N]
    sumexpr = foldl((a, b) -> :($a + $b), (Symbol("g", i) for i in 1:N))
    ties = [:($(Symbol("g", i)).amplitude -> g1.amplitude * 0.5) for i in 2:N]
    consblock = Expr(:block, :(g1.amplitude in (0.0, Inf)), ties...)
    Core.eval(@__MODULE__, quote
        let
            _cm = @model $(Expr(:block, decls..., sumexpr))
            @constrain _cm $consblock
            _cm
        end
    end)
end

# Handwritten N-bump: p = [A, μ₁, σ₁, μ₂, σ₂, …]
function hand_nbump(p, x, N)
    A = p[1]
    y = zero.(x)
    for i in 1:N
        μ, σ = p[2i], p[2i + 1]
        Ai = i == 1 ? A : 0.5 * A
        y .+= @. Ai * exp(-((x - μ) / σ)^2 / 2)
    end
    return y
end

Ns = [2, 4, 8, 16, 32, 64]

sweep_af   = Float64[]
sweep_hand = Float64[]
sweep_wp   = Float64[]

println("\n== Scaling: N Gaussians, all amplitudes tied ==")
@printf("  %4s  %5s  %12s  %12s  %12s  %8s\n",
    "N", "free", "withparams", "AstroFit", "handwritten", "ratio")

function bench_one(con_cm, x, N)
    pc = AstroFit.params(con_cm)
    @assert render(withparams(con_cm, pc), x) ≈ hand_nbump(pc, x, N)

    bw = @benchmark withparams($con_cm, $pc)
    ba = @benchmark render(withparams($con_cm, $pc), $x)
    bh = @benchmark hand_nbump($pc, $x, $N)

    tw, ta, th = median(bw).time, median(ba).time, median(bh).time
    push!(sweep_wp, tw)
    push!(sweep_af, ta)
    push!(sweep_hand, th)

    @printf("  %4d  %5d  %10.1f ns  %10.1f ns  %10.1f ns  %6.2f x\n",
        N, nfree(con_cm), tw, ta, th, ta / th)
end

for N in Ns
    con_cm = make_models(N)
    x = collect(range(0.0, N + 1.0; length = 400))
    Base.invokelatest(bench_one, con_cm, x, N)
end

# ============================================================================
# 3. Plot
# ============================================================================

fig = Figure(size = (900, 450), fontsize = 14)

# left: render times (log-log)
ax1 = Axis(fig[1, 1];
    title = "Constrained render time",
    xlabel = "N components", ylabel = "Time (µs)",
    xscale = log2, yscale = log10,
    xticks = Ns,
    xtickformat = xs -> string.(Int.(xs)),
)
scatterlines!(ax1, Ns, sweep_af ./ 1e3;
    label = "AstroFit", color = :dodgerblue, linewidth = 2, markersize = 10)
scatterlines!(ax1, Ns, sweep_hand ./ 1e3;
    label = "Handwritten", color = :tomato, linewidth = 2, markersize = 10)
axislegend(ax1; position = :lt)

# right: ratio as bars
ax2 = Axis(fig[1, 2];
    title = "AstroFit / Handwritten",
    xlabel = "N components", ylabel = "Ratio",
    xticks = (eachindex(Ns), string.(Ns)),
    yticks = [0.9, 0.95, 1.0, 1.05],
    yminorticksvisible = false,
)
ratios = sweep_af ./ sweep_hand
barplot!(ax2, eachindex(Ns), ratios;
    color = [r <= 1.0 ? :dodgerblue : :tomato for r in ratios],
    strokewidth = 0.5, strokecolor = :grey50)
hlines!(ax2, [1.0]; color = :black, linestyle = :dash, linewidth = 1.5)
ylims!(ax2, 0.88, 1.08)

# withparams cost as annotation on the left panel
let wp_label = join([@sprintf("N=%d: %dns", Ns[i], round(Int, sweep_wp[i])) for i in eachindex(Ns)], "  ")
    ha_label = @sprintf("Hα+[NII] fixed case: %.2fx (5 free, 151 pts)", t_af / t_hand)
    Label(fig[2, :], ha_label * "    ·    withparams overhead: " * wp_label;
        fontsize = 11, color = :grey50, halign = :center)
end

save("bench/scaling.png", fig; px_per_unit = 2)
println("\nsaved -> bench/scaling.png")
