# Shared definitions for the `withparams` overhead benchmarks.
#
# The handwritten kernels mirror the exact render formulas in src/model.jl:
#   Gaussian1D : A * exp(-((x - μ)/σ)^2 / 2)      (model.jl:22)
#   Linear1D   : slope * x + intercept            (model.jl:39)
# Each constrained kernel takes (p, x::Vector) and returns y::Vector, so it
# computes the same thing as `render(withparams(cm, p), x)`. Its constraints
# are hardcoded by hand and its `p` ordering matches AstroFit's free parameter
# order, so the same parameter vector drives both implementations.

using AstroFit

# Rest-frame wavelengths (Å), matching examples/halpha_nii_fit.jl
const λ_Ha = 6562.8
const λ_NII_r = 6583.4
const λ_NII_b = 6548.1

# ===========================================================================
# Part A.1 — small model: Linear1D + Gaussian1D
# ===========================================================================

function small_models()
    free = @model begin
        cont = Linear1D(slope = 0.1, intercept = 1.0)
        line = Gaussian1D(amplitude = 5.0, mean = 0.0, sigma = 1.5)
        cont + line
    end
    con = free
    @constrain con begin
        cont.slope = 0.1
        line.amplitude in (0.0, Inf)
        line.mean in (-5.0, 5.0)
        line.sigma in (0.3, 10.0)
    end
    return free, con
end

# slope fixed at 0.1 → 4 free: p = [intercept, A, μ, σ]
function hand_small_con(p, x)
    intercept, A, μ, σ = p
    return @. 0.1 * x + intercept + A * exp(-((x - μ) / σ)^2 / 2)
end

# ===========================================================================
# Part A.2 — realistic Hα + [NII] complex (from examples/halpha_nii_fit.jl)
# ===========================================================================

function complex_models()
    free = @model begin
        cont = Linear1D(slope = 0.0, intercept = 1.0)
        ha = Gaussian1D(amplitude = 10.0, mean = λ_Ha, sigma = 3.0)
        nii_r = Gaussian1D(amplitude = 3.0, mean = λ_NII_r, sigma = 3.0)
        nii_b = Gaussian1D(amplitude = 1.0, mean = λ_NII_b, sigma = 3.0)
        cont + ha + nii_r + nii_b
    end
    con = free
    @constrain con begin
        cont.intercept in (0.0, Inf)
        ha.amplitude in (0.0, Inf)
        ha.mean in (λ_Ha - 30, λ_Ha + 30)
        ha.sigma in (0.5, 20.0)
        nii_b.amplitude -> ha.amplitude / 3.0
        nii_r.amplitude -> (3.06 / 3.0) * ha.amplitude
        nii_r.mean -> (λ_NII_r / λ_Ha) * ha.mean
        nii_b.mean -> (λ_NII_b / λ_Ha) * ha.mean
        nii_r.sigma -> ha.sigma
        nii_b.sigma -> ha.sigma
    end
    return free, con
end

# bounds + 6 ties → 5 free:  p = [slope, intercept, ha.A, ha.μ, ha.σ]
# ties baked in: nii_b.A = A/3, nii_r.A = 3.06/3·A, centroids via λ-ratios, σ shared
function hand_complex_con(p, x)
    s, ic, A, μ, σ = p
    rA = (3.06 / 3.0) * A
    bA = A / 3.0
    rμ = (λ_NII_r / λ_Ha) * μ
    bμ = (λ_NII_b / λ_Ha) * μ
    return @. s * x + ic +
        A * exp(-((x - μ) / σ)^2 / 2) +
        rA * exp(-((x - rμ) / σ)^2 / 2) +
        bA * exp(-((x - bμ) / σ)^2 / 2)
end

# ===========================================================================
# Part B — scaling sweep: sum of N Gaussians, built programmatically.
#
# con: @bound g1.amplitude, and @tie gᵢ.amplitude = g1.amplitude·0.5 (i>1)
#      → 2N+1 free. The @model/@constrain blocks are assembled as Exprs and
#      eval'd once per N, so the macros resolve names→optics for us.
#
# AstroFit's constrained parameter order for this generated tree follows the
# leaf/field walk used by params/withparams:
#   [g1.amplitude, g1.mean, g1.sigma, g2.mean, g2.sigma, ..., gN.mean, gN.sigma]
# ===========================================================================

function nbump_models(N)
    decls = [
        :(
                $(Symbol("g", i)) =
                Gaussian1D(amplitude = 1.0, mean = $(Float64(i)), sigma = 1.0)
            )
            for i in 1:N
    ]
    sumexpr = foldl((a, b) -> :($a + $b), (Symbol("g", i) for i in 1:N))
    modelblock = Expr(:block, decls..., sumexpr)

    ties = [
        :($(Symbol("g", i)).amplitude -> g1.amplitude * 0.5)
            for i in 2:N
    ]
    consblock = Expr(:block, :(g1.amplitude in (0.0, Inf)), ties...)

    return Core.eval(
        @__MODULE__, quote
            let
                _free = @model $modelblock
                _con = _free
                @constrain _con $consblock
                (_free, _con)
            end
        end
    )
end

function hand_nbump_con(p, x, N)
    A = p[1]
    y = zero.(x)
    for i in 1:N
        μ = p[2i]
        σ = p[2i + 1]
        Ai = i == 1 ? A : 0.5 * A
        y .+= @. Ai * exp(-((x - μ) / σ)^2 / 2)
    end
    return y
end
