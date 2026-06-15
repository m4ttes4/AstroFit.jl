# Shared definitions for the constraint-overhead benchmarks.
#
# The handwritten kernels mirror the exact render formulas in src/model.jl:
#   Gaussian1D : A * exp(-((x - μ)/σ)^2 / 2)      (model.jl:22)
#   Linear1D   : slope * x + intercept            (model.jl:39)
# Each kernel takes (p, x::Vector) and returns y::Vector, so it computes the
# same thing as `render(withparams(cm, p), x)`. The constrained kernels bake
# the ties straight into the code; their `p` ordering matches AstroFit's free
# parameter order (`free_lenses`/spec order) so the *same* p drives both.

using AstroFit

# Rest-frame wavelengths (Å), matching examples/halpha_nii_fit.jl
const λ_Ha    = 6562.8
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
    con = @constrain free begin
        @fix   cont.slope = 0.1
        @bound line.amplitude in (0.0, Inf)
        @bound line.mean      in (-5.0, 5.0)
        @bound line.sigma     in (0.3, 10.0)
    end
    free, con
end

# all 5 params free:           p = [slope, intercept, A, μ, σ]
function hand_small_free(p, x)
    slope, intercept, A, μ, σ = p
    @. slope * x + intercept + A * exp(-((x - μ) / σ)^2 / 2)
end

# slope fixed at 0.1 → 4 free: p = [intercept, A, μ, σ]
function hand_small_con(p, x)
    intercept, A, μ, σ = p
    @. 0.1 * x + intercept + A * exp(-((x - μ) / σ)^2 / 2)
end

# ===========================================================================
# Part A.2 — realistic Hα + [NII] complex (from examples/halpha_nii_fit.jl)
# ===========================================================================

function complex_models()
    free = @model begin
        cont  = Linear1D(slope = 0.0, intercept = 1.0)
        ha    = Gaussian1D(amplitude = 10.0, mean = λ_Ha,    sigma = 3.0)
        nii_r = Gaussian1D(amplitude = 3.0,  mean = λ_NII_r, sigma = 3.0)
        nii_b = Gaussian1D(amplitude = 1.0,  mean = λ_NII_b, sigma = 3.0)
        cont + ha + nii_r + nii_b
    end
    con = @constrain free begin
        @bound cont.intercept in (0.0, Inf)
        @bound ha.amplitude   in (0.0, Inf)
        @bound ha.mean        in (λ_Ha - 30, λ_Ha + 30)
        @bound ha.sigma       in (0.5, 20.0)
        @tie   nii_b.amplitude = ha.amplitude / 3.0
        @tie   nii_r.amplitude = (3.06 / 3.0) * ha.amplitude
        @tie   nii_r.mean      = (λ_NII_r / λ_Ha) * ha.mean
        @tie   nii_b.mean      = (λ_NII_b / λ_Ha) * ha.mean
        @tie   nii_r.sigma     = ha.sigma
        @tie   nii_b.sigma     = ha.sigma
    end
    free, con
end

# all 11 params free:
# p = [slope, intercept, ha.A, ha.μ, ha.σ, nr.A, nr.μ, nr.σ, nb.A, nb.μ, nb.σ]
function hand_complex_free(p, x)
    s, ic, hA, hμ, hσ, rA, rμ, rσ, bA, bμ, bσ = p
    @. s * x + ic +
       hA * exp(-((x - hμ) / hσ)^2 / 2) +
       rA * exp(-((x - rμ) / rσ)^2 / 2) +
       bA * exp(-((x - bμ) / bσ)^2 / 2)
end

# bounds + 6 ties → 5 free:  p = [slope, intercept, ha.A, ha.μ, ha.σ]
# ties baked in: nii_b.A = A/3, nii_r.A = 3.06/3·A, centroids via λ-ratios, σ shared
function hand_complex_con(p, x)
    s, ic, A, μ, σ = p
    rA = (3.06 / 3.0) * A
    bA = A / 3.0
    rμ = (λ_NII_r / λ_Ha) * μ
    bμ = (λ_NII_b / λ_Ha) * μ
    @. s * x + ic +
       A  * exp(-((x - μ)  / σ)^2 / 2) +
       rA * exp(-((x - rμ) / σ)^2 / 2) +
       bA * exp(-((x - bμ) / σ)^2 / 2)
end

# ===========================================================================
# Part B — scaling sweep: sum of N Gaussians, built programmatically.
#
# free: all 3N params free.
# con : @bound g1.amplitude, and @tie gᵢ.amplitude = g1.amplitude·0.5 (i>1)
#       → 2N+1 free. The @model/@constrain blocks are assembled as Exprs and
#       eval'd once per N, so the macros resolve names→optics for us (no manual
#       optic building, and no handwritten kernel whose param layout would have
#       to track spec order at every N).
# ===========================================================================

function nbump_models(N)
    decls = [:( $(Symbol("g", i)) =
                  Gaussian1D(amplitude = 1.0, mean = $(Float64(i)), sigma = 1.0) )
             for i in 1:N]
    sumexpr    = foldl((a, b) -> :($a + $b), (Symbol("g", i) for i in 1:N))
    modelblock = Expr(:block, decls..., sumexpr)

    ties      = [:( @tie $(Symbol("g", i)).amplitude = g1.amplitude * 0.5 )
                 for i in 2:N]
    consblock = Expr(:block, :( @bound g1.amplitude in (0.0, Inf) ), ties...)

    Core.eval(@__MODULE__, quote
        let
            _free = @model $modelblock
            _con  = @constrain _free $consblock
            (_free, _con)
        end
    end)
end
