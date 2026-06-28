# Prior entry points. Real implementations live in AstroFitDistributionsExt —
# priors are a "Distributions loaded" feature, so the bare stubs throw a helpful
# error when Distributions isn't loaded. The `priors` field on CompiledModel is
# structural (compiled.jl); only setting/checking/evaluating them needs Distributions.

function _missing_distributions()
    throw(
        ArgumentError(
            "Bayesian prior evaluation requires Distributions.jl. " *
                "Load it with `using Distributions` before calling logprior/logposterior."
        )
    )
end

"""
    setprior(cm::CompiledModel, leaf::Symbol, field::Symbol, dist) -> CompiledModel

Attach a prior distribution `dist` to parameter `leaf.field`. Requires `Distributions.jl`.

See also: [`logprior`](@ref), [`@prior`](@ref)
"""
setprior(args...) = _missing_distributions()

"""
    logprior(cm::CompiledModel, p) -> Float64

Evaluate the sum of log-prior densities at parameter vector `p`. Requires `Distributions.jl`.

See also: [`setprior`](@ref), [`logposterior`](@ref)
"""
logprior(args...) = _missing_distributions()

function _logprior(cm::CompiledModel, p)
    priors = getfield(cm, :priors)
    (priors === nothing || isempty(priors)) && return 0.0
    logprior(cm, p)
end

# Validate priors target free parameters. Real method in DistributionsExt; the
# Nothing case (no priors) is a no-op that needs no Distributions.
_validate_priors(::CompiledModel{<:Any, Nothing}) = nothing

# Build reference distributions for sampling: prior if set, else Uniform from
# bounds. Used by AstroFitPigeonsExt; implemented in AstroFitDistributionsExt
# (which Pigeons co-triggers, so the method is always available there).
_reference_dists(args...) = _missing_distributions()
