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

# Validate priors target free parameters. Real method in DistributionsExt; the
# Nothing case (no priors) is a no-op that needs no Distributions.
_validate_priors(::CompiledModel{<:Any, Nothing}) = nothing

# Resolve (leaf,field)=>dist priors into a Vector aligned with paramnames(cm)/p,
# once per ObjectiveFunction construction instead of per logprior call. Real
# method (which also validates every free parameter has a prior) lives in
# DistributionsExt; the Nothing case needs no Distributions.
_resolve_priors(::CompiledModel{<:Any, Nothing}, names) = nothing
