module AstroFitDistributionsExt

using AstroFit
using Distributions: logpdf, Distribution

import AstroFit: logprior, setprior, _validate_priors, _resolve_priors

function setprior(cm::AstroFit.CompiledModel, leaf::Symbol, field::Symbol, dist)
    AstroFit._masterfree(cm, leaf, field) || throw(
        ArgumentError(
            "prior target `$leaf.$field` must be a free parameter (Free or Bounded)"
        )
    )
    priors = getfield(cm, :priors)
    key = (leaf, field)
    existing = priors === nothing ? () : priors
    filtered = Tuple(p for p in existing if first(p) != key)
    return AstroFit.CompiledModel(getfield(cm, :tree), (filtered..., (key, dist)))
end

function _validate_priors(cm::AstroFit.CompiledModel)
    priors = getfield(cm, :priors)
    priors === nothing && return nothing
    for ((leaf, field), _) in priors
        AstroFit._masterfree(cm, leaf, field) || throw(
            ArgumentError(
                "prior on `$leaf.$field` targets a parameter that is not free (must be Free or Bounded)"
            )
        )
    end
    return nothing
end

# Resolve (leaf,field)=>dist priors into a Vector aligned with paramnames(cm)/p,
# once per ObjectiveFunction construction. Every free parameter must have a
# prior — logposterior/logprior are Bayesian-only entry points and there is no
# implicit fallback (bounds are not priors); this throws immediately if any
# free parameter is missing one, rather than silently contributing 0. Narrowed
# to a concrete small-Union eltype (not Vector{Any}) so logprior's hot-path
# loop dispatches logpdf statically per element instead of dynamically.
function _resolve_priors(cm::AstroFit.CompiledModel, names)
    priors = getfield(cm, :priors)
    priors === nothing && return nothing

    prior_map = Dict(Symbol(leaf, :_, field) => dist for ((leaf, field), dist) in priors)
    length(prior_map) == length(priors) || throw(ArgumentError(
        "duplicate prior target in @constrain block"
    ))

    resolved = map(names) do name
        dist = get(prior_map, name, nothing)
        dist === nothing && throw(ArgumentError(
            "parameter `$name` has no prior — every free parameter needs one for Bayesian inference"
        ))
        dist isa Distribution || throw(ArgumentError(
            "prior for `$name` must be a Distribution, got $(typeof(dist))"
        ))
        dist
    end
    length(prior_map) == length(names) || throw(ArgumentError(
        "prior set for a target that isn't a free parameter of this model"
    ))

    U = Union{unique(typeof.(resolved))...}
    return Vector{U}(resolved)
end

function logprior(f::AstroFit.ObjectiveFunction, p)
    dists = f.priors
    dists === nothing && throw(ArgumentError(
        "no priors set on this model — logprior/logposterior require every " *
            "free parameter to have a prior (0/$(length(p)) set)"
    ))
    s = 0.0
    for i in eachindex(dists, p)
        s += logpdf(dists[i], p[i])
    end
    return s
end

end
