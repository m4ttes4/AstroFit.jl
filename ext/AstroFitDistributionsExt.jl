module AstroFitDistributionsExt

using AstroFit
using Distributions: logpdf, Uniform

import AstroFit: logprior, setprior, _validate_priors, _reference_dists

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

function logprior(cm::AstroFit.CompiledModel, p)
    priors = getfield(cm, :priors)
    (priors === nothing || isempty(priors)) && return 0.0
    names = AstroFit.paramnames(cm)
    s = 0.0
    for ((leaf, field), dist) in priors
        target = Symbol(leaf, :_, field)
        idx = findfirst(==(target), names)
        s += logpdf(dist, p[idx])
    end
    return s
end

logprior(cm::AstroFit.CompiledModel) = logprior(cm, AstroFit.params(cm))

# Reference distribution per parameter: user prior if set, else Uniform from
# finite bounds. Throws if a parameter has neither.
function _reference_dists(cm::AstroFit.CompiledModel, names, lower, upper)
    priors = getfield(cm, :priors)
    prior_map = Dict{Symbol, Any}()
    if priors !== nothing
        for ((leaf, field), dist) in priors
            prior_map[Symbol(leaf, :_, field)] = dist
        end
    end
    return map(eachindex(names)) do i
        name = names[i]
        haskey(prior_map, name) && return prior_map[name]
        lo, hi = lower[i], upper[i]
        isfinite(lo) && isfinite(hi) && return Uniform(lo, hi)
        throw(ArgumentError(
            "Parameter `$name` has no prior and no finite bounds. " *
            "Set a prior with `@constrain` or provide an explicit reference distribution."
        ))
    end
end

end
