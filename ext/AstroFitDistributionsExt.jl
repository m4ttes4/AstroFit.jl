module AstroFitDistributionsExt

using AstroFit
using Distributions: logpdf

import AstroFit: logprior, setprior, _validate_priors

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

end
