module AstroFitPigeonsExt

using AstroFit
using Pigeons: Pigeons, DistributionLogPotential
using Pigeons.Random: AbstractRNG
using Distributions: product_distribution

function Pigeons.initialization(f::AstroFit.ObjectiveFunction, rng::AbstractRNG, ::Int)
    f.statistic === AstroFit.logposterior || throw(ArgumentError(
        "Pigeons requires a log-density statistic. " *
        "Use `ObjectiveFunction(cm, x, y, err; statistic = logposterior)`."
    ))
    _check_priors(f)
    return [rand(rng, dist) for dist in f.priors]
end


function Pigeons.sample_names(x::Array, p::Pigeons.InterpolatedLogPotential)
    target = p.path.target
    if target isa AstroFit.ObjectiveFunction
        return [Symbol.(target.names); :log_density]
    end
    return [map(i -> Symbol("param_$i"), 1:length(x)); :log_density]
end

function Pigeons.default_reference(f::AstroFit.ObjectiveFunction)
    f.statistic === AstroFit.logposterior || throw(ArgumentError(
        "Pigeons requires a log-density statistic. " *
        "Use `ObjectiveFunction(cm, x, y, err; statistic = logposterior)`."
    ))
    _check_priors(f)
    return DistributionLogPotential(product_distribution(f.priors))
end

# The reference distribution *is* the prior — reusing f.priors (rather than a
# separate bounds-derived fallback) keeps reference and target sharing the
# same support automatically: truncated iff the user truncated the prior.
_check_priors(f::AstroFit.ObjectiveFunction) = f.priors === nothing && throw(ArgumentError(
    "no priors set on this model — Pigeons requires every free parameter " *
        "to have a prior (0/$(f.ndim) set)"
))

end
