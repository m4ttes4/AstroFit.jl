module AstroFitPigeonsExt

using AstroFit
using Pigeons: Pigeons, DistributionLogPotential
using Pigeons.Random: AbstractRNG
using Distributions: product_distribution

function Pigeons.initialization(f::AstroFit.ObjectiveFunction, rng::AbstractRNG, ::Int)
    _check_target(f)
    return [rand(rng, dist) for dist in f.priors]
end

# Dispatch on our target type inside the interpolated potential, so Pigeons'
# own `sample_names(::Array, _)` still serves every other target.
const _AstroFitPotential = Pigeons.InterpolatedLogPotential{
    <:Pigeons.InterpolatingPath{<:Any, <:AstroFit.ObjectiveFunction},
}
Pigeons.sample_names(::Array, p::_AstroFitPotential) = [Symbol.(p.path.target.names); :log_density]

function Pigeons.default_reference(f::AstroFit.ObjectiveFunction)
    _check_target(f)
    return DistributionLogPotential(product_distribution(f.priors))
end

# The reference distribution *is* the prior — reusing f.priors (rather than a
# separate bounds-derived fallback) keeps reference and target sharing the
# same support automatically: truncated iff the user truncated the prior.
function _check_target(f::AstroFit.ObjectiveFunction)
    f.statistic === AstroFit.logposterior || throw(ArgumentError(
        "Pigeons requires a log-density statistic. " *
        "Use `ObjectiveFunction(cm, x, y, err; statistic = logposterior)`."
    ))
    f.priors === nothing && throw(ArgumentError(
        "no priors set on this model — Pigeons requires every free parameter " *
            "to have a prior (0/$(f.ndim) set)"
    ))
    return nothing
end

end
