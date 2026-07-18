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
    return rand(rng, f.cm)
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
    dists = AstroFit._reference_dists(f.cm, f.names, f.lower, f.upper)
    return DistributionLogPotential(product_distribution(dists))
end

end
