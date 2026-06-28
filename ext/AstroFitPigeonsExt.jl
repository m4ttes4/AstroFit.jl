module AstroFitPigeonsExt

using AstroFit
using Pigeons: Pigeons, DistributionLogPotential
using Pigeons.Random: AbstractRNG
using Distributions: product_distribution

function Pigeons.initialization(f::AstroFit.ObjectiveFunction, rng::AbstractRNG, ::Int)
    f.statistic === Val(:chi2) && throw(ArgumentError(
        "Pigeons requires a log-density statistic, got :chi2. " *
        "Use `ObjectiveFunction(cm, x, y, err; statistic = :neglogposterior)` or `:negloglikelihood`."
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
    f.statistic === Val(:chi2) && throw(ArgumentError(
        "Pigeons requires a log-density statistic, got :chi2. " *
        "Use `ObjectiveFunction(cm, x, y, err; statistic = :neglogposterior)` or `:negloglikelihood`."
    ))
    dists = AstroFit._reference_dists(f.cm, f.names, f.lower, f.upper)
    return DistributionLogPotential(product_distribution(dists))
end

end
