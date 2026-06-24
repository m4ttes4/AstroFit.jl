module AstroFitPigeonsExt

using AstroFit
using Pigeons: Pigeons, DistributionLogPotential
using Pigeons.Random: AbstractRNG
using Distributions: Uniform, product_distribution

function Pigeons.initialization(f::AstroFit.ObjectiveFunction, ::AbstractRNG, ::Int)
    f.statistic === Val(:chi2) && throw(ArgumentError(
        "Pigeons requires a log-density statistic, got :chi2. " *
        "Use `ObjectiveFunction(cm, x, y, err; statistic = :neglogposterior)` or `:negloglikelihood`."
    ))
    return AstroFit.params(f.cm)
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
    priors = getfield(f.cm, :priors)
    names = f.names
    prior_map = Dict{Symbol, Any}()
    if priors !== nothing
        for ((leaf, field), dist) in priors
            prior_map[Symbol(leaf, :_, field)] = dist
        end
    end
    dists = map(eachindex(names)) do i
        name = names[i]
        if haskey(prior_map, name)
            return prior_map[name]
        end
        lo, hi = f.lower[i], f.upper[i]
        isfinite(lo) && isfinite(hi) && return Uniform(lo, hi)
        throw(ArgumentError(
            "Parameter `$name` has no prior and no finite bounds. " *
            "Set a prior with `@constrain` or provide an explicit `reference` to `pigeons()`."
        ))
    end
    return DistributionLogPotential(product_distribution(dists))
end

end
