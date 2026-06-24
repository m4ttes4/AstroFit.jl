module AstroFitLogDensityProblemsExt

using AstroFit
using LogDensityProblems: LogDensityProblems

function _check_posterior_statistic(f::AstroFit.ObjectiveFunction)
    f.statistic === Val(:logposterior) && return nothing
    throw(ArgumentError(
        "LogDensityProblems requires `statistic = :logposterior`, " *
        "got `$(typeof(f.statistic).parameters[1])`. " *
        "Construct with `ObjectiveFunction(cm, x, y, err; statistic = :logposterior)`."
    ))
end

function LogDensityProblems.logdensity(f::AstroFit.ObjectiveFunction, p)
    _check_posterior_statistic(f)
    return f(p)
end

function LogDensityProblems.dimension(f::AstroFit.ObjectiveFunction)
    _check_posterior_statistic(f)
    return length(f.lower)
end

LogDensityProblems.capabilities(::Type{<:AstroFit.ObjectiveFunction}) =
    LogDensityProblems.LogDensityOrder{0}()

end
