module AstroFitLogDensityProblemsExt

using AstroFit
using LogDensityProblems

function LogDensityProblems.logdensity(f::AstroFit.ObjectiveFunction, p)
    return AstroFit.logposterior(f, p)
end

LogDensityProblems.dimension(f::AstroFit.ObjectiveFunction) = f.ndim

LogDensityProblems.capabilities(::Type{<:AstroFit.ObjectiveFunction}) =
    LogDensityProblems.LogDensityOrder{0}()

end
