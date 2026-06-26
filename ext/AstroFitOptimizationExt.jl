module AstroFitOptimizationExt

using AstroFit
import Optimization: OptimizationFunction, OptimizationProblem
using Optimization: AutoForwardDiff
using ForwardDiff

function OptimizationFunction(
        cm::AstroFit.CompiledModel, x, y, err = nothing;
        statistic = :chi2, adtype = AutoForwardDiff(), kwargs...
    )
    f = AstroFit.ObjectiveFunction(cm, x, y, err; statistic)
    return OptimizationFunction(f, adtype; kwargs...)
end

function OptimizationFunction(
        f::AstroFit.ObjectiveFunction; adtype = AutoForwardDiff(), kwargs...
    )
    return OptimizationFunction(f, adtype; kwargs...)
end

function OptimizationProblem(
        cm::AstroFit.CompiledModel, x, y, err = nothing;
        statistic = :chi2, adtype = AutoForwardDiff(), kwargs...
    )
    optf = AstroFit.ObjectiveFunction(cm, x, y, err; statistic)
    u0 = params(cm)
    lb, ub = optf.lower, optf.upper

    f = OptimizationFunction(optf; adtype)

    return if all(isinf, lb) && all(isinf, ub)
        OptimizationProblem(f, u0; kwargs...)
    else
        OptimizationProblem(f, u0; lb, ub, kwargs...)
    end
end

end
