module AstroFitOptimizationExt

using AstroFit
import Optimization: OptimizationFunction, OptimizationProblem
using Optimization: AutoForwardDiff
using ForwardDiff

function OptimizationFunction(cm::AstroFit.CompiledModel, x, y, err = nothing;
                              adtype = AutoForwardDiff())
    f = objective(cm, x, y; err)
    OptimizationFunction((u, _p) -> f(u), adtype)
end

function OptimizationProblem(cm::AstroFit.CompiledModel, x, y, err = nothing;
                             adtype = AutoForwardDiff(), kwargs...)
    optf   = OptimizationFunction(cm, x, y, err; adtype)
    u0     = params(cm)
    lb, ub = bounds(cm)
    if all(isinf, lb) && all(isinf, ub)
        OptimizationProblem(optf, u0; kwargs...)
    else
        OptimizationProblem(optf, u0; lb, ub, kwargs...)
    end
end

end
