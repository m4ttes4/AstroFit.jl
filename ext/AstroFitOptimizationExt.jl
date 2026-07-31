module AstroFitOptimizationExt

using AstroFit
import Optimization: OptimizationFunction, OptimizationProblem
using Optimization: AutoForwardDiff
using ForwardDiff

# Single-pass forward mode: seed all `n` free params at once (`Chunk{n}`) instead of
# ForwardDiff's default heuristic (caps chunk ~12, so >12 params take several padded
# passes). Measured 20–44% faster gradients across 17–122 params — the objective is a
# scalar per-point render loop, so one Dual{n} sweep beats multiple partial sweeps. Costs
# a one-time compile hit (Dual{n} specialization) that amortizes after ~10³ gradient
# calls; an optimizer/sampler does far more.
# ponytail: full chunk unconditionally. Ceiling is huge n (Dual{200}+ compile latency
# balloons superlinearly) — target models are <30 params; pass your own `adtype` if you
# build pathologically wide models.
_fullchunk(n) = AutoForwardDiff(chunksize = n)

function OptimizationFunction(
        cm::AstroFit.CompiledModel, x, y, err = nothing;
        statistic = AstroFit.chi2, adtype = _fullchunk(AstroFit.nfree(cm)), kwargs...
    )
    f = AstroFit.ObjectiveFunction(cm, x, y, err; statistic)
    return OptimizationFunction(f, adtype; kwargs...)
end

function OptimizationFunction(
        f::AstroFit.ObjectiveFunction; adtype = _fullchunk(f.ndim), kwargs...
    )
    return OptimizationFunction(f, adtype; kwargs...)
end

function OptimizationProblem(
        cm::AstroFit.CompiledModel, x, y, err = nothing;
        statistic = AstroFit.chi2, adtype = _fullchunk(AstroFit.nfree(cm)), kwargs...
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
