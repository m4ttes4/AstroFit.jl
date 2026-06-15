module AstroFitOptimizationExt

using AstroFit
using Optimization: OptimizationFunction, OptimizationProblem, AutoForwardDiff
# Loaded for its side effect: registers the ForwardDiff backend that the default
# `AutoForwardDiff()` adtype dispatches to (via DifferentiationInterface).
using ForwardDiff

# Pick the loss form once, at problem-build time, so the closure handed to
# Optimization is monomorphic: the solve loop stays type-stable and the
# ForwardDiff Duals flow cleanly through `withparams` (see project.md §3.4).
# `x` is the independent variable: a single array (1D) or a tuple of arrays
# (ND, e.g. (X, Y) for an image) — splatted into `render` via `_coords`.
function _objective(cm, x, y, err)
    if err === nothing
        return u -> sum(abs2, render(withparams(cm, u), AstroFit._coords(x)...) .- y)  # unweighted LSQ
    elseif isempty(getfield(cm, :priors))
        return u -> -AstroFit.loglikelihood(cm, u, x, y, err)       # MLE
    else
        return u -> -logposterior(cm, u, x, y, err)                # MAP (needs Distributions)
    end
end

function OptimizationFunction(cm::AstroFit.CompiledModel, x, y, err = nothing;
                              adtype = AutoForwardDiff())
    loss = _objective(cm, x, y, err)
    OptimizationFunction((u, _p) -> loss(u), adtype)
end

function OptimizationProblem(cm::AstroFit.CompiledModel, x, y, err = nothing;
                             adtype = AutoForwardDiff(), kwargs...)
    optf   = OptimizationFunction(cm, x, y, err; adtype)
    u0     = paramvector(cm)
    lb, ub = bounds_vectors(getfield(cm, :spec))
    # Omit bounds entirely when nothing is constrained, so unconstrained
    # optimizers (plain BFGS / NelderMead) accept the problem.
    if all(isinf, lb) && all(isinf, ub)
        OptimizationProblem(optf, u0; kwargs...)
    else
        OptimizationProblem(optf, u0; lb, ub, kwargs...)
    end
end

end
