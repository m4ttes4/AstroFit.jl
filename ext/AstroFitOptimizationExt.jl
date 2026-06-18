module AstroFitOptimizationExt

using AstroFit
# `import` (non `using ...:`) perché aggiungiamo metodi a queste funzioni di
# Optimization per CompiledModel; definire un metodo su un binding importato con
# `using` emette un warning sotto Julia ≥ 1.12.
import Optimization: OptimizationFunction, OptimizationProblem
using Optimization: AutoForwardDiff
# Caricato per il side effect: registra il backend ForwardDiff su cui il default
# `AutoForwardDiff()` dispatcha (via DifferentiationInterface). La ext è
# co-triggered su ForwardDiff, quindi è sempre presente quando questa si attiva.
using ForwardDiff

"""
    OptimizationFunction(cm::CompiledModel, x, y, err = nothing; adtype = AutoForwardDiff())

Impacchetta per Optimization.jl la stessa funzione-obiettivo del core,
`objective(cm, x, y; err)` = `u -> -logposterior(cm, u, x, y, err)`. Senza prior è
la negative log-likelihood gaussiana; con `err = nothing` è l'obiettivo a varianza
unitaria (≡ minimi quadrati); con prior è la negative log-posterior (richiede
Distributions.jl).
"""
function OptimizationFunction(cm::AstroFit.CompiledModel, x, y, err = nothing;
                              adtype = AutoForwardDiff())
    f = objective(cm, x, y; err)
    OptimizationFunction((u, _p) -> f(u), adtype)
end

"""
    OptimizationProblem(cm::CompiledModel, x, y, err = nothing; adtype = AutoForwardDiff(), kwargs...)

Costruisce il problema partendo dai valori liberi correnti di `cm`. I bound (da
`@bound`) sono passati come `lb`/`ub` quando presenti — in quel caso usa un solver
box-aware (es. `Fminbox(LBFGS())`). Richiede `using Optimization, ForwardDiff`.
"""
function OptimizationProblem(cm::AstroFit.CompiledModel, x, y, err = nothing;
                             adtype = AutoForwardDiff(), kwargs...)
    optf   = OptimizationFunction(cm, x, y, err; adtype)
    u0     = paramvector(cm)
    lb, ub = bounds_vectors(getfield(cm, :spec))
    # Niente bound quando nulla è vincolato, così i solver non vincolati
    # (BFGS / NelderMead) accettano il problema.
    if all(isinf, lb) && all(isinf, ub)
        OptimizationProblem(optf, u0; kwargs...)
    else
        OptimizationProblem(optf, u0; lb, ub, kwargs...)
    end
end

end
