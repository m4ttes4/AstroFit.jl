function _missing_distributions()
    throw(ArgumentError(
        "Bayesian prior evaluation requires Distributions.jl. " *
        "Load it with `using Distributions` before calling logprior/logposterior."))
end

logprior(args...) = _missing_distributions()

# Independent coordinates: a single array (1D fit) or a tuple of arrays (ND fit,
# e.g. (X, Y) for an image). Normalized to a tuple so `render` can be splatted.
_coords(x::Tuple) = x
_coords(x)        = (x,)

function _check_data(x, y, err)
    all(c -> length(c) == length(y), _coords(x)) || throw(ArgumentError(
        "each coordinate array must have the same length as `y`"))
    err === nothing && return nothing               # noise-free fit: unit variance
    length(y) == length(err) || throw(ArgumentError(
        "`y` and `err` must have the same length"))
    all(>(0), err) || throw(ArgumentError("all `err` values must be positive"))
    nothing
end

"""
    loglikelihood(cm::CompiledModel, x, y, err)
    loglikelihood(cm::CompiledModel, p, x, y, err)

Gaussian log-likelihood of the data `y` (at coordinates `x`) under the model.

`err` is the per-point standard deviation as a vector matching `y`, or `nothing`
for a **noise-free fit**: the likelihood then assumes unit variance, so its
maximiser coincides with the least-squares solution. The second form rebuilds the
model from a flat free-parameter vector `p` first.
"""
function loglikelihood(cm::CompiledModel, x, y, err)
    _check_data(x, y, err)
    χ2 = _chi2(getfield(cm, :model), _coords(x), y, err)
    err === nothing && return -0.5 * χ2 - length(y) / 2 * log(2π)
    -0.5 * χ2 - sum(log, err) - length(y) / 2 * log(2π)
end

# Fused Σ residual² (weighted by `err` unless `nothing`). Iterates pointwise so no
# prediction/residual arrays are materialised — the bottleneck in the fit/AD loop.
# `_check_data` has validated the lengths, so the indexing is `@inbounds`.
function _chi2(model, coords, y, err)
    sum(eachindex(y)) do i
        r = render(model, map(c -> @inbounds(c[i]), coords)...) - @inbounds(y[i])
        err === nothing ? abs2(r) : abs2(r / @inbounds(err[i]))
    end
end

loglikelihood(cm::CompiledModel, p, x, y, err) =
    loglikelihood(withparams(cm, p), x, y, err)

# A model with no priors needs no Distributions: skip logprior entirely so the
# posterior collapses to the likelihood without forcing the optional dependency.
logposterior(cm::CompiledModel, x, y, err) =
    (isempty(getfield(cm, :priors)) ? 0.0 : logprior(cm)) +
    loglikelihood(cm, x, y, err)

logposterior(cm::CompiledModel, p, x, y, err) =
    (isempty(getfield(cm, :priors)) ? 0.0 : logprior(cm, p)) +
    loglikelihood(cm, p, x, y, err)

"""
    objective(cm::CompiledModel, x, y; err = nothing)

Return the scalar function `u -> -logposterior(cm, u, x, y, err)` to **minimise**
over the flat free-parameter vector `u` (start from `paramvector(cm)`; bounds from
`bounds_vectors(cm.spec)`). Solver-agnostic and AD-friendly — hand it to any
minimiser. With no priors it is the negative log-likelihood; `err = nothing` gives
the unit-variance / least-squares target; with priors it is the negative
log-posterior (needs Distributions.jl).
"""
objective(cm::CompiledModel, x, y; err = nothing) =
    u -> -logposterior(cm, u, x, y, err)
