function _missing_distributions()
    throw(ArgumentError(
        "Bayesian prior evaluation requires Distributions.jl. " *
        "Load it with `using Distributions` before calling logprior/logposterior."))
end

logprior(args...) = _missing_distributions()

function _check_data(x, y, err)
    length(x) == length(y) || throw(ArgumentError(
        "`x` and `y` must have the same length"))
    length(y) == length(err) || throw(ArgumentError(
        "`y` and `err` must have the same length"))
    all(>(0), err) || throw(ArgumentError("all `err` values must be positive"))
    nothing
end

function loglikelihood(cm::CompiledModel, x, y, err)
    _check_data(x, y, err)
    residual = (render(cm, x) .- y) ./ err
    -0.5 * sum(abs2, residual) - sum(log, err) - length(y) / 2 * log(2π)
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
