function _missing_distributions()
    throw(ArgumentError(
        "Bayesian prior evaluation requires Distributions.jl. " *
        "Load it with `using Distributions` before calling logprior/logposterior."))
end

logprior(args...) = _missing_distributions()
setprior(args...) = _missing_distributions()

_coords(x::Tuple) = x
_coords(x)        = (x,)

function _check_data(x, y, err)
    all(c -> length(c) == length(y), _coords(x)) || throw(ArgumentError(
        "each coordinate array must have the same length as `y`"))
    err === nothing && return nothing
    length(y) == length(err) || throw(ArgumentError(
        "`y` and `err` must have the same length"))
    all(>(0), err) || throw(ArgumentError("all `err` values must be positive"))
    nothing
end

function _chi2(model, coords, y, err)
    sum(eachindex(y)) do i
        r = render(model, map(c -> @inbounds(c[i]), coords)...) - @inbounds(y[i])
        err === nothing ? abs2(r) : abs2(r / @inbounds(err[i]))
    end
end

function _loglikelihood(model, x, y, err)
    _check_data(x, y, err)
    χ2 = _chi2(model, _coords(x), y, err)
    err === nothing ? -0.5 * χ2 - length(y) / 2 * log(2π) :
                      -0.5 * χ2 - sum(log, err) - length(y) / 2 * log(2π)
end

loglikelihood(cm::CompiledModel, p, x, y, err) =
    _loglikelihood(withparams(cm, p), x, y, err)

loglikelihood(cm::CompiledModel, x, y, err) =
    loglikelihood(cm, params(cm), x, y, err)

logposterior(cm::CompiledModel, p, x, y, err) =
    (getfield(cm, :priors) === nothing ? 0.0 : logprior(cm, p)) +
    loglikelihood(cm, p, x, y, err)

logposterior(cm::CompiledModel, x, y, err) =
    logposterior(cm, params(cm), x, y, err)

objective(cm::CompiledModel, x, y; err = nothing) =
    u -> -logposterior(cm, u, x, y, err)
