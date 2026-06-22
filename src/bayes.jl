function _missing_distributions()
    throw(
        ArgumentError(
            "Bayesian prior evaluation requires Distributions.jl. " *
                "Load it with `using Distributions` before calling logprior/logposterior."
        )
    )
end

logprior(args...) = _missing_distributions()
setprior(args...) = _missing_distributions()

_coords(x::Tuple) = x
_coords(x) = (x,)

function _check_data(x, y, err)
    all(c -> length(c) == length(y), _coords(x)) || throw(
        ArgumentError(
            "each coordinate array must have the same length as `y`"
        )
    )
    err === nothing && return nothing
    length(y) == length(err) || throw(
        ArgumentError(
            "`y` and `err` must have the same length"
        )
    )
    all(>(0), err) || throw(ArgumentError("all `err` values must be positive"))
    return nothing
end

function _chi2(model, coords, y, err)
    return sum(eachindex(y)) do i
        r = render(model, map(c -> @inbounds(c[i]), coords)...) - @inbounds(y[i])
        err === nothing ? abs2(r) : abs2(r / @inbounds(err[i]))
    end
end

function _loglikelihood(model, x, y, err)
    _check_data(x, y, err)
    χ2 = _chi2(model, _coords(x), y, err)
    return err === nothing ? -0.5 * χ2 - length(y) / 2 * log(2π) :
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


    
struct PosteriorTarget{CM, X, Y, E}
    cm::CM
    x::X
    y::Y
    err::E
    lb::Vector{Float64}
    ub::Vector{Float64}
    names::Vector{Symbol}
end

function PosteriorTarget(cm::CompiledModel, x, y, err = nothing)
    _check_data(x, y, err)
    lb, ub = bounds(cm)
    return PosteriorTarget(cm, x, y, err, lb, ub, paramnames(cm))
end

function (lp::PosteriorTarget)(p)
    for i in eachindex(p)
        (p[i] < lp.lb[i] || p[i] > lp.ub[i]) && return -Inf
    end
    return logposterior(lp.cm, p, lp.x, lp.y, lp.err)
end
