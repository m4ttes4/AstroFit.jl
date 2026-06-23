function _missing_distributions()
    throw(
        ArgumentError(
            "Bayesian prior evaluation requires Distributions.jl. " *
                "Load it with `using Distributions` before calling logprior/logposterior."
        )
    )
end

setprior(args...) = _missing_distributions()
logprior(args...) = _missing_distributions()

function _logprior(cm::CompiledModel, p)
    priors = getfield(cm, :priors)
    (priors === nothing || isempty(priors)) && return 0.0
    logprior(cm, p)
end

_coords(x::Tuple) = x
_coords(x) = (x,)

function chi2(model, coords, y, err)
    return sum(eachindex(y)) do i
        r = render(model, map(c -> @inbounds(c[i]), coords)...) - @inbounds(y[i])
        err === nothing ? abs2(r) : abs2(r / @inbounds(err[i]))
    end
end

function check_data(x, y, err)
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

struct ObjectiveFunction{CM, C, Y, E, S}
    cm::CM
    coords::C
    y::Y
    err::E
    lower::Vector{Float64}
    upper::Vector{Float64}
    names::Vector{Symbol}
    statistic::S
end

function ObjectiveFunction(cm::CompiledModel, x, y, err = nothing; statistic = :chi2)
    check_data(x, y, err)
    lower, upper = bounds(cm)
    return ObjectiveFunction(
        cm,
        _coords(x),
        y,
        err,
        Float64.(lower),
        Float64.(upper),
        paramnames(cm),
        Val(statistic),
    )
end

(f::ObjectiveFunction)(p) = _evaluate(f.statistic, f, p)
(f::ObjectiveFunction)(p, _) = f(p)

_evaluate(::Val{:chi2}, f, p) = chi2(f, p)
_evaluate(::Val{:negloglikelihood}, f, p) = -loglikelihood(f, p)
_evaluate(::Val{:logposterior}, f, p) = logposterior(f, p)
_evaluate(::Val{:neglogposterior}, f, p) = -logposterior(f, p)
_evaluate(::Val{S}, _, _) where {S} = throw(ArgumentError("unknown statistic: $S"))

chi2(f::ObjectiveFunction, p) = chi2(withparams(f.cm, p), f.coords, f.y, f.err)

function loglikelihood(f::ObjectiveFunction, p)
    χ2 = chi2(f, p)
    n = length(f.y)
    return f.err === nothing ?
        -0.5 * χ2 - n / 2 * log(2π) :
        -0.5 * χ2 - sum(log, f.err) - n / 2 * log(2π)
end

function _inside_bounds(f::ObjectiveFunction, p)
    for i in eachindex(p)
        (f.lower[i] <= p[i] <= f.upper[i]) || return false
    end
    return true
end

function logposterior(f::ObjectiveFunction, p)
    _inside_bounds(f, p) || return -Inf
    return _logprior(f.cm, p) + loglikelihood(f, p)
end
