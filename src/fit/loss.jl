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

# Generic (multi-D) — peel first iteration for type stability with ForwardDiff Duals
function chi2(model, coords, y, ::Nothing)
    fi = firstindex(y)
    acc = @inbounds abs2(render(model, map(c -> c[fi], coords)...) - y[fi])
    @inbounds for i in (fi+1):lastindex(y)
        acc += abs2(render(model, map(c -> c[i], coords)...) - y[i])
    end
    return acc
end

function chi2(model, coords, y, err)
    fi = firstindex(y)
    r = @inbounds render(model, map(c -> c[fi], coords)...) - y[fi]
    acc = abs2(r / @inbounds err[fi])
    @inbounds for i in (fi+1):lastindex(y)
        r = render(model, map(c -> c[i], coords)...) - y[i]
        acc += abs2(r / err[i])
    end
    return acc
end

# 1D fast path — direct indexing, no map/splat
function chi2(model, coords::Tuple{AbstractVector}, y, ::Nothing)
    x = coords[1]
    fi = firstindex(y)
    acc = @inbounds abs2(render(model, x[fi]) - y[fi])
    @inbounds for i in (fi+1):lastindex(y)
        acc += abs2(render(model, x[i]) - y[i])
    end
    return acc
end

function chi2(model, coords::Tuple{AbstractVector}, y, err)
    x = coords[1]
    fi = firstindex(y)
    r = @inbounds render(model, x[fi]) - y[fi]
    acc = abs2(r / @inbounds err[fi])
    @inbounds for i in (fi+1):lastindex(y)
        r = render(model, x[i]) - y[i]
        acc += abs2(r / err[i])
    end
    return acc
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
    _loglike_const::Float64
end

function ObjectiveFunction(cm::CompiledModel, x, y, err = nothing; statistic = :chi2)
    check_data(x, y, err)
    lower, upper = bounds(cm)
    n = length(y)
    llc = err === nothing ?
        -n / 2 * log(2π) :
        -sum(log, err) - n / 2 * log(2π)
    return ObjectiveFunction(
        cm,
        _coords(x),
        y,
        err,
        Float64.(lower),
        Float64.(upper),
        paramnames(cm),
        Val(statistic),
        llc,
    )
end

(f::ObjectiveFunction)(p) = _evaluate(f.statistic, f, p)
(f::ObjectiveFunction)(p, _) = f(p) # Optimization.jl convention

_evaluate(::Val{:chi2}, f, p) = chi2(f, p)
_evaluate(::Val{:negloglikelihood}, f, p) = -loglikelihood(f, p)
_evaluate(::Val{:logposterior}, f, p) = logposterior(f, p)
_evaluate(::Val{:neglogposterior}, f, p) = -logposterior(f, p)
_evaluate(::Val{S}, _, _) where {S} = throw(ArgumentError("unknown statistic: $S"))

@inline chi2(f::ObjectiveFunction, p) = chi2(withparams(f.cm, p), f.coords, f.y, f.err)

loglikelihood(f::ObjectiveFunction, p) = -0.5 * chi2(f, p) + f._loglike_const

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
