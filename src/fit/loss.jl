function _missing_distributions()
    throw(
        ArgumentError(
            "Bayesian prior evaluation requires Distributions.jl. " *
                "Load it with `using Distributions` before calling logprior/logposterior."
        )
    )
end

"""
    setprior(cm::CompiledModel, leaf::Symbol, field::Symbol, dist) -> CompiledModel

Attach a prior distribution `dist` to parameter `leaf.field`. Requires `Distributions.jl`.

See also: [`logprior`](@ref), [`@prior`](@ref)
"""
setprior(args...) = _missing_distributions()

"""
    logprior(cm::CompiledModel, p) -> Float64

Evaluate the sum of log-prior densities at parameter vector `p`. Requires `Distributions.jl`.

See also: [`setprior`](@ref), [`logposterior`](@ref)
"""
logprior(args...) = _missing_distributions()

function _logprior(cm::CompiledModel, p)
    priors = getfield(cm, :priors)
    (priors === nothing || isempty(priors)) && return 0.0
    logprior(cm, p)
end

_coords(x::Tuple) = x
_coords(x) = (x,)

"""
    chi2(model, coords, y, err)
    chi2(f::ObjectiveFunction, p)

Compute the χ² statistic (sum of squared residuals, optionally weighted by `1/err²`).

The two-argument form substitutes parameters `p` into the objective's compiled model
before evaluating.

See also: [`loglikelihood`](@ref), [`ObjectiveFunction`](@ref)
"""
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

"""
    check_data(x, y, err)

Validate that coordinate, data, and error arrays have consistent lengths and that
all `err` values (if provided) are positive. Throws `ArgumentError` on failure.
"""
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

"""
    ObjectiveFunction(cm::CompiledModel, x, y, [err]; statistic=:chi2)

Callable objective wrapping a [`CompiledModel`](@ref), data, and optional errors.

Calling `f(p)` evaluates the chosen statistic at parameter vector `p`.
Supports `f(p, _)` for the two-argument convention used by Optimization.jl.

# Arguments
- `cm::CompiledModel`: the compiled model to evaluate
- `x`: coordinate data (single vector for 1D, tuple of vectors for multi-D)
- `y`: observed data values
- `err`: optional per-point errors (standard deviations)

# Keywords
- `statistic::Symbol=:chi2`: objective to compute. One of `:chi2`,
  `:negloglikelihood`, `:logposterior`, `:neglogposterior`

# Examples
```julia
f = ObjectiveFunction(cm, x, y, err)
f(p)                     # χ² at parameter vector p
f = ObjectiveFunction(cm, x, y; statistic=:neglogposterior)
f(p)                     # -log posterior at p
```

See also: [`chi2`](@ref), [`loglikelihood`](@ref), [`logposterior`](@ref)
"""
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

"""
    loglikelihood(f::ObjectiveFunction, p) -> Float64

Compute the Gaussian log-likelihood at parameter vector `p`: `-0.5 * χ² + const`.

See also: [`chi2`](@ref), [`logposterior`](@ref)
"""
loglikelihood(f::ObjectiveFunction, p) = -0.5 * chi2(f, p) + f._loglike_const

function _inside_bounds(f::ObjectiveFunction, p)
    for i in eachindex(p)
        (f.lower[i] <= p[i] <= f.upper[i]) || return false
    end
    return true
end

"""
    logposterior(f::ObjectiveFunction, p) -> Float64

Compute the log-posterior: `logprior(cm, p) + loglikelihood(f, p)`.

Returns `-Inf` if `p` is outside the parameter bounds.

See also: [`loglikelihood`](@ref), [`logprior`](@ref)
"""
function logposterior(f::ObjectiveFunction, p)
    _inside_bounds(f, p) || return -Inf
    return _logprior(f.cm, p) + loglikelihood(f, p)
end
