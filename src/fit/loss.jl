
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
    ObjectiveFunction(cm::CompiledModel, x, y, [err]; statistic=chi2)

Callable objective wrapping a [`CompiledModel`](@ref), data, and optional errors.

Calling `f(p)` calls `statistic(f, p)`. Supports `f(p, _)` for the two-argument
convention used by Optimization.jl.

# Arguments
- `cm::CompiledModel`: the compiled model to evaluate
- `x`: coordinate data (single vector for 1D, tuple of vectors for multi-D)
- `y`: observed data values
- `err`: optional per-point errors (standard deviations)

# Keywords
- `statistic`: callable with signature `(f::ObjectiveFunction, p) -> Float64`,
  called as `f(p)`. Default [`chi2`](@ref). [`loglikelihood`](@ref),
  [`logposterior`](@ref), [`negloglikelihood`](@ref), [`neglogposterior`](@ref)
  already have this shape and can be passed directly; any user function or
  closure with the same signature works too.

# Examples
```julia
f = ObjectiveFunction(cm, x, y, err)
f(p)                     # χ² at parameter vector p
f = ObjectiveFunction(cm, x, y, err; statistic=neglogposterior)
f(p)                     # -log posterior at p

# custom likelihood, e.g. Poisson counts
poisson_ll(f, p) = begin
    m = withparams(f.cm, p)
    sum(i -> logpdf(Poisson(render(m, f.coords[1][i])), f.y[i]), eachindex(f.y))
end
f = ObjectiveFunction(cm, x, y; statistic=poisson_ll)
f(p)                     # poisson_ll(f, p)
```

See also: [`chi2`](@ref), [`loglikelihood`](@ref), [`logposterior`](@ref)
"""
struct ObjectiveFunction{CM, C, Y, E, S, PR}
    cm::CM
    coords::C
    y::Y
    err::E
    lower::Vector{Float64}
    upper::Vector{Float64}
    names::Vector{Symbol}
    statistic::S
    ndim::Int
    _loglike_const::Float64
    priors::PR
end

function ObjectiveFunction(cm::CompiledModel, x, y, err = nothing; statistic = chi2)
    check_data(x, y, err)
    lower, upper = bounds(cm)
    n = length(y)
    llc = err === nothing ?
        -n / 2 * log(2π) :
        -sum(log, err) - n / 2 * log(2π)
    names = paramnames(cm)
    return ObjectiveFunction(
        cm,
        _coords(x),
        y,
        err,
        Float64.(lower),
        Float64.(upper),
        names,
        statistic,
        nfree(cm),
        llc,
        _resolve_priors(cm, names),
    )
end

(f::ObjectiveFunction)(p) = f.statistic(f, p)
(f::ObjectiveFunction)(p, _) = f(p) # Optimization.jl convention

@inline chi2(f::ObjectiveFunction, p) = chi2(withparams(f.cm, p), f.coords, f.y, f.err)

"""
    loglikelihood(f::ObjectiveFunction, p) -> Float64

Compute the Gaussian log-likelihood at parameter vector `p`: `-0.5 * χ² + const`.

See also: [`chi2`](@ref), [`logposterior`](@ref)
"""
@inline loglikelihood(f::ObjectiveFunction, p) = -0.5 * chi2(f, p) + f._loglike_const

"""
    negloglikelihood(f::ObjectiveFunction, p) -> Float64

`-loglikelihood(f, p)`. Convenience for use as `statistic`.
"""
negloglikelihood(f::ObjectiveFunction, p) = -loglikelihood(f, p)

"""
    logposterior(f::ObjectiveFunction, p) -> Float64

Compute the log-posterior: `logprior(cm, p) + loglikelihood(f, p)`.

`Bounded` parameters are not automatically rejected outside their bounds —
attach an explicit `@prior leaf.field ~ Uniform(lower, upper)` (or a
`Truncated` prior) if you need that enforced as `-Inf`. Requires
`Distributions.jl`.

See also: [`loglikelihood`](@ref), [`logprior`](@ref)
"""
function logposterior(f::ObjectiveFunction, p)
    return logprior(f, p) + loglikelihood(f, p)
end

"""
    neglogposterior(f::ObjectiveFunction, p) -> Float64

`-logposterior(f, p)`. Convenience for use as `statistic`.
"""
neglogposterior(f::ObjectiveFunction, p) = -logposterior(f, p)
