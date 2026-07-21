_coords(x::Tuple) = x
_coords(x) = (x,)


# NOTE rewrite chi2 in a more idiomatic way
"""
    chi2(model, coords, y, err)
    chi2(f::ObjectiveFunction, p)

Compute the χ² statistic (sum of squared residuals, optionally weighted by `1/err²`).

The two-argument form substitutes parameters `p` into the objective's compiled model
before evaluating.

Pointwise models take a scalar loop that never materializes the prediction;
models containing a kernel ([`AbstractKernel`](@ref)) cannot be evaluated one
point at a time, so they render once over the whole coordinate array. The split
is by [`evalstyle`](@ref) and folds at compile time.

See also: [`loglikelihood`](@ref), [`ObjectiveFunction`](@ref)
"""
chi2(model, coords, y, err) = _chi2(evalstyle(model), model, coords, y, err)

@inline _chi2(::Pointwise, model, coords, y, err) = _chi2p(model, coords, y, err)

# Domainwise: `_eval`, not `render` — the tree materializes an array only where a
# kernel forces one, and the compound nodes above it stay a lazy `Broadcasted`.
# Indexing that in the residual sum means the prediction is never assembled into
# an array of its own, so an objective call allocates the kernel's working arrays
# and nothing else.
function _chi2(::Domainwise, model, coords, y, ::Nothing)
    μ = _eval(model, coords...)
    _checkpred(μ, y)
    return sum(i -> abs2(μ[i] - y[i]), eachindex(y))
end

function _chi2(::Domainwise, model, coords, y, err)
    μ = _eval(model, coords...)
    _checkpred(μ, y)
    return sum(i -> abs2((μ[i] - y[i]) / err[i]), eachindex(y))
end

# A kernel that is not size-preserving would otherwise silently mis-align the
# residual (or throw far from the cause).
function _checkpred(μ, y)
    size(μ) == size(y) || throw(
        DimensionMismatch(
            "model prediction has size $(size(μ)) but data has size $(size(y)) — " *
                "a kernel must return an array the same size as its input"
        )
    )
    return nothing
end

# `_eval` (render.jl) is the same lazy `Broadcasted` that `render`/`render!` build,
# reused here instead of indexing one coordinate array at a time: a single loop
# then serves both coordinate forms — a flat list of co-shaped points, and the
# grid form (a column `x` against a row `y`) that a kernel needs. Nothing is
# materialized: `μ[i]` evaluates `render` at that point.
function _chi2p(model, coords, y, ::Nothing)
    μ = _eval(model, coords...)
    return sum(i -> abs2(μ[i] - y[i]), eachindex(y))
end

function _chi2p(model, coords, y, err)
    μ = _eval(model, coords...)
    return sum(i -> abs2((μ[i] - y[i]) / err[i]), eachindex(y))
end

# 1D fast path — direct indexing, no map/splat
function _chi2p(model, coords::Tuple{AbstractVector}, y, ::Nothing)
    x = coords[1]
    fi = firstindex(y)
    acc = @inbounds abs2(render(model, x[fi]) - y[fi])
    @inbounds for i in (fi + 1):lastindex(y)
        acc += abs2(render(model, x[i]) - y[i])
    end
    return acc
end

function _chi2p(model, coords::Tuple{AbstractVector}, y, err)
    x = coords[1]
    fi = firstindex(y)
    r = @inbounds render(model, x[fi]) - y[fi]
    acc = abs2(r / @inbounds err[fi])
    @inbounds for i in (fi + 1):lastindex(y)
        r = render(model, x[i]) - y[i]
        acc += abs2(r / err[i])
    end
    return acc
end

# One rule covers both coordinate forms: the coordinates must broadcast to
# exactly the shape of `y`. That admits the flat point list (co-shaped arrays)
# and the grid form (one axis per dimension — a column `x` against a row `y`),
# and it rejects two plain vectors against an image, which would otherwise render
# the diagonal instead of the image.
function _checkcoords(coords, y)
    shape = try
        Base.Broadcast.broadcast_shape(map(axes, coords)...)
    catch
        nothing                       # incompatible axes: report it as our own error
    end
    shape == axes(y) || throw(
        ArgumentError(
            "coordinates must broadcast to the shape of `y`: got sizes $(map(size, coords)) against $(size(y))"
        )
    )
    return nothing
end

"""
    check_data(x, y, err)

Validate that the coordinates broadcast to the shape of `y`, that `err` (if
provided) matches `y` in length, and that all `err` values are positive. Throws
`ArgumentError` on failure.

Coordinates may be given either as a flat point list (every coordinate array
co-shaped with `y`) or in grid form (one axis per dimension, shaped so they
broadcast — a column `x` against a row `y`). The grid form is what a model
containing an [`AbstractKernel`](@ref) needs, since a kernel is handed the whole
image.
"""
function check_data(x, y, err)
    _checkcoords(_coords(x), y)
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
# is it necessary to save the logconstant?
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
