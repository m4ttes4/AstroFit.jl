"""
    ObjectiveFunction(cm::CompiledModel, points, y, [err]; statistic=chi2)

Callable objective wrapping a [`CompiledModel`](@ref), data, and optional errors.

Calling `f(p)` calls `statistic(f, p)`. Supports `f(p, _)` for the two-argument
convention used by Optimization.jl.

The inputs are validated once, here: `points` follows [`render`](@ref)'s rules (a
vector for 1D, `CartesianIndices(img)`, `Coords(x, y, …)`, any array of points),
`axes(points) == axes(y)`, and `err` (if given) matches `y` and is positive. The
model must produce one value per point. A masked fit is a mask on the points:
`ObjectiveFunction(cm, CartesianIndices(img)[mask], img[mask])`.

# Arguments
- `cm::CompiledModel`: the compiled model to evaluate
- `points`: the array of points the data were taken at
- `y`: observed data values, one per point
- `err`: optional per-point errors (standard deviations)

# Keywords
- `statistic`: callable with signature `(f::ObjectiveFunction, p) -> Float64`,
  called as `f(p)`. Default [`chi2`](@ref). [`loglikelihood`](@ref),
  [`logposterior`](@ref), [`negloglikelihood`](@ref), [`neglogposterior`](@ref)
  already have this shape and can be passed directly; any user function or
  closure with the same signature works too.

# Examples
```julia
f = ObjectiveFunction(cm, λ, flux, err)
f(p)                     # χ² at parameter vector p
f = ObjectiveFunction(cm, λ, flux, err; statistic=neglogposterior)
f(p)                     # -log posterior at p

# custom likelihood, e.g. Poisson counts
poisson_ll(f, p) = begin
    m = withparams(f.cm, p)
    sum(i -> logpdf(Poisson(render(m, f.points[i])), f.y[i]), eachindex(f.y))
end
f = ObjectiveFunction(cm, t, counts; statistic=poisson_ll)
f(p)                     # poisson_ll(f, p)
```

See also: [`chi2`](@ref), [`loglikelihood`](@ref), [`logposterior`](@ref)
"""
struct ObjectiveFunction{CM, P, Y, E, S, PR}
    cm::CM
    points::P
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

function ObjectiveFunction(
        cm::CompiledModel{<:AbstractModel{<:Any, 1}}, points::AbstractArray, y::AbstractArray, err = nothing;
        statistic = chi2
    )
    isempty(y) && throw(ArgumentError("ObjectiveFunction needs at least one data point"))
    axes(points) == axes(y) ||
        throw(DimensionMismatch("point axes $(axes(points)) do not match data axes $(axes(y))"))
    # render's rules on a one-point view: same method, same error, constant memory.
    render(cm, view(points, map(a -> first(a):first(a), axes(points))...))
    if err !== nothing
        axes(err) == axes(y) ||
            throw(DimensionMismatch("err axes $(axes(err)) do not match data axes $(axes(y))"))
        all(>(0), err) || throw(ArgumentError("all `err` values must be positive"))
    end
    lower, upper = bounds(cm)
    n = length(y)
    llc = err === nothing ?
        -n / 2 * log(2π) :
        -sum(log, err) - n / 2 * log(2π)
    names = paramnames(cm)
    return ObjectiveFunction(
        cm,
        points,
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
ObjectiveFunction(cm::CompiledModel{<:AbstractModel{<:Any, 1}}, points, y, err = nothing; kwargs...) =
    throw(ArgumentError("ObjectiveFunction takes an array of points and an array of data, got $(typeof(points)) and $(typeof(y))"))
ObjectiveFunction(cm::CompiledModel{<:AbstractModel{I, O}}, points, y, err = nothing; kwargs...) where {I, O} =
    throw(ArgumentError("a fit needs one value per point; this model produces $O"))

(f::ObjectiveFunction)(p) = f.statistic(f, p)
(f::ObjectiveFunction)(p, _) = f(p) # Optimization.jl convention

"""
    chi2(f::ObjectiveFunction, p)

The χ² statistic at parameter vector `p`: the sum of squared residuals, weighted by
`1/err²` when the objective has errors. The prediction is never materialized.

See also: [`loglikelihood`](@ref), [`ObjectiveFunction`](@ref)
"""
function chi2(f::ObjectiveFunction, p)
    m, A, y, err = withparams(f.cm, p), f.points, f.y, f.err
    Is = CartesianIndices(axes(y))
    I1 = first(Is)
    r1 = render(m, A[I1]) - y[I1]
    # The accumulator takes the first term's type (a Dual under ForwardDiff), so it
    # stays concrete through the loop; this is why empty data is rejected.
    s = zero(abs2(err === nothing ? r1 : r1 / err[I1]))
    # @inbounds is sound: the constructor checked the axes and the fields are immutable.
    @inbounds for I in Is
        r = render(m, A[I]) - y[I]
        s += abs2(err === nothing ? r : r / err[I])
    end
    return s
end

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

Compute the log-posterior: `logprior(f, p) + loglikelihood(f, p)`.

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
