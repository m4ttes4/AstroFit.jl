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



struct ObjectiveFunction{CM, C, Y, U}
    cm::CM 
    coords::C 
    y::Y
    err::U
    lower::Vector{Float64}
    upper::Vector{Float64}
end

function ObjectiveFunction(cm::CompiledModel, x, y, err=nothing)
    check_data(x, y, err)
    lb, ub = bounds(cm)
    ObjectiveFunction(cm, _coords(x), y, err, lb, ub)
end

function (f::ObjectiveFunction)(p)
    model = withparams(f.cm, p)
    return chi2(model, f.coords, f.y, f.err)
end
(f::ObjectiveFunction)(p, _) = f(p) #Optimization.jl convention
