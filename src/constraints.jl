abstract type AbstractConstraint end

# Position in the spec is identity: constraints carry no parameter index.
# withparams (@generated) assigns p-slots by a compile-time walk in tree order.

# A free parameter — gets its own slot in p.
struct Free <: AbstractConstraint end

# A free parameter constrained to [lower, upper].
struct Bounded{T} <: AbstractConstraint
    lower::T
    upper::T
end
function Bounded(lower, upper)
    lo, hi = promote(lower, upper)
    (isnan(lo) || isnan(hi)) && throw(ArgumentError("Bounded: lower and upper must not be NaN"))
    lo ≥ hi && throw(ArgumentError("Bounded: requires lower < upper, got lower=$lo, upper=$hi"))
    Bounded(lo, hi)
end

# A parameter pinned to a constant — consumes no slot in p.
struct Fixed{T} <: AbstractConstraint
    value::T
end

# value = f(master₁ … masterₙ). Masters are referenced by path and must be free.
# Paths live in the type so the @generated withparams can read them (decision 8).
struct Tied{Paths, F} <: AbstractConstraint
    f::F
end
Tied(paths::Tuple, f::F) where {F} = Tied{paths, F}(f)
