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
Bounded(lower, upper) = Bounded(promote(lower, upper)...)

# A parameter pinned to a constant — consumes no slot in p.
struct Fixed{T} <: AbstractConstraint
    value::T
end

# value = f(master₁ … masterₙ). Masters are referenced by path and must be free.
# Paths live in the type so the @generated withparams can read them (decision 8).
struct Tied{Paths,F} <: AbstractConstraint
    f::F
end
Tied(paths::Tuple, f::F) where {F} = Tied{paths,F}(f)
