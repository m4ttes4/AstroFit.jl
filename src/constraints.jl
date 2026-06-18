abstract type AbstractConstraint end

struct Free{I} <: AbstractConstraint end

struct Fixed{T} <: AbstractConstraint
    value::T
end
Fixed() = Fixed(nothing)

struct Bounded{I,T} <: AbstractConstraint
    lower::T
    upper::T
end
Bounded{I}(lower::T, upper::T) where {I,T} = Bounded{I,T}(lower, upper)
Bounded{I}(lower, upper) where I = Bounded{I}(promote(lower, upper)...)

struct Tied{Is,F} <: AbstractConstraint
    f::F
end
Tied{Is}(f::F) where {Is,F} = Tied{Is,F}(f)

resolve(::Free{I}, p) where I = p[I]
resolve(c::Fixed, _) = c.value
resolve(::Bounded{I}, p) where I = p[I]
@generated function resolve(c::Tied{Is}, p) where Is
    args = [:(p[$i]) for i in Is]
    :(c.f($(args...)))
end
