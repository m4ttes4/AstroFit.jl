struct Free end

struct Fixed{T}
    value::T
end
Fixed() = Fixed(nothing)

struct Bounded{T}
    lower::T
    upper::T
end
Bounded(lower, upper) = Bounded(promote(lower, upper)...)

struct Tied{F, Ms<:Tuple}
    f::F
    masters::Ms
end
