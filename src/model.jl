abstract type AbstractModel end

Base.broadcastable(m::AbstractModel) = Ref(m)

render(m::AbstractModel, xs::AbstractArray...) = render.(m, xs...)

function render!(out::AbstractArray, m::AbstractModel, xs...)
    out .= render.(Ref(m), xs...)
    return out
end
