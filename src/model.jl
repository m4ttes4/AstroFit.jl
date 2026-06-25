abstract type AbstractModel end

Base.broadcastable(m::AbstractModel) = (m,)

render(m::AbstractModel, xs::AbstractArray...) = render.(m, xs...)

function render!(out::AbstractArray, m::AbstractModel, xs...)
    out .= render.(m, xs...)
    return out
end
