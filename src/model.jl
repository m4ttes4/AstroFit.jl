abstract type AbstractModel{I, O} end

Base.broadcastable(m::AbstractModel) = (m,)
