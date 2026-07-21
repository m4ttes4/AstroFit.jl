abstract type AbstractModel end

Base.broadcastable(m::AbstractModel) = (m,)
