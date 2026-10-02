abstract type AbstractModel{I, O} end

# A model broadcasts as a scalar. `Ref`, not a 1-tuple: a tuple would broadcast a
# 0-dimensional input to a 1-element vector.
Base.broadcastable(m::AbstractModel) = Ref(m)
