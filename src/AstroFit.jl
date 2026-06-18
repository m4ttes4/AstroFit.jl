module AstroFit

include("constraints.jl")
include("model.jl")
include("compound.jl")
include("compiled.jl")

export AbstractModel, Gaussian1D, Const1D, Linear1D
export AbstractConstraint, Free, Fixed, Bounded, Tied, resolve
export CompiledModel, withparams
export render

end
