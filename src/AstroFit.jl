module AstroFit

include("constraints.jl")
include("model.jl")
include("compound.jl")
include("compiled.jl")
include("withparams.jl")
include("params.jl")
include("constrain.jl")
include("macro.jl")
include("bayes.jl")
include("show.jl")

export AbstractModel, Gaussian1D, Const1D, Linear1D
export AbstractConstraint, Free, Fixed, Bounded, Tied
export CompiledModel, withparams, params, nfree, bounds, paramnames
export setconstraint, validate
export logprior, logposterior, objective, setprior
export @model, @fix, @bound, @free, @tie, @prior, @constrain
export render

end
