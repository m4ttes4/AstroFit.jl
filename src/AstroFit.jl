module AstroFit

include("constraints.jl")
include("model.jl")
include("zoo/models1d.jl")
include("zoo/models2d.jl")
include("compound.jl")
include("compiled.jl")
include("withparams.jl")
include("params.jl")
include("constrain.jl")
include("macro.jl")
include("zoo/recipes1d.jl")
include("bayes.jl")
include("show.jl")

export AbstractModel
export Gaussian1D, Const1D, Linear1D, Lorentzian1D, Voigt1D
export PowerLaw1D, BlackBody1D, BrokenPowerLaw1D, Exponential1D
export Gaussian2D, Sersic2D, Moffat2D, Beta2D
export AbstractConstraint, Free, Fixed, Bounded, Tied
export CompiledModel, withparams, params, nfree, bounds, paramnames
export setconstraint, validate
export loglikelihood, logprior, logposterior, objective, setprior, PosteriorTarget
export @model, @fix, @bound, @free, @tie, @prior, @constrain
export render, render!
export emission_line, absorption_line, doublet, powerlaw_continuum, blackbody_continuum

end
