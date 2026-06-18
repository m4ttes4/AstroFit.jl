module AstroFit

using Accessors
using Accessors: PropertyLens
import Accessors: set

include("constraints.jl")
include("model.jl")
include("compound.jl")
include("params.jl")
include("macro.jl")
include("zoo/spectral1d.jl")
include("zoo/galaxy2d.jl")
include("bayes.jl")
include("show.jl")

export AbstractModel, Gaussian1D, Const1D, Linear1D, Gaussian2D, ExponentialDisk2D
export Sum, Difference, Product, Quotient, Pipe
export Free, Fixed, Bounded, Tied
export render, render!
export CompiledModel, ComponentRef, compile, withparams, resolve
export free_lenses, gather, scatter, bounds_vectors, nfree, freevals, paramvector
export logprior, logposterior, objective
export @model, @constrain
export @fix, @bound, @tie, @free, @prior
export @set                      # re-exported from Accessors
export ConstantBackground1D, LinearContinuum1D
export EmissionLine1D, AbsorptionLine1D, EmissionDoublet1D
export EmissionLineSpectrum1D, AbsorptionLineSpectrum1D
export GalaxyGaussianLineProfile2D, GalaxyExponentialLineProfile2D

end
