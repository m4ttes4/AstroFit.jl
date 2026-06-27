using BenchmarkTools, AstroFit

const SUITE = BenchmarkGroup()

const X1     = collect(0.0:0.01:12.0)
const XS_128  = collect(range(0.0, 12.0, length = 128))
const XS_1024 = collect(range(0.0, 12.0, length = 1_024))
const XS_8192 = collect(range(0.0, 12.0, length = 8_192))

include("render.jl")
include("withparams.jl")
include("grid.jl")
