"""
    AbstractKernel{I, O} <: AbstractModel{I, O}

A model whose fields default to [`Fixed`](@ref): a kernel is normally a known
calibration input, so it adds no optimizer slot unless a field is freed with
[`@free`](@ref) ([ADR-0004](docs/adr/0004-kernel-fields-fixed-by-default.md)).
Otherwise it is an ordinary scalar model:

```julia
struct Saturate{L} <: AbstractKernel{1, 1}
    level::L
end
AstroFit.evaluate(k::Saturate, f::Number) = min(f, k.level)
```

Operations on whole arrays (convolution, rebinning) are not supported yet.
"""
abstract type AbstractKernel{I, O} <: AbstractModel{I, O} end
