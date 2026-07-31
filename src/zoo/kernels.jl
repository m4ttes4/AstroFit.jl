"""
    GaussianPSF(; sigma)

Gaussian point-spread function, applied by direct convolution over the sample
grid. `sigma` is in **samples**, not physical units — divide the physical width
by the grid step when constructing it (see
[ADR-0001](docs/adr/0001-kernel-grid-contract.md)).

The sampled weights are normalized to sum to 1, and the normalization is
recomputed per output point so truncation at the array edges conserves flux
instead of darkening the borders. Both divisions run inside the derivative path,
so a `@free` `sigma` differentiates correctly under ForwardDiff.

`ponytail:` direct convolution, O(N·σ) — an FFT would be faster for very wide
kernels but breaks ForwardDiff; swap only if a profile demands it.

The support is truncated at 4σ, so the sampled half-width steps by one whenever
`4σ` crosses an integer and the output is discontinuous there. The jump is the
weight entering at the edge — order `exp(-8)` before renormalization, measured at
~2e-9 relative — so it is invisible to a fit, but a finite-difference check
straddling such a σ (e.g. σ = 1.5) will disagree with the analytic derivative.
The derivative is the correct one.

# Examples
```julia
cm = @model begin
    line = Gaussian1D(amplitude = 1.0, mean = 0.0, sigma = 1.0)
    psf  = GaussianPSF(sigma = 2.0)
    line |> psf
end
render(cm, x)             # convolved profile
@free cm.psf.sigma        # opt in to fitting the PSF width
```

See also: [`AbstractKernel`](@ref)
"""
Base.@kwdef struct GaussianPSF{T <: Real} <: AbstractKernel
    sigma::T
end

function render(k::GaussianPSF, ys::AbstractVector)
    σ = k.sigma
    σ > 0 || throw(ArgumentError("GaussianPSF: sigma must be positive, got $σ"))
    # Truncate at 4σ, at least one neighbour. The half-width is structural, so it
    # must not carry a derivative — ForwardDiff's own `ceil(::Type, ::Dual)`
    # drops it, which is what keeps a free `sigma` differentiable here.
    h = max(1, ceil(Int, 4σ))
    w = [exp(-abs2(d / σ) / 2) for d in (-h):h]

    out = similar(ys, promote_type(eltype(ys), eltype(w)))
    n = length(ys)
    lo = firstindex(ys)
    @inbounds for i in eachindex(ys)
        acc = zero(eltype(out))
        wsum = zero(eltype(w))
        for (j, d) in enumerate((-h):h)
            m = i + d
            (lo <= m < lo + n) || continue      # clamp: renormalize over what fits
            acc += w[j] * ys[m]
            wsum += w[j]
        end
        out[i] = acc / wsum
    end
    return out
end
