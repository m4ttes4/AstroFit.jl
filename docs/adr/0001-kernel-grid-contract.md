# Kernel grid contract: index-as-grid, no physical spacing

A **Kernel** receives only intensities (`render(k, xs::AbstractArray)`) and treats the array index as its grid; kernel widths are expressed in samples, not in arcsec/Å. Threading a physical spacing `dx` through `Leaf`, `Pipe` and `chi2` was rejected as speculative: the grids used in practice are uniform, so index space and physical space differ only by a constant the user applies once when constructing the kernel.

The **Array Render** of a Kernel is size-preserving: the output has the same shape as the input. This is what lets `chi2` compare `μ` against `y` without runtime shape checks. Edge handling (clamping, zero-padding, ...) is left to each individual Kernel, not fixed by the framework.

**Precondition**: on a non-uniform grid (e.g. log-linear wavelength) an index-space kernel is not a physically constant-width convolution. This is documented, not enforced.
