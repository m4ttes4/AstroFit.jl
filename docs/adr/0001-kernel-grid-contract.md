# Kernel grid contract: index-as-grid, no physical spacing

A **Kernel** receives only intensities (`render(k, xs::AbstractArray)`) and treats the array index as its grid; kernel widths are expressed in samples, not in arcsec/Å. Threading a physical spacing `dx` through `Leaf`, `Pipe` and `chi2` was rejected as speculative: the grids used in practice are uniform, so index space and physical space differ only by a constant the user applies once when constructing the kernel.

The **Array Render** of a Kernel preserves its input axes. Framework evaluation checks this at each kernel, before a downstream broadcast can conceal an invalid singleton output. `chi2` separately checks the final prediction against the data. Direct calls to user-defined array methods remain the author's responsibility. Edge handling (clamping, zero-padding, ...) is left to each individual Kernel, not fixed by the framework. See [ADR-0007](0007-rendering-protocol.md).

**Precondition**: on a non-uniform grid (e.g. log-linear wavelength) an index-space kernel is not a physically constant-width convolution. This is documented, not enforced.
