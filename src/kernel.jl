"""
    AbstractKernel <: AbstractModel

A model whose value at one point depends on neighbouring values, so it cannot be
written as a scalar `render(k, x::Number)`. A PSF convolution is the motivating
case.

Implement one like any other model, but define the *array* render instead of the
scalar one:

```julia
struct BoxKernel <: AbstractKernel
    width::Int
end

AstroFit.render(k::BoxKernel, ys::AbstractVector) = ...   # same length as ys
```

Contract:

- **Intensities in, intensities out.** A kernel's array argument is the values
  produced upstream, not coordinates — array index is the grid
  ([ADR-0001](docs/adr/0001-kernel-grid-contract.md)), so widths are in samples
  and the grid is assumed uniform.
- **Axes-preserving.** `axes(render(k, ys)) == axes(ys)`. Framework evaluation
  checks this immediately after each kernel, before broadcasting can hide a
  mismatch. A direct call to a user-defined `render(k, ys)` invokes that method
  itself, so its author remains responsible for this contract. Edge handling is
  the kernel's own choice.
- **Whatever shape the upstream model produced.** A 2D kernel is handed an image,
  which means the model has to be rendered over *grid-form* coordinates: one axis
  per dimension, shaped so they broadcast — a column against a row,
  `render(m, x, reshape(y, 1, :))`. Two plain vectors broadcast to the diagonal,
  not the image; `check_data` rejects that at construction
  ([ADR-0006](docs/adr/0006-grid-form-coordinates.md)).
- Kernel fields default to `Fixed` (a kernel is normally a known calibration
  input); use `@free` to fit one — see
  [ADR-0004](docs/adr/0004-kernel-fields-fixed-by-default.md).

Composition follows ordinary model semantics: `model |> psf` convolves, and the
result composes further — `(model |> psf) + continuum`, `(model |> psf) * t`,
`model |> psf1 |> psf2`.
"""
abstract type AbstractKernel{I, O} <: AbstractModel{I, O} end
