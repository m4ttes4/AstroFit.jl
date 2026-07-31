# Grid-form coordinates belong to the caller, validated by one broadcast rule

A multi-dimensional model takes its coordinates in one of two forms, and both are
accepted everywhere — `render`, `render!`, `check_data`, `chi2`:

- **flat point list** — every coordinate array co-shaped with `y`, the form that
  scattered points and the existing pointwise path already use;
- **grid form** — one axis per dimension, shaped so they broadcast: a column `x`
  against a row `y`, `render(m, x, reshape(y, 1, :))`.

The rule that admits both is a single line: *the coordinates must broadcast to
exactly the shape of `y`*. `check_data` enforces it, and `_chi2p` reads the
coordinates through a lazy `Broadcasted` so one loop serves both forms without
materializing anything.

This replaces the mechanism in [ADR-0003](0003-grid-form-data-for-kernel-fits.md),
which had the framework reshape flat axis vectors inside the kernel branch. That
was never implemented, and it cannot cover the case that motivated it: `render(m,
x, y)` is a public entry point with no `y`-data to infer a grid from, so the
reshape would have to be repeated in `render` and `chi2` independently — and the
trap would stay open in `render`, which is exactly where a 2D **Kernel** is handed
a vector instead of an image.

Two plain equal-length vectors against an image are therefore *rejected*, not
repaired: they broadcast to the diagonal, and `check_data` throws at construction
instead of fitting the wrong thing. Shaping an axis is zero-copy (`reshape` and
`'` both share memory), so the cost of moving the responsibility to the caller is
one call, paid once.

**Consequence for the zoo.** A linear `eachindex(out, xs, ys)` loop cannot span a
column against a row, so the 2D `render!` methods broadcast a shared `_rot2d`
helper with the trig and the divisions hoisted out of it. Measured at 256×256
against the linear loop they replace: Gaussian2D 275.8 → 301.5 µs (1.09x),
Sersic2D 1715.5 → 1711.3 µs (1.00x), both still zero-allocation, and the grid
form is the faster of the two coordinate layouts (283.4 µs for Gaussian2D). The
constants are passed as broadcast arguments rather than captured in a closure;
the closure form measured 1.30x.

**What the grid form buys.** Coordinates stop scaling with the image: a 256×256
fit carries 4 KB of axes instead of 1 MB of coordinate matrices, and the
100×100 example in the README drops from 160 KB to 1.6 KB.

## A third form: no coordinates at all

`render(m, image)` and `render!(out, m)` take a matrix and render over its **index
space** — the coordinates are the array's own indices. This is the same
index-as-grid convention [ADR-0001](0001-kernel-grid-contract.md) already fixed
for a **Kernel**, extended to the model that feeds it, so the two halves of
`model |> psf` finally measure their grid the same way.

The consequence is stated, not hidden: **model parameters are then in pixels.** A
`Gaussian2D(x0 = -3.5)` over a `range(-8, 8; length = 100)` axis becomes
`x0 = 30.6` in index space. Callers who want physical units keep using the grid
form above; the two live side by side.

Three constraints shaped the implementation:

- **It is sugar, not a second engine.** `render(m, image)` forwards to
  `render(m, axes(image, 1), reshape(axes(image, 2), 1, :))`. Reshaping a range is
  lazy, so the index grid costs no coordinate memory and inherits the hoisted,
  zero-allocation broadcast the grid form already had. `render` allocates exactly
  its output array; `render!(out, m)` allocates nothing.
- **`AbstractMatrix` only, never `AbstractArray`.** A lone *vector* has to keep
  meaning "coordinate values", or every 1D render would change meaning. The
  index-grid reading is 2D-only by dispatch.
- **Routed through `evalstyle`, not around it.** A bare **Kernel** handed a matrix
  still reads it as intensities, and a kernel-bearing tree recurses until a
  pointwise leaf turns the template into the image the kernel convolves. A
  vector-only kernel handed a matrix still reports what it needs instead of
  falling into the new path.

`render!(out, m)` takes no template: `out` supplies the grid *and* receives the
values. A `render!(out, m, image)` parallel to `render(m, image)` was rejected
because its two matrix arguments could only ever disagree.
`render!(out, cm, image)` gets no special case: the matrix takes the ordinary
route — a template on the pointwise branch, intensities on the domainwise one,
exactly as for a bare model — so it agrees with `render!(out, cm)` when the two
grids agree and throws `DimensionMismatch` when they do not. A deprecation shim
on the `CompiledModel` entry point was rejected: it would have made
`CompiledModel` the one type where a lone matrix means something different, and
dispatch cannot tell a template from a matrix of coordinate values anyway.

## One evaluator behind both entry points

`render` and `render!` share a single internal function:

```julia
_eval(::Pointwise,  m, xs...) = instantiate(broadcasted(render, (m,), xs...))
_eval(::Domainwise, m, xs...) = _arender(m, xs...)

render(m, xs...)       = materialize(_eval(m, xs...))
render!(out, m, xs...) = (out .= _eval(m, xs...); out)
```

The pointwise branch returns an *un-materialized* `Broadcasted`, so `render`
materializes it and `render!` writes it into a buffer — the same fused traversal
either way. The domainwise branch stays lazy too: `_arender` composes its children
with `broadcasted` rather than `.+`, so an array appears only where a kernel
forces one and the compound nodes above it are still a `Broadcasted`. On
`(g |> psf) + c` over 512 points that halves an objective call's allocation —
16832 B in 14 allocations down to 8512 B in 8 — because neither the pointwise
sibling nor the sum itself is assembled any more. Per-call time barely moves
(7.84 → 7.63 µs median); the gain is memory traffic over a fit loop, where 20k
evaluations went from 322.6 to 163.9 MiB and GC from 7.5 to 3.6 ms.
This replaced four
parallel helpers (`_render`, `_render!`, `_imrender`, and `chi2`'s own
`_lazyrender`), each of which had re-implemented the same style split; `chi2` now
indexes the very same object.

Measured before and after the consolidation, unchanged to the microsecond:
Gaussian2D `render!` 301.2 → 301.4 µs, Sersic2D 1709.5 → 1711.1 µs, 2D χ² 1039.9
→ 1039.8 µs, 1D χ² over 5000 points 19.3 µs, every one of them at zero
allocations. Against the handwritten baseline: render 0.80x, χ² 0.92x.

**The single-matrix rule belongs inside `_eval`, not at the entry point.** A first
attempt rewrote `render(m, image)` to axes unconditionally, on the reasoning that
a kernel's own `render(k, ::AbstractMatrix)` is more specific and would win first.
It is not reachable: in a compiled tree the kernel sits behind a `Leaf`, so the
call lands on the generic method and the template rewrite ate the intensities the
kernel was supposed to convolve. Placing the rewrite on the `Pointwise` branch of
`_eval` keeps `Leaf`-wrapped kernels on the intensity path.
