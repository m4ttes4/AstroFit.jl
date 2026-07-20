# Kernel fits require grid-form data

Fitting a model that contains a **Kernel** requires the observed data in grid form — `y` an array whose shape is the grid, with one coordinate axis vector per dimension — instead of the flat point-list form (`length(coord) == length(y)` for every coordinate) that the pointwise path uses. Convolution needs the neighbourhood structure that flattening destroys, so the grid has to live in the data layout rather than be inferred from an unpromised flattening order.

The axis vectors are stored flat and reshaped to broadcast shapes **by the kernel branch itself**, not by the caller: axis `k` is reshaped to extend along dimension `k`, so a 2D render becomes `render.(m, xaxis, permutedims(yaxis))` and yields a matrix. Requiring pre-shaped axes from the caller was rejected as an undocumented trap — two plain equal-length vectors broadcast elementwise and silently return the diagonal instead of the image (verified), and unequal lengths throw `DimensionMismatch`.

`check_data` and `chi2` branch on `_haskernel`; the pointwise path keeps the flat form and its zero-allocation scalar loop untouched. Passing scattered points to a kernel-bearing model fails at construction rather than returning a wrong χ².
