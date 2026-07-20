# Kernel composition: selective array-native nodes, guarded by `_haskernel`

Compound nodes get an **Array Render** (`render(node, xs::AbstractArray...)`) that recurses structurally, but only behind a `_haskernel` guard: a tree with no **Kernel** takes the existing fused scalar broadcast, unchanged. This was chosen over making every node array-native unconditionally, which would replace the single-pass broadcast with per-node intermediate arrays and regress the ≤1.0x-vs-handwritten benchmark. Because `_haskernel` dispatches on a concrete value's type it folds to a compile-time constant, so kernel-free models pay nothing.

Restricting the override to `Pipe` alone was rejected: it makes `(source |> psf) + background` — a convolved component summed with an unconvolved one — a `MethodError`, and that is a real modelling pattern.

Coordinate arity stays variadic (`xs...`) throughout, since a **Kernel** must work on 2D images; the **Kernel** itself still takes exactly one intensity array, per [ADR-0001](0001-kernel-grid-contract.md).
