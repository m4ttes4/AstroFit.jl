# Compiled constraints with structured spec

The current `CompiledModel` uses a flat tuple of `(optic, constraint)` pairs as its spec. `withparams` scatters free values into the model tree via optics, then resolves ties in a separate pass — both implemented as `@generated` functions that unroll lens-based `set` chains. This works but is more complex than necessary: optics dominate the hot path, `free_lenses`/`tied_entries`/`scatter`/`_resolve` are all separate `@generated` functions, and `Registry`/`ComponentRef` exist mainly to bridge the gap between the flat spec and the named component tree.

We decided to restructure `CompiledModel` around a **NamedTuple spec that mirrors the component tree** and **constraint types that carry their parameter index as a type parameter** and know how to resolve themselves. `withparams` becomes a single `@generated` function that walks the model type and spec in parallel, calls `resolve` on each constraint, and reconstructs the tree bottom-up — no optics, no scatter/resolve split.

This was chosen over incremental cleanup of the current design because the flat-spec-with-optics architecture forces every operation (filtering, scattering, resolving) to be a separate `@generated` function, and adding features (new constraint types, nested prefabs) requires threading optics through multiple layers. The structured spec eliminates the indirection and makes the system extensible: defining a new constraint is just a struct with an index type parameter plus a `resolve` method.

## Considered options

- **Keep flat spec, clean up internals.** Would reduce code but not complexity — optics and the scatter/resolve split remain fundamental.
- **Store a compiled closure `f` in CompiledModel.** Flexible but opaque to Julia's compiler — cannot inline into the fit loop, risking allocations on the hot path.

## Consequences

- `Registry`, `ComponentRef`, `free_lenses`, `scatter`, `_resolve`, and the `_set` lens-chain machinery are eliminated.
- Accessors remains a dependency for `@set` (not hot path) and user-facing property access.
- New `AbstractConstraint` supertype and `resolve` protocol — extending the constraint system requires only a struct definition and a `resolve` method.
- `@model` and `@constrain` keep their current syntax; internal code generation changes.
