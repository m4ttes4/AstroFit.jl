# AstroFit

Parametric model-fitting library for astronomy. Users compose named model components with a DSL, attach constraints, and fit against data.

## Language

**Model**:
An immutable, stack-allocated struct (`<: AbstractModel{N}`) that maps N-dimensional coordinates to a scalar value via `render`.
_Avoid_: function, kernel

**CompiledModel**:
A model tree bundled with its constraint spec, priors, and component names. The unit of fitting — owns the parameter vector bridge and the hot-path `withparams` rebuild.
_Avoid_: fitted model, wrapper

**Spec**:
A `NamedTuple` mirroring the component tree, where each leaf is a tuple of constraints (one per field of the corresponding model). Serves as both constraint storage and component registry.
_Avoid_: parameter map, config

**Constraint**:
A type (`<: AbstractConstraint`) that knows how to resolve a single parameter from the free-parameter vector `p`. Each constraint carries its index into `p` as a type parameter. Built-in kinds: `Free`, `Fixed`, `Bounded`, `Tied`.
_Avoid_: rule, modifier

**Component**:
A named leaf in the model tree — a user-given symbol (e.g. `g1`, `Ha`) bound to an `AbstractModel` or a prefab `CompiledModel`.
_Avoid_: submodel, node

**Prefab**:
A `CompiledModel` used as a leaf inside `@model`. Its constraints nest under the component name in the parent spec.
_Avoid_: template, factory

**Render**:
Evaluate a model at given coordinates, producing a scalar. `render(m, x...)` is the only evaluation entry point.
_Avoid_: evaluate, call, predict

**Resolve (constraint)**:
Compute the concrete parameter value from the free-parameter vector `p`. Dispatches on constraint type: `resolve(::Free{I}, p) = p[I]`, `resolve(::Tied{Is,F}, p) = f(p[Is]...)`, etc.
_Avoid_: apply, compute

**Resolve (ties)**:
Recompute all `Tied` parameters in a model tree from their master values. In the new design this is subsumed by per-constraint `resolve` inside `withparams`.

## Relationships

- A **CompiledModel** contains exactly one **Model** tree and one **Spec**
- A **Spec** mirrors the **Component** tree: each key is a component name, each leaf value is a tuple of **Constraints**
- A **Constraint** resolves one parameter from the free-parameter vector via `resolve`
- A **Prefab** is a **CompiledModel** whose **Spec** nests under its component name when composed via `@model`

## Example dialogue

> **Dev:** "When a user writes `@fix Ha.amplitude = 5.0` inside `@constrain`, what happens?"
> **Domain expert:** "The macro parses the path `(:Ha, :amplitude)`, navigates the spec to find `Ha`'s constraint tuple, and replaces the constraint at the `amplitude` position with `Fixed(5.0)`. The model tree is updated and a new CompiledModel is returned."

> **Dev:** "What does `withparams` do in the hot path?"
> **Domain expert:** "It takes a flat parameter vector `p`, walks the spec and model type in parallel, calls `resolve` on each constraint to get the value, reconstructs each leaf model positionally, and rebuilds the tree bottom-up. No optics, no intermediate scatter/resolve passes."

## Flagged ambiguities

- "resolve" is used for two distinct operations: (1) computing a single parameter value from `p` via a constraint's `resolve` method, and (2) the legacy operation of recomputing all tied parameters in a tree. In the new design, meaning (2) is subsumed by (1) — `withparams` calls per-constraint `resolve` and there is no separate tie-resolution pass.
