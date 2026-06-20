# AstroFit

Parametric model-fitting library for astronomy. Users compose named model components with a DSL, attach constraints, and fit against data.

## Language

**Model**:
An immutable, stack-allocated struct (`<: AbstractModel{N}`) that maps N-dimensional coordinates to a scalar value via `render`.
_Avoid_: function, kernel

**CompiledModel**:
A single annotated model tree plus priors. The unit of fitting — owns the parameter vector bridge and the hot-path `withparams` rebuild. Two fields only: `tree`, `priors`. The tree is the sole state (current values live in the leaf models); everything else — parameter count, initial vector `p₀`, name→slot map — is derived from the tree on demand, never cached. The accessors `nfree`, `params` (`p₀`), `bounds`, and `paramnames` expose these views, each walking the tree in the same DFS order `withparams` assigns slots.
_Avoid_: fitted model, wrapper

**Leaf**:
A named node in the annotated tree (`Leaf{name} <: AbstractModel`): carries the component's model values and a positional tuple of constraints (one per field). The user name lives in its type. Constraint storage and component registry are the leaves themselves — there is no separate spec.
_Avoid_: spec, parameter map, config

**Constraint**:
A type (`<: AbstractConstraint`) describing one field of a leaf model. Its **position in its `Leaf`'s constraint tuple is its identity** — it carries no parameter index. Built-in kinds: `Free` (a singleton), `Bounded` (bounds only), `Fixed` (value only), `Tied` (a function plus paths to its free masters). `withparams` assigns each `Free`/`Bounded` a slot in `p` by a compile-time walk in tree order.
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
Compute the concrete parameter value from the free-parameter vector `p`. Subsumed by `withparams`' compile-time walk: a `Free`/`Bounded` reads its assigned slot `p[k]` (slot from position, not stored), a `Fixed` yields its value, a `Tied` applies `f` to its masters' slots. `Tied` masters must be free (no chaining).
_Avoid_: apply, compute

**Resolve (ties)**:
Recompute all `Tied` parameters in a model tree from their master values. In the new design this is subsumed by per-constraint `resolve` inside `withparams`.

## Relationships

- A **CompiledModel** contains exactly one annotated tree (operator nodes + **Leaf** nodes) and its priors
- The tree's operator nodes are model compound types (`Sum`, …); each **Leaf** is tagged with its **Component** name and holds a positional tuple of **Constraints**
- A **Constraint**'s position in its **Leaf**'s tuple identifies its parameter; `withparams` assigns `Free`/`Bounded` slots in `p` by a compile-time walk of the tree
- A **Tied** constraint references one or more free masters by path; it reads many, writes one
- A **Prefab** is a **CompiledModel** whose tree nests as a **Leaf** under its component name when composed via `@model`

## Example dialogue

> **Dev:** "When a user writes `@fix Ha.amplitude = 5.0` inside `@constrain`, what happens?"
> **Domain expert:** "The macro parses the path `(:Ha, :amplitude)`, navigates the tree to the `Leaf{:Ha}`, and replaces the constraint at the `amplitude` position with `Fixed(5.0)`. Because indices are structural, nothing is renumbered — a new CompiledModel is returned."

> **Dev:** "What does `withparams` do in the hot path?"
> **Domain expert:** "It takes a flat parameter vector `p` and, in a `@generated` walk of the annotated tree type, emits each parameter directly — a `Free`/`Bounded` becomes a literal `p[k]`, a `Fixed` its value, a `Tied` its function applied to its masters' slots — then reconstructs each leaf positionally and rebuilds the tree bottom-up. The Leaf wrappers are stripped: it returns the bare model tree, which the ordinary compound `render` evaluates. No optics, no separate spec, no per-constraint index."

## Flagged ambiguities

- "resolve" is used for two distinct operations: (1) computing a single parameter value from `p` via a constraint's `resolve` method, and (2) the legacy operation of recomputing all tied parameters in a tree. In the new design, meaning (2) is subsumed by (1) — `withparams` calls per-constraint `resolve` and there is no separate tie-resolution pass.
