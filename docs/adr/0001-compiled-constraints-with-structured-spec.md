# Compiled constraints with a structured spec

We restructured `CompiledModel` around a **single annotated model tree**: operator
nodes are the model compound types (`Sum`, `Pipe`, …) and each leaf is a `Leaf{name}`
carrying the component's values plus its positional tuple of constraints. This replaces
both the old flat `(optic, constraint)` spec (with its `Registry`/`scatter`/`_resolve`
machinery) and the intermediate two-tree design (a model tree plus a parallel
`NamedTuple` spec). `withparams` walks the one tree and rebuilds it bottom-up. This
eliminates the optics layer, the parallel-tree sync invariant, and synthetic operator
keys; the system stays extensible — a new constraint is a struct plus a rule for how
`withparams` reads it.

This document records the design decisions reached for that restructure. They were
worked out collaboratively (a design grill plus a three-way review) and supersede
the original proposal in this file, which had constraints carry their parameter
index as a type parameter — that approach was reversed (see decision 1).

## Decisions

1. **Structural parameter index, not index-in-the-type (design X).**
   Constraints do not store their slot in `p`. `Free` is a singleton; `Bounded`
   carries only bounds; `Fixed` carries only its value. `withparams` is
   `@generated` and assigns `p`-slots by a compile-time walk of the spec in tree
   order, emitting `p[k]` as literals.
   - *Chosen over* index-in-the-type (`Free{I}`, design Y), where `resolve` would be
     plain dispatch and `withparams` plain recursion. One of the two complexities is
     irreducible: a dense `p` forces contiguous indices, so Y must **renumber on
     every edit** and rewrite a prefab's constraint types on composition; X confines
     all of that to one `@generated` function and keeps constraints clean values.
   - *Why X:* renumbering is correctness-critical mutation on the interactive
     edit/compose path — exactly where a silent off-by-one corrupts which parameter
     maps where. One contained, test-pinned `@generated` function is the safer place
     for the unavoidable complexity. Under X, editing a constraint is a single
     swap and **prefabs compose without rewriting** (their spec drops in unchanged
     and positions re-derive at compile time).

2. **Position is the optic for self-describing constraints.** A constraint's
   location in the nested spec already identifies its parameter, so `Free`/`Bounded`/
   `Fixed` need no path. Only `Tied` carries a path-optic, because it references a
   *different* node.

3. **`Tied` references free masters only — no chaining.** A `Tied`'s masters must be
   free parameters, not other `Tied`s. Chains collapse by function composition (no
   added expressive power), so the restriction loses nothing; allowing them would
   break the uniform single-pass resolution and force compile-time dependency
   ordering + cycle detection. Validated at compile time (a master that is not free
   is an error).

4. **`Tied` reads many masters, writes one slot.** `value = f(master₁ … masterₙ)`,
   one output per `Tied`. A free value shared across N slots is N `Tied`s pointing at
   the same free master.

5. **No general "linking" primitive.** The four kinds (`Free`/`Bounded`/`Fixed`/
   `Tied`) cover every real use. Reusable physical couplings (line-ratio, shared
   width, redshift) live in **model component types** (e.g. `Redshift1D`,
   `EmissionDoublet1D`), not in the constraint system.

6. **`CompiledModel` fuses model and spec into one annotated tree.** Two fields:
   `tree`, `priors`. There is no separate spec tree, hence no parallel-tree sync
   invariant: an edit touches one structure. (Supersedes the earlier three-field
   `model`/`spec`/`priors` shape.)

7. **Constraints annotate the tree leaves; no separate spec.** Operator nodes are the
   model compound types (`Sum`, …). Each leaf is a `Leaf{name}` (`<: AbstractModel`)
   carrying the component's current/default model values and a plain `Tuple` of
   constraints positional to the model's fields. `name` is a type parameter (the user
   name); field names stay owned by the model type. No `NamedTuple`, no synthetic
   operator keys — structure is the compound node types plus tuple position.
   (Supersedes the earlier nested-`NamedTuple` spec.)

8. **Navigation is O(1) at runtime, resolved from the type.** `cm.ha` → `getproperty`
   lifts the name to `Val{:ha}` and a recursive `_nav` walks the tree for `Leaf{:ha}`.
   No stored path, no desync. It does **not** need to be `@generated`: because the name
   is in the type (`Val{name}`/`Leaf{name}`), each leaf resolves by dispatch to the leaf
   or `nothing`, and the `h === nothing` test folds at compile time — inference collapses
   the dead branches to a concrete return type (verified: `@inferred` and `@allocated == 0`).
   Plain dispatch was chosen over `@generated` as the simpler form that meets the same
   guarantee; `@generated` stays the fallback if a pathologically deep tree defeats
   inlining. `getproperty` routes the real fields (`tree`, `priors`) to `getfield`, so a
   leaf named `tree`/`priors` is not navigable; a missing name is an `ArgumentError`.

## Implementation invariants

- Leaf reconstruction must go through `ConstructionBase.constructorof` (via
  `Accessors`), never `typeof(m)(...)`, or ForwardDiff loses derivatives under
  `Dual` retyping.
- `resolve(::Tied)` need not be `@generated` (a single `ntuple(k -> p[Is[k]], Val(n))`
  suffices); under X, `Tied` resolution inlines into `withparams` anyway.
- The hot path must stay allocation-free and type-stable — pin with the existing
  `@inferred` / `@allocated == 0` tests for both `Float64` and `Dual`.

## Status of the branch

The design-X refactor is in place. Done: the four constraint types; the `Leaf{name,M,C}`
leaf type with `render` delegation; the two-field `CompiledModel{T,P}` (`tree`, `priors`)
per decisions 6/7; `withparams` (`src/withparams.jl`); `getproperty` navigation (decision
8, plain recursion, no `@generated`); the parameter-introspection layer (`src/params.jl`);
the constraint-application engine `setconstraint`/`validate` (`src/constrain.jl`, the
design-X rewrite of the old `paths`/renumbering file); and the surface macros — `@model`
plus the constraint verbs `@fix`/`@bound`/`@free`/`@tie` and the `@constrain` block
(`src/macro.jl`). Only out of scope for this cut: prefab nesting (a `CompiledModel` as a
leaf).

`params.jl` exposes the free-slot views the optimizer needs: `nfree`, `params` (the
current `p₀`), `bounds` (an `(lower, upper)` pair — `Free`→`(-Inf,Inf)`, `Bounded`→its
bounds), and `paramnames` (`:<leaf>_<field>` labels for fit output). All four are plain
recursion — setup-time, not hot path — and walk the tree in the **same DFS order**
`withparams`' `_slotmap!` assigns slots, so they line up slot-for-slot with `p`. Nothing
structurally forces those two walks to agree; the round-trip test (free fields → sentinels
through `withparams(cm, params(cm))`) is what pins the coupling against drift. Values are
gathered as tuples then `collect`ed, so an all-`Fixed`/`Tied` leaf contributes `()` without
poisoning the element type via `vcat`.

`@model` (`src/macro.jl`) is built. It takes a `begin … end` block: each `name = expr`
binds `name` to `Leaf{:name}(expr, all-Free defaults)`; the trailing expression is the
composition. That composition is emitted untouched — the compound operators evaluate it
and, since `Leaf <: AbstractModel`, build the annotated tree directly, so the macro never
parses it (the `∘`/`|>` swap is Julia's to handle). It emits the composition **once**
(the earlier "emit twice — raw + wrapped" was for the abandoned two-tree design; with a
single annotated tree only the wrapped form exists). Leaf names are left unescaped so
macro hygiene renames them to gensyms — no leak into caller scope — while model
expressions are escaped (they may use caller variables). Default constraints are all
`Free`; the constraint verbs / `@constrain` edit them. A leaf used more than once in the composition
is rejected at construction (`_compiled`), since its `(name, field)` slots would collide
in `withparams`; sharing a value is `Tied`'s job. Prefab nesting (a `CompiledModel` as a
leaf) is out of scope for this first cut.

`withparams(cm, p)` is `@generated` and returns the **bare** model tree (Leaf wrappers
stripped), so the hot path is the same straight-line rebuild + `render` as a plain
compound model. Two passes over the tree *type*: pass 1 assigns each `Free`/`Bounded`
field its slot in `p` by position (keyed by `(leaf-name, field)`); pass 2 emits the
reconstruction — `Free`/`Bounded` → `p[k]`, `Fixed` → its runtime value, `Tied` → its
function applied to its masters' slots (multi-master, in path order). Forward references
work because pass 1 completes before pass 2. Leaf models are rebuilt via
`ConstructionBase.constructorof` (from `Accessors`), keeping ForwardDiff `Dual`
derivatives. A `Tied` master that is not a free parameter is a `KeyError` at
specialization time if it reaches `withparams`; in practice `validate` (below) catches
it first with a clear `ArgumentError` — that is decision 3's "validated at compile time".
Verified: `@inferred` and `@allocated == 0` on `Float64` and `Dual`, correct gradients.

`setconstraint`/`validate` (`src/constrain.jl`) are the constraint-application engine.
`setconstraint(cm, leaf, field, c)` is the write-analog of `_nav`: navigation *finds* a
leaf, but the tree is immutable, so editing rebuilds the spine root→leaf (`_setleaf`) and
swaps one tuple slot (`Base.setindex`) — pure, priors kept, no renumber (slots are
structural). `validate(cm)` walks the tree and checks every `Tied`'s masters are Free or
Bounded (exist and free), throwing `ArgumentError` otherwise; it subsumes cycle detection
(a tie can't reach another tie) and returns `cm` so it chains. The two are kept separate:
`setconstraint` is a dumb local edit, `validate` is the whole-tree check run eagerly by
the macro layer — not per-edit, so a forward tie whose master is fixed-then-freed in one
batch isn't wrongly rejected.

The constraint macros (`src/macro.jl`) lower to that engine. Each standalone verb
(`@fix`, `@bound`, `@free`, `@tie`, `@prior`) has a distinct operator and **auto-rebinds**
the model variable (no `m = @fix m…` needed): `@fix m.l.f = v` or `@fix m.l.f` (current
value), `@bound m.l.f in (lo, hi)`, `@tie m.l.f -> expr`, `@prior m.l.f ~ dist`,
`@free m.l.f`. `@tie m.g2.x -> f(m.g1.y, …)` walks the RHS, turning each `m.leaf.field`
master into a fresh lambda argument (paths collected in order, non-path code left intact —
function calls, arithmetic, and constants are preserved). **Master variables are captured
*live*, not frozen** — the lambda closes over caller bindings by reference, so a
literal/`const` coefficient is what you get either way, but a captured *reassigned* variable
both tracks later changes and boxes, breaking the zero-alloc hot path (use a literal or
`const`). `@constrain m begin … end` is the ergonomic block: leaf names are bare (a gensym
root is injected), each line is dispatched by AST shape (no `@verb` prefix except `@free`):
`=` → fix, `->` → tie, `in` → bound, `~` → prior, bare path → fix-at-current. The block
rejects a **duplicate target at expansion time**, auto-rebinds the model variable, and
runs `validate` once at the end. Verified: a macro-generated literal tie stays
`@allocated == 0` and `@inferred`; the block catches both a duplicate target and a tie
broken by a later edit.
