using Accessors: PropertyLens
import Accessors: set

# ---------------------------------------------------------------------------
# free_lenses / tied_entries — selection happens at compile time from the
# spec's type (the constraint types are static), so the hot path sees a
# plain tuple expression: no recursion, no inference limits on long specs.
# ---------------------------------------------------------------------------

_constraint_type(T) = T.parameters[2]

@generated function free_lenses(spec::Tuple)
    idx = [k for (k, T) in enumerate(spec.parameters)
           if _constraint_type(T) <: Union{Free, Bounded}]
    Expr(:tuple, (:(spec[$k][1]) for k in idx)...)
end

@generated function tied_entries(spec::Tuple)
    idx = [k for (k, T) in enumerate(spec.parameters)
           if _constraint_type(T) <: Tied]
    Expr(:tuple, (:(spec[$k]) for k in idx)...)
end

# ---------------------------------------------------------------------------
# scatter / resolve — write free params back; recompute Tied dependents.
# Both are fully unrolled by @generated: each `set` changes the model's
# type, and a recursive formulation hits Julia's inference recursion limit
# on big models (→ dynamic dispatch → allocations in the fit loop).
# ---------------------------------------------------------------------------

gather(model, spec) = map(l -> l(model), free_lenses(spec))

# _set — whole-lens set with the composition flattened at compile time.
# Accessors' recursive set over ComposedFunction trips Julia's inference
# recursion limiter when the value type differs from the tree's (e.g. Dual
# into a Float64 tree) → dynamic dispatch → allocations in the fit loop.
# Emitting the get/set chain as straight-line code over primitive lenses
# keeps it inferable. Falls back to Accessors.set for non-singleton optics.

_primitive_lenses(::Type{ComposedFunction{O, I}}) where {O, I} =
    vcat(_primitive_lenses(I), _primitive_lenses(O))
_primitive_lenses(::Type{typeof(identity)}) = []
_primitive_lenses(::Type{T}) where {T} =
    Base.issingletontype(T) ? Any[T.instance] : nothing

function _set_expr(obj, lens, v, prims)
    n = length(prims)
    n == 0 && return v                       # identity lens: replace the object
    block = Expr(:block, Expr(:meta, :inline))
    ovars = [gensym(:o) for _ in 1:n]
    push!(block.args, :($(ovars[1]) = $obj))
    for k in 2:n
        push!(block.args, :($(ovars[k]) = $(prims[k - 1])($(ovars[k - 1]))))
    end
    sv = v
    for k in n:-1:1
        nsv = gensym(:s)
        push!(block.args, :($nsv = set($(ovars[k]), $(prims[k]), $sv)))
        sv = nsv
    end
    push!(block.args, sv)
    block
end

# N = 3
# begin
#     @inline
#     o1 = obj        # radice
#     o2 = a(o1)      # scendi con lente a
#     o3 = b(o2)      # scendi con lente b
#     # risali impostando
#     s1 = set(o3, c, v)   # imposta foglia
#     s2 = set(o2, b, s1)  # risali con subtree aggiornato
#     s3 = set(o1, a, s2)  # risali alla radice
#     s3
# end
@generated function _set(obj, lens, v)
    prims = try
        _primitive_lenses(lens)
    catch
        nothing
    end
    prims === nothing && return :(set(obj, lens, v))
    _set_expr(:obj, :lens, :v, prims)
end

scatter(model, spec, vals) = _scatter(model, free_lenses(spec), vals)

# N = 3
# begin
#     @inline
#     m0 = model
#     m1 = _set(m0, lenses[1], @inbounds(vals[1]))
#     m2 = _set(m1, lenses[2], @inbounds(vals[2]))
#     m3 = _set(m2, lenses[3], @inbounds(vals[3]))
#     m3
# end
@generated function _scatter(model, lenses::Tuple, vals)
    N = length(lenses.parameters)
    out  = Expr(:block, Expr(:meta, :inline), :(m0 = model))
    prev = :m0
    for k in 1:N
        mk = Symbol(:m, k)
        push!(out.args, :($mk = _set($prev, lenses[$k], @inbounds(vals[$k]))))
        prev = mk
    end
    push!(out.args, prev)
    out
end

resolve(model, spec) = _resolve(model, tied_entries(spec))

@inline _apply_tie(model, target, c::Tied) =
    _set(model, target, c.f(map(l -> l(model), c.masters)...))

@generated function _resolve(model, entries::Tuple)
    N = length(entries.parameters)
    out  = Expr(:block, Expr(:meta, :inline), :(m0 = model))
    prev = :m0
    for k in 1:N
        mk = Symbol(:m, k)
        push!(out.args, :($mk = _apply_tie($prev, entries[$k][1], entries[$k][2])))
        prev = mk
    end
    push!(out.args, prev)
    out
end

# ---------------------------------------------------------------------------
# CompiledModel — model tree + spec + priors + names registry
# ---------------------------------------------------------------------------

struct CompiledModel{M, S<:Tuple, P<:Tuple, R}
    model::M    # always tie-resolved (invariant I1)
    spec::S     # NTuple of (optic, constraint)
    priors::P   # NTuple of (optic, prior distribution)
    names::R    # NamedTuple: component name → optic
end

# Single gate that establishes I1: every constructor goes through here.
_compiled(model, spec, names, priors=()) =
    CompiledModel(resolve(model, spec), spec, priors, names)

function compile(model, spec, names=(;); priors=())
    _compiled(_apply_fixed(model, spec), spec, names, priors)
end

_apply_fixed(model, ::Tuple{}) = model
function _apply_fixed(model, spec::Tuple)
    _apply_fixed(_set_fixed(model, first(spec)...), Base.tail(spec))
end
_set_fixed(model, lens, c::Fixed)         = set(model, lens, c.value)
_set_fixed(model, lens, ::Fixed{Nothing}) = model
_set_fixed(model, lens, c)               = model

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

nfree(cm::CompiledModel)      = length(free_lenses(cm.spec))
freevals(cm::CompiledModel)   = map(l -> l(cm.model), free_lenses(cm.spec))
paramvector(cm::CompiledModel) = collect(Float64, freevals(cm))

function bounds_vectors(spec)
    pairs = _free_pairs(spec)
    (Float64[_lower(p[2]) for p in pairs], Float64[_upper(p[2]) for p in pairs])
end

_free_pairs(::Tuple{}) = ()
function _free_pairs(spec::Tuple)
    _cons_pair(first(spec)..., _free_pairs(Base.tail(spec)))
end
_cons_pair(lens, ::Fixed, rest) = rest
_cons_pair(lens, ::Tied,  rest) = rest
_cons_pair(lens, c,       rest) = ((lens, c), rest...)

_lower(::Free)     = -Inf
_lower(b::Bounded) =  b.lower
_upper(::Free)     =  Inf
_upper(b::Bounded) =  b.upper

# ---------------------------------------------------------------------------
# withparams — rebuild from flat parameter vector (hot path)
# ---------------------------------------------------------------------------

withparams(cm::CompiledModel, p) =
    _compiled(_scatter(getfield(cm, :model), free_lenses(getfield(cm, :spec)), p),
              getfield(cm, :spec), getfield(cm, :names), getfield(cm, :priors))

# ---------------------------------------------------------------------------
# Optic composition / comparison helpers
# ---------------------------------------------------------------------------

# Compose two optics avoiding useless identity wrappers.
_compose(::typeof(identity), ::typeof(identity)) = identity
_compose(::typeof(identity), o) = o
_compose(o, ::typeof(identity)) = o
_compose(a, b) = a ⨟ b

# Structural comparison robust to composition associativity:
# (a ⨟ b) ⨟ c and a ⨟ (b ⨟ c) flatten to the same lens sequence.
_optic_leaves(o::ComposedFunction) = (_optic_leaves(o.inner)..., _optic_leaves(o.outer)...)
_optic_leaves(::typeof(identity)) = ()
_optic_leaves(o) = (o,)
_same_optic(a, b) = _optic_leaves(a) == _optic_leaves(b)

# ---------------------------------------------------------------------------
# Registry — names entry for a prefab: optic to the subtree + its sub-names
# ---------------------------------------------------------------------------

struct Registry{O, R<:NamedTuple}
    optic::O   # optic from the parent tree root to the prefab subtree
    names::R   # the prefab's own registry, relative to that subtree
end

# ---------------------------------------------------------------------------
# ComponentRef — cursor into a named component (read/write path)
# ---------------------------------------------------------------------------

struct ComponentRef{CM<:CompiledModel, O, R<:NamedTuple}
    root::CM
    optic::O   # optic from the root tree to this component's subtree
    names::R   # sub-registry, relative to this subtree (empty for plain leaves)
end

_ref_or_value(root, prefix, r::Registry) =
    ComponentRef(root, _compose(prefix, r.optic), r.names)
function _ref_or_value(root, prefix, optic)
    full = _compose(prefix, optic)
    sub  = full(getfield(root, :model))
    sub isa AbstractModel ? ComponentRef(root, full, (;)) : sub
end

# cm.narrow → ComponentRef;  cm.model/spec/names → field directly
function Base.getproperty(cm::CompiledModel, s::Symbol)
    (s === :model || s === :spec || s === :priors || s === :names) && return getfield(cm, s)
    getindex(cm, s)
end

# cm[:narrow] — explicit, unambiguous access (also when a component name
# collides with a reserved field)
function Base.getindex(cm::CompiledModel, s::Symbol)
    names = getfield(cm, :names)
    hasproperty(names, s) || throw(ArgumentError(
        "no component `$s`; available: $(join(keys(names), ", "))"))
    _ref_or_value(cm, identity, getproperty(names, s))
end

Base.propertynames(cm::CompiledModel) =
    (:model, :spec, :priors, :names, keys(getfield(cm, :names))...)

# cm.narrow.amplitude → value; cm.Ha.line → deeper ComponentRef
function Base.getproperty(ref::ComponentRef, s::Symbol)
    root  = getfield(ref, :root)
    optic = getfield(ref, :optic)
    names = getfield(ref, :names)
    hasproperty(names, s) && return _ref_or_value(root, optic, getproperty(names, s))
    getproperty(optic(getfield(root, :model)), s)
end

Base.propertynames(ref::ComponentRef) =
    (keys(getfield(ref, :names))...,
     propertynames(getfield(ref, :optic)(getfield(getfield(ref, :root), :model)))...)

# ---------------------------------------------------------------------------
# @set support: Accessors decomposes `@set cm.a.b = v` into nested set calls.
# The inner set (on the ComponentRef) knows the root, so the constraint check
# happens there; the outer set (on the CompiledModel) swaps the subtree and
# re-resolves the ties (invariant I1).
# ---------------------------------------------------------------------------

function _check_set(spec::Tuple, optic, v)
    for (t, c) in spec
        _same_optic(t, optic) || continue
        c isa Tied && throw(ArgumentError(
            "cannot set a Tied parameter: it is computed from its master(s); set those instead"))
        if c isa Bounded
            (c.lower <= v <= c.upper) || throw(ArgumentError(
                "value $v outside bounds ($(c.lower), $(c.upper))"))
        end
        return nothing
    end
    nothing
end

function set(cm::CompiledModel, ::PropertyLens{s}, v) where {s}
    (s === :model || s === :spec || s === :priors || s === :names) && throw(ArgumentError(
        "field `$s` of a CompiledModel is read-only"))
    names = getfield(cm, :names)
    hasproperty(names, s) || throw(ArgumentError(
        "no component `$s`; available: $(join(keys(names), ", "))"))
    entry = getproperty(names, s)
    optic = entry isa Registry ? entry.optic : entry
    v isa AbstractModel || _check_set(getfield(cm, :spec), optic, v)
    _compiled(set(getfield(cm, :model), optic, v), getfield(cm, :spec), names,
              getfield(cm, :priors))
end

function set(ref::ComponentRef, ::PropertyLens{s}, v) where {s}
    root    = getfield(ref, :root)
    base    = getfield(ref, :optic)
    names   = getfield(ref, :names)
    subtree = base(getfield(root, :model))
    if hasproperty(names, s)
        # replacing a sub-component subtree: the param-level check already
        # happened in the deeper set call. For scalar entries from an
        # implicit prefab registry, this is the param-level set.
        entry = getproperty(names, s)
        optic = entry isa Registry ? entry.optic : entry
        full = _compose(base, optic)
        full(getfield(root, :model)) isa AbstractModel ||
            _check_set(getfield(root, :spec), full, v)
        return set(subtree, optic, v)
    end
    _check_set(getfield(root, :spec), _compose(base, PropertyLens{s}()), v)
    set(subtree, PropertyLens{s}(), v)
end

# ---------------------------------------------------------------------------
# Calling a CompiledModel = evaluation (not parameter rebuild)
# ---------------------------------------------------------------------------

render(cm::CompiledModel, x...) = render(getfield(cm, :model), x...)

# ---------------------------------------------------------------------------
# Algebra on CompiledModel outside @model would silently drop the spec —
# informative error instead (compose inside @model so constraints travel).
# ---------------------------------------------------------------------------

const _COMPOSE_HINT = "cannot compose CompiledModels outside @model: their " *
                      "constraints would be lost. Compose inside @model " *
                      "(e.g. `@model Ha + Hb`) so the constraints travel."

for op in (:+, :-, :*, :/, :∘, :|>)
    @eval Base.$op(::CompiledModel, ::CompiledModel) = throw(ArgumentError(_COMPOSE_HINT))
    @eval Base.$op(::CompiledModel, ::AbstractModel) = throw(ArgumentError(_COMPOSE_HINT))
    @eval Base.$op(::AbstractModel, ::CompiledModel) = throw(ArgumentError(_COMPOSE_HINT))
end
