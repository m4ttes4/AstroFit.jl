# ---------------------------------------------------------------------------
# Compile-time AST helpers
# ---------------------------------------------------------------------------

# Canonicalize a composition expression to strictly-binary form so the model
# tree has a single fold authority. `a + b + c + d` parses as one n-ary call
# `+(a,b,c,d)`; Base.afoldl would left-fold it at runtime. We left-fold it here
# instead — emitting `((a+b)+c)+d` — and use this same canonical form for BOTH
# the optic walker and the builder closure, so the two can never disagree (and
# we no longer depend on Base.afoldl's fold order). `-`, `/`, `∘`, `|>` already
# parse as nested binary calls; we just recurse into their operands.
function _canonicalize(expr)
    expr isa Symbol && return expr
    Meta.isexpr(expr, :call) || return expr
    op, args = expr.args[1], expr.args[2:end]
    if op in (:+, :*) && length(args) > 2
        return foldl((l, r) -> Expr(:call, op, l, _canonicalize(r)),
                     args[2:end]; init = _canonicalize(args[1]))
    end
    Expr(:call, op, map(_canonicalize, args)...)
end

# Walk the composition expression and return ordered list of (name => optic_path).
# optic_path is a Vector{Symbol} of :left/:right steps from the root.
# a ∘ b  → Pipe(b, a): a→.right, b→.left
# a |> b → Pipe(a, b): a→.left,  b→.right
# +,-,*,/ → left→.left, right→.right
# Expects a canonical (binary) expression: see _canonicalize.
function _walk_optics(expr, path::Vector{Symbol} = Symbol[])
    if expr isa Symbol
        return Pair{Symbol, Vector{Symbol}}[expr => copy(path)]
    end
    Meta.isexpr(expr, :call) || _model_expr_error(expr)
    op = expr.args[1]
    length(expr.args) == 3 || _model_expr_error(expr)
    a, b = expr.args[2], expr.args[3]
    if op in (:+, :-, :*, :/)
        return [_walk_optics(a, [path; :left]); _walk_optics(b, [path; :right])]
    elseif op == :∘       # a ∘ b → Pipe(b, a)
        return [_walk_optics(a, [path; :right]); _walk_optics(b, [path; :left])]
    elseif op == :|>      # a |> b → Pipe(a, b)
        return [_walk_optics(a, [path; :left]); _walk_optics(b, [path; :right])]
    end
    _model_expr_error(expr)
end

function _model_expr_error(expr)
    error("@model: cannot extract component names from `$expr`. " *
          "Each leaf must be a named symbol (bind anonymous models to a name first); " *
          "compose with +, -, *, /, ∘, or |>.")
end

# Build an Accessors optic expression from a path of field names.
function _path_to_optic_expr(path::Vector{Symbol})
    isempty(path) && return :(identity)
    result = :(Accessors.PropertyLens{$(QuoteNode(path[1]))}())
    for i in 2:length(path)
        result = :($result ⨟ Accessors.PropertyLens{$(QuoteNode(path[i]))}())
    end
    result
end

# Replace each leaf symbol with its closure argument (operators untouched).
function _substitute_leaves(expr, argmap)
    if expr isa Symbol
        return get(argmap, expr, expr)
    elseif Meta.isexpr(expr, :call)
        return Expr(:call, expr.args[1],
                    (_substitute_leaves(a, argmap) for a in expr.args[2:end])...)
    end
    expr
end

const _RESERVED = (:model, :spec, :priors, :names)

# ---------------------------------------------------------------------------
# @model — runtime builder
# ---------------------------------------------------------------------------

_strip_leaf(m::AbstractModel) = m
_strip_leaf(cm::CompiledModel) = getfield(cm, :model)
_strip_leaf(x) = throw(ArgumentError(
    "@model: each leaf must be an AbstractModel or a CompiledModel (prefab), " *
    "got $(typeof(x))"))

# Identity check: every leaf optic must hit exactly the value that went into the
# tree. Now redundant by construction — the walker and the builder closure share
# one canonical fold (see _canonicalize), so the map is correct by derivation.
# Kept as a debug `@assert` (see _build_model) to guard against future edits to
# the macro that might reintroduce a divergence. Returns Bool.
function _identity_check(tree, names, optics, bare)
    all(((n, o, b),) -> o(tree) === b, zip(names, optics, bare))
end

# Every parameter of a bare leaf is Free by default (explicit spec entries),
# so the authored model is fittable as-is and @constrain only overrides.
_leaf_free_spec(prefix, m::T) where {T<:AbstractModel} =
    map(f -> (_compose(prefix, PropertyLens{f}()), Free()), fieldnames(T))

# Re-rooting a prefab spec under its leaf prefix: the prefix applies to the
# target optic AND to every master optic inside a Tied (critical point 1).
_reroot(prefix, c::Tied) = Tied(c.f, map(m -> _compose(prefix, m), c.masters))
_reroot(prefix, c) = c

_leaf_spec(o, m::AbstractModel) = _leaf_free_spec(o, m)
_leaf_spec(o, cm::CompiledModel) =
    map(((t, c),) -> (_compose(o, t), _reroot(o, c)), getfield(cm, :spec))

_leaf_priors(o, ::AbstractModel) = ()
_leaf_priors(o, cm::CompiledModel) =
    map(((t, p),) -> (_compose(o, t), p), getfield(cm, :priors))

_collect_spec(::Tuple{}, ::Tuple{}) = ()
_collect_spec(optics::Tuple, values::Tuple) =
    (_leaf_spec(first(optics), first(values))...,
     _collect_spec(Base.tail(optics), Base.tail(values))...)

_collect_priors(::Tuple{}, ::Tuple{}) = ()
_collect_priors(optics::Tuple, values::Tuple) =
    (_leaf_priors(first(optics), first(values))...,
     _collect_priors(Base.tail(optics), Base.tail(values))...)

_registry_entry(o, ::AbstractModel) = o
_registry_entry(o, cm::CompiledModel) = Registry(o, getfield(cm, :names))

function _build_model(f, names::Tuple{Vararg{Symbol}}, optics::Tuple, values::Tuple)
    bare = map(_strip_leaf, values)
    tree = f(bare...)
    @assert _identity_check(tree, names, optics, bare) "@model: walker optic→field map mismatch (internal bug)"
    spec     = _collect_spec(optics, values)
    priors   = _collect_priors(optics, values)
    registry = NamedTuple{names}(map(_registry_entry, optics, values))
    _compiled(tree, spec, registry, priors)
end

"""
    @model expr
    @model begin
        name₁ = model₁
        name₂ = model₂
        name₁ + name₂
    end

Construct a `CompiledModel` with a named component registry.

**Block form**: bindings define component names; the final expression is the
composition tree. **Inline form**: `@model g1 + g2` — names are taken from the
in-scope variable symbols.

Components compose with `+`, `-`, `*`, `/` (pointwise) or `∘` / `|>` (pipe).
Every leaf must be a named symbol bound to an `AbstractModel` or a
`CompiledModel` (prefab); a prefab's constraints travel, namespaced under the
leaf name. Every bare-leaf parameter starts `Free`.
"""
macro model(expr)
    bindings  = Pair{Symbol, Any}[]
    comp_expr = nothing

    if Meta.isexpr(expr, :block)
        for stmt in expr.args
            stmt isa LineNumberNode && continue
            if Meta.isexpr(stmt, :(=)) && stmt.args[1] isa Symbol
                push!(bindings, stmt.args[1] => stmt.args[2])
            elseif comp_expr === nothing
                comp_expr = stmt
            else
                error("@model: multiple composition expressions in block")
            end
        end
    else
        comp_expr = expr
    end

    comp_expr === nothing && error("@model: no composition expression found")

    comp_expr = _canonicalize(comp_expr)
    optic_pairs = _walk_optics(comp_expr)
    leafnames   = Symbol[first(p) for p in optic_pairs]
    allunique(leafnames) ||
        error("@model: duplicate component name(s): " *
              "$(join(unique(n for n in leafnames if count(==(n), leafnames) > 1), ", "))")
    for n in leafnames
        n in _RESERVED && error("@model: `$n` is a reserved name (CompiledModel field)")
    end

    optic_exprs = [_path_to_optic_expr(last(p)) for p in optic_pairs]
    argmap  = Dict{Symbol, Symbol}(n => gensym(n) for n in leafnames)
    closure = Expr(:->, Expr(:tuple, (argmap[n] for n in leafnames)...),
                   _substitute_leaves(comp_expr, argmap))
    names_tuple = Expr(:tuple, (QuoteNode(n) for n in leafnames)...)
    call = :(_build_model($closure, $names_tuple,
                          ($(optic_exprs...),),
                          ($((esc(n) for n in leafnames)...),)))

    isempty(bindings) && return call
    let_head = Expr(:block, (:($(esc(k)) = $(esc(v))) for (k, v) in bindings)...)
    Expr(:let, let_head, call)
end

# ---------------------------------------------------------------------------
# @constrain — runtime name resolution
# ---------------------------------------------------------------------------

function _lookup_name(names::NamedTuple, s::Symbol, full)
    hasproperty(names, s) || throw(ArgumentError(
        "@constrain: unknown name `$s` in `$(join(full, '.'))`; " *
        "available: $(join(keys(names), ", "))"))
    getproperty(names, s)
end

_resolve_path(cm::CompiledModel, path::Tuple) =
    _resolve_entry(getfield(cm, :names), path, path)

function _resolve_entry(names::NamedTuple, rest::Tuple, full)
    entry = _lookup_name(names, first(rest), full)
    _descend(entry, Base.tail(rest), full)
end

_descend(r::Registry, rest::Tuple, full) =
    _compose(r.optic, _resolve_entry(r.names, rest, full))
_descend(r::Registry, ::Tuple{}, full) = throw(ArgumentError(
    "@constrain: `$(join(full, '.'))` names a component, not a parameter"))
_descend(optic, ::Tuple{}, full) = optic
function _descend(optic, rest::Tuple, full)
    o = optic
    for s in rest
        o = _compose(o, PropertyLens{s}())
    end
    o
end

# Naked-model support: wrap a single leaf with an implicit registry
# (param name → its own lens) and an all-Free spec.
_iscompound(::Union{Sum, Difference, Product, Quotient, Pipe}) = true
_iscompound(::AbstractModel) = false

_implicit_names(m::T) where {T<:AbstractModel} =
    NamedTuple{fieldnames(T)}(map(f -> PropertyLens{f}(), fieldnames(T)))

_as_compiled(cm::CompiledModel) = cm
function _as_compiled(m::AbstractModel)
    _iscompound(m) && throw(ArgumentError(
        "@constrain: a hand-built compound has no component names; " *
        "build it with @model so the components are addressable"))
    _compiled(m, _leaf_free_spec(identity, m), _implicit_names(m))
end
_as_compiled(x) = throw(ArgumentError(
    "@constrain: expected a Model or CompiledModel, got $(typeof(x))"))

# ---------------------------------------------------------------------------
# @constrain — merge, validation, application
# ---------------------------------------------------------------------------

# Within a block the last entry per target wins (same rule as the override
# of factory constraints).
function _dedupe_last(entries::Tuple)
    out = ()
    for e in entries
        out = (filter(x -> !_same_optic(first(x), first(e)), out)..., e)
    end
    out
end

_filter_overridden(old::Tuple, new::Tuple) =
    filter(e -> !any(n -> _same_optic(first(n), first(e)), new), old)

# Validation on the merged view (conflicts can emerge only after the merge):
# V1 no tie chains, V2 no self-tie, V3 sane bounds. V4 (current value inside
# bounds) needs the tree and runs in _constrain after @fix application.
function _validate_spec(spec::Tuple)
    for (t, c) in spec
        if c isa Bounded
            c.lower < c.upper || throw(ArgumentError(
                "invalid bounds ($(c.lower), $(c.upper)): need lo < hi"))
        elseif c isa Tied
            for m in c.masters
                _same_optic(m, t) && throw(ArgumentError(
                    "self-tie: a parameter cannot be tied to itself"))
                for (t2, c2) in spec
                    c2 isa Tied && _same_optic(m, t2) && throw(ArgumentError(
                        "tie chain: a master is itself a Tied target; " *
                        "tie directly to its master(s) instead"))
                end
            end
        end
    end
    nothing
end

function _check_bounds(model, spec::Tuple)
    for (t, c) in spec
        c isa Bounded || continue
        v = t(model)
        (c.lower <= v <= c.upper) || throw(ArgumentError(
            "current value $v is outside the new bounds ($(c.lower), $(c.upper)); " *
            "set the value first or widen the bounds (no silent clamp)"))
    end
    nothing
end

function _constraint_for(spec::Tuple, optic)
    for (t, c) in spec
        _same_optic(t, optic) && return c
    end
    nothing
end

function _validate_priors(spec::Tuple, priors::Tuple)
    for (t, p) in priors
        c = _constraint_for(spec, t)
        c isa Union{Free, Bounded} && continue
        c isa Fixed && throw(ArgumentError(
            "@prior: cannot attach a prior to a Fixed parameter"))
        c isa Tied && throw(ArgumentError(
            "@prior: cannot attach a prior to a Tied parameter"))
        throw(ArgumentError("@prior: target is not a free parameter"))
    end
    nothing
end

function _constrain(cm::CompiledModel, entries::Tuple, prior_entries::Tuple=())
    model = getfield(cm, :model)
    for (t, c) in entries
        t(model) isa AbstractModel && throw(ArgumentError(
            "@constrain: a constraint target must be a parameter, not a component"))
    end
    for (t, p) in prior_entries
        t(model) isa AbstractModel && throw(ArgumentError(
            "@prior: target must be a parameter, not a component"))
    end
    new    = _dedupe_last(entries)
    merged = (_filter_overridden(getfield(cm, :spec), new)..., new...)
    new_priors = _dedupe_last(prior_entries)
    priors = (_filter_overridden(getfield(cm, :priors), new_priors)..., new_priors...)
    _validate_spec(merged)
    _validate_priors(merged, priors)
    # @fix values from THIS block write into the tree once; older Fixed
    # entries already wrote theirs (the tree is the single source of truth)
    model = _apply_fixed(model, new)
    _check_bounds(model, merged)
    _compiled(model, merged, getfield(cm, :names), priors)
end

"""
    @constrain model begin
        @fix   component.param = value
        @fix   component.param            # fix at current value
        @bound component.param in (lo, hi)
        @tie   component.param = expr(other.param, ...)
        @free  component.param
        @prior component.param ~ Distribution(args...)
    end

Attach constraints and return a new `CompiledModel`. Accepts the result of
`@model`, an existing `CompiledModel` (prefab — constraints merge by name,
the new ones win) or a naked single leaf model (params addressed without
prefix). Each `@tie` RHS may reference any number of `component.param`
masters; they are auto-detected. `@prior` stores statistical priors separately
from mechanical constraints.
"""
macro constrain(model_expr, block)
    Meta.isexpr(block, :block) ||
        error("@constrain: second argument must be a begin...end block")

    cm_var       = gensym("cm")
    spec_entries = Any[]
    prior_entries = Any[]

    path_expr(path) = Expr(:tuple, (QuoteNode(s) for s in path)...)
    optic_expr(path) = :(_resolve_path($cm_var, $(path_expr(path))))

    for stmt in block.args
        stmt isa LineNumberNode && continue
        Meta.isexpr(stmt, :macrocall) ||
            error("@constrain: unexpected `$stmt`; use @fix, @bound, @tie, @free")

        kw   = stmt.args[1]
        args = [a for a in stmt.args[2:end] if !(a isa LineNumberNode)]

        if kw == Symbol("@fix")
            if length(args) == 1 && Meta.isexpr(args[1], :(=))
                path = _extract_path(args[1].args[1])
                path === nothing && error("@fix: invalid path `$(args[1].args[1])`")
                push!(spec_entries,
                      :($(optic_expr(path)), Fixed($(esc(args[1].args[2])))))
            elseif length(args) == 1
                path = _extract_path(args[1])
                path === nothing && error("@fix: invalid path `$(args[1])`")
                push!(spec_entries, :($(optic_expr(path)), Fixed()))
            else
                error("@fix: expected `@fix param` or `@fix param = value`")
            end

        elseif kw == Symbol("@bound")
            length(args) == 1 || error("@bound: expected `@bound param in (lo, hi)`")
            arg = args[1]
            (Meta.isexpr(arg, :call) && arg.args[1] === :in) ||
                error("@bound: expected `param in (lo, hi)`, got `$arg`")
            path   = _extract_path(arg.args[2])
            bounds = arg.args[3]
            (Meta.isexpr(bounds, :tuple) && length(bounds.args) == 2) ||
                error("@bound: bounds must be a 2-tuple (lo, hi), got `$bounds`")
            path === nothing && error("@bound: invalid path `$(arg.args[2])`")
            lo, hi = esc(bounds.args[1]), esc(bounds.args[2])
            push!(spec_entries, :($(optic_expr(path)), Bounded($lo, $hi)))

        elseif kw == Symbol("@tie")
            length(args) == 1 && Meta.isexpr(args[1], :(=)) ||
                error("@tie: expected `@tie target = expr`")
            target_path = _extract_path(args[1].args[1])
            target_path === nothing && error("@tie: invalid target path")
            rhs                   = args[1].args[2]
            replaced_rhs, masters = _extract_and_replace_masters(rhs)
            isempty(masters) &&
                error("@tie: no `component.param` refs found in `$rhs`")
            master_optics = [optic_expr(collect(p)) for (p, _) in masters]
            argnames      = [n for (_, n) in masters]
            closure       = esc(Expr(:->, Expr(:tuple, argnames...), replaced_rhs))
            push!(spec_entries,
                  :($(optic_expr(target_path)), Tied($closure, ($(master_optics...),))))

        elseif kw == Symbol("@free")
            length(args) == 1 || error("@free: expected `@free param`")
            path = _extract_path(args[1])
            path === nothing && error("@free: invalid path `$(args[1])`")
            push!(spec_entries, :($(optic_expr(path)), Free()))

        elseif kw == Symbol("@prior")
            length(args) == 1 || error("@prior: expected `@prior param ~ distribution`")
            arg = args[1]
            (Meta.isexpr(arg, :call) && arg.args[1] === :~ && length(arg.args) == 3) ||
                error("@prior: expected `param ~ distribution`, got `$arg`")
            path = _extract_path(arg.args[2])
            path === nothing && error("@prior: invalid path `$(arg.args[2])`")
            push!(prior_entries, :($(optic_expr(path)), $(esc(arg.args[3]))))

        else
            error("@constrain: unknown keyword $kw; use @fix, @bound, @tie, @free, @prior")
        end
    end

    spec_expr = Expr(:tuple, spec_entries...)
    prior_expr = Expr(:tuple, prior_entries...)

    quote
        local $cm_var = _as_compiled($(esc(model_expr)))
        _constrain($cm_var, $spec_expr, $prior_expr)
    end
end

# Extract a tuple of symbols from a dotted path expression.
# narrow.amplitude → (:narrow, :amplitude);  plain_sym → (:plain_sym,)
# Returns nothing if the expression is not a valid path.
function _extract_path(expr)
    expr isa Symbol && return (expr,)
    Meta.isexpr(expr, :.) && expr.args[2] isa QuoteNode || return nothing
    inner = _extract_path(expr.args[1])
    inner === nothing && return nothing
    return (inner..., expr.args[2].value::Symbol)
end

# Walk the @tie RHS: replace every dotted reference (component.field) with a
# fresh gensym argument, collecting (path, argname) pairs as master entries.
function _extract_and_replace_masters(expr)
    masters  = Pair{Tuple{Vararg{Symbol}}, Symbol}[]
    replaced = _replace_masters_inner(expr, masters)
    return replaced, masters
end

function _replace_masters_inner(expr, masters)
    path = _extract_path(expr)
    # length ≥ 2: genuine component.field dotted ref (not a bare symbol like :+)
    if path !== nothing && length(path) >= 2
        argname = gensym("m")
        push!(masters, path => argname)
        return argname
    end
    expr isa Expr || return expr
    Expr(expr.head, [_replace_masters_inner(a, masters) for a in expr.args]...)
end
