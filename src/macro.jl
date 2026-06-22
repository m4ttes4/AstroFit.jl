# @model begin
#     g1 = Gaussian1D(...)        # leaf: name = model expression
#     g2 = Gaussian1D(...)
#     g1 + g2                     # the composition (one trailing expression)
# end
#
# Each `name = expr` binds `name` to a Leaf{:name}(model, all-Free constraints). The
# composition is left untouched: the compound operators evaluate it and, since
# Leaf <: AbstractModel, build the annotated tree directly — the macro never parses it.
# Constraints stay all-Free here; @constrain edits them later.
macro model(blk)
    blk isa Expr && blk.head === :block || error("@model expects a begin…end block")
    defs = Any[]
    final = nothing
    for s in blk.args
        s isa LineNumberNode && continue
        if s isa Expr && s.head === :(=)
            name, mexpr = s.args[1], s.args[2]
            name isa Symbol || error("@model: leaf name must be a symbol, got `$name`")
            name in (:tree, :priors) && error("@model: `$name` is a reserved CompiledModel field name")
            m = gensym(name)
            # Leaf names stay unescaped: macro hygiene renames them to gensyms,
            # consistently in the binding and in `final`, so they don't leak into the
            # caller. Only the model expression is escaped (it may use caller variables).
            push!(defs, :(local $m = $(esc(mexpr))))
            push!(defs, :($name = Leaf{$(QuoteNode(name))}($m, _defaults($m))))
        else
            final === nothing || error("@model: expected one composition expression, got also `$s`")
            final = s
        end
    end
    final === nothing && error("@model: missing composition expression")
    return quote
        $(defs...)
        _compiled($final)
    end
end

# Default constraints for a fresh leaf: every field free.
_defaults(m) = ntuple(_ -> Free(), fieldcount(typeof(m)))

# Wrap the tree, rejecting a leaf used more than once — its (name, field) slots would
# collide in withparams. Sharing a value across components is Tied's job, not aliasing.
function _compiled(tree)
    names = Symbol[]
    _leafnames!(names, tree)
    allunique(names) || error(
        "@model: leaf(s) used more than once: " *
            join(unique(n for n in names if count(==(n), names) > 1), ", ")
    )
    return CompiledModel(tree, nothing)
end

_leafnames!(acc, ::Leaf{name}) where {name} = (push!(acc, name); acc)
_leafnames!(acc, m) = (_leafnames!(acc, m.left); _leafnames!(acc, m.right); acc)

# ---------------------------------------------------------------------------
# Helpers shared by standalone macros and @constrain block.
# ---------------------------------------------------------------------------

# Split a `model.leaf.field` access → (model_expr, :leaf, :field).
function _splitpath(e)
    e isa Expr && e.head === :. && e.args[1] isa Expr && e.args[1].head === :. ||
        error("expected `model.leaf.field`, got `$e`")
    return (e.args[1].args[1], e.args[1].args[2].value, e.args[2].value)
end

# @tie RHS → (paths_expr, lambda). Every `root.leaf.field` becomes a fresh arg (masters
# collected in order); all other code is left intact.
function _tiewalk(rhs, root)
    paths = Tuple{Symbol, Symbol}[]; args = Symbol[]
    walk(x) =
    if x isa Expr && x.head === :. && x.args[1] isa Expr &&
            x.args[1].head === :. && x.args[1].args[1] == root
        (_, l, f) = _splitpath(x); g = gensym(); push!(paths, (l, f)); push!(args, g); g
    elseif x isa Expr
        Expr(x.head, map(walk, x.args)...)
    else
        x
    end
    newrhs = walk(rhs)
    pe = Expr(:tuple, (Expr(:tuple, QuoteNode(l), QuoteNode(f)) for (l, f) in paths)...)
    return (pe, Expr(:->, Expr(:tuple, args...), newrhs))
end

_setexpr(root, l, f, c) = :(setconstraint($root, $(QuoteNode(l)), $(QuoteNode(f)), $c))

# Expression for Fixed(current_value) — reads the field from the model at macro-expansion time.
_fixcurrent(root, l, f) = :(Fixed(getfield(getproperty($root, $(QuoteNode(l))).model, $(QuoteNode(f)))))

# ---------------------------------------------------------------------------
# Standalone constraint macros. Each auto-rebinds the model variable:
#   @fix m.leaf.field = value   →   m = setconstraint(m, :leaf, :field, Fixed(value))
# ---------------------------------------------------------------------------

macro fix(a)
    return if a isa Expr && a.head === :(=)
        (r, l, f) = _splitpath(a.args[1])
        r isa Symbol || error("nested paths require @constrain block")
        c = :(Fixed($(esc(a.args[2]))))
        :($(esc(r)) = $(_setexpr(esc(r), l, f, c)))
    elseif a isa Expr && a.head === :.
        (r, l, f) = _splitpath(a)
        r isa Symbol || error("nested paths require @constrain block")
        :($(esc(r)) = $(_setexpr(esc(r), l, f, _fixcurrent(esc(r), l, f))))
    else
        error("@fix expects `model.leaf.field` or `model.leaf.field = value`")
    end
end

macro tie(a)
    a isa Expr && a.head === :-> || error("@tie expects `model.leaf.field -> expression`")
    lhs = a.args[1]
    rhs_block = a.args[2]
    rhs = rhs_block isa Expr && rhs_block.head === :block ? rhs_block.args[end] : rhs_block
    (r, l, f) = _splitpath(lhs)
    r isa Symbol || error("nested paths require @constrain block")
    (pe, lam) = _tiewalk(rhs, r)
    c = :(Tied($pe, $(esc(lam))))
    return :($(esc(r)) = $(_setexpr(esc(r), l, f, c)))
end

macro bound(a)
    a isa Expr && a.head === :call && a.args[1] === :in &&
        length(a.args) == 3 && a.args[3] isa Expr && a.args[3].head === :tuple &&
        length(a.args[3].args) == 2 ||
        error("@bound expects `model.leaf.field in (lo, hi)`")
    (r, l, f) = _splitpath(a.args[2])
    r isa Symbol || error("nested paths require @constrain block")
    lo, hi = a.args[3].args
    c = :(Bounded($(esc(lo)), $(esc(hi))))
    return :($(esc(r)) = $(_setexpr(esc(r), l, f, c)))
end

macro free(p)
    (r, l, f) = _splitpath(p)
    r isa Symbol || error("nested paths require @constrain block")
    return :($(esc(r)) = $(_setexpr(esc(r), l, f, :(Free()))))
end

macro prior(a)
    a isa Expr && a.head === :call && a.args[1] === :~ && length(a.args) == 3 ||
        error("@prior expects `model.leaf.field ~ distribution`")
    (r, l, f) = _splitpath(a.args[2])
    r isa Symbol || error("nested paths require @constrain block")
    return :($(esc(r)) = setprior($(esc(r)), $(QuoteNode(l)), $(QuoteNode(f)), $(esc(a.args[3]))))
end

# ---------------------------------------------------------------------------
# @constrain block — syntactic sugar with auto-rebind.
#
#   @constrain m begin
#       narrow.amplitude              # fix at current value
#       narrow.amplitude = 1.0        # fix at value
#       broad.mean -> narrow.mean     # tie
#       narrow.mean in (-1, 1)        # bound
#       narrow.mean ~ Normal(0, 1)    # prior
#       @free narrow.mean             # free
#   end
# ---------------------------------------------------------------------------

# Prefix bare leaf.field with the gensym root: narrow.amplitude → g.narrow.amplitude.
# ponytail: also catches dotted caller data (`= point.x`) — snapshot to a local first.
_inject(root, e) =
if e isa Expr && e.head === :. && e.args[1] isa Symbol && e.args[2] isa QuoteNode
    Expr(:., Expr(:., root, QuoteNode(e.args[1])), e.args[2])
elseif e isa Expr
    Expr(e.head, (_inject(root, a) for a in e.args)...)
else
    e
end

macro constrain(cm, blk)
    blk isa Expr && blk.head === :block || error("@constrain expects a begin…end block")
    cm isa Symbol || error("@constrain: first argument must be a variable name")
    g = gensym(:cm); seen = Set{Tuple{Symbol, Symbol}}()
    out = Any[:($g = $(esc(cm)))]
    for s in blk.args
        s isa LineNumberNode && continue
        s = _inject(g, s)

        if s isa Expr && s.head === :.
            (_, l, f) = _splitpath(s)
            _checkdup!(seen, l, f)
            push!(out, :($g = $(_setexpr(g, l, f, _fixcurrent(g, l, f)))))

        elseif s isa Expr && s.head === :(=)
            (_, l, f) = _splitpath(s.args[1])
            _checkdup!(seen, l, f)
            push!(out, :($g = $(_setexpr(g, l, f, :(Fixed($(esc(s.args[2]))))))))

        elseif s isa Expr && s.head === :->
            lhs = s.args[1]
            rhs_block = s.args[2]
            rhs = rhs_block isa Expr && rhs_block.head === :block ? rhs_block.args[end] : rhs_block
            (_, l, f) = _splitpath(lhs)
            (pe, lam) = _tiewalk(rhs, g)
            _checkdup!(seen, l, f)
            push!(out, :($g = $(_setexpr(g, l, f, :(Tied($pe, $(esc(lam))))))))

        elseif s isa Expr && s.head === :call && length(s.args) >= 3 && s.args[1] === :in
            tup = s.args[3]
            tup isa Expr && tup.head === :tuple && length(tup.args) == 2 ||
                error("@constrain: `in` expects `path in (lo, hi)`")
            (_, l, f) = _splitpath(s.args[2])
            lo, hi = tup.args
            _checkdup!(seen, l, f)
            push!(out, :($g = $(_setexpr(g, l, f, :(Bounded($(esc(lo)), $(esc(hi))))))))

        elseif s isa Expr && s.head === :call && length(s.args) == 3 && s.args[1] === :~
            (_, l, f) = _splitpath(s.args[2])
            push!(out, :($g = setprior($g, $(QuoteNode(l)), $(QuoteNode(f)), $(esc(s.args[3])))))

        elseif s isa Expr && s.head === :macrocall && s.args[1] === Symbol("@free")
            fargs = [x for x in s.args[3:end] if !(x isa LineNumberNode)]
            length(fargs) == 1 || error("@free expects one path")
            (_, l, f) = _splitpath(fargs[1])
            _checkdup!(seen, l, f)
            push!(out, :($g = $(_setexpr(g, l, f, :(Free())))))

        else
            error("@constrain: unrecognized expression `$s`")
        end
    end
    push!(out, :($(esc(cm)) = validate($g)))
    return Expr(:block, out...)
end

function _checkdup!(seen, l, f)
    (l, f) in seen && error("@constrain: `$l.$f` constrained twice")
    return push!(seen, (l, f))
end
