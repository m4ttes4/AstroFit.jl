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
    quote
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
    allunique(names) || error("@model: leaf(s) used more than once: " *
        join(unique(n for n in names if count(==(n), names) > 1), ", "))
    CompiledModel(tree, nothing)
end

_leafnames!(acc, l::Leaf{name}) where {name} = (push!(acc, name); acc)
_leafnames!(acc, m) = (_leafnames!(acc, m.left); _leafnames!(acc, m.right); acc)

# ---------------------------------------------------------------------------
# Constraint verbs + @constrain block. All lower to the setconstraint/validate
# engine (constrain.jl). The verbs are pure (return a new CompiledModel, user
# rebinds); @constrain threads them and adds the two block-only checks.
# ---------------------------------------------------------------------------

# Split a `model.leaf.field` access → (model_expr, :leaf, :field). Field names are
# QuoteNode-wrapped, so the path is (. (. model (quote leaf)) (quote field)).
function _splitpath(e)
    e isa Expr && e.head === :. && e.args[1] isa Expr && e.args[1].head === :. ||
        error("expected `model.leaf.field`, got `$e`")
    (e.args[1].args[1], e.args[1].args[2].value, e.args[2].value)
end

# @tie RHS → (paths_expr, lambda). Every `root.leaf.field` becomes a fresh arg (masters
# collected in order); all other code is left intact. ponytail: non-path caller variables
# are captured LIVE by the closure — use a literal or `const` coefficient, since a captured
# reassigned variable both tracks later changes and boxes (breaking withparams' zero-alloc).
function _tiewalk(rhs, root)
    paths = Tuple{Symbol,Symbol}[]; args = Symbol[]
    walk(x) =
        if x isa Expr && x.head === :. && x.args[1] isa Expr &&
           x.args[1].head === :. && x.args[1].args[1] == root
            (_, l, f) = _splitpath(x); g = gensym(); push!(paths, (l, f)); push!(args, g); g
        elseif x isa Expr
            Expr(x.head, map(walk, x.args)...)
        else; x end
    newrhs = walk(rhs)                      # walk first, then read the collected paths/args
    pe = Expr(:tuple, (Expr(:tuple, QuoteNode(l), QuoteNode(f)) for (l, f) in paths)...)
    (pe, Expr(:->, Expr(:tuple, args...), newrhs))
end

# The single verb-name → (root, leaf, field, constraint_expr) map. Both the standalone
# verb macros and the @constrain block share this, so the mapping lives in one place.
function _verb(name, args)
    if name === Symbol("@fix")
        (r, l, f) = _splitpath(args[1].args[1]); (r, l, f, :(Fixed($(esc(args[1].args[2])))))
    elseif name === Symbol("@bound")
        (r, l, f) = _splitpath(args[1]); (r, l, f, :(Bounded($(esc(args[2])), $(esc(args[3])))))
    elseif name === Symbol("@free")
        (r, l, f) = _splitpath(args[1]); (r, l, f, :(Free()))
    elseif name === Symbol("@tie")
        (r, l, f) = _splitpath(args[1].args[1])
        (pe, lam) = _tiewalk(args[1].args[2], r); (r, l, f, :(Tied($pe, $(esc(lam)))))
    else
        error("@constrain: unknown verb `$name`")
    end
end

_setexpr(root, l, f, c) = :(setconstraint($root, $(QuoteNode(l)), $(QuoteNode(f)), $c))

macro fix(a)
    a isa Expr && a.head === :(=) || error("@fix expects `path = value`")
    (r, l, f, c) = _verb(Symbol("@fix"), (a,)); _setexpr(esc(r), l, f, c)
end
macro bound(p, lo, hi)
    (r, l, f, c) = _verb(Symbol("@bound"), (p, lo, hi)); _setexpr(esc(r), l, f, c)
end
macro free(p)
    (r, l, f, c) = _verb(Symbol("@free"), (p,)); _setexpr(esc(r), l, f, c)
end
macro tie(a)
    a isa Expr && a.head === :(=) || error("@tie expects `path = rhs`")
    (r, l, f, c) = _verb(Symbol("@tie"), (a,)); _setexpr(esc(r), l, f, c)
end
macro prior(a)
    a isa Expr && a.head === :call && a.args[1] === :~ && length(a.args) == 3 ||
        error("@prior expects `model.leaf.field ~ distribution`")
    (r, l, f) = _splitpath(a.args[2])
    :(setprior($(esc(r)), $(QuoteNode(l)), $(QuoteNode(f)), $(esc(a.args[3]))))
end

# Prefix the block's gensym root onto bare paths: `g1.field` → `g.g1.field`, so the bare
# in-block form reduces to the deep form one splitter handles. ponytail: this also catches
# dotted caller data (`= point.x`) as a path — snapshot such values to a local first.
_inject(root, e) =
    if e isa Expr && e.head === :. && e.args[1] isa Symbol && e.args[2] isa QuoteNode
        Expr(:., Expr(:., root, QuoteNode(e.args[1])), e.args[2])
    elseif e isa Expr
        Expr(e.head, (_inject(root, a) for a in e.args)...)
    else; e end

# @constrain cm begin @fix g1.x = 5; @tie g2.y = 0.3*g1.x; … end
# Implicit model root (bare leaf names), threads the verbs, rejects a duplicate target at
# expansion time, and runs validate(cm) at the end (catches a tie broken by a later edit).
macro constrain(cm, blk)
    blk isa Expr && blk.head === :block || error("@constrain expects a begin…end block")
    g = gensym(:cm); seen = Set{Tuple{Symbol,Symbol}}()
    out = Any[:($g = $(esc(cm)))]
    for s in blk.args
        s isa LineNumberNode && continue
        s isa Expr && s.head === :macrocall || error("@constrain: expected a verb, got `$s`")
        args = Tuple(_inject(g, x) for x in s.args[3:end] if !(x isa LineNumberNode))
        (_, l, f, c) = _verb(s.args[1], args)
        (l, f) in seen && error("@constrain: `$l.$f` constrained twice")
        push!(seen, (l, f))
        push!(out, :($g = $(_setexpr(g, l, f, c))))
    end
    push!(out, :(validate($g)))
    Expr(:block, out...)
end
