"""
    @model begin
        name₁ = ModelExpr(...)
        name₂ = ModelExpr(...)
        name₁ + name₂
    end

Build a [`CompiledModel`](@ref) from named leaf models and a composition expression.

Each `name = expr` line creates a [`Leaf`](@ref) with all-[`Free`](@ref) constraints.
The final expression (e.g. `name₁ + name₂`) defines how leaves combine; compound
operators (`+`, `*`, `|`) build the tree. Constraints are set afterwards via
[`@constrain`](@ref) or the standalone macros ([`@fix`](@ref), [`@bound`](@ref),
[`@tie`](@ref), [`@free`](@ref), [`@prior`](@ref)).

# Examples
```julia
m = @model begin
    g1 = Gaussian1D(amplitude=1.0, mean=0.0, stddev=1.0)
    g2 = Gaussian1D(amplitude=0.5, mean=3.0, stddev=0.8)
    g1 + g2
end
```

See also: [`@constrain`](@ref), [`CompiledModel`](@ref)
"""
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

"""
    _defaults(m) -> NTuple{N, Free}

Return a tuple of [`Free`](@ref) constraints, one per field of `m`.
"""
_defaults(m) = ntuple(_ -> Free(), fieldcount(typeof(m)))

"""
    _compiled(tree) -> CompiledModel

Wrap `tree` in a [`CompiledModel`](@ref), rejecting duplicate leaf names.

Duplicate leaves would collide in [`withparams`](@ref); use [`Tied`](@ref) to share
values across components instead.
"""
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

"""
    @fix model.leaf.field = value
    @fix model.leaf.field

Fix a parameter to `value`, or to its current value if no `=` is given.

# Examples
```julia
@fix m.g1.amplitude = 1.0   # fix to explicit value
@fix m.g1.amplitude          # fix at current value
```

See also: [`@free`](@ref), [`@bound`](@ref), [`@tie`](@ref), [`@constrain`](@ref)
"""
macro fix(a)
    return if a isa Expr && a.head === :(=)
        (r, l, f) = _splitpath(a.args[1])
        r isa Symbol || error("nested paths require @constrain block")
        c = :(Fixed($(esc(a.args[2]))))
        :($(esc(r)) = validate($(_setexpr(esc(r), l, f, c))))
    elseif a isa Expr && a.head === :.
        (r, l, f) = _splitpath(a)
        r isa Symbol || error("nested paths require @constrain block")
        :($(esc(r)) = validate($(_setexpr(esc(r), l, f, _fixcurrent(esc(r), l, f)))))
    else
        error("@fix expects `model.leaf.field` or `model.leaf.field = value`")
    end
end

"""
    @tie model.leaf.field -> expression

Tie a parameter to an expression of other model parameters.

The right-hand side may reference other `model.leaf.field` paths, which become
the master parameters. The tied parameter is computed from them via the given
expression and consumes no free-parameter slot.

# Examples
```julia
@tie m.g2.mean -> m.g1.mean + 0.5
@tie m.g2.stddev -> 2 * m.g1.stddev
```

See also: [`Tied`](@ref), [`@fix`](@ref), [`@constrain`](@ref)
"""
macro tie(a)
    a isa Expr && a.head === :-> || error("@tie expects `model.leaf.field -> expression`")
    lhs = a.args[1]
    rhs_block = a.args[2]
    rhs = rhs_block isa Expr && rhs_block.head === :block ? rhs_block.args[end] : rhs_block
    (r, l, f) = _splitpath(lhs)
    r isa Symbol || error("nested paths require @constrain block")
    (pe, lam) = _tiewalk(rhs, r)
    c = :(Tied($pe, $(esc(lam))))
    return :($(esc(r)) = validate($(_setexpr(esc(r), l, f, c))))
end

"""
    @bound model.leaf.field in (lower, upper)

Constrain a parameter to the interval `[lower, upper]`.

# Examples
```julia
@bound m.g1.amplitude in (0.0, 10.0)
@bound m.g1.mean in (-5.0, 5.0)
```

See also: [`Bounded`](@ref), [`@fix`](@ref), [`@free`](@ref), [`@constrain`](@ref)
"""
macro bound(a)
    a isa Expr && a.head === :call && a.args[1] === :in &&
        length(a.args) == 3 && a.args[3] isa Expr && a.args[3].head === :tuple &&
        length(a.args[3].args) == 2 ||
        error("@bound expects `model.leaf.field in (lo, hi)`")
    (r, l, f) = _splitpath(a.args[2])
    r isa Symbol || error("nested paths require @constrain block")
    lo, hi = a.args[3].args
    c = :(Bounded($(esc(lo)), $(esc(hi))))
    return :($(esc(r)) = validate($(_setexpr(esc(r), l, f, c))))
end

"""
    @free model.leaf.field

Remove any constraint on a parameter, making it free again.

# Examples
```julia
@free m.g1.amplitude
```

See also: [`Free`](@ref), [`@fix`](@ref), [`@bound`](@ref), [`@constrain`](@ref)
"""
macro free(p)
    (r, l, f) = _splitpath(p)
    r isa Symbol || error("nested paths require @constrain block")
    return :($(esc(r)) = validate($(_setexpr(esc(r), l, f, :(Free())))))
end

"""
    @prior model.leaf.field ~ distribution

Attach a prior distribution to a parameter.

# Examples
```julia
using Distributions
@prior m.g1.amplitude ~ Normal(1.0, 0.5)
@prior m.g1.mean ~ Uniform(-5.0, 5.0)
```

See also: [`setprior`](@ref), [`logprior`](@ref), [`@constrain`](@ref)
"""
macro prior(a)
    a isa Expr && a.head === :call && a.args[1] === :~ && length(a.args) == 3 ||
        error("@prior expects `model.leaf.field ~ distribution`")
    (r, l, f) = _splitpath(a.args[2])
    r isa Symbol || error("nested paths require @constrain block")
    return :($(esc(r)) = validate(setprior($(esc(r)), $(QuoteNode(l)), $(QuoteNode(f)), $(esc(a.args[3])))))
end

"""
    @constrain model begin
        leaf.field                    # fix at current value
        leaf.field = value            # fix at explicit value
        leaf.field -> expression      # tie to other parameters
        leaf.field in (lo, hi)        # bound to interval
        leaf.field ~ distribution     # attach prior
        @free leaf.field              # unconstrain
    end

Apply multiple constraints to a [`CompiledModel`](@ref) in a single block.

Paths are written as `leaf.field` (without the model prefix). Each constraint form
mirrors its standalone macro ([`@fix`](@ref), [`@tie`](@ref), [`@bound`](@ref),
[`@prior`](@ref), [`@free`](@ref)). A given `leaf.field` may only appear once in the
block (except for priors, which are orthogonal to constraints). The model variable is
automatically rebound after validation.

# Examples
```julia
@constrain m begin
    g1.amplitude = 1.0
    g1.mean in (-5.0, 5.0)
    g2.mean -> m.g1.mean + 0.5
    g2.stddev ~ Normal(1.0, 0.2)
    @free g1.stddev
end
```

See also: [`@model`](@ref), [`validate`](@ref)
"""
:(@constrain)

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
