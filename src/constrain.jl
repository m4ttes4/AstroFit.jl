using Accessors: constructorof

# Constraint-application engine (design X). setconstraint is the write-analog of _nav:
# navigation *finds* a leaf, but the tree is immutable, so editing rebuilds root→leaf.
# Pure — returns a new CompiledModel, no renumber (slots are structural), priors kept.
# The verb macros (@fix/@bound/@free/@tie) lower to this; validate is the eager check.

# Swap the constraint at one leaf field. Reuses getproperty (_nav) to read+existence-check
# the leaf; _setleaf rebuilds only the spine down to it.
function setconstraint(cm::CompiledModel, leaf::Symbol, field::Symbol, c::AbstractConstraint)
    l = getproperty(cm, leaf)                       # ArgumentError if leaf missing
    i = findfirst(==(field), fieldnames(typeof(l.model)))
    i === nothing && throw(ArgumentError(
        "no field `$field` in `$leaf` ($(nameof(typeof(l.model))))"))
    newleaf = Leaf{leaf}(l.model, Base.setindex(l.constraints, c, i))
    CompiledModel(_setleaf(getfield(cm, :tree), Val(leaf), newleaf), getfield(cm, :priors))
end

_setleaf(::Leaf{n}, ::Val{n}, new) where {n} = new
_setleaf(l::Leaf, ::Val, new) = l
_setleaf(node, v::Val, new) =
    constructorof(typeof(node))(_setleaf(node.left, v, new), _setleaf(node.right, v, new))

# validate(cm): every Tied master must be a free parameter (Free/Bounded) — no chaining,
# no tie to a fixed/missing slot. Subsumes cycle detection (a tie can't reach another tie).
# Returns cm so it chains. Throws on the first offending tie.
function validate(cm::CompiledModel)
    _vnode(getfield(cm, :tree), cm)
    _validate_priors(cm)
    cm
end

_validate_priors(::CompiledModel{<:Any,Nothing}) = nothing
_vnode(n, cm) = (_vnode(n.left, cm); _vnode(n.right, cm); nothing)
function _vnode(l::Leaf{lname}, cm) where {lname}
    for (i, c) in enumerate(l.constraints)
        c isa Tied || continue
        for (mleaf, mfield) in _tiepaths(c)
            _masterfree(cm, mleaf, mfield) || throw(ArgumentError(
                "tie on `$lname.$(fieldname(typeof(l.model), i))` references " *
                "`$mleaf.$mfield`, which is not a free parameter (it must exist and be Free or Bounded)"))
        end
    end
    nothing
end

_tiepaths(::Tied{Paths}) where {Paths} = Paths

# Is (leaf, field) a free slot? Missing leaf/field → not free (ponytail: one boolean, the
# caller's message covers both "missing" and "fixed").
function _masterfree(cm, leaf::Symbol, field::Symbol)
    l = _nav(getfield(cm, :tree), Val(leaf))
    l === nothing && return false
    i = findfirst(==(field), fieldnames(typeof(l.model)))
    i === nothing ? false : _isfree(l.constraints[i])
end
