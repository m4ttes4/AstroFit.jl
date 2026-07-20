# Parameter introspection over a CompiledModel: the free-slot views the optimizer needs.
# Every function walks the tree in the SAME DFS order withparams' _slotmap! assigns slots
# (left→right subtree; within a leaf, fields in order; only Free/Bounded count). Plain
# recursion — setup-time, not hot path, so no @generated. Tuples + collect (not vcat) so an
# all-Fixed/Tied leaf contributes () without poisoning the element type.

using Random: Random, AbstractRNG

_isfree(c) = c isa Free || c isa Bounded

nfree(cm::CompiledModel) = _nfree(getfield(cm, :tree))
_nfree(l::Leaf) = count(_isfree, l.constraints)
_nfree(n) = _nfree(n.left) + _nfree(n.right)

# Current free-parameter values — the p₀ the optimizer starts from. Fields no longer share
# a type, so an Int field beside Float64 ones would give a Vector{Real}: promote here to keep
# the optimizer's vector concrete. `promote()` on nothing yields Vector{Union{}}, hence the guard.
function params(cm::CompiledModel)
    vals = _params(getfield(cm, :tree))
    return isempty(vals) ? Float64[] : collect(promote(vals...))
end
_params(l::Leaf) = Tuple(getfield(l.model, i) for (i, c) in enumerate(l.constraints) if _isfree(c))
_params(n) = (_params(n.left)..., _params(n.right)...)

# Box bounds as (lower, upper): Free → (-Inf, Inf), Bounded → its (lower, upper).
function bounds(cm::CompiledModel)
    bs = _bounds(getfield(cm, :tree))
    return collect(first.(bs)), collect(last.(bs))
end
_bound(::Free) = (-Inf, Inf)
_bound(b::Bounded) = (b.lower, b.upper)
_bounds(l::Leaf) = Tuple(_bound(c) for c in l.constraints if _isfree(c))
_bounds(n) = (_bounds(n.left)..., _bounds(n.right)...)

# Slot labels for fit output: :<leafname>_<fieldname>.
paramnames(cm::CompiledModel) = collect(_pnames(getfield(cm, :tree)))
_pnames(l::Leaf{name}) where {name} =
    Tuple(
    Symbol(name, :_, fieldname(typeof(l.model), i))
        for (i, c) in enumerate(l.constraints) if _isfree(c)
)
_pnames(n) = (_pnames(n.left)..., _pnames(n.right)...)

# rand: sample free parameters uniformly within bounds.
function Random.rand(rng::AbstractRNG, cm::CompiledModel)
    lb, ub = bounds(cm)
    names = paramnames(cm)
    for i in eachindex(lb, ub)
        (isfinite(lb[i]) && isfinite(ub[i])) || throw(ArgumentError(
            "parameter `$(names[i])` has no finite bounds — set bounds with `@bound` before sampling"))
    end
    return [lb[i] + (ub[i] - lb[i]) * rand(rng) for i in eachindex(lb, ub)]
end
Random.rand(cm::CompiledModel) = rand(Random.default_rng(), cm)
