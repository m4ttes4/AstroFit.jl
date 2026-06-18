struct CompiledModel{M, S<:NamedTuple, P}
    model::M
    spec::S
    priors::P
end

CompiledModel(model, spec) = CompiledModel(model, spec, ())

# ---------------------------------------------------------------------------
# Property access: cm.g1 → submodel, cm.g1.amplitude → value
# ---------------------------------------------------------------------------

function Base.getproperty(cm::CompiledModel, s::Symbol)
    s in (:model, :spec, :priors) && return getfield(cm, s)
    spec = getfield(cm, :spec)
    model = getfield(cm, :model)
    idx = findfirst(==(s), keys(spec))
    idx === nothing && throw(ArgumentError("no component `$s`; available: $(keys(spec))"))
    getfield(model, fieldnames(typeof(model))[idx])
end

Base.propertynames(cm::CompiledModel) =
    (:model, :spec, :priors, keys(getfield(cm, :spec))...)

# ---------------------------------------------------------------------------
# Render: delegate to the tree
# ---------------------------------------------------------------------------

render(cm::CompiledModel, x...) = render(getfield(cm, :model), x...)

# ---------------------------------------------------------------------------
# withparams — @generated rebuild from flat parameter vector
# ---------------------------------------------------------------------------

# Compile-time helper: is this model type a compound (has AbstractModel children)?
_iscompound(M::Type) = any(F -> F <: AbstractModel, fieldtypes(M))

# Compile-time helper: emit the expression to rebuild a leaf model from its spec.
#
#   _emit_leaf(Gaussian1D{Float64}, :(spec.g1), :p)
#   → :(Gaussian1D(resolve(spec.g1[1], p), resolve(spec.g1[2], p), resolve(spec.g1[3], p)))
#
function _emit_leaf(LeafType, spec_expr, p)
    nfields = length(fieldnames(LeafType))
    args = [:(resolve($spec_expr[$i], $p)) for i in 1:nfields]
    :($(nameof(LeafType))($(args...)))
end

# Compile-time helper: emit the expression to rebuild a compound or leaf,
# walking M (model type) and S (spec type) in parallel.
#
# For our example Sum{Gaussian1D, Gaussian1D} with spec (g1=(...), g2=(...)):
#
#   _emit_rebuild(Sum{...}, NamedTuple{(:g1,:g2),...}, :spec, :p)
#   → quote
#       child1 = Gaussian1D(resolve(spec.g1[1], p), ...)
#       child2 = Gaussian1D(resolve(spec.g2[1], p), ...)
#       Sum(child1, child2)
#     end
#
function _emit_rebuild(M, S, spec_expr, p)
    if !_iscompound(M)
        return _emit_leaf(M, spec_expr, p)
    end

    spec_keys = fieldnames(S)
    tree_fields = fieldnames(M)
    child_types = fieldtypes(M)

    block = Expr(:block)
    child_vars = Symbol[]
    for (i, (child_T, spec_key)) in enumerate(zip(child_types, spec_keys))
        var = Symbol(:child, i)
        push!(child_vars, var)
        child_spec = :($spec_expr.$spec_key)
        child_S = fieldtype(S, i)
        push!(block.args, :($var = $(_emit_rebuild(child_T, child_S, child_spec, p))))
    end
    push!(block.args, :($(nameof(M))($(child_vars...))))
    block
end

# ---------------------------------------------------------------------------
# Utilities — spec traversal (not hot path)
# ---------------------------------------------------------------------------

_isfree(::Free) = true
_isfree(::Bounded) = true
_isfree(::AbstractConstraint) = false

_free_index(::Free{I}) where I = I
_free_index(::Bounded{I}) where I = I

# Walk a (possibly nested) spec, calling f(constraint) on each leaf constraint.
function _foreach_constraint(f, spec::NamedTuple)
    for v in values(spec)
        _foreach_constraint(f, v)
    end
end
function _foreach_constraint(f, constraints::Tuple)
    for c in constraints
        f(c)
    end
end

nfree(cm::CompiledModel) = nfree(getfield(cm, :spec))
function nfree(spec::NamedTuple)
    n = 0
    _foreach_constraint(c -> _isfree(c) && (n += 1), spec)
    n
end

function freevals(cm::CompiledModel)
    spec = getfield(cm, :spec)
    model = getfield(cm, :model)
    pairs = Tuple{Int,Float64}[]
    _foreach_spec_with_model(spec, model) do c, val
        _isfree(c) && push!(pairs, (_free_index(c), Float64(val)))
    end
    sort!(pairs; by=first)
    Tuple(v for (_, v) in pairs)
end

function _foreach_spec_with_model(f, spec::NamedTuple, model)
    for (i, v) in enumerate(values(spec))
        child = getfield(model, fieldnames(typeof(model))[i])
        _foreach_spec_with_model(f, v, child)
    end
end
function _foreach_spec_with_model(f, constraints::Tuple, model)
    for (i, c) in enumerate(constraints)
        f(c, getfield(model, fieldnames(typeof(model))[i]))
    end
end

paramvector(cm::CompiledModel) = collect(Float64, freevals(cm))

function bounds_vectors(cm::CompiledModel)
    pairs = Tuple{Int,Float64,Float64}[]
    _foreach_constraint(getfield(cm, :spec)) do c
        if c isa Bounded
            push!(pairs, (_free_index(c), c.lower, c.upper))
        elseif c isa Free
            push!(pairs, (_free_index(c), -Inf, Inf))
        end
    end
    sort!(pairs; by=first)
    (Float64[lo for (_, lo, _) in pairs], Float64[hi for (_, _, hi) in pairs])
end

# ---------------------------------------------------------------------------
# withparams — @generated rebuild from flat parameter vector
# ---------------------------------------------------------------------------

@generated function withparams(cm::CompiledModel{M,S,P}, p) where {M,S,P}
    spec_expr = :(getfield(cm, :spec))
    rebuild = _emit_rebuild(M, S, spec_expr, :p)
    quote
        spec = getfield(cm, :spec)
        tree = $rebuild
        CompiledModel(tree, spec, getfield(cm, :priors))
    end
end
