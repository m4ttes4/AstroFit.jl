# Leaf + CompiledModel structure, plus getproperty navigation. withparams lives in
# withparams.jl; the setconstraint/validate edit engine in constrain.jl.

# A named leaf in the annotated tree. `name` (the user symbol) lives in the type so
# navigation resolves from the type; `constraints` is a Tuple positional to `model`'s
# fields. Being an AbstractModel, the compound operators (Sum, …) compose leaves directly.
struct Leaf{name, M, C} <: AbstractModel
    model::M
    constraints::C
end
Leaf{name}(model::M, constraints::C) where {name, M, C} = Leaf{name, M, C}(model, constraints)

@inline render(l::Leaf, x::Number...) = render(l.model, x...)
render!(out::AbstractArray, l::Leaf, x...) = render!(out, l.model, x...)

# A single annotated model tree (compound nodes + Leaf leaves) plus priors.
struct CompiledModel{T, P}
    tree::T
    priors::P
end
render(cm::CompiledModel, x...) = render(getfield(cm, :tree), x...)
render!(out::AbstractArray, cm::CompiledModel, x...) = render!(out, getfield(cm, :tree), x...)

# Navigation: cm.g1 returns the Leaf tagged :g1. Plain dispatch recursion — `name` is in
# the type (Val{name}/Leaf{name}), so each leaf resolves statically to the leaf or
# `nothing`, and `h === nothing` folds at compile time. No @generated needed.
_nav(l::Leaf{name}, ::Val{name}) where {name} = l
_nav(::Leaf, ::Val) = nothing
_nav(n, v::Val) = (h = _nav(n.left, v); h === nothing ? _nav(n.right, v) : h)

function Base.getproperty(cm::CompiledModel, name::Symbol)
    return name === :tree || name === :priors ? getfield(cm, name) :
        let leaf = _nav(getfield(cm, :tree), Val(name))
            leaf === nothing && throw(ArgumentError("no component `$name` in model"))
            leaf
    end
end
