using ConstructionBase: constructorof

"""
    _slotmap!(map, T, counter) -> Dict{Tuple{Symbol,Symbol}, Int}

Walk the tree *type* `T` and assign each [`Free`](@ref) or [`Bounded`](@ref) field a
positional index in the flat parameter vector `p`. The mapping is keyed by
`(leaf_name, field_name)` and filled in left-to-right tree order.

This runs at compile time inside the `@generated withparams`.
"""
function _slotmap!(map, T, counter)
    if T <: Leaf
        name, M, C = T.parameters
        for (fname, Ci) in zip(fieldnames(M), fieldtypes(C))
            if Ci <: Free || Ci <: Bounded
                counter[] += 1
                map[(name, fname)] = counter[]
            end
        end
    else  # compound node: left, right
        L, R = T.parameters
        _slotmap!(map, L, counter)
        _slotmap!(map, R, counter)
    end
    return map
end

"""
    _fieldexpr(Ci, name, fname, i, acc, slots) -> Expr | Symbol

Return the expression that computes one leaf field's value at runtime, depending on
its constraint type:

- [`Free`](@ref) / [`Bounded`](@ref): references the pre-loaded local `_pN` for this
  field's slot in the parameter vector.
- [`Fixed`](@ref): reads the constant from the persistent tree (`acc.constraints[i].value`).
- [`Tied`](@ref): calls the tie function with the master parameters' locals as arguments.
"""
function _fieldexpr(Ci, name, fname, i, acc, slots)
    return if Ci <: Free || Ci <: Bounded
        Symbol("_p", slots[(name, fname)])
    elseif Ci <: Fixed
        :(($acc).constraints[$i].value)
    elseif Ci <: Tied
        args = (Symbol("_p", slots[path]) for path in Ci.parameters[1])
        :(($acc).constraints[$i].f($(args...)))
    else
        error("withparams: unknown constraint $Ci")
    end
end

"""
    _ctorexpr(M, fields) -> Expr

Build the constructor call that rebuilds one leaf model of type `M` from the
per-field expressions `fields`.

Fields sharing a type parameter are `promote`d together, so a dual number in one
slot lifts its siblings — without it a `Gaussian1D{T<:Real}` holding one `Dual`
and two `Float64`s could not be constructed, which is what makes ForwardDiff
work.

The grouping is by *which* type parameter, read from the struct declaration at
compile time, and every other field is passed through untouched. Both halves
matter once a model holds something other than plain numbers — a measured
instrumental PSF is the motivating case:

```julia
struct BigPSF{T <: Real, V <: AbstractVector} <: AbstractKernel
    sigma::T          # ─┐ promoted together: they must agree
    scale::T          # ─┘
    taps::V           # its own parameter — promoting it against sigma would throw
    halfwidth::Int    # concrete — promoting it to Float64 breaks the constructor
    edge::Symbol      # nothing to promote at all
end
```

Promoting everything together throws
`promotion of types Vector{Float64} and Float64 failed to change any arguments`;
promoting every `<:Number` field turns `halfwidth` into a `Float64` (and, under
AD, into a `Dual`) and the constructor no longer matches.

When one group covers every field — every model in the zoo — the emitted
expression is exactly the `promote(...)` call this function replaced, so the
common path costs nothing.
"""
function _ctorexpr(M, fields)
    ctor = constructorof(M)

    # Declared field types, from the generic struct rather than this instance:
    # `Gaussian1D{Float64}` says `Float64`, `Gaussian1D` says `T`, and only the
    # latter tells us which fields are required to agree.
    decl = Base.unwrap_unionall(M.name.wrapper).types
    groups = Dict{Any, Vector{Int}}()
    for (i, F) in enumerate(decl)
        F isa TypeVar && push!(get!(groups, F, Int[]), i)
    end
    linked = [g for g in values(groups) if length(g) > 1]

    isempty(linked) && return :($ctor($(fields...)))
    length(linked) == 1 && length(linked[1]) == fieldcount(M) &&
        return :($ctor(promote($(fields...))...))

    args = copy(fields)
    pre = Expr[]
    for g in linked
        pr = gensym(:promoted)
        push!(pre, :($pr = promote($(fields[g]...))))
        for (k, i) in enumerate(g)
            args[i] = :($pr[$k])
        end
    end
    return Expr(:block, pre..., :($ctor($(args...))))
end

"""
    _treeexpr(T, acc, slots) -> Expr

Generate the expression that reconstructs one subtree from parameter values.

For a [`Leaf`](@ref), emits a constructor call with each field produced by
[`_fieldexpr`](@ref). For a compound node, recurses into `.left` and `.right` and
wraps them in the original compound-node constructor.

The result is an *annotated* tree: each [`Leaf`](@ref) is rebuilt with the new
model values while its `constraints` tuple is carried over unchanged from the
persistent tree, so the rebuilt tree stays navigable and re-fittable.

`acc` is the expression that reaches this node in the persistent runtime tree;
it is needed to read [`Fixed`](@ref) values, [`Tied`](@ref) functions, and the
leaf `constraints` carried into the rebuilt tree.
"""
function _treeexpr(T, acc, slots)
    return if T <: Leaf
        name, M, C = T.parameters
        fields = [
            _fieldexpr(fieldtypes(C)[i], name, fieldnames(M)[i], i, acc, slots)
                for i in 1:fieldcount(M)
        ]
        :(Leaf{$(QuoteNode(name))}($(_ctorexpr(M, fields)), ($acc).constraints))
    else
        L, R = T.parameters
        :(
            $(T.name.wrapper)(
                $(_treeexpr(L, :(($acc).left), slots)),
                $(_treeexpr(R, :(($acc).right), slots))
            )
        )
    end
end

"""
    withparams(cm::CompiledModel, p) -> CompiledModel

Rebuild the model tree stored in `cm` using parameter values from the flat vector `p`.

Each element of `p` maps to one [`Free`](@ref) or [`Bounded`](@ref) parameter, assigned
in left-to-right tree order. [`Fixed`](@ref) parameters keep their stored value;
[`Tied`](@ref) parameters are computed from their master parameters via the stored
function.

The result is a new [`CompiledModel`](@ref) wrapping the rebuilt annotated tree
(constraints and priors carried over from `cm`), so it is navigable (`result.leaf`),
renderable and re-fittable just like `cm`. This function is `@generated`: the
slot-to-field mapping is resolved at compile time from the tree's type, so the runtime
cost is a straight-line sequence of loads and constructor calls with no dynamic dispatch.

# Arguments
- `cm::CompiledModel`: compiled model containing the annotated tree and constraints
- `p`: flat parameter vector of length [`nfree(cm)`](@ref)

# Examples
```julia
m = @model begin
    g = Gaussian1D(amplitude=1.0, mean=0.0, sigma=1.0)
    g
end
fit = withparams(m, [2.0, 0.5, 1.5])    # amplitude=2, mean=0.5, sigma=1.5
render(fit, 0.0)                          # evaluate at x=0
fit.g.model.amplitude                     # navigable: 2.0
```

See also: [`CompiledModel`](@ref), [`params`](@ref), [`nfree`](@ref), [`render`](@ref)
"""
@generated function withparams(cm::CompiledModel, p)
    T = cm.parameters[1]
    slots = _slotmap!(Dict{Tuple{Symbol, Symbol}, Int}(), T, Ref(0))
    nslots = length(slots)
    loads = [:($(Symbol("_p", i)) = @inbounds p[$i]) for i in 1:nslots]
    tree = _treeexpr(T, :(getfield(cm, :tree)), slots)
    return Expr(:block, loads..., :(CompiledModel($tree, getfield(cm, :priors))))
end

"""
    withparams(cm::CompiledModel; kwargs...) -> CompiledModel

Rebuild the model tree stored in `cm`, overriding selected free parameters by name.

Each keyword must match an entry of [`paramnames(cm)`](@ref), i.e. take the form
`<leaf>_<field>` (e.g. `g_amplitude` for field `amplitude` of leaf `g`). Parameters not
mentioned keep their current value. This is a convenience front-end for interactive use —
tweak a couple of parameters and re-render without building the full parameter vector:

```julia
render(withparams(m; g_mean = 0.5), x)
```

Internally the keywords are written into a copy of [`params(cm)`](@ref) and forwarded to
the positional `withparams(cm, p)`, so the two methods are always consistent. Prefer the
positional method in hot loops (optimizers, samplers): this one allocates the parameter
vector and resolves names at runtime.

Only [`Free`](@ref) and [`Bounded`](@ref) parameters can be set this way. Naming a
[`Fixed`](@ref) or [`Tied`](@ref) parameter (or a nonexistent one) throws an
`ArgumentError` listing the available names; changing a fixed value is a constraint
edit — use [`@constrain`](@ref) instead.

# Arguments
- `cm::CompiledModel`: compiled model containing the annotated tree and constraints
- `kwargs...`: free-parameter overrides keyed by `<leaf>_<field>` name

# Examples
```julia
m = @model begin
    g = Gaussian1D(amplitude=1.0, mean=0.0, sigma=1.0)
    g
end
tall = withparams(m; g_amplitude = 5.0)   # mean, sigma keep current values
tall.g.model.amplitude                    # 5.0
tall.g.model.sigma                        # 1.0
```

See also: [`withparams(cm, p)`](@ref withparams), [`params`](@ref), [`paramnames`](@ref)
"""
function withparams(cm::CompiledModel; kwargs...)
    p = params(cm)
    names = paramnames(cm)
    for (k, v) in kwargs
        i = findfirst(==(k), names)
        i === nothing && throw(
            ArgumentError(
                "withparams: no free parameter `$k` — available: $(join(names, ", "))"
            )
        )
        p[i] = v
    end
    return withparams(cm, p)
end
