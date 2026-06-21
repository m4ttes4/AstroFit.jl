using Accessors: constructorof

# withparams(cm, p): rebuild the bare model tree from a flat parameter vector.
# @generated: slot assignment is positional (decision X) and resolved here at compile
# time by walking the tree *type*; only Fixed values and Tied functions are read from
# the persistent tree at runtime. Returns the bare tree (Leaf wrappers stripped) so the
# hot path is the same straight-line rebuild + render as a plain compound model — ADR 0001.

# Pass 1: give each Free/Bounded field its slot in `p`, keyed by (leaf name, field).
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

# How one leaf field's value is computed.
function _fieldexpr(Ci, name, fname, i, acc, slots)
    return if Ci <: Free || Ci <: Bounded
        :(p[$(slots[(name, fname)])])
    elseif Ci <: Fixed
        :(($acc).constraints[$i].value)
    elseif Ci <: Tied
        args = (:(p[$(slots[path])]) for path in Ci.parameters[1])
        :(($acc).constraints[$i].f($(args...)))
    else
        error("withparams: unknown constraint $Ci")
    end
end

# Expression that reconstructs one subtree, bare (no Leaf wrapper). `acc` reaches this
# node in the runtime tree, needed only for Fixed/Tied runtime fields.
function _treeexpr(T, acc, slots)
    return if T <: Leaf
        name, M, C = T.parameters
        fields = (
            _fieldexpr(fieldtypes(C)[i], name, fieldnames(M)[i], i, acc, slots)
                for i in 1:fieldcount(M)
        )
        :($(constructorof(M))($(fields...)))
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

@generated function withparams(cm::CompiledModel, p)
    T = cm.parameters[1]
    slots = _slotmap!(Dict{Tuple{Symbol, Symbol}, Int}(), T, Ref(0))
    return _treeexpr(T, :(getfield(cm, :tree)), slots)
end
