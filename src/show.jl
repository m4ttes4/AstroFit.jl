# Pretty display. Compact `show` rebuilds the algebraic expression; the MIME
# text/plain CompiledModel show draws a colored, flattened tree.
# ponytail: colors via stdlib printstyled (honors IOContext :color); no Crayons dep.

const _NODE = Union{Sum, Difference, Product, Quotient, Pipe}
const _ASSOC = Union{Sum, Product}            # only these flatten into sibling chains
const _TIED_COLOR = 208                      # ANSI 256-color orange

_opsym(::Sum) = "+"
_opsym(::Difference) = "-"
_opsym(::Product) = "*"
_opsym(::Quotient) = "/"
_opsym(::Pipe) = "|>"

_leafname(::Leaf{n}) where {n} = n
_fmt(v::AbstractFloat) = string(round(v; sigdigits = 4))
_fmt(v) = string(v)

# --- compact one-line expression: leaves print as their name, bare components as-is.
# Parenthesise a child only when its operator differs from the parent (keeps same-op
# chains like `a + b + c` flat, but guards `a + b*c` vs `(a+b)*c`).
function _expr(node, parentop = nothing)
    node isa Leaf && return string(_leafname(node))
    node isa _NODE || return sprint(show, node)
    op = _opsym(node)
    inner = "$(_expr(node.left, op)) $op $(_expr(node.right, op))"
    return (parentop !== nothing && parentop != op) ? "($inner)" : inner
end

Base.show(io::IO, m::_NODE) = print(io, _expr(m))
Base.show(io::IO, l::Leaf) = print(io, _leafname(l))
Base.show(io::IO, cm::CompiledModel) = print(io, _expr(getfield(cm, :tree)))

# --- constraint summaries
_counts(cm::CompiledModel) = _counts(getfield(cm, :tree))
_counts(n) = _add(_counts(n.left), _counts(n.right))
function _counts(l::Leaf)
    cs = l.constraints
    return (
        free = count(c -> c isa Free, cs),
        bounds = count(c -> c isa Bounded, cs),
        fixed = count(c -> c isa Fixed, cs),
        tied = count(c -> c isa Tied, cs),
    )
end
_add(a, b) = (
    free = a.free + b.free,
    bounds = a.bounds + b.bounds,
    fixed = a.fixed + b.fixed,
    tied = a.tied + b.tied,
)

# --- flattened children of a tree node
_kids(node) = node isa _ASSOC ?
    _flatten!(AbstractModel[], node, _opsym(node)) :
    AbstractModel[node.left, node.right]
function _flatten!(acc, node, op)
    if node isa _ASSOC && _opsym(node) == op
        _flatten!(acc, node.left, op)
        _flatten!(acc, node.right, op)
    else
        push!(acc, node)
    end
    return acc
end

# --- the tree view
function _header(io, title, counts)
    printstyled(io, title; bold = true)
    for (n, label, color) in (
            (counts.free, "free", :green),
            (counts.bounds, "bounds", :blue),
            (counts.fixed, "fixed", :red),
            (counts.tied, "tied", _TIED_COLOR),
        )
        printstyled(io, "  ·  "; color = :light_black)
        _stat(io, n, label, color)
    end
    return println(io)
end

function Base.show(io::IO, ::MIME"text/plain", l::Leaf)
    _header(io, "Leaf", _counts(l))
    return _tree(io, l, "", true, true)
end

function Base.show(io::IO, ::MIME"text/plain", cm::CompiledModel)
    tree = getfield(cm, :tree)
    _header(io, "CompiledModel", _counts(cm))
    print(io, "formula: ")
    println(io, _expr(tree))
    return _tree(io, tree, "", true, true)
end

function _stat(io, n, label, color)
    print(io, n, " ")
    return printstyled(io, label; color, bold = true)
end

function _tree(io, node, prefix, islast, isroot = false)
    isroot || print(io, prefix, islast ? "└─ " : "├─ ")
    child = isroot ? prefix : prefix * (islast ? "   " : "│  ")
    return if node isa Leaf
        _leafline(io, node); println(io)
        fields = fieldnames(typeof(node.model))
        isempty(fields) && return
        width = maximum(length(string(f)) for f in fields)
        for (i, f) in enumerate(fields)
            _fieldline(
                io, child, i == length(fields), f,
                getfield(node.model, f), node.constraints[i], width
            )
        end
    else
        printstyled(io, _opsym(node), "\n"; color = :light_black, bold = true)
        kids = _kids(node)
        for (i, k) in enumerate(kids)
            _tree(io, k, child, i == length(kids))
        end
    end
end

function _leafline(io, l)
    printstyled(io, _leafname(l); color = :cyan, bold = true)
    printstyled(io, " :: "; color = :light_black)
    return printstyled(io, nameof(typeof(l.model)); color = :blue)
end

function _fieldline(io, prefix, islast, f, v, c, width)
    print(io, prefix, islast ? "└─ " : "├─ ")
    print(io, rpad(string(f), width))
    print(io, "  ", rpad(_fieldvalue(v, c), 8), "  ")
    _constraint(io, c)
    return println(io)
end

_fieldvalue(v, c::Fixed) = _fmt(c.value)
_fieldvalue(v, _) = _fmt(v)

function _constraint(io, ::Free)
    return printstyled(io, "free"; color = :green, bold = true)
end
function _constraint(io, c::Bounded)
    printstyled(io, "bounds"; color = :blue, bold = true)
    return print(io, " [", _fmt(c.lower), ", ", _fmt(c.upper), "]")
end
function _constraint(io, ::Fixed)
    return printstyled(io, "fixed"; color = :red, bold = true)
end
function _constraint(io, c::Tied)
    printstyled(io, "tied"; color = _TIED_COLOR, bold = true)
    return print(io, " -> ", _masters(c))
end
_masters(::Tied{P}) where {P} = join(("$l.$f" for (l, f) in P), ", ")
