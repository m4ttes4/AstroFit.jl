# ---------------------------------------------------------------------------
# Display — compact tree of named components with parameter values and
# constraints. Constraint colors: free=green, fixed=red, bounded=yellow,
# tied=ANSI orange (only when the IO supports color).
# ---------------------------------------------------------------------------

_typename(m) = string(nameof(typeof(m)))

_fmt(v) = sprint(print, v; context = :compact => true)

function _find_constraint(spec::Tuple, optic)
    for (t, c) in spec
        _same_optic(t, optic) && return c
    end
    nothing
end

# Map every leaf-parameter optic (as its flattened lens sequence) to its
# dotted path, so Tied masters can be displayed by name.
function _collect_paths!(d, names, prefix_optic, prefix, model)
    for (k, entry) in pairs(names)
        if entry isa Registry
            _collect_paths!(d, entry.names, _compose(prefix_optic, entry.optic),
                            string(prefix, k, "."), model)
        else
            full = _compose(prefix_optic, entry)
            sub  = full(model)
            if sub isa AbstractModel
                for f in fieldnames(typeof(sub))
                    lens = _compose(full, PropertyLens{f}())
                    d[_optic_leaves(lens)] = string(prefix, k, ".", f)
                end
            else
                d[_optic_leaves(full)] = string(prefix, k)
            end
        end
    end
    d
end

_master_path(d, optic) = get(() -> sprint(print, optic), d, _optic_leaves(optic))

function _print_orange(io, xs...)
    if get(io, :color, false)
        print(io, "\e[38;5;208m")
        print(io, xs...)
        print(io, "\e[39m")
    else
        print(io, xs...)
    end
end

function _constraint_counts(spec::Tuple)
    free = fixed = bounded = tied = 0
    for (_, c) in spec
        if c isa Free
            free += 1
        elseif c isa Fixed
            fixed += 1
        elseif c isa Bounded
            bounded += 1
        elseif c isa Tied
            tied += 1
        end
    end
    (; free, fixed, bounded, tied)
end

function _component_count(names)
    n = 0
    for (_, entry) in pairs(names)
        n += 1
        entry isa Registry && (n += _component_count(entry.names))
    end
    n
end

function _print_count(io, label, n, color)
    print(io, "  ", rpad(label * ":", 12))
    color === nothing ? print(io, n) : printstyled(io, n; color)
    println(io)
end

_print_name(io, name) = printstyled(io, name; bold = true)
_print_type(io, type) = printstyled(io, type; color = :cyan)

_print_constraint(io, ::Free, d) = printstyled(io, "free"; color = :green)
_print_constraint(io, ::Fixed, d) = printstyled(io, "fixed"; color = :red)
_print_constraint(io, c::Bounded, d) =
    printstyled(io, "bounded [", _fmt(c.lower), ", ", _fmt(c.upper), "]"; color = :yellow)
_print_constraint(io, c::Tied, d) =
    _print_orange(io, "tied to ", join((_master_path(d, o) for o in c.masters), ", "))

function _param_line(io, prefix, label, w, value, c, d)
    print(io, prefix, rpad(label, w), " = ", _fmt(value))
    if c !== nothing
        print(io, "  ")
        _print_constraint(io, c, d)
    end
    println(io)
end

function _show_params(io, m::AbstractModel, prefix_optic, spec, prefix, d)
    fns = fieldnames(typeof(m))
    w = maximum((length(string(f)) for f in fns); init = 0)
    n = length(fns)
    for (i, f) in enumerate(fns)
        branch = i == n ? "└─ " : "├─ "
        c = _find_constraint(spec, _compose(prefix_optic, PropertyLens{f}()))
        _param_line(io, prefix * branch, string(f), w, getfield(m, f), c, d)
    end
end

function _component_header(io, prefix, branch, name, type)
    print(io, prefix, branch)
    _print_name(io, string(name))
    print(io, " :: ")
    _print_type(io, type)
    println(io)
end

function _show_names(io, model, spec, names, prefix_optic, prefix, d)
    w = maximum((length(string(k)) for (k, entry) in pairs(names)
                 if !(entry isa Registry) &&
                    !(_compose(prefix_optic, entry)(model) isa AbstractModel));
                init = 0)
    entries = collect(pairs(names))
    n = length(entries)
    for (i, (k, entry)) in enumerate(entries)
        last = i == n
        branch = last ? "└─ " : "├─ "
        child_prefix = prefix * (last ? "   " : "│  ")
        if entry isa Registry
            full = _compose(prefix_optic, entry.optic)
            _component_header(io, prefix, branch, k, _typename(full(model)))
            _show_names(io, model, spec, entry.names, full, child_prefix, d)
        else
            full = _compose(prefix_optic, entry)
            sub  = full(model)
            if sub isa AbstractModel
                _component_header(io, prefix, branch, k, _typename(sub))
                _show_params(io, sub, full, spec, child_prefix, d)
            else
                _param_line(io, prefix * branch, string(k), w, sub, _find_constraint(spec, full), d)
            end
        end
    end
end

function _show_summary(io, cm::CompiledModel)
    counts = _constraint_counts(getfield(cm, :spec))
    printstyled(io, "AstroFit.CompiledModel"; bold = true)
    println(io)
    _print_count(io, "components", _component_count(getfield(cm, :names)), nothing)
    _print_count(io, "free", counts.free, :green)
    _print_count(io, "fixed", counts.fixed, :red)
    _print_count(io, "bounded", counts.bounded, :yellow)
    print(io, "  ", rpad("tied:", 12))
    _print_orange(io, counts.tied)
    println(io)
end

function Base.show(io::IO, ::MIME"text/plain", cm::CompiledModel)
    model = getfield(cm, :model)
    spec  = getfield(cm, :spec)
    names = getfield(cm, :names)
    buf   = IOContext(IOBuffer(), io)
    _show_summary(buf, cm)
    println(buf)
    d = _collect_paths!(Dict{Any, String}(), names, identity, "", model)
    if isempty(keys(names))
        model isa AbstractModel ?
            _show_params(buf, model, identity, spec, "  ", d) :
            println(buf, "  ", model)
    else
        _show_names(buf, model, spec, names, identity, "  ", d)
    end
    print(io, chomp(String(take!(buf.io))))
end

Base.show(io::IO, cm::CompiledModel) =
    print(io, "CompiledModel(", length(keys(getfield(cm, :names))),
          " components, ", nfree(cm), " free)")

function Base.show(io::IO, ::MIME"text/plain", ref::ComponentRef)
    root  = getfield(ref, :root)
    optic = getfield(ref, :optic)
    names = getfield(ref, :names)
    model = getfield(root, :model)
    spec  = getfield(root, :spec)
    buf   = IOContext(IOBuffer(), io)
    printstyled(buf, "AstroFit.ComponentRef"; bold = true)
    print(buf, " :: ")
    _print_type(buf, _typename(optic(model)))
    println(buf)
    d = _collect_paths!(Dict{Any, String}(), getfield(root, :names), identity, "", model)
    isempty(keys(names)) ?
        _show_params(buf, optic(model), optic, spec, "  ", d) :
        _show_names(buf, model, spec, names, optic, "  ", d)
    print(io, chomp(String(take!(buf.io))))
end

Base.show(io::IO, ref::ComponentRef) =
    print(io, "ComponentRef(", _typename(getfield(ref, :optic)(getfield(getfield(ref, :root), :model))), ")")
