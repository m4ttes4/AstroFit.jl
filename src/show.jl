# ---------------------------------------------------------------------------
# Display — minimal tree of named components with parameter values and
# constraints. Constraints are color-coded: free=green, fixed=gray,
# bounded=yellow, tied=magenta (only when the IO supports color).
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

_print_constraint(io, ::Free, d)    = printstyled(io, "free"; color = :green)
_print_constraint(io, ::Fixed, d)   = printstyled(io, "fixed"; color = :light_black)
_print_constraint(io, c::Bounded, d) =
    printstyled(io, "∈ [", _fmt(c.lower), ", ", _fmt(c.upper), "]"; color = :yellow)
_print_constraint(io, c::Tied, d) =
    printstyled(io, "tied(", join((_master_path(d, o) for o in c.masters), ", "), ")";
                color = :magenta)

function _param_line(io, indent, label, w, value, c, d)
    print(io, indent, rpad(label, w), " = ", _fmt(value))
    if c !== nothing
        print(io, "  ")
        _print_constraint(io, c, d)
    end
    println(io)
end

function _show_params(io, m::AbstractModel, prefix_optic, spec, indent, d)
    fns = fieldnames(typeof(m))
    w = maximum((length(string(f)) for f in fns); init = 0)
    for f in fns
        c = _find_constraint(spec, _compose(prefix_optic, PropertyLens{f}()))
        _param_line(io, indent, string(f), w, getfield(m, f), c, d)
    end
end

function _show_names(io, model, spec, names, prefix_optic, indent, d)
    w = maximum((length(string(k)) for (k, entry) in pairs(names)
                 if !(entry isa Registry) &&
                    !(_compose(prefix_optic, entry)(model) isa AbstractModel));
                init = 0)
    for (k, entry) in pairs(names)
        if entry isa Registry
            full = _compose(prefix_optic, entry.optic)
            println(io, indent, k, " :: ", _typename(full(model)))
            _show_names(io, model, spec, entry.names, full, indent * "  ", d)
        else
            full = _compose(prefix_optic, entry)
            sub  = full(model)
            if sub isa AbstractModel
                println(io, indent, k, " :: ", _typename(sub))
                _show_params(io, sub, full, spec, indent * "  ", d)
            else
                _param_line(io, indent, string(k), w, sub, _find_constraint(spec, full), d)
            end
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", cm::CompiledModel)
    model = getfield(cm, :model)
    spec  = getfield(cm, :spec)
    names = getfield(cm, :names)
    buf   = IOContext(IOBuffer(), io)
    n = nfree(cm)
    printstyled(buf, "CompiledModel"; bold = true)
    println(buf, "  (", n, " free parameter", n == 1 ? "" : "s", ")")
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
    println(buf, "ComponentRef :: ", _typename(optic(model)))
    d = _collect_paths!(Dict{Any, String}(), getfield(root, :names), identity, "", model)
    isempty(keys(names)) ?
        _show_params(buf, optic(model), optic, spec, "  ", d) :
        _show_names(buf, model, spec, names, optic, "  ", d)
    print(io, chomp(String(take!(buf.io))))
end

Base.show(io::IO, ref::ComponentRef) =
    print(io, "ComponentRef(", _typename(getfield(ref, :optic)(getfield(getfield(ref, :root), :model))), ")")
