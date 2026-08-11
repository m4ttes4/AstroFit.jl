# Driver for the AstroFit benchmark suite: runs it, prints a report, saves both
# the raw trials and the printed report next to each other.
#
#   julia --project=examples --startup-file=no bench/bench_run.jl [group]
#
# A positional argument restricts the run to ONE top-level group, e.g. `render`.
#
# Environment:
#   ASTROFIT_BENCH_QUICK=1        smoke mode, 0.05 s per benchmark (timings are junk)
#   ASTROFIT_BENCH_OUTDIR=<dir>   where results.json / report.txt land
#   ASTROFIT_BENCH_BASELINE=<f>   a previous results.json to judge this run against

include(joinpath(@__DIR__, "bench_suite.jl"))

using BenchmarkTools, Printf, Dates, Statistics

const QUICK = get(ENV, "ASTROFIT_BENCH_QUICK", "0") == "1"

# `@benchmarkable` snapshots DEFAULT_PARAMETERS when the suite is *built* — that
# already happened in the include above, so mutating them here would be ignored.
# `run` kwargs are applied per benchmark at execution time and do take effect.
const RUNOPTS = QUICK ? (; seconds = 0.05) : (;)

const GROUP = isempty(ARGS) ? nothing : ARGS[1]

# Re-wrap the selected group instead of returning it bare, so key paths (and
# therefore the RATIOS lookups) are identical to those of a full run.
function selectgroup(suite, name)
    name === nothing && return suite
    haskey(suite, name) || error(
        "unknown benchmark group $(repr(name)); available: " *
            join(sort!(string.(collect(keys(suite)))), ", ")
    )
    sub = BenchmarkGroup()
    sub[name] = suite[name]
    return sub
end

const RESULTS = run(selectgroup(SUITE, GROUP); verbose = true, RUNOPTS...)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

pathlabel(path) = join(path, "/")
aspath(k) = k isa Union{Tuple, AbstractVector} ? collect(k) : [k]

# BenchmarkTools' `getindex` is a `get!`: a missing key silently INSERTS an empty
# group. Walk with `haskey` so a filtered run neither lies about a leaf nor
# mutates the tree we are about to serialize.
function findleaf(group, path)
    node = group
    for k in path
        (node isa BenchmarkGroup && haskey(node, k)) || return nothing
        node = node[k]
    end
    return node isa BenchmarkGroup ? nothing : node
end

# Sorted (path, trial) leaves — a BenchmarkGroup is a Dict, so without this the
# row order changes between runs and two reports cannot be diffed.
sortedleaves(group) = sort(BenchmarkTools.leaves(group); by = kv -> pathlabel(first(kv)))

function printheader(io)
    println(io, "="^78)
    println(io, "AstroFit benchmark suite")
    println(io, "="^78)
    @printf(io, "%-10s %s\n", "date", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    @printf(io, "%-10s %s\n", "julia", VERSION)
    @printf(io, "%-10s %s\n", "cpu", Sys.CPU_NAME)
    @printf(io, "%-10s %d\n", "threads", Threads.nthreads())
    @printf(io, "%-10s %s\n", "group", GROUP === nothing ? "all" : GROUP)
    @printf(
        io, "%-10s %s\n", "mode",
        QUICK ? "QUICK (0.05 s/benchmark) — timings are NOT meaningful" : "full"
    )
    println(io)
    @printf(io, "  %-6s %6s %8s\n", "case", "nfree", "points")
    for (name, case) in pairs(CASES)
        @printf(io, "  %-6s %6d %8d\n", string(name), nfree(case.cm), length(case.y))
    end
    println(io)
    return
end

function printgroup(io, name, group)
    rows = sortedleaves(group)
    isempty(rows) && return
    w = maximum(kv -> length(pathlabel(first(kv))), rows)
    w = max(w, length("benchmark"))
    println(io, "[", name, "]")
    @printf(io, "  %s  %12s  %8s  %11s\n", rpad("benchmark", w), "median", "allocs", "memory")
    println(io, "  ", "-"^(w + 37))
    for (path, trial) in rows
        m = median(trial)
        @printf(
            io, "  %s  %12s  %8d  %11s\n",
            rpad(pathlabel(path), w),
            BenchmarkTools.prettytime(time(m)),
            allocs(m),
            BenchmarkTools.prettymemory(memory(m)),
        )
    end
    println(io)
    return
end

function printratios(io, results)
    rows = [(aspath(a), aspath(h)) for (a, h) in RATIOS]
    isempty(rows) && return
    println(io, "[ratios]")
    println(io, "  AstroFit median / handwritten median — <= 1.00x means AstroFit is at")
    println(io, "  or faster than the equivalent handwritten code.")
    wa = maximum(r -> length(pathlabel(r[1])), rows)
    wh = maximum(r -> length(pathlabel(r[2])), rows)
    for (apath, hpath) in rows
        a = findleaf(results, apath)
        h = findleaf(results, hpath)
        cell = if a === nothing || h === nothing
            "n/a"                     # filtered run: one of the two was not measured
        else
            @sprintf("%.2fx", time(median(a)) / time(median(h)))
        end
        @printf(
            io, "  %s  vs  %s  %8s\n",
            rpad(pathlabel(apath), wa), rpad(pathlabel(hpath), wh), cell
        )
    end
    println(io)
    return
end

function printjudgement(io, results, path)
    baseline = BenchmarkTools.load(path)[1]        # `load` always returns a Vector
    # `judge` walks the target's keys and skips those absent from the baseline,
    # so a filtered run against a full baseline compares the intersection.
    verdicts = judge(median(results), median(baseline))
    changed = filter(kv -> !isinvariant(last(kv)), sortedleaves(verdicts))
    println(io, "[regressions] vs ", path)
    if isempty(changed)
        println(io, "  every leaf invariant.")
    else
        w = maximum(kv -> length(pathlabel(first(kv))), changed)
        for (p, t) in changed
            @printf(
                io, "  %s  time %-11s %8s   memory %-11s %8s\n",
                rpad(pathlabel(p), w),
                time(t), @sprintf("%.2fx", time(ratio(t))),
                memory(t), @sprintf("%.2fx", memory(ratio(t))),
            )
        end
    end
    println(io)
    return
end

const REPORT = let io = IOBuffer()
    printheader(io)
    for key in sort(collect(keys(RESULTS)); by = string)
        printgroup(io, string(key), RESULTS[key])
    end
    printratios(io, RESULTS)
    baseline = get(ENV, "ASTROFIT_BENCH_BASELINE", "")
    isempty(baseline) || printjudgement(io, RESULTS, baseline)
    String(take!(io))
end

print(REPORT)

const OUTDIR = get(
    ENV, "ASTROFIT_BENCH_OUTDIR",
    joinpath(@__DIR__, "results", Dates.format(now(), "yyyy-mm-dd_HHMMSS")),
)
mkpath(OUTDIR)
# Save the full trials, not `median(results)`: the medians are recoverable from
# them, and a baseline of medians could not be re-judged.
BenchmarkTools.save(joinpath(OUTDIR, "results.json"), RESULTS)
write(joinpath(OUTDIR, "report.txt"), REPORT)
println("saved to ", OUTDIR)
