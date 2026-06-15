@testitem "@model block: registry, default-Free spec, evaluation" tags=[:authoring] begin
    using AstroFit

    m = @model begin
        narrow = Gaussian1D(amplitude=2.0, sigma=1.0)
        broad  = Gaussian1D(amplitude=0.5, sigma=8.0)
        narrow + broad
    end

    @test m isa CompiledModel
    @test m.model isa Sum
    @test nfree(m) == 6                       # every leaf param starts Free
    @test m.narrow.amplitude == 2.0
    @test m.broad.sigma == 8.0
    @test m[:narrow].amplitude == 2.0         # explicit indexing
    @test render(m, 0.0) == render(m.model.left, 0.0) + render(m.model.right, 0.0)
end

@testitem "@model inline and mixed forms" tags=[:authoring] begin
    using AstroFit

    g1 = Gaussian1D(amplitude=2.0, mean=0.0, sigma=1.0)
    g2 = Gaussian1D(amplitude=1.0, mean=5.0, sigma=2.0)

    two = @model g1 + g2
    @test two.g1.sigma == 1.0
    @test two.g2.mean == 5.0

    with_base = @model begin
        base = Const1D(value=0.5)
        g1 + g2 + base
    end
    @test with_base.base.value == 0.5
    @test with_base.g1.amplitude == 2.0
    @test render(with_base, 0.0) == render(g1, 0.0) + render(g2, 0.0) + 0.5
end

@testitem "addressing through operators: nesting and Pipe inversion" tags=[:authoring] begin
    using AstroFit

    # cont + line - absor → cont at .left.left, line at .left.right, absor at .right
    m = @model begin
        cont  = Linear1D(slope=0.0, intercept=1.0)
        line  = Gaussian1D(amplitude=1.0, mean=0.0, sigma=1.0)
        absor = Gaussian1D(amplitude=0.3, mean=0.0, sigma=3.0)
        cont + line - absor
    end
    @test m.model.left.left isa Linear1D
    @test m.cont.intercept == 1.0
    @test m.line.amplitude == 1.0
    @test m.absor.sigma == 3.0

    # a |> b → Pipe(a, b): a inner (.left), b outer (.right)
    inner = Linear1D(slope=2.0, intercept=1.0)
    outer = Gaussian1D(amplitude=1.0, mean=0.0, sigma=1.0)
    shaped = @model inner |> outer
    @test shaped.model.left isa Linear1D
    @test shaped.inner.slope == 2.0
    @test shaped.outer.sigma == 1.0
    @test render(shaped, 0.5) == render(outer, render(inner, 0.5))

    # a ∘ b → Pipe(b, a): a outer (.right), b inner (.left)
    composed = @model outer ∘ inner
    @test composed.model.left isa Linear1D
    @test composed.outer.amplitude == 1.0
    @test render(composed, 0.5) == render(outer, render(inner, 0.5))
end

@testitem "@constrain: fix/bound/tie/free, I1 and round-trip vs compile" tags=[:authoring] begin
    using AstroFit
    using Accessors: @optic

    m = @model begin
        narrow = Gaussian1D(amplitude=2.0, sigma=1.0)
        broad  = Gaussian1D(amplitude=0.5, sigma=8.0)
        narrow + broad
    end
    cm = @constrain m begin
        @fix   narrow.amplitude = 1.0
        @bound narrow.mean      in (-1, 1)
        @bound broad.sigma      in (0, Inf)
        @tie   broad.mean       = narrow.mean
        @tie   broad.amplitude  = narrow.amplitude / 3
    end

    @test cm.narrow.amplitude == 1.0          # @fix wrote into the tree
    @test cm.broad.mean == cm.narrow.mean     # identity tie resolved (I1)
    @test cm.broad.amplitude == 1.0 / 3       # functional tie resolved
    @test nfree(cm) == 3                      # narrow.mean, narrow.sigma, broad.sigma

    # same constraints by hand through the low-level engine API
    hand = compile(m.model, (
        (@optic(_.left.mean),       Bounded(-1.0, 1.0)),
        (@optic(_.left.sigma),      Free()),
        (@optic(_.right.sigma),     Bounded(0.0, Inf)),
        (@optic(_.left.amplitude),  Fixed(1.0)),
        (@optic(_.right.mean),      Tied(identity, (@optic(_.left.mean),))),
        (@optic(_.right.amplitude), Tied(a -> a / 3, (@optic(_.left.amplitude),))),
    ))
    @test nfree(hand) == nfree(cm)
    @test sort(paramvector(hand)) == sort(paramvector(cm))
    @test render(cm, 0.7) == render(hand, 0.7)

    # @fix at current value; @free releases
    cm2 = @constrain cm begin
        @fix  narrow.sigma
        @free narrow.mean
    end
    @test nfree(cm2) == 2                     # sigma now fixed, mean plain Free
    @test cm2.narrow.sigma == 1.0             # current value kept
    lo, hi = bounds_vectors(cm2.spec)
    @test (-1.0 ∉ lo) && (1.0 ∉ hi)           # the old Bounded on mean is gone
end

@testitem "naked single model: implicit registry" tags=[:authoring] begin
    using AstroFit

    g = Gaussian1D(amplitude=2.0, mean=0.0, sigma=1.0)
    cm = @constrain g begin
        @fix   sigma = 2.0
        @bound amplitude in (0, 10)
    end
    @test cm.sigma == 2.0
    @test nfree(cm) == 2                      # amplitude (Bounded) + mean (Free)

    cm2 = @set cm.amplitude = 5.0
    @test cm2.amplitude == 5.0
    @test_throws ArgumentError @set cm.amplitude = 11.0   # out of bounds
end

@testitem "prefab: constraints travel namespaced; merge/override by name" tags=[:authoring, :prefab] begin
    using AstroFit

    function EmissionLine(; center, flux=1.0)
        m = @model begin
            line = Gaussian1D(mean=center, amplitude=flux)
            line
        end
        @constrain m begin
            @bound line.sigma in (0, Inf)
        end
    end

    spectrum = @model begin
        cont = Linear1D(slope=0.0, intercept=1.0)
        Ha   = EmissionLine(center=6563.0)
        Hb   = EmissionLine(center=4861.0)
        cont + Ha + Hb
    end

    @test spectrum.Ha.line.mean == 6563.0     # hierarchical read
    @test spectrum.Hb.line.mean == 4861.0     # same prefab type, no collision
    @test nfree(spectrum) == 8                # 2 (cont) + 3 + 3
    lo, _ = bounds_vectors(spectrum.spec)
    @test count(==(0.0), lo) == 2             # both factory bounds traveled

    cm = @constrain spectrum begin
        @bound Ha.line.sigma in (1, 5)        # restrict the factory bound
        @tie   Hb.line.sigma = Ha.line.sigma  # tie the two widths
        @free  Ha.line.amplitude              # explicit release
    end
    @test cm.Hb.line.sigma == cm.Ha.line.sigma
    @test nfree(cm) == 7                      # Hb.line.sigma became Tied
    lo2, hi2 = bounds_vectors(cm.spec)
    @test (1.0 in lo2) && (5.0 in hi2)        # override applied
    @test count(==(0.0), lo2) == 0            # Hb factory bound replaced by the tie
end

@testitem "prefab with internal tie: masters re-rooted (critical point 1)" tags=[:authoring, :prefab, :tied] begin
    using AstroFit
    using Accessors: @set

    function DoubleLine(; center)
        m = @model begin
            main = Gaussian1D(mean=center, amplitude=2.0, sigma=1.0)
            echo = Gaussian1D(mean=center + 10.0, amplitude=99.0, sigma=1.0)
            main + echo
        end
        @constrain m begin
            @tie echo.amplitude = main.amplitude / 2
        end
    end

    big = @model begin
        base = Const1D(value=0.0)
        D = DoubleLine(center=100.0)
        base + D
    end

    @test big.D.echo.amplitude == 1.0         # tie resolved on the PREFIXED master

    # I1 through the prefab tie: setting the master updates the dependent
    big2 = @set big.D.main.amplitude = 4.0
    @test big2.D.main.amplitude == 4.0
    @test big2.D.echo.amplitude == 2.0

    # withparams flows through the re-rooted tie too
    p = paramvector(big)
    i = findfirst(==(2.0), p)                 # main.amplitude slot
    p[i] = 6.0
    @test withparams(big, p).D.echo.amplitude == 3.0
end

@testitem "@set: new CompiledModel, I1 on master, Tied/bounds errors" tags=[:authoring] begin
    using AstroFit
    using Accessors: @set

    m = @model begin
        narrow = Gaussian1D(amplitude=2.0, sigma=1.0)
        broad  = Gaussian1D(amplitude=0.5, sigma=8.0)
        narrow + broad
    end
    cm = @constrain m begin
        @bound narrow.sigma in (0.1, 10)
        @tie   broad.mean   = narrow.mean
    end

    cm2 = @set cm.narrow.amplitude = 3.0
    @test cm2.narrow.amplitude == 3.0
    @test cm.narrow.amplitude == 2.0          # immutability

    cm3 = @set cm.narrow.mean = 1.5           # master of the tie
    @test cm3.broad.mean == 1.5               # dependent already updated (I1)

    @test_throws ArgumentError @set cm.broad.mean = 9.9     # Tied target
    @test_throws ArgumentError @set cm.narrow.sigma = 99.0  # out of bounds
    cm4 = @set cm.narrow.sigma = 5.0                        # inside bounds
    @test cm4.narrow.sigma == 5.0
end

@testitem "@constrain validation: V1-V4 and bad paths" tags=[:authoring] begin
    using AstroFit

    m = @model begin
        a = Gaussian1D(amplitude=2.0, sigma=1.0)
        b = Gaussian1D(amplitude=0.5, sigma=8.0)
        a + b
    end

    # V3: insane bounds
    @test_throws ArgumentError (@constrain m begin
        @bound a.sigma in (5, 1)
    end)
    # V4: current value outside the new bounds (never a silent clamp)
    @test_throws ArgumentError (@constrain m begin
        @bound a.amplitude in (10, 20)
    end)
    # V2: self-tie
    @test_throws ArgumentError (@constrain m begin
        @tie a.sigma = 2 * a.sigma
    end)
    # V1: tie chain across merged view (master is itself Tied)
    tied = @constrain m begin
        @tie b.sigma = a.sigma
    end
    @test_throws ArgumentError (@constrain tied begin
        @tie b.amplitude = b.sigma + 1.0
    end)
    # unknown component / parameter
    @test_throws ArgumentError (@constrain m begin
        @fix c.sigma = 1.0
    end)
    # component (not parameter) as target
    @test_throws ArgumentError (@constrain m begin
        @free a
    end)
    # last one wins inside the same block (@tie then @fix)
    cm = @constrain m begin
        @tie b.sigma = a.sigma
        @fix b.sigma = 3.0
    end
    @test cm.b.sigma == 3.0
    @test nfree(cm) == 5                      # b.sigma fixed, everything else Free
end

@testitem "informative errors: scalars, CompiledModel algebra, hand-built compound" tags=[:authoring] begin
    using AstroFit

    g1 = Gaussian1D(amplitude=2.0, sigma=1.0)
    g2 = Gaussian1D(amplitude=1.0, sigma=2.0)

    @test_throws ArgumentError 2 * g1
    @test_throws ArgumentError g1 + 1
    @test_throws ArgumentError -g1

    cma = @model g1 + g2
    cmb = @model g2 + g1
    @test_throws ArgumentError cma + cmb      # compose inside @model instead
    @test_throws ArgumentError cma * g1
    @test_throws ArgumentError g1 |> cmb

    s = g1 + g2                                # hand-built compound, no names
    @test_throws ArgumentError (@constrain s begin
        @fix amplitude = 1.0
    end)
end

@testitem "@model expansion errors: anonymous leaf, duplicates, reserved" tags=[:authoring] begin
    using AstroFit

    # anonymous leaf inline
    @test_throws Exception macroexpand(@__MODULE__,
        :(AstroFit.@model g1 + Gaussian1D(amplitude=1.0)))
    # same symbol twice = sibling collision
    @test_throws Exception macroexpand(@__MODULE__,
        :(AstroFit.@model g1 + g1))
    # reserved names
    @test_throws Exception macroexpand(@__MODULE__,
        :(AstroFit.@model begin
              model = Gaussian1D()
              names = Gaussian1D()
              model + names
          end))
end

@testitem "@model identity check trips on a wrong walker map" tags=[:authoring] begin
    using AstroFit
    using Accessors: @optic

    g1 = Gaussian1D(amplitude=2.0, sigma=1.0)
    g2 = Gaussian1D(amplitude=1.0, sigma=2.0)
    # deliberately swapped optics: the guardrail (now a debug @assert) must fire,
    # not resolve silently
    @test_throws AssertionError AstroFit._build_model(
        (a, b) -> Sum(a, b), (:x, :y),
        (@optic(_.right), @optic(_.left)), (g1, g2))
end

@testitem "authored model: withparams stays inferred and allocation-free" tags=[:authoring, :perf] begin
    using AstroFit
    using Test: @inferred

    m = @model begin
        cont = Linear1D(slope=0.0, intercept=1.0)
        line = Gaussian1D(amplitude=2.0, mean=6563.0, sigma=2.0)
        cont + line
    end
    cm = @constrain m begin
        @fix   cont.slope = 0.0
        @bound line.sigma in (0.1, 50)
        @tie   line.amplitude = cont.intercept * 2
    end

    p = paramvector(cm)
    @inferred withparams(cm, p)
    probe(cm, p) = withparams(cm, p)
    probe(cm, p)                               # warmup
    @test @allocated(probe(cm, p)) == 0
end

@testitem "hot loop: zero-alloc rebuild on big prefab models, Float64 and Dual" tags=[:authoring, :perf, :autodiff] begin
    using AstroFit
    using ForwardDiff
    using Test: @inferred

    function EmissionLine(; center, flux=1.0)
        m = @model begin
            line = Gaussian1D(mean=center, amplitude=flux)
            line
        end
        @constrain m begin
            @bound line.sigma in (0, Inf)
        end
    end

    # continuum + 6 lines, 5 cross-prefab ties, 20-entry merged spec: big
    # enough that a recursive formulation of the rebuild hits Julia's
    # inference recursion limit (the regression this test pins down)
    big = @model begin
        cont = Linear1D(slope=0.0, intercept=1.0)
        L1 = EmissionLine(center=6563.0)
        L2 = EmissionLine(center=4861.0)
        L3 = EmissionLine(center=4340.0)
        L4 = EmissionLine(center=4102.0)
        L5 = EmissionLine(center=3970.0)
        L6 = EmissionLine(center=3889.0)
        cont + L1 + L2 + L3 + L4 + L5 + L6
    end
    cm = @constrain big begin
        @tie L2.line.sigma = L1.line.sigma
        @tie L3.line.sigma = L1.line.sigma
        @tie L4.line.sigma = L1.line.sigma
        @tie L5.line.sigma = L1.line.sigma
        @tie L6.line.sigma = L1.line.sigma
    end
    @test nfree(cm) == 15

    probe(cm, p) = withparams(cm, p)
    p = paramvector(cm)
    @inferred withparams(cm, p)
    probe(cm, p)                               # warmup
    @test @allocated(probe(cm, p)) == 0

    # Dual params: the rebuilt tree changes eltype — must stay inferred
    # and allocation-free for ForwardDiff-based optimizers
    pd = ForwardDiff.Dual{Nothing}.(p, 1.0)
    @inferred withparams(cm, pd)
    probe(cm, pd)                              # warmup
    @test @allocated(probe(cm, pd)) == 0

    # full optimization-style loss: gradient flows, ties stay consistent
    xs = collect(range(3800.0, 7000.0, length=200))
    ys = render(cm, xs)
    loss(v) = sum(abs2, render(withparams(cm, v), xs) .- ys)
    @test loss(p) == 0.0
    g = ForwardDiff.gradient(loss, p)
    @test all(isfinite, g)
end
