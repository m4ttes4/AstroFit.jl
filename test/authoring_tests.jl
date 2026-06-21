@testitem "@model block: registry, default-Free leaves, evaluation" tags=[:authoring] begin
    using AstroFit

    cm = @model begin
        narrow = Gaussian1D(amplitude=2.0, sigma=1.0)
        broad = Gaussian1D(amplitude=0.5, sigma=8.0)
        narrow + broad
    end

    @test cm isa CompiledModel
    @test getfield(cm, :tree) isa AstroFit.Sum
    @test nfree(cm) == 6
    @test cm.narrow.model.amplitude == 2.0
    @test cm.broad.model.sigma == 8.0
    @test cm.narrow.constraints == (Free(), Free(), Free())
    @test render(cm, 0.0) == render(cm.narrow.model, 0.0) + render(cm.broad.model, 0.0)
end

@testitem "@model requires named leaves in a begin block" tags=[:authoring] begin
    using AstroFit

    g1 = Gaussian1D(amplitude=2.0, mean=0.0, sigma=1.0)
    g2 = Gaussian1D(amplitude=1.0, mean=5.0, sigma=2.0)

    @test_throws Exception macroexpand(@__MODULE__, :(AstroFit.@model $g1 + $g2))

    cm = @model begin
        left = g1
        right = g2
        left + right
    end

    @test cm.left.model === g1
    @test cm.right.model === g2
    @test render(cm, 0.0) == render(g1, 0.0) + render(g2, 0.0)
end

@testitem "compound operators preserve tree shape and render semantics" tags=[:authoring] begin
    using AstroFit

    cm = @model begin
        cont = Linear1D(slope=0.0, intercept=1.0)
        line = Gaussian1D(amplitude=1.0, mean=0.0, sigma=1.0)
        absor = Gaussian1D(amplitude=0.3, mean=0.0, sigma=3.0)
        cont + line - absor
    end

    tree = getfield(cm, :tree)
    @test tree isa AstroFit.Difference
    @test tree.left isa AstroFit.Sum
    @test tree.left.left === cm.cont
    @test tree.left.right === cm.line
    @test tree.right === cm.absor

    shaped = @model begin
        inner = Linear1D(slope=2.0, intercept=1.0)
        outer = Gaussian1D(amplitude=1.0, mean=0.0, sigma=1.0)
        inner |> outer
    end
    @test render(shaped, 0.5) == render(shaped.outer.model, render(shaped.inner.model, 0.5))

    composed = @model begin
        inner = Linear1D(slope=2.0, intercept=1.0)
        outer = Gaussian1D(amplitude=1.0, mean=0.0, sigma=1.0)
        outer ∘ inner
    end
    @test render(composed, 0.5) == render(composed.outer.model, render(composed.inner.model, 0.5))
end

@testitem "@constrain block edits constraints and auto-rebinds" tags=[:authoring] begin
    using AstroFit

    cm = @model begin
        narrow = Gaussian1D(amplitude=2.0, sigma=1.0)
        broad = Gaussian1D(amplitude=0.5, sigma=8.0)
        narrow + broad
    end

    old = cm
    @constrain cm begin
        narrow.amplitude = 1.0
        narrow.mean in (-1, 1)
        broad.sigma in (0, Inf)
        broad.mean -> narrow.mean
    end

    @test cm !== old
    @test cm.narrow.constraints[1] isa Fixed
    @test cm.narrow.constraints[1].value == 1.0
    @test cm.narrow.constraints[2] isa Bounded
    @test cm.broad.constraints[2] isa Tied
    @test cm.broad.constraints[3] isa Bounded
    @test nfree(cm) == 4

    rebuilt = withparams(cm, params(cm))
    @test rebuilt.left.amplitude == 1.0
    @test rebuilt.right.mean == rebuilt.left.mean
end

@testitem "@constrain block supports fix-current and @free" tags=[:authoring] begin
    using AstroFit

    cm = @model begin
        line = Gaussian1D(amplitude=2.0, mean=0.0, sigma=1.0)
        line
    end

    @constrain cm begin
        line.mean in (-1, 1)
    end
    @test cm.line.constraints[2] isa Bounded

    @constrain cm begin
        line.sigma
        @free line.mean
    end

    @test cm.line.constraints[2] isa Free
    @test cm.line.constraints[3] isa Fixed
    @test cm.line.constraints[3].value == 1.0
end

@testitem "standalone macros auto-rebind" tags=[:authoring] begin
    using AstroFit

    cm = @model begin
        a = Gaussian1D(amplitude=2.0, sigma=1.0)
        b = Gaussian1D(amplitude=0.5, sigma=8.0)
        a + b
    end

    old = cm
    @fix cm.a.amplitude = 1.0
    @test cm !== old
    @test cm.a.constraints[1] isa Fixed
    @test cm.a.constraints[1].value == 1.0

    @bound cm.a.sigma in (0, 10)
    @test cm.a.constraints[3] isa Bounded

    @free cm.a.sigma
    @test cm.a.constraints[3] isa Free

    @tie cm.b.mean -> cm.a.mean
    @test cm.b.constraints[2] isa Tied
    @test withparams(cm, params(cm)).right.mean == withparams(cm, params(cm)).left.mean
end

@testitem "setconstraint is the low-level edit API" tags=[:authoring] begin
    using AstroFit

    cm = @model begin
        line = Gaussian1D(amplitude=2.0, mean=0.0, sigma=1.0)
        line
    end

    cm2 = setconstraint(cm, :line, :sigma, Bounded(0.1, 10.0))
    @test cm.line.constraints[3] isa Free
    @test cm2.line.constraints[3] isa Bounded
    @test bounds(cm2) == ([-Inf, -Inf, 0.1], [Inf, Inf, 10.0])

    @test_throws ArgumentError setconstraint(cm, :missing, :sigma, Free())
    @test_throws ArgumentError setconstraint(cm, :line, :missing, Free())
end

@testitem "@constrain validation rejects bad tie masters and bad paths" tags=[:authoring] begin
    using AstroFit

    cm = @model begin
        a = Gaussian1D(amplitude=2.0, sigma=1.0)
        b = Gaussian1D(amplitude=0.5, sigma=8.0)
        a + b
    end

    @test_throws ArgumentError (m -> @constrain m begin
        a.sigma -> 2 * a.sigma
    end)(cm)

    fixed = cm
    @constrain fixed begin
        a.sigma = 1.0
    end
    @test_throws ArgumentError (m -> @constrain m begin
        b.sigma -> a.sigma
    end)(fixed)

    @test_throws ArgumentError (m -> @constrain m begin
        c.sigma = 1.0
    end)(cm)

    @test_throws ErrorException macroexpand(@__MODULE__,
        :(AstroFit.@constrain cm begin @free a end))
end

@testitem "@model expansion errors: duplicates and reserved names" tags=[:authoring] begin
    using AstroFit

    @test_throws Exception (@model begin
        g = Gaussian1D()
        g + g
    end)

    @test_throws Exception macroexpand(@__MODULE__,
        :(AstroFit.@model begin
              tree = Gaussian1D()
              tree
          end))

    @test_throws Exception macroexpand(@__MODULE__,
        :(AstroFit.@model begin
              priors = Gaussian1D()
              priors
          end))
end

@testitem "authored model: withparams stays inferred and allocation-free" tags=[:authoring, :perf] begin
    using AstroFit
    using Test: @inferred

    cm = @model begin
        cont = Linear1D(slope=0.0, intercept=1.0)
        line = Gaussian1D(amplitude=2.0, mean=6563.0, sigma=2.0)
        cont + line
    end

    @constrain cm begin
        cont.slope = 0.0
        line.sigma in (0.1, 50)
        line.amplitude -> 2 * cont.intercept
    end

    p = params(cm)
    @inferred withparams(cm, p)
    probe(cm, p) = withparams(cm, p)
    probe(cm, p)
    @test @allocated(probe(cm, p)) == 0
end

@testitem "hot loop: zero-alloc rebuild with Dual parameters" tags=[:authoring, :perf, :autodiff] begin
    using AstroFit
    using ForwardDiff
    using Test: @inferred

    cm = @model begin
        cont = Linear1D(slope=0.0, intercept=1.0)
        l1 = Gaussian1D(amplitude=2.0, mean=6563.0, sigma=2.0)
        l2 = Gaussian1D(amplitude=1.0, mean=4861.0, sigma=2.0)
        l3 = Gaussian1D(amplitude=0.5, mean=4340.0, sigma=2.0)
        cont + l1 + l2 + l3
    end

    @constrain cm begin
        l2.sigma -> l1.sigma
        l3.sigma -> l1.sigma
        l2.amplitude -> l1.amplitude / 3
        l3.amplitude -> l1.amplitude / 4
    end

    probe(cm, p) = withparams(cm, p)
    p = params(cm)
    @inferred withparams(cm, p)
    probe(cm, p)
    @test @allocated(probe(cm, p)) == 0

    pd = ForwardDiff.Dual{Nothing}.(p, 1.0)
    @inferred withparams(cm, pd)
    probe(cm, pd)
    @test @allocated(probe(cm, pd)) == 0

    xs = collect(range(4300.0, 6700.0; length=200))
    ys = render(withparams(cm, p), xs)
    loss(v) = sum(abs2, render(withparams(cm, v), xs) .- ys)
    @test loss(p) == 0.0
    g = ForwardDiff.gradient(loss, p)
    @test all(isfinite, g)
end
