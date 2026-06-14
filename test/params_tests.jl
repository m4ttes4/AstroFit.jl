@testitem "round-trip Free/Fixed/Bounded" tags=[:core, :params] begin
    using AstroFit
    using Accessors: @optic

    model = Sum(Gaussian1D(amplitude=2.0, mean=0.0, sigma=1.0),
                Gaussian1D(amplitude=3.0, mean=5.0, sigma=1.5))
    spec = (
        (@optic(_.left.amplitude), Free()),
        (@optic(_.left.mean),      Fixed(0.0)),
        (@optic(_.right.sigma),    Bounded(0.1, 10.0)),
    )
    cm = compile(model, spec)

    @test nfree(cm) == 2                       # amplitude + sigma; mean is Fixed
    @test paramvector(cm) == [2.0, 1.5]

    rebuilt = withparams(cm, paramvector(cm))
    @test rebuilt.model.left.amplitude == 2.0
    @test rebuilt.model.left.mean      == 0.0  # Fixed, untouched
    @test rebuilt.model.right.sigma    == 1.5

    lower, upper = bounds_vectors(cm.spec)
    @test lower == [-Inf, 0.1]
    @test upper == [Inf, 10.0]
end

@testitem "bounds_vectors excludes Tied" tags=[:core, :params, :tied] begin
    using AstroFit
    using Accessors: @optic

    model = Sum(Gaussian1D(amplitude=2.0, sigma=1.0), Gaussian1D(amplitude=9.0))
    spec = (
        (@optic(_.left.amplitude),  Bounded(0.0, 100.0)),
        (@optic(_.left.sigma),      Free()),
        (@optic(_.right.amplitude), Tied(a -> 2a, (@optic(_.left.amplitude),))),
    )
    cm = compile(model, spec)

    lower, upper = bounds_vectors(cm.spec)       # must not MethodError on Tied
    @test length(lower) == nfree(cm) == 2        # tied target excluded
    @test lower == [0.0, -Inf]
    @test upper == [100.0, Inf]
end

@testitem "Fixed value overrides the model field; Fixed() keeps it" tags=[:core, :params] begin
    using AstroFit
    using Accessors: @optic

    # field initialized to 1.0, but spec fixes it to 5.0 → model becomes 5.0
    model = Gaussian1D(amplitude=1.0, mean=0.0, sigma=1.0)
    spec = (
        (@optic(_.amplitude), Free()),
        (@optic(_.sigma),     Fixed(5.0)),
    )
    cm = compile(model, spec)

    @test cm.model.sigma == 5.0                  # _apply_fixed wrote it once
    @test nfree(cm) == 1
    m = withparams(cm, paramvector(cm)).model
    @test m.sigma == 5.0
    @test m.amplitude == 1.0

    # Fixed with an Int into a Float64 field must promote, not error
    spec2 = ((@optic(_.amplitude), Free()), (@optic(_.sigma), Fixed(3)))
    cm2 = compile(model, spec2)
    @test cm2.model.sigma == 3.0

    # Fixed() marker: fixes at the *current* tree value, does not overwrite
    spec3 = ((@optic(_.amplitude), Free()), (@optic(_.sigma), Fixed()))
    cm3 = compile(model, spec3)
    @test cm3.model.sigma == 1.0                 # tree value untouched
    @test nfree(cm3) == 1                        # still excluded from the fit
    @test withparams(cm3, [7.0]).model.sigma == 1.0
end

@testitem "tie: b.sigma = a.sigma" tags=[:core, :tied] begin
    using AstroFit
    using Accessors: @optic

    model = Sum(Gaussian1D(sigma=1.0), Gaussian1D(sigma=99.0))
    spec = (
        (@optic(_.left.sigma),  Free()),
        (@optic(_.right.sigma), Tied(s -> s, (@optic(_.left.sigma),))),
    )
    cm = compile(model, spec)

    @test nfree(cm) == 1                        # only the master is free
    @test paramvector(cm) == [1.0]

    m = withparams(cm, [4.0]).model
    @test m.left.sigma  == 4.0
    @test m.right.sigma == 4.0                  # tied to the master
end

@testitem "invariant I1: compile resolves ties immediately" tags=[:core, :tied] begin
    using AstroFit
    using Accessors: @optic

    # right.sigma starts stale at 99.0; compile must already resolve it
    model = Sum(Gaussian1D(sigma=2.0), Gaussian1D(sigma=99.0))
    spec = (
        (@optic(_.left.sigma),  Free()),
        (@optic(_.right.sigma), Tied(s -> 3s, (@optic(_.left.sigma),))),
    )
    cm = compile(model, spec)

    @test cm.model.right.sigma == 6.0           # resolved at construction, no call needed
end

@testitem "calling a CompiledModel evaluates; vector call errors" tags=[:core, :params] begin
    using AstroFit
    using Accessors: @optic

    model = Sum(Gaussian1D(amplitude=2.0, mean=0.0, sigma=1.0),
                Const1D(value=1.0))
    spec = ((@optic(_.left.amplitude), Free()),)
    cm = compile(model, spec)

    @test render(cm, 0.0) == 3.0                # peak + constant
    xs = [-1.0, 0.0, 1.0]
    @test render(cm, xs) == [render(cm, -1.0), render(cm, 0.0), render(cm, 1.0)]
end

@testitem "tied functional: amp2 = 2*amp1 and multi-master" tags=[:core, :tied] begin
    using AstroFit
    using Accessors: @optic

    # single-master functional relation
    m1 = Sum(Gaussian1D(amplitude=3.0), Gaussian1D(amplitude=99.0))
    s1 = (
        (@optic(_.left.amplitude),  Free()),
        (@optic(_.right.amplitude), Tied(a -> 2a, (@optic(_.left.amplitude),))),
    )
    cm1 = compile(m1, s1)
    @test withparams(cm1, [5.0]).model.right.amplitude == 10.0

    # two-master relation: right.amplitude = left.amplitude * left.mean
    m2 = Sum(Gaussian1D(amplitude=2.0, mean=3.0), Gaussian1D(amplitude=99.0))
    s2 = (
        (@optic(_.left.amplitude),  Free()),
        (@optic(_.left.mean),       Free()),
        (@optic(_.right.amplitude), Tied((a, b) -> a * b,
                                         (@optic(_.left.amplitude), @optic(_.left.mean)))),
    )
    cm2 = compile(m2, s2)
    @test nfree(cm2) == 2
    @test withparams(cm2, [2.0, 3.0]).model.right.amplitude == 6.0
end

@testitem "tied with Fixed master" tags=[:core, :tied] begin
    using AstroFit
    using Accessors: @optic

    model = Sum(Gaussian1D(sigma=2.0), Gaussian1D(sigma=99.0))
    spec = (
        (@optic(_.left.sigma),  Fixed(2.0)),
        (@optic(_.right.sigma), Tied(s -> 3s, (@optic(_.left.sigma),))),
    )
    cm = compile(model, spec)

    @test nfree(cm) == 0
    @test paramvector(cm) == Float64[]

    m = withparams(cm, Float64[]).model
    @test m.left.sigma  == 2.0                  # Fixed
    @test m.right.sigma == 6.0                  # 3 * fixed master
end

@testitem "autodiff flows through ties" tags=[:core, :tied, :autodiff] begin
    using AstroFit
    using Accessors: @optic
    using ForwardDiff

    # right.amplitude = 2*left.amplitude; both Gaussians peak at x=0.
    # loss(p) = withparams(cm, p)(0.0) = p[1] + 2*p[1] = 3*p[1]  ⇒  d/dp = 3.
    model = Sum(Gaussian1D(amplitude=1.0, mean=0.0, sigma=1.0),
                Gaussian1D(amplitude=1.0, mean=0.0, sigma=1.0))
    spec = (
        (@optic(_.left.amplitude),  Free()),
        (@optic(_.right.amplitude), Tied(a -> 2a, (@optic(_.left.amplitude),))),
    )
    cm = compile(model, spec)

    loss(p) = render(withparams(cm, p), 0.0)
    g = ForwardDiff.gradient(loss, [1.0])
    @test all(isfinite, g)
    @test g ≈ [3.0]
end

@testitem "hot path: withparams type-stable and allocation-free" tags=[:core, :perf] begin
    using AstroFit
    using Accessors: @optic
    using ForwardDiff
    using Test: @inferred

    model = Sum(Gaussian1D(amplitude=2.0, sigma=1.0),
                Gaussian1D(amplitude=3.0, sigma=1.5))

    # with a Tied entry in the spec
    spec_tied = (
        (@optic(_.left.amplitude),  Free()),
        (@optic(_.left.sigma),      Bounded(0.1, 10.0)),
        (@optic(_.right.amplitude), Tied(a -> 2a, (@optic(_.left.amplitude),))),
    )
    # and without
    spec_plain = (
        (@optic(_.left.amplitude),  Free()),
        (@optic(_.left.sigma),      Bounded(0.1, 10.0)),
    )

    probe(cm, p) = withparams(cm, p)

    for spec in (spec_tied, spec_plain)
        cm = compile(model, spec)

        # Float64
        p = paramvector(cm)
        @inferred withparams(cm, p)
        probe(cm, p)                             # warmup
        @test @allocated(probe(cm, p)) == 0

        # ForwardDiff.Dual — eltype must flow through and stay allocation-free
        D = ForwardDiff.Dual{Nothing}
        pd = D.(p, 1.0)
        @inferred withparams(cm, pd)
        probe(cm, pd)                            # warmup
        @test @allocated(probe(cm, pd)) == 0
    end
end

@testitem "fit-loop rendering cost is negligible vs grid evaluation" tags=[:perf] begin
    using AstroFit
    using Accessors: @optic

    model = Sum(Sum(Linear1D(slope=0.0, intercept=1.0),
                    Gaussian1D(amplitude=2.0, mean=6563.0, sigma=2.0)),
                Gaussian1D(amplitude=0.7, mean=4861.0, sigma=2.0))
    spec = (
        ((@optic _.left.left.slope),       Free()),
        ((@optic _.left.left.intercept),   Free()),
        ((@optic _.left.right.amplitude),  Bounded(0.0, Inf)),
        ((@optic _.left.right.mean),       Free()),
        ((@optic _.left.right.sigma),      Bounded(0.0, Inf)),
        ((@optic _.right.amplitude),       Tied(a -> a/3, ((@optic _.left.right.amplitude),))),
        ((@optic _.right.mean),            Fixed(4861.0)),
        ((@optic _.right.sigma),           Tied(identity, ((@optic _.left.right.sigma),))),
    )
    cm = compile(model, spec)
    p  = paramvector(cm)
    xs = collect(range(4000.0, 7000.0, length=2000))

    rebuild(cm, p) = withparams(cm, p)
    rebuild(cm, p)                               # warmup
    @test @allocated(rebuild(cm, p)) == 0        # param rebuild: zero alloc

    # the design's fit loop: rebuild once per iteration, evaluate N times;
    # the only allocation is the broadcast output vector
    ys = render(withparams(cm, p), xs)
    loss(p) = sum(abs2, render(withparams(cm, p), xs) .- ys)
    @test loss(p) == 0.0
    t_render = @elapsed (for _ in 1:1000; rebuild(cm, p); end)
    t_eval   = @elapsed (for _ in 1:1000; render(withparams(cm, p), xs); end)
    @test t_render < t_eval                      # grid evaluation dominates
end

@testitem "gather/scatter excludes Fixed and Tied" tags=[:core, :params] begin
    using AstroFit
    using Accessors: @optic

    model = Sum(Gaussian1D(amplitude=2.0, sigma=1.0), Gaussian1D(sigma=9.0))
    spec = (
        (@optic(_.left.amplitude), Free()),
        (@optic(_.left.sigma),     Fixed(1.0)),
        (@optic(_.right.sigma),    Tied(s -> s, (@optic(_.left.sigma),))),
    )
    @test gather(model, spec) == (2.0,)          # only the Free amplitude
    m2 = scatter(model, spec, (7.0,))
    @test m2.left.amplitude == 7.0
end
