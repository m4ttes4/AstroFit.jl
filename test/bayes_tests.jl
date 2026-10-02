@testitem "priors: stored separately from mechanical constraints" tags = [:bayes] begin
    using AstroFit
    using Distributions

    cm = @model begin
        line = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        line
    end

    prior = LogNormal(0.0, 1.0)
    @constrain cm begin
        line.sigma in (0, Inf)
        line.sigma ~ prior
    end

    @test nfree(cm) == 3
    @test AstroFit.params(cm) == [2.0, 0.0, 1.0]
    @test length(getfield(cm, :priors)) == 1
    @test only(getfield(cm, :priors))[2] === prior
    @test bounds(cm) == ([-Inf, -Inf, 0.0], [Inf, Inf, Inf])
end

@testitem "priors: user priors override by target" tags = [:bayes] begin
    using AstroFit
    using Distributions

    cm = @model begin
        line = Gaussian1D(amplitude = 1.0, mean = 0.0, sigma = 1.0)
        line
    end

    factory = LogNormal(0.0, 1.0)
    user = LogNormal(1.0, 0.5)

    @constrain cm begin
        line.sigma ~ factory
    end
    @constrain cm begin
        line.sigma ~ user
    end

    priors = getfield(cm, :priors)
    @test length(priors) == 1
    @test only(priors)[1] == (:line, :sigma)
    @test only(priors)[2] === user
end

@testitem "priors survive Prefab composition" tags = [:bayes, :prefab] begin
    using AstroFit
    using Distributions

    prefab = @model begin
        line = Gaussian1D()
        line
    end
    @constrain prefab begin
        line.amplitude ~ Normal()
        line.mean ~ Normal()
        line.sigma ~ LogNormal()
    end

    cm = @model begin
        halpha = prefab
        halpha
    end
    objective = ObjectiveFunction(cm, [0.0], render(cm, [0.0]); statistic = logposterior)

    @test isfinite(objective(AstroFit.params(cm)))
end

@testitem "priors reject fixed and tied targets" tags = [:bayes] begin
    using AstroFit
    using Distributions

    cm = @model begin
        a = Gaussian1D(amplitude = 2.0, sigma = 1.0)
        b = Gaussian1D(amplitude = 1.0, sigma = 9.0)
        a + b
    end

    fixed = cm
    @test_throws ArgumentError (
        m -> @constrain m begin
            a.sigma
            a.sigma ~ LogNormal(0.0, 1.0)
        end
    )(fixed)

    tied = cm
    @test_throws ArgumentError (
        m -> @constrain m begin
            b.sigma -> a.sigma
            b.sigma ~ LogNormal(0.0, 1.0)
        end
    )(tied)
end

@testitem "chi2: weighted residuals" tags = [:bayes] begin
    using AstroFit

    cm = @model begin
        c = Const1D(value = 2.0)
        c
    end

    x = [1.0, 2.0, 3.0]
    y = [2.0, 3.0, 0.0]
    err = [1.0, 2.0, 4.0]
    # residuals 0, -1, 2: weighted 0 + (1/2)² + (2/4)², unweighted 0 + 1 + 4
    @test chi2(ObjectiveFunction(cm, x, y, err), params(cm)) == 0.5
    @test chi2(ObjectiveFunction(cm, x, y), params(cm)) == 5.0
end

@testitem "ObjectiveFunction: 1D evaluation and Optimization.jl convention" tags = [:bayes] begin
    using AstroFit

    cm = @model begin
        g = Gaussian1D(amplitude = 1.0, mean = 0.0, sigma = 1.0)
        g
    end
    x = collect(-2.0:0.5:2.0)
    y = render(cm, x)
    u = params(cm)

    f = ObjectiveFunction(cm, x, y)
    @test f(u) == 0.0
    @test f(u, nothing) == f(u)

    u2 = u .+ 0.3
    @test f(u2) > 0.0
    @test f(u) <= f(u2)
end

@testitem "ObjectiveFunction: 2D evaluation" tags = [:bayes] begin
    using AstroFit

    cm = @model begin
        g = Gaussian2D(amplitude = 2.0, x0 = 0.0, y0 = 0.0, sigma = 1.0, q = 1.0, theta = 0.0)
        g
    end
    c = collect(-2.0:0.5:2.0)
    pts = Coords(c, c)
    y = render(cm, pts)
    u = params(cm)

    f = ObjectiveFunction(cm, pts, y)
    @test f(u) == 0.0
    @test f(u .+ 0.1) > 0.0

    # A masked fit is a mask on the points: corrupted pixels outside it are ignored.
    pix = CartesianIndices((9, 9))
    img = render(cm, pix)
    bad = copy(img); bad[1, :] .= 1.0e6
    keep = trues(9, 9); keep[1, :] .= false
    @test ObjectiveFunction(cm, pix[keep], bad[keep])(u) == 0.0
    @test ObjectiveFunction(cm, pix, bad)(u) > 1.0e12
end

@testitem "ObjectiveFunction: statistic is a callable" tags = [:bayes] begin
    using AstroFit
    using Distributions

    cm = @model begin
        g = Gaussian1D(amplitude = 1.0, mean = 0.0, sigma = 1.0)
        g
    end
    @constrain cm begin
        g.amplitude ~ Normal(1.0, 1.0)
        g.mean ~ Normal(0.0, 1.0)
        g.sigma ~ Normal(1.0, 1.0)
    end
    x = collect(-2.0:0.5:2.0)
    y = render(cm, x)
    u = AstroFit.params(cm)

    f = ObjectiveFunction(cm, x, y; statistic = logposterior)
    @test f(u) == logposterior(f, u)

    doublechi2(f, p) = 2 * chi2(f, p)
    fc = ObjectiveFunction(cm, x, y; statistic = doublechi2)
    @test fc(u) == 2 * chi2(fc, u)
end

@testitem "ObjectiveFunction: Bayesian extensions ignore statistic" tags = [:bayes] begin
    using AstroFit, LogDensityProblems, Pigeons, Random, Distributions

    cm = @model begin
        g = Gaussian1D(amplitude = 1.0, mean = 0.0, sigma = 1.0)
        g
    end
    @bound cm.g.amplitude in (0, 10)
    @bound cm.g.mean in (-5, 5)
    @bound cm.g.sigma in (0.1, 10)
    @constrain cm begin
        g.amplitude ~ Uniform(0, 10)
        g.mean ~ Uniform(-5, 5)
        g.sigma ~ Uniform(0.1, 10)
    end
    x = collect(-2.0:0.5:2.0)
    y = render(cm, x)
    u = AstroFit.params(cm)

    f = ObjectiveFunction(cm, x, y) # default statistic = chi2
    @test LogDensityProblems.logdensity(f, u) == logposterior(f, u)
    @test LogDensityProblems.dimension(f) == f.ndim

    @test_throws ArgumentError Pigeons.initialization(f, Random.default_rng(), 1)
    @test_throws ArgumentError Pigeons.default_reference(f)

    fp = ObjectiveFunction(cm, x, y; statistic = logposterior)
    @test Pigeons.initialization(fp, Random.default_rng(), 1) isa Vector{Float64}
    @test Pigeons.default_reference(fp) isa Pigeons.DistributionLogPotential

    # sample_names labels our targets by parameter name, and leaves Pigeons'
    # default in charge of every other target.
    potential(target) = Pigeons.InterpolatedLogPotential(
        Pigeons.InterpolatingPath(Pigeons.default_reference(fp), target, Pigeons.LinearInterpolator()), 0.5
    )
    @test Pigeons.sample_names(u, potential(fp)) == [:g_amplitude, :g_mean, :g_sigma, :log_density]
    @test Pigeons.sample_names(u, potential(Pigeons.toy_mvn_target(3))) ==
        [:param_1, :param_2, :param_3, :log_density]
end

@testitem "ObjectiveFunction: data validation" tags = [:bayes] begin
    using AstroFit

    cm = @model begin
        c = Const1D(value = 1.0)
        c
    end
    x = [1.0, 2.0, 3.0]
    y = [1.0, 1.0, 1.0]

    @test_throws DimensionMismatch ObjectiveFunction(cm, x, y[1:2])
    @test_throws DimensionMismatch ObjectiveFunction(cm, x, y, [1.0, 1.0])
    @test_throws ArgumentError ObjectiveFunction(cm, x, y, [1.0, 0.0, 1.0])
    @test_throws ArgumentError ObjectiveFunction(cm, x, y, [1.0, -1.0, 1.0])
    @test_throws "at least one data point" ObjectiveFunction(cm, Float64[], Float64[])
    @test_throws "takes an array of points" ObjectiveFunction(cm, (x,), y)  # the old tuple form

    # render's rules, checked once at construction on a single point
    disk = @model begin
        d = Gaussian2D()
        d
    end
    img = zeros(3, 4)
    @test_throws "an array of numbers with 2 dimension(s) is not a set of points" ObjectiveFunction(disk, img, img)

    # one value per point is part of the signature
    struct Rot{T <: Real} <: AbstractModel{2, 2}
        theta::T
    end
    rot = @model begin
        r = Rot(0.1)
        r
    end
    @test_throws "a fit needs one value per point; this model produces 2" ObjectiveFunction(rot, CartesianIndices(img), img)
end

@testitem "ObjectiveFunction: allocation-free hot path" tags = [:bayes] begin
    using AstroFit

    cm = @model begin
        g = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        g
    end
    mk(n) = (x = collect(range(-3, 3; length = n)); (x, render(cm, x), fill(0.1, n)))

    x1, y1, e1 = mk(50)
    x2, y2, e2 = mk(5000)
    f1 = ObjectiveFunction(cm, x1, y1, e1)
    f2 = ObjectiveFunction(cm, x2, y2, e2)
    u = params(cm)
    f1(u); f2(u)

    a1 = @allocated f1(u)
    a2 = @allocated f2(u)
    @test a1 == a2
    @test a2 < 512

    # 2D points, pixels or physical axes: the prediction is never materialized.
    cm2 = @model begin
        g = Gaussian2D(amplitude = 2.0, x0 = 5.0, y0 = 5.0, sigma = 1.0, q = 1.0, theta = 0.0)
        g
    end
    v = params(cm2)
    for mk in (
            n -> CartesianIndices((n, n)),
            n -> (c = collect(range(0.0, 10.0; length = n)); Coords(c, c)),
        )
        local f1, f2 = (ObjectiveFunction(cm2, mk(n), render(cm2, mk(n))) for n in (10, 60))
        f1(v); f2(v)
        @test @allocated(f1(v)) == @allocated(f2(v))
        @test @allocated(f2(v)) < 512
    end
end

@testitem "ObjectiveFunction: chi2 is inferred and differentiable on 2D points" tags = [:bayes] begin
    using AstroFit, ForwardDiff, Test

    cm = @model begin
        a = Gaussian2D(amplitude = 2.0, x0 = 0.3, y0 = -0.2, sigma = 1.1, q = 0.8, theta = 0.4)
        b = Sersic2D(amplitude = 1.0, x0 = -0.5, y0 = 0.4, r_eff = 1.5, n = 1.5, q = 0.9, theta = 0.1)
        a + b
    end
    pts = Coords(collect(range(-3.0, 3.0; length = 15)), collect(range(-2.0, 2.0; length = 11)))
    f = ObjectiveFunction(cm, pts, render(cm, pts), fill(0.1, size(pts)))
    p = params(cm) .+ 0.05 .* (1:nfree(cm)) ./ nfree(cm)  # away from the minimum

    pd = ForwardDiff.Dual{Nothing}.(p, 1.0)
    @test @inferred(chi2(f, pd)) isa ForwardDiff.Dual
    # the fused residual broadcast over dual parameters agrees with the χ² loop
    md = withparams(cm, pd)
    @test sum(abs2, (render.(md, pts) .- f.y) ./ f.err) ≈ chi2(f, pd)

    # Central differences are an independent reference: error ~h² ≈ 1e-12, rounding ~eps/h ≈ 1e-10.
    g = ForwardDiff.gradient(f, p)
    h = 1.0e-6
    for i in eachindex(p)
        step = zeros(length(p)); step[i] = h
        @test g[i] ≈ (f(p .+ step) - f(p .- step)) / 2h rtol = 1.0e-6
    end
end

@testitem "ObjectiveFunction: fits through Optimization" tags = [:bayes] begin
    using AstroFit
    using Optimization, OptimizationOptimJL

    x = collect(-10.0:0.2:10.0)
    truth = @model begin
        line = Gaussian1D(amplitude = 2.0, mean = 1.0, sigma = 1.0)
        cont = Const1D(value = 0.5)
        line + cont
    end
    y = render(truth, x)

    start = @model begin
        line = Gaussian1D(amplitude = 1.0, mean = 0.0, sigma = 2.0)
        cont = Const1D(value = 0.1)
        line + cont
    end
    sol = solve(OptimizationProblem(start, x, y), Optim.LBFGS())
    @test sol.u ≈ [2.0, 1.0, 1.0, 0.5] rtol = 1.0e-4
end
