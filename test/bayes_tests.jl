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
    @test logprior(cm) == logpdf(prior, cm.line.model.sigma)
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

@testitem "priors reject fixed and tied targets" tags = [:bayes] begin
    using AstroFit
    using Distributions

    cm = @model begin
        a = Gaussian1D(amplitude = 2.0, sigma = 1.0)
        b = Gaussian1D(amplitude = 1.0, sigma = 9.0)
        a + b
    end

    fixed = cm
    @constrain fixed begin
        a.sigma
    end
    @test_throws ArgumentError (
        m -> @constrain m begin
            a.sigma ~ LogNormal(0.0, 1.0)
        end
    )(fixed)

    tied = cm
    @constrain tied begin
        b.sigma -> a.sigma
    end
    @test_throws ArgumentError (
        m -> @constrain m begin
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
    expected = sum(abs2, (render(cm, x) .- y) ./ err)

    @test chi2(cm.tree, (x,), y, err) == expected
    @test chi2(cm.tree, (x,), y, nothing) == sum(abs2, render(cm, x) .- y)
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
    xs = repeat(collect(-2.0:0.5:2.0), outer = 9)
    ys = repeat(collect(-2.0:0.5:2.0), inner = 9)
    y = render(cm, xs, ys)
    u = params(cm)

    f = ObjectiveFunction(cm, (xs, ys), y)
    @test f(u) ≈ 0.0 atol = 1e-20
    @test f(u .+ 0.1) > 0.0
end

@testitem "ObjectiveFunction: statistic is a callable" tags = [:bayes] begin
    using AstroFit

    cm = @model begin
        g = Gaussian1D(amplitude = 1.0, mean = 0.0, sigma = 1.0)
        g
    end
    x = collect(-2.0:0.5:2.0)
    y = render(cm, x)
    u = params(cm)

    f = ObjectiveFunction(cm, x, y; statistic = logposterior)
    @test f(u) == logposterior(f, u)

    doublechi2(f, p) = 2 * chi2(f, p)
    fc = ObjectiveFunction(cm, x, y; statistic = doublechi2)
    @test fc(u) == 2 * chi2(fc, u)
end

@testitem "ObjectiveFunction: Bayesian extensions ignore statistic" tags = [:bayes] begin
    using AstroFit, LogDensityProblems, Pigeons, Random

    cm = @model begin
        g = Gaussian1D(amplitude = 1.0, mean = 0.0, sigma = 1.0)
        g
    end
    @bound cm.g.amplitude in (0, 10)
    @bound cm.g.mean in (-5, 5)
    @bound cm.g.sigma in (0.1, 10)
    x = collect(-2.0:0.5:2.0)
    y = render(cm, x)
    u = params(cm)

    f = ObjectiveFunction(cm, x, y) # default statistic = chi2
    @test LogDensityProblems.logdensity(f, u) == logposterior(f, u)
    @test LogDensityProblems.dimension(f) == f.ndim

    @test_throws ArgumentError Pigeons.initialization(f, Random.default_rng(), 1)
    @test_throws ArgumentError Pigeons.default_reference(f)

    fp = ObjectiveFunction(cm, x, y; statistic = logposterior)
    @test Pigeons.initialization(fp, Random.default_rng(), 1) isa Vector{Float64}
    @test Pigeons.default_reference(fp) isa Pigeons.DistributionLogPotential
end

@testitem "ObjectiveFunction: data validation" tags = [:bayes] begin
    using AstroFit

    cm = @model begin
        c = Const1D(value = 1.0)
        c
    end
    x = [1.0, 2.0, 3.0]
    y = [1.0, 1.0, 1.0]

    @test_throws ArgumentError ObjectiveFunction(cm, x, y[1:2])
    @test_throws ArgumentError ObjectiveFunction(cm, x, y, [1.0, 0.0, 1.0])
    @test_throws ArgumentError ObjectiveFunction(cm, x, y, [1.0, -1.0, 1.0])
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
end
