@testitem "priors: stored separately from mechanical constraints" tags=[:bayes] begin
    using AstroFit
    using Distributions

    cm = @model begin
        line = Gaussian1D(amplitude=2.0, mean=0.0, sigma=1.0)
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

@testitem "priors: user priors override by target" tags=[:bayes] begin
    using AstroFit
    using Distributions

    cm = @model begin
        line = Gaussian1D(amplitude=1.0, mean=0.0, sigma=1.0)
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

@testitem "priors reject fixed and tied targets" tags=[:bayes] begin
    using AstroFit
    using Distributions

    cm = @model begin
        a = Gaussian1D(amplitude=2.0, sigma=1.0)
        b = Gaussian1D(amplitude=1.0, sigma=9.0)
        a + b
    end

    fixed = cm
    @constrain fixed begin
        a.sigma
    end
    @test_throws ArgumentError (m -> @constrain m begin
        a.sigma ~ LogNormal(0.0, 1.0)
    end)(fixed)

    tied = cm
    @constrain tied begin
        b.sigma -> a.sigma
    end
    @test_throws ArgumentError (m -> @constrain m begin
        b.sigma ~ LogNormal(0.0, 1.0)
    end)(tied)
end

@testitem "loglikelihood: Gaussian independent errors" tags=[:bayes] begin
    using AstroFit

    cm = @model begin
        c = Const1D(value=2.0)
        c
    end

    x = [1.0, 2.0, 3.0]
    y = [2.0, 3.0, 0.0]
    err = [1.0, 2.0, 4.0]
    residual = (render(cm, x) .- y) ./ err
    expected = -0.5 * sum(abs2, residual) - sum(log, err) - length(y) / 2 * log(2π)

    @test AstroFit.loglikelihood(cm, x, y, err) == expected
    @test AstroFit.loglikelihood(cm, params(cm), x, y, err) == expected
    @test_throws ArgumentError AstroFit.loglikelihood(cm, x, y, [1.0, 0.0, 1.0])
    @test_throws ArgumentError AstroFit.loglikelihood(cm, x, y[1:2], err)
end

@testitem "loglikelihood: noise-free fit uses unit variance" tags=[:bayes] begin
    using AstroFit

    cm = @model begin
        c = Const1D(value=2.0)
        c
    end

    x = [1.0, 2.0, 3.0]
    y = [2.0, 3.0, 0.0]
    r = render(cm, x) .- y
    expected = -0.5 * sum(abs2, r) - length(y) / 2 * log(2π)

    @test AstroFit.loglikelihood(cm, x, y, nothing) == expected
    @test AstroFit.loglikelihood(cm, params(cm), x, y, nothing) == expected
    @test AstroFit.loglikelihood(cm, x, y, nothing) ==
          AstroFit.loglikelihood(cm, x, y, ones(length(y)))
    @test_throws ArgumentError AstroFit.loglikelihood(cm, x, y[1:2], nothing)
    @test logposterior(cm, x, y, nothing) == AstroFit.loglikelihood(cm, x, y, nothing)
end

@testitem "loglikelihood: allocation-free hot path" tags=[:bayes] begin
    using AstroFit

    cm = @model begin
        g = Gaussian1D(amplitude=2.0, mean=0.0, sigma=1.0)
        g
    end
    mk(n) = (x = collect(range(-3, 3; length=n)); (x, render(cm, x), fill(0.1, n)))

    x1, y1, e1 = mk(50)
    x2, y2, e2 = mk(5000)
    AstroFit.loglikelihood(cm, x1, y1, e1)
    AstroFit.loglikelihood(cm, x2, y2, e2)

    a1 = @allocated AstroFit.loglikelihood(cm, x1, y1, e1)
    a2 = @allocated AstroFit.loglikelihood(cm, x2, y2, e2)
    @test a1 == a2
    @test a2 < 512
end

@testitem "objective: solver-agnostic minimisation target" tags=[:bayes] begin
    using AstroFit

    cm = @model begin
        g = Gaussian1D(amplitude=1.0, mean=0.0, sigma=1.0)
        g
    end
    x = collect(-2.0:0.5:2.0)
    y = render(cm, x)
    u = params(cm)

    f0 = objective(cm, x, y)
    @test f0(u) == -logposterior(cm, u, x, y, nothing)

    err = fill(0.5, length(y))
    fe = objective(cm, x, y; err=err)
    @test fe(u) == -logposterior(cm, u, x, y, err)

    u2 = u .+ 0.3
    @test f0(u2) == -logposterior(cm, u2, x, y, nothing)
    @test f0(u) <= f0(u2)
end
