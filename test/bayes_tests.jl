@testitem "priors: stored separately from mechanical constraints" tags=[:bayes] begin
    using AstroFit

    m = @model begin
        line = Gaussian1D(amplitude=2.0, mean=0.0, sigma=1.0)
        line
    end

    cm = @constrain m begin
        @bound line.sigma in (0, Inf)
        @prior line.sigma ~ :positive_width
    end

    @test nfree(cm) == 3
    @test paramvector(cm) == [2.0, 0.0, 1.0]
    @test length(getfield(cm, :priors)) == 1
    @test only(getfield(cm, :priors))[2] == :positive_width
    @test bounds_vectors(cm.spec) == ([-Inf, -Inf, 0.0], [Inf, Inf, Inf])
end

@testitem "priors: prefab priors travel and user priors override by target" tags=[:bayes, :prefab] begin
    using AstroFit

    function PriorLine(; center)
        m = @model begin
            line = Gaussian1D(amplitude=1.0, mean=center, sigma=1.0)
            line
        end
        @constrain m begin
            @prior line.sigma ~ :factory
        end
    end

    spectrum = @model begin
        Ha = PriorLine(center=6563.0)
        Hb = PriorLine(center=4861.0)
        Ha + Hb
    end

    @test length(getfield(spectrum, :priors)) == 2

    cm = @constrain spectrum begin
        @prior Ha.line.sigma ~ :user
    end

    priors = getfield(cm, :priors)
    @test length(priors) == 2
    @test count(==(:user), last.(priors)) == 1
    @test count(==(:factory), last.(priors)) == 1
end

@testitem "priors: invalid targets and missing Distributions fallback" tags=[:bayes] begin
    using AstroFit

    m = @model begin
        a = Gaussian1D(amplitude=2.0, sigma=1.0)
        b = Gaussian1D(amplitude=1.0, sigma=9.0)
        a + b
    end

    fixed = @constrain m begin
        @fix a.sigma
    end
    @test_throws ArgumentError @constrain fixed begin
        @prior a.sigma ~ :fixed
    end

    tied = @constrain m begin
        @tie b.sigma = a.sigma
    end
    @test_throws ArgumentError @constrain tied begin
        @prior b.sigma ~ :tied
    end

    cm = @constrain m begin
        @prior a.sigma ~ :dummy
    end
    @test_throws ArgumentError logprior(cm)
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
    @test AstroFit.loglikelihood(cm, paramvector(cm), x, y, err) == expected
    @test_throws ArgumentError AstroFit.loglikelihood(cm, x, y, [1.0, 0.0, 1.0])
    @test_throws ArgumentError AstroFit.loglikelihood(cm, x, y[1:2], err)
end

@testitem "loglikelihood: noise-free fit (err = nothing) uses unit variance" tags=[:bayes] begin
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
    @test AstroFit.loglikelihood(cm, paramvector(cm), x, y, nothing) == expected
    # identical to a σ=1 weighted likelihood
    @test AstroFit.loglikelihood(cm, x, y, nothing) ==
          AstroFit.loglikelihood(cm, x, y, ones(length(y)))
    # coordinates are still validated when err is omitted
    @test_throws ArgumentError AstroFit.loglikelihood(cm, x, y[1:2], nothing)
    # no priors → logposterior falls back to the likelihood
    @test logposterior(cm, x, y, nothing) == AstroFit.loglikelihood(cm, x, y, nothing)
end

@testitem "loglikelihood: allocation-free hot path (no per-point arrays)" tags=[:bayes] begin
    using AstroFit

    cm = @model begin
        g = Gaussian1D(amplitude=2.0, mean=0.0, sigma=1.0)
        g
    end
    mk(n) = (x = collect(range(-3, 3; length=n)); (x, render(cm, x), fill(0.1, n)))

    x1, y1, e1 = mk(50)
    x2, y2, e2 = mk(5000)
    AstroFit.loglikelihood(cm, x1, y1, e1)          # warm up both sizes
    AstroFit.loglikelihood(cm, x2, y2, e2)

    # the old implementation materialised O(n) prediction/residual arrays; the
    # fused version must not — allocations stay constant as n grows 100×.
    a1 = @allocated AstroFit.loglikelihood(cm, x1, y1, e1)
    a2 = @allocated AstroFit.loglikelihood(cm, x2, y2, e2)
    @test a1 == a2
    @test a2 < 256
end

@testitem "objective: solver-agnostic minimisation target (no Optimization needed)" tags=[:bayes] begin
    using AstroFit

    cm = @model begin
        g = Gaussian1D(amplitude=1.0, mean=0.0, sigma=1.0)
        g
    end
    x = collect(-2.0:0.5:2.0)
    y = render(cm, x)
    u = paramvector(cm)

    # noise-free target = negative unit-variance log-likelihood
    f0 = objective(cm, x, y)
    @test f0(u) == -logposterior(cm, u, x, y, nothing)

    # weighted target threads err through
    err = fill(0.5, length(y))
    fe = objective(cm, x, y; err=err)
    @test fe(u) == -logposterior(cm, u, x, y, err)

    # callable on a perturbed vector — what a BYO minimiser does each step
    u2 = u .+ 0.3
    @test f0(u2) == -logposterior(cm, u2, x, y, nothing)
    @test f0(u) <= f0(u2)            # minimum sits at the (noise-free) truth
end
