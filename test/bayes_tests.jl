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

@testitem "Distributions extension: logprior and logposterior" tags=[:bayes, :extension] begin
    using AstroFit
    using Distributions

    cm = @model begin
        line = Gaussian1D(amplitude=2.0, mean=0.0, sigma=1.0)
        line
    end

    cm = @constrain cm begin
        @prior line.amplitude ~ Normal(2.0, 0.5)
        @prior line.sigma ~ LogNormal(0.0, 0.2)
    end

    expected = logpdf(Normal(2.0, 0.5), cm.line.amplitude) +
               logpdf(LogNormal(0.0, 0.2), cm.line.sigma)
    @test logprior(cm) == expected
    @test logprior(cm, paramvector(cm)) == expected

    x = [0.0, 1.0]
    y = render(cm, x)
    err = [0.1, 0.2]
    @test logposterior(cm, x, y, err) ==
          logprior(cm) + AstroFit.loglikelihood(cm, x, y, err)
    @test logposterior(cm, paramvector(cm), x, y, err) ==
          logprior(cm) + AstroFit.loglikelihood(cm, x, y, err)
end
