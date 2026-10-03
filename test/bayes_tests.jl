@testitem "logposterior is the log prior plus the Gaussian log likelihood" tags = [:bayes] begin
    using AstroFit, Distributions

    cm = @model begin
        g = Gaussian1D(amplitude = 1.0, mean = 0.0, sigma = 1.0)
        g
    end
    @constrain cm begin
        g.sigma in (0, Inf)
        g.amplitude ~ Normal(1.0, 1.0)
        g.mean ~ Normal(0.0, 1.0)
        g.sigma ~ LogNormal(0.0, 1.0)
    end
    @test bounds(cm) == ([-Inf, -Inf, 0.0], [Inf, Inf, Inf])  # priors are not bounds

    x = collect(-2.0:0.5:2.0)
    err = fill(0.2, length(x))
    y = render(cm, x) .+ 0.1 .* sin.(x)  # off the model, deterministically
    f = ObjectiveFunction(cm, x, y, err; statistic = logposterior)
    p = [1.2, 0.1, 0.9]

    # Distributions is the independent reference for both terms.
    ll = sum(logpdf.(Normal.(render(withparams(cm, p), x), err), y))
    lp = logpdf(Normal(1.0, 1.0), 1.2) + logpdf(Normal(0.0, 1.0), 0.1) + logpdf(LogNormal(0.0, 1.0), 0.9)
    @test loglikelihood(f, p) ≈ ll
    @test logprior(f, p) ≈ lp
    @test f(p) ≈ lp + ll
    @test neglogposterior(f, p) ≈ -(lp + ll)
end

@testitem "the last prior set on a parameter wins, and survives later constraint blocks" tags = [:bayes] begin
    using AstroFit, Distributions

    cm = @model begin
        line = Gaussian1D(amplitude = 1.0, mean = 0.0, sigma = 1.0)
        line
    end
    @constrain cm begin
        line.amplitude ~ Normal()
        line.mean ~ Normal()
        line.sigma ~ LogNormal(0.0, 1.0)
    end
    @prior cm.line.sigma ~ LogNormal(1.0, 0.5)
    @constrain cm begin  # resets the constraints, not the priors
        line.sigma in (0, Inf)
    end
    @test bounds(cm) == ([-Inf, -Inf, 0.0], [Inf, Inf, Inf])

    x = [0.0, 1.0]
    f = ObjectiveFunction(cm, x, render(cm, x); statistic = logposterior)
    p = [1.0, 0.0, 2.0]
    @test logprior(f, p) ≈ logpdf(Normal(), 1.0) + logpdf(Normal(), 0.0) + logpdf(LogNormal(1.0, 0.5), 2.0)
end

@testitem "priors target free parameters, and every free parameter needs one" tags = [:bayes] begin
    using AstroFit, Distributions

    cm = @model begin
        a = Gaussian1D()
        b = Gaussian1D()
        a + b
    end
    @test_throws "prior target `a.sigma` must be a free parameter" (
        m -> @constrain m begin
            a.sigma
            a.sigma ~ LogNormal()
        end
    )(cm)
    @test_throws "prior target `b.sigma` must be a free parameter" (
        m -> @constrain m begin
            b.sigma -> a.sigma
            b.sigma ~ LogNormal()
        end
    )(cm)

    partial = cm
    @prior partial.a.sigma ~ LogNormal()
    @test_throws "parameter `a_amplitude` has no prior" ObjectiveFunction(partial, [0.0], [0.0])
    @test_throws "no priors set on this model" logposterior(ObjectiveFunction(cm, [0.0], [0.0]), params(cm))
end

@testitem "priors survive Prefab composition" tags = [:bayes, :prefab] begin
    using AstroFit, Distributions

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
    f = ObjectiveFunction(cm, [0.0], [1.0]; statistic = logposterior)
    p = [1.1, 0.2, 0.9]
    @test logprior(f, p) ≈ logpdf(Normal(), 1.1) + logpdf(Normal(), 0.2) + logpdf(LogNormal(), 0.9)
end

@testitem "LogDensityProblems and Pigeons see the log posterior, labeled by parameter name" tags = [:bayes] begin
    using AstroFit, LogDensityProblems, Pigeons, Random, Distributions

    cm = @model begin
        g = Gaussian1D(amplitude = 1.0, mean = 0.0, sigma = 1.0)
        g
    end
    @constrain cm begin
        g.amplitude in (0, 10)
        g.mean in (-5, 5)
        g.sigma in (0.1, 10)
        g.amplitude ~ Uniform(0, 10)
        g.mean ~ Uniform(-5, 5)
        g.sigma ~ Uniform(0.1, 10)
    end
    x = collect(-2.0:0.5:2.0)
    y = render(cm, x)
    u = params(cm)

    # LogDensityProblems ignores `statistic`: the density is always the log posterior.
    f = ObjectiveFunction(cm, x, y)
    @test LogDensityProblems.logdensity(f, u) ≈ logposterior(f, u)
    @test LogDensityProblems.dimension(f) == 3

    # Pigeons draws from the priors, so only a log-density objective qualifies.
    needs = "Pigeons requires a log-density statistic"
    @test_throws needs Pigeons.initialization(f, Random.Xoshiro(1), 1)
    @test_throws needs Pigeons.default_reference(f)

    fp = ObjectiveFunction(cm, x, y; statistic = logposterior)
    init = Pigeons.initialization(fp, Random.Xoshiro(1), 1)
    @test init isa Vector{Float64}
    lo, hi = bounds(cm)
    @test all(lo .<= init .<= hi)  # inside every prior's support
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
