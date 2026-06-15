@testitem "Optimization extension: MLE recovery via native solve" tags=[:optimization, :extension] begin
    using AstroFit
    using Optimization, OptimizationOptimJL
    using Optimization.SciMLBase: successful_retcode

    truth = @model begin
        g = Gaussian1D(amplitude=3.0, mean=1.0, sigma=0.8)
        g
    end
    x = collect(-3.0:0.2:5.0)
    y = render(truth, x)                 # noise-free: minimum sits exactly at truth
    err = fill(0.1, length(x))

    cm = @model begin
        g = Gaussian1D(amplitude=1.0, mean=0.0, sigma=1.0)
        g
    end

    prob = OptimizationProblem(cm, x, y, err)          # default AutoForwardDiff
    @test prob.lb === nothing && prob.ub === nothing   # free-only model: no bounds passed
    sol = solve(prob, LBFGS())
    @test successful_retcode(sol)

    fit = withparams(cm, sol.u)
    @test fit.g.amplitude ≈ 3.0 atol=1e-3
    @test fit.g.mean      ≈ 1.0 atol=1e-3
    @test fit.g.sigma     ≈ 0.8 atol=1e-3
end

@testitem "Optimization extension: bounded fit passes lb/ub" tags=[:optimization, :extension] begin
    using AstroFit
    using Optimization, OptimizationOptimJL
    using Optimization.SciMLBase: successful_retcode

    truth = @model begin
        g = Gaussian1D(amplitude=3.0, mean=1.0, sigma=0.8)
        g
    end
    x = collect(-3.0:0.2:5.0)
    y = render(truth, x)

    cm = @model begin
        g = Gaussian1D(amplitude=1.0, mean=0.0, sigma=1.0)
        g
    end
    cm = @constrain cm begin
        @bound g.amplitude in (0.0, Inf)
        @bound g.mean      in (-2.0, 2.0)
        @bound g.sigma     in (0.1, 5.0)
    end

    err = fill(0.1, length(x))
    prob = OptimizationProblem(cm, x, y, err)
    @test prob.lb == [0.0, -2.0, 0.1]
    @test prob.ub == [Inf, 2.0, 5.0]

    sol = solve(prob, Fminbox(LBFGS()))
    @test successful_retcode(sol)
    @test all(prob.lb .<= sol.u .<= prob.ub)

    fit = withparams(cm, sol.u)
    @test fit.g.amplitude ≈ 3.0 atol=1e-2
    @test fit.g.mean      ≈ 1.0 atol=1e-2
    @test fit.g.sigma     ≈ 0.8 atol=1e-2
end

@testitem "Optimization extension: unweighted least-squares (err omitted)" tags=[:optimization, :extension] begin
    using AstroFit
    using Optimization, OptimizationOptimJL

    truth = @model begin
        g = Gaussian1D(amplitude=2.0, mean=0.5, sigma=1.2)
        g
    end
    x = collect(-3.0:0.2:5.0)
    y = render(truth, x)

    cm = @model begin
        g = Gaussian1D(amplitude=1.0, mean=0.0, sigma=1.0)
        g
    end

    # objective is the plain sum of squared residuals
    optf = OptimizationFunction(cm, x, y)
    @test optf.f(paramvector(cm), nothing) == sum(abs2, render(cm, x) .- y)

    sol = solve(OptimizationProblem(cm, x, y), NelderMead())
    fit = withparams(cm, sol.u)
    @test fit.g.amplitude ≈ 2.0 atol=1e-2
    @test fit.g.mean      ≈ 0.5 atol=1e-2
    @test fit.g.sigma     ≈ 1.2 atol=1e-2
end

@testitem "Optimization extension: priors yield MAP" tags=[:optimization, :extension] begin
    using AstroFit
    using Optimization, OptimizationOptimJL
    using Distributions

    truth = @model begin
        g = Gaussian1D(amplitude=3.0, mean=1.0, sigma=0.8)
        g
    end
    x = collect(-3.0:0.2:5.0)
    y = render(truth, x)
    err = fill(0.1, length(x))

    cm = @model begin
        g = Gaussian1D(amplitude=1.0, mean=0.0, sigma=1.0)
        g
    end

    # MLE alone recovers amplitude ≈ 3; a tight prior far above pulls the MAP up
    mle = withparams(cm, solve(OptimizationProblem(cm, x, y, err), LBFGS()).u)
    @test mle.g.amplitude ≈ 3.0 atol=1e-2

    prior_cm = @constrain cm begin
        @prior g.amplitude ~ Normal(10.0, 0.05)
    end
    map_fit = withparams(prior_cm, solve(OptimizationProblem(prior_cm, x, y, err), LBFGS()).u)
    @test map_fit.g.amplitude > 5.0
end
