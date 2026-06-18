@testitem "Optimization extension: MLE recovery via native solve" tags=[:optimization, :extension] begin
    using AstroFit
    using Optimization, OptimizationOptimJL
    using ForwardDiff   # co-triggers AstroFitOptimizationExt (default AutoForwardDiff)
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
    using ForwardDiff   # co-triggers AstroFitOptimizationExt (default AutoForwardDiff)
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
    using ForwardDiff   # co-triggers AstroFitOptimizationExt (default AutoForwardDiff)

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

    # objective is the negative unit-variance log-likelihood (∝ least squares)
    optf = OptimizationFunction(cm, x, y)
    @test optf.f(paramvector(cm), nothing) == -AstroFit.loglikelihood(cm, x, y, nothing)

    sol = solve(OptimizationProblem(cm, x, y), NelderMead())
    fit = withparams(cm, sol.u)
    @test fit.g.amplitude ≈ 2.0 atol=1e-2
    @test fit.g.mean      ≈ 0.5 atol=1e-2
    @test fit.g.sigma     ≈ 1.2 atol=1e-2
end

@testitem "Optimization extension: priors yield MAP" tags=[:optimization, :extension] begin
    using AstroFit
    using Optimization, OptimizationOptimJL
    using ForwardDiff   # co-triggers AstroFitOptimizationExt (default AutoForwardDiff)
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

@testitem "Optimization extension: 2D fit with tuple coordinates" tags=[:optimization, :extension] begin
    using AstroFit
    using Optimization, OptimizationOptimJL
    using ForwardDiff   # co-triggers AstroFitOptimizationExt (default AutoForwardDiff)
    using Optimization.SciMLBase: successful_retcode

    truth = @model begin
        g = Gaussian2D(amplitude=5.0, x0=0.5, y0=-0.3, sigma_x=1.0, sigma_y=0.7, theta=0.0)
        g
    end
    coord = collect(-3.0:0.3:3.0)
    X = [x for x in coord, y in coord]
    Y = [y for x in coord, y in coord]
    data = render(truth, X, Y)                # noise-free: minimum sits at truth
    err  = fill(0.05, size(data))

    cm = @model begin
        g = Gaussian2D(amplitude=1.0, x0=0.0, y0=0.0, sigma_x=1.5, sigma_y=1.5, theta=0.0)
        g
    end
    cm = @constrain cm begin
        @bound g.amplitude in (0.0, Inf)
        @fix   g.theta     = 0.0           # avoid the theta/σx-σy degeneracy
        @bound g.sigma_x   in (0.1, 5.0)
        @bound g.sigma_y   in (0.1, 5.0)
    end

    # unweighted-branch plumbing: objective renders with both coordinates
    optf = OptimizationFunction(cm, (X, Y), data)
    @test optf.f(paramvector(cm), nothing) == -AstroFit.loglikelihood(cm, (X, Y), data, nothing)

    prob = OptimizationProblem(cm, (X, Y), data, err)
    sol  = solve(prob, Fminbox(LBFGS()))
    @test successful_retcode(sol)

    fit = withparams(cm, sol.u)
    @test fit.g.amplitude ≈ 5.0 atol=1e-2
    @test fit.g.x0        ≈ 0.5 atol=1e-2
    @test fit.g.y0        ≈ -0.3 atol=1e-2
    @test fit.g.sigma_x   ≈ 1.0 atol=1e-2
    @test fit.g.sigma_y   ≈ 0.7 atol=1e-2
end
