@testitem "chi2 is the sum of squared, error-weighted residuals" tags = [:fit] begin
    using AstroFit

    cm = @model begin
        c = Const1D(value = 2.0)
        c
    end
    x = [1.0, 2.0, 3.0]
    y = [2.0, 3.0, 0.0]
    err = [1.0, 2.0, 4.0]
    # residuals 0, -1, 2: weighted 0 + (1/2)² + (2/4)², unweighted 0 + 1 + 4
    @test chi2(ObjectiveFunction(cm, x, y, err), params(cm)) ≈ 0.5
    @test chi2(ObjectiveFunction(cm, x, y), params(cm)) ≈ 5.0

    # f(p) is statistic(f, p); f(p, _) is the Optimization.jl convention.
    f = ObjectiveFunction(cm, x, y, err; statistic = (f, p) -> 2chi2(f, p))
    @test f(params(cm)) ≈ 1.0
    @test f(params(cm), nothing) ≈ 1.0
end

@testitem "the objective vanishes at the truth on any set of points, masks included" tags = [:fit, :twod] begin
    using AstroFit

    # The data are rendered from the model itself, so at the truth every residual is
    # exactly zero: `== 0.0` is the claim, not a tolerance shortcut.
    cm = @model begin
        g = Gaussian2D(amplitude = 2.0, x0 = 4.0, y0 = 5.0, sigma = 1.0, q = 0.8, theta = 0.3)
        g
    end
    u = params(cm)
    c = collect(1.0:9.0)
    for pts in (Coords(c, c), CartesianIndices((9, 9)), [(1.0, 2.0), (4.0, 5.5)])
        f = ObjectiveFunction(cm, pts, render(cm, pts))
        @test f(u) == 0.0
        @test f(u .+ 0.1) > 0.0
    end

    # A masked fit is a mask on the points: corrupted pixels outside it are ignored.
    pix = CartesianIndices((9, 9))
    bad = render(cm, pix)
    bad[1, :] .= 1.0e6
    keep = trues(9, 9)
    keep[1, :] .= false
    @test ObjectiveFunction(cm, pix[keep], bad[keep])(u) == 0.0
    @test ObjectiveFunction(cm, pix, bad)(u) > 1.0e12
end

@testitem "ObjectiveFunction validates its data once, at construction" tags = [:fit] begin
    using AstroFit

    cm = @model begin
        c = Const1D(value = 1.0)
        c
    end
    x = [1.0, 2.0, 3.0]
    y = [1.0, 1.0, 1.0]

    @test_throws DimensionMismatch ObjectiveFunction(cm, x, y[1:2])
    @test_throws DimensionMismatch ObjectiveFunction(cm, x, y, [1.0, 1.0])
    @test_throws "all `err` values must be positive" ObjectiveFunction(cm, x, y, [1.0, 0.0, 1.0])
    @test_throws "all `err` values must be positive" ObjectiveFunction(cm, x, y, [1.0, -1.0, 1.0])
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

@testitem "the objective allocates nothing, whatever the number of points" tags = [:fit, :perf] setup = [Fixtures] begin
    using AstroFit

    # The prediction is never materialized: 1D with and without errors, pixels, axes.
    allocs(f, p) = @allocated f(p)
    cm = Fixtures.constrained
    u = params(cm)
    x = collect(range(-3.0, 8.0; length = 5000))
    for f in (ObjectiveFunction(cm, x, render(cm, x)), ObjectiveFunction(cm, x, render(cm, x), fill(0.1, 5000)))
        f(u)
        @test allocs(f, u) == 0
    end

    cm2 = @model begin
        g = Gaussian2D(amplitude = 2.0, x0 = 5.0, y0 = 5.0, sigma = 1.0, q = 1.0, theta = 0.0)
        g
    end
    v = params(cm2)
    c = collect(range(0.0, 10.0; length = 60))
    for pts in (CartesianIndices((60, 60)), Coords(c, c))
        f = ObjectiveFunction(cm2, pts, render(cm2, pts))
        f(v)
        @test allocs(f, v) == 0
    end
end

@testitem "chi2 is inferred and differentiable on 2D points" tags = [:fit, :autodiff, :twod] setup = [Fixtures] begin
    using AstroFit, ForwardDiff

    cm = @model begin
        a = Gaussian2D(amplitude = 2.0, x0 = 0.3, y0 = -0.2, sigma = 1.1, q = 0.8, theta = 0.4)
        b = Sersic2D(amplitude = 1.0, x0 = -0.5, y0 = 0.4, r_eff = 1.5, n = 1.5, q = 0.9, theta = 0.1)
        a + b
    end
    pts = Coords(collect(range(-3.0, 3.0; length = 15)), collect(range(-2.0, 2.0; length = 11)))
    f = ObjectiveFunction(cm, pts, render(cm, pts), fill(0.1, size(pts)))
    p = params(cm) .+ 0.05 .* (1:nfree(cm)) ./ nfree(cm)  # away from the minimum

    @test @inferred(chi2(f, ForwardDiff.Dual{Nothing}.(p, 1.0))) isa ForwardDiff.Dual
    @test ForwardDiff.gradient(f, p) ≈ Fixtures.fdgrad(f, p) rtol = 1.0e-6
end

@testitem "OptimizationProblem recovers the parameters the data were made from" tags = [:fit] begin
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
