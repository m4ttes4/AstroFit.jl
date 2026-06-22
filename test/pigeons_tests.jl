@testitem "PosteriorTarget: callable returns logposterior inside bounds" tags = [:pigeons] begin
    using AstroFit

    cm = @model begin
        g = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        g
    end
    @constrain cm begin
        g.amplitude in (0.1, 10.0)
        g.mean in (-5.0, 5.0)
        g.sigma in (0.1, 5.0)
    end

    x = collect(-3.0:0.5:3.0)
    y = render(cm, x) .+ 0.01 .* ones(length(x))
    err = fill(0.1, length(x))

    target = PosteriorTarget(cm, x, y, err)

    p = params(cm)
    @test target(p) == logposterior(cm, p, x, y, err)
    @test target(p) > -Inf
end

@testitem "PosteriorTarget: returns -Inf outside bounds" tags = [:pigeons] begin
    using AstroFit

    cm = @model begin
        g = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        g
    end
    @constrain cm begin
        g.amplitude in (0.1, 10.0)
        g.mean in (-5.0, 5.0)
        g.sigma in (0.1, 5.0)
    end

    x = [1.0, 2.0, 3.0]
    y = [1.0, 1.0, 1.0]
    target = PosteriorTarget(cm, x, y)

    @test target([-1.0, 0.0, 1.0]) == -Inf
    @test target([2.0, 0.0, -0.5]) == -Inf
end

@testitem "PosteriorTarget: LogDensityProblems interface" tags = [:pigeons] begin
    using AstroFit
    using Pigeons
    using LogDensityProblems

    cm = @model begin
        g = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        g
    end
    @constrain cm begin
        g.amplitude in (0.1, 10.0)
        g.mean in (-5.0, 5.0)
        g.sigma in (0.1, 5.0)
    end

    x = [1.0, 2.0, 3.0]
    y = [1.0, 1.0, 1.0]
    err = fill(0.1, 3)
    target = PosteriorTarget(cm, x, y, err)

    @test LogDensityProblems.dimension(target) == 3
    @test LogDensityProblems.logdensity(target, params(cm)) == target(params(cm))
    @test LogDensityProblems.capabilities(typeof(target)) isa
        LogDensityProblems.LogDensityOrder{0}
end

@testitem "PosteriorTarget: Pigeons integration" tags = [:pigeons] begin
    using AstroFit
    using Pigeons

    cm = @model begin
        c = Const1D(value = 3.0)
        c
    end
    @constrain cm begin
        c.value in (0.0, 10.0)
    end

    x = [1.0, 2.0, 3.0]
    y = [3.0, 3.0, 3.0]
    err = fill(0.5, 3)
    target = PosteriorTarget(cm, x, y, err)
    ref = PosteriorTarget(cm, x, zeros(3), fill(1e6, 3))

    pt = pigeons(target = target, reference = ref, n_rounds = 4, n_chains = 4)
    @test pt isa Pigeons.PT
end
