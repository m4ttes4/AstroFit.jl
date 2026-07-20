@testitem "params and bounds follow the withparams slot order" tags = [:core, :params] begin
    using AstroFit

    cm = @model begin
        left = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        right = Gaussian1D(amplitude = 3.0, mean = 5.0, sigma = 1.5)
        left + right
    end

    @constrain cm begin
        left.mean = 0.0
        right.sigma in (0.1, 10.0)
    end

    @test nfree(cm) == 5
    @test params(cm) == [2.0, 1.0, 3.0, 5.0, 1.5]
    @test paramnames(cm) == [
        :left_amplitude, :left_sigma,
        :right_amplitude, :right_mean, :right_sigma,
    ]

    lower, upper = bounds(cm)
    @test lower == [-Inf, -Inf, -Inf, -Inf, 0.1]
    @test upper == [Inf, Inf, Inf, Inf, 10.0]

    rebuilt = withparams(cm, params(cm))
    @test rebuilt.left.model.amplitude == 2.0
    @test rebuilt.left.model.mean == 0.0
    @test rebuilt.right.model.sigma == 1.5
end

@testitem "params is a concrete vector even for mixed field types" tags = [:core, :params] begin
    using AstroFit

    # Fields no longer share a type parameter, so an integer literal stays an Int in the
    # struct. Without promotion in `params` the optimizer would get a Vector{Real}.
    cm = @model begin
        g = Gaussian1D(amplitude = 1, mean = 0.0, sigma = 1.0)
        g
    end
    @test cm.g.model.amplitude === 1
    @test params(cm) isa Vector{Float64}

    # every parameter fixed: no values to promote, still a usable empty vector
    allfixed = @fix cm.g.amplitude
    allfixed = @fix allfixed.g.mean
    allfixed = @fix allfixed.g.sigma
    @test nfree(allfixed) == 0
    @test params(allfixed) isa Vector{Float64}
    @test isempty(params(allfixed))
end

@testitem "bounds excludes fixed and tied slots" tags = [:core, :params, :tied] begin
    using AstroFit

    cm = @model begin
        left = Gaussian1D(amplitude = 2.0, sigma = 1.0)
        right = Gaussian1D(amplitude = 9.0)
        left + right
    end

    @constrain cm begin
        left.amplitude in (0.0, 100.0)
        right.amplitude -> 2 * left.amplitude
    end

    lower, upper = bounds(cm)
    @test length(lower) == length(upper) == nfree(cm) == 5
    @test lower[1] == 0.0
    @test upper[1] == 100.0
    @test :right_amplitude ∉ paramnames(cm)
end

@testitem "fix at explicit and current values" tags = [:core, :params] begin
    using AstroFit

    cm = @model begin
        line = Gaussian1D(amplitude = 1.0, mean = 0.0, sigma = 1.0)
        line
    end

    @constrain cm begin
        line.amplitude = 5.0
        line.sigma
    end

    @test cm.line.constraints[1] isa Fixed
    @test cm.line.constraints[1].value == 5.0
    @test cm.line.constraints[3] isa Fixed
    @test cm.line.constraints[3].value == 1.0
    @test params(cm) == [0.0]

    rebuilt = withparams(cm, [2.0])
    @test rebuilt.line.model.amplitude == 5.0
    @test rebuilt.line.model.mean == 2.0
    @test rebuilt.line.model.sigma == 1.0
end

@testitem "ties are resolved by withparams" tags = [:core, :tied] begin
    using AstroFit

    cm = @model begin
        left = Gaussian1D(amplitude = 2.0, mean = 3.0, sigma = 1.0)
        right = Gaussian1D(amplitude = 99.0, mean = 0.0, sigma = 9.0)
        left + right
    end

    @constrain cm begin
        right.amplitude -> left.amplitude * left.mean
        right.sigma -> left.sigma
    end

    @test nfree(cm) == 4
    m = withparams(cm, [2.0, 3.0, 4.0, 0.0])
    @test m.right.model.amplitude == 6.0
    @test m.right.model.sigma == 4.0
end

@testitem "validate rejects ties to non-free masters" tags = [:core, :tied] begin
    using AstroFit

    cm = @model begin
        left = Gaussian1D(sigma = 2.0)
        right = Gaussian1D(sigma = 9.0)
        left + right
    end

    @test_throws ArgumentError (
        m -> @constrain m begin
            left.sigma = 2.0
            right.sigma -> left.sigma
        end
    )(cm)
end

@testitem "rendering a CompiledModel works for scalars and arrays" tags = [:core, :params] begin
    using AstroFit

    cm = @model begin
        line = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        base = Const1D(value = 1.0)
        line + base
    end

    @test render(cm, 0.0) == 3.0
    xs = [-1.0, 0.0, 1.0]
    @test render(cm, xs) == [render(cm, x) for x in xs]
end

@testitem "autodiff flows through tied parameters" tags = [:core, :tied, :autodiff] begin
    using AstroFit
    using ForwardDiff

    cm = @model begin
        left = Gaussian1D(amplitude = 1.0, mean = 0.0, sigma = 1.0)
        right = Gaussian1D(amplitude = 1.0, mean = 0.0, sigma = 1.0)
        left + right
    end

    @constrain cm begin
        right.amplitude -> 2 * left.amplitude
    end

    loss(p) = render(withparams(cm, p), 0.0)
    g = ForwardDiff.gradient(loss, params(cm))
    @test all(isfinite, g)
    @test g[1] ≈ 3.0
end

@testitem "withparams is inferred and allocation-free" tags = [:core, :perf] begin
    using AstroFit
    using ForwardDiff
    using Test: @inferred

    cm = @model begin
        left = Gaussian1D(amplitude = 2.0, sigma = 1.0)
        right = Gaussian1D(amplitude = 3.0, sigma = 1.5)
        left + right
    end

    @constrain cm begin
        left.sigma in (0.1, 10.0)
        right.amplitude -> 2 * left.amplitude
    end

    probe(cm, p) = withparams(cm, p)

    p = params(cm)
    @inferred withparams(cm, p)
    probe(cm, p)
    @test @allocated(probe(cm, p)) == 0

    pd = ForwardDiff.Dual{Nothing}.(p, 1.0)
    @inferred withparams(cm, pd)
    probe(cm, pd)
    @test @allocated(probe(cm, pd)) == 0
end

@testitem "withparams keyword method overrides by name" tags = [:core, :params] begin
    using AstroFit

    cm = @model begin
        left = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        right = Gaussian1D(amplitude = 3.0, mean = 5.0, sigma = 1.5)
        left + right
    end

    # partial override: untouched parameters keep current values
    m = withparams(cm; left_amplitude = 7.0, right_mean = 4.0)
    @test m.left.model.amplitude == 7.0
    @test m.right.model.mean == 4.0
    @test m.left.model.sigma == 1.0
    @test m.right.model.amplitude == 3.0

    # no kwargs: identity on values
    @test params(withparams(cm)) == params(cm)

    # result is a CompiledModel: renderable and re-usable
    @test render(m, 0.0) == render(withparams(cm, [7.0, 0.0, 1.0, 3.0, 4.0, 1.5]), 0.0)

    # unknown name throws with the available names listed
    @test_throws ArgumentError withparams(cm; bogus = 1.0)
    err = try
        withparams(cm; bogus = 1.0)
    catch e
        e
    end
    @test occursin("left_amplitude", err.msg)
end

@testitem "withparams keyword method rejects fixed and tied parameters" tags = [:core, :params, :tied] begin
    using AstroFit

    cm = @model begin
        left = Gaussian1D(amplitude = 2.0, sigma = 1.0)
        right = Gaussian1D(amplitude = 3.0, sigma = 1.5)
        left + right
    end

    @constrain cm begin
        left.sigma = 1.0
        right.amplitude -> 2 * left.amplitude
    end

    @test_throws ArgumentError withparams(cm; left_sigma = 2.0)
    @test_throws ArgumentError withparams(cm; right_amplitude = 9.0)

    # free parameters still settable; tie recomputed from the new master
    m = withparams(cm; left_amplitude = 5.0)
    @test m.left.model.amplitude == 5.0
    @test m.right.model.amplitude == 10.0
    @test m.left.model.sigma == 1.0
end
