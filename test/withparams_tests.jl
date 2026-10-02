@testitem "withparams is inferred and allocation-free, for Float64 and Dual parameters" tags = [:core, :perf] setup = [Fixtures] begin
    using AstroFit, ForwardDiff

    # Fixed, Bounded and Tied fields together: the case the generated code must fold.
    cm = Fixtures.constrained
    probe(cm, p) = withparams(cm, p)
    p = params(cm)
    pd = ForwardDiff.Dual{Nothing}.(p, 1.0)

    @inferred withparams(cm, p)
    probe(cm, p)
    @test @allocated(probe(cm, p)) == 0

    @inferred withparams(cm, pd)
    probe(cm, pd)
    @test @allocated(probe(cm, pd)) == 0
end

@testitem "gradients flow through Fixed and Tied fields" tags = [:core, :tied, :autodiff] setup = [Fixtures] begin
    using AstroFit, ForwardDiff

    cm = Fixtures.constrained
    x = [0.5, 3.0, 7.0]
    loss(p) = sum(render(withparams(cm, p), x))
    p = [1.5, 3.0, 0.5, 2.0, 6.0]
    @test ForwardDiff.gradient(loss, p) ≈ Fixtures.fdgrad(loss, p) rtol = 1.0e-6
end

@testitem "the keyword method sets free parameters by name" tags = [:core, :params] setup = [Fixtures] begin
    using AstroFit

    cm = Fixtures.constrained
    m = withparams(cm; a_amplitude = 3.0, b_mean = 6.0)
    @test params(m) == [1.0, 3.0, 0.0, 1.0, 6.0]  # the other names keep their values
    @test render(m, 6.0) ≈ 0.1 * 6 + 1 + 3exp(-18) + 6  # b's tied amplitude follows a's
    @test params(withparams(cm)) == params(cm)

    @test_throws "no free parameter `bogus` — available: cont_intercept, a_amplitude" withparams(cm; bogus = 1.0)
    @test_throws "no free parameter `cont_slope`" withparams(cm; cont_slope = 0.2)  # Fixed
    @test_throws "no free parameter `b_amplitude`" withparams(cm; b_amplitude = 9.0)  # Tied
end

@testitem "params is a concrete Float64 vector, even with integer fields or no free parameter" tags = [:core, :params] begin
    using AstroFit

    # Each field has its own type parameter, so an integer literal stays an Int in the
    # struct. Without promotion in `params` the optimizer would get a Vector{Real}.
    cm = @model begin
        g = Gaussian1D(amplitude = 1, mean = 0.0, sigma = 1.0)
        g
    end
    @test cm.g.model.amplitude === 1
    @test params(cm) isa Vector{Float64}

    allfixed = @fix cm.g.amplitude
    allfixed = @fix allfixed.g.mean
    allfixed = @fix allfixed.g.sigma
    @test params(allfixed) isa Vector{Float64}
    @test isempty(params(allfixed))
end

@testitem "a model that already carries duals can be re-parameterized" tags = [:core, :autodiff] setup = [Fixtures] begin
    using AstroFit, ForwardDiff

    # `<: Real` (not `<: AbstractFloat`) is the bound that admits a Dual, and a Dual of
    # a Dual, which second-order AD produces.
    cm = Fixtures.constrained
    d1 = withparams(cm, ForwardDiff.Dual.(params(cm), 1.0))
    d2 = withparams(d1, ForwardDiff.Dual.(params(d1), 1.0))
    @test d2.a.model.sigma isa ForwardDiff.Dual{<:Any, <:ForwardDiff.Dual}
    @test d2.b.model.sigma isa ForwardDiff.Dual{<:Any, <:ForwardDiff.Dual}  # a Tied field follows its master
    @test d2.cont.model.slope === 0.1  # Fixed: stays a Float64
end

@testitem "a `_cache_` field is derived state, rebuilt from the parameters" tags = [:authoring] begin
    using AstroFit, ForwardDiff

    struct Tilted{S <: Real, T <: Real, C} <: AstroFit.AbstractModel{1, 1}
        sigma::S
        theta::T
        _cache_::C
        function Tilted(sigma::S, theta::T) where {S <: Real, T <: Real}
            c = (cos(theta) / sigma,)
            return new{S, T, typeof(c)}(sigma, theta, c)
        end
    end
    AstroFit.evaluate(m::Tilted, x::Number) = m._cache_[1] * x

    cm = @model begin
        t = Tilted(2.0, 0.5)
        t
    end
    @test paramnames(cm) == [:t_sigma, :t_theta]
    @test length.(bounds(cm)) == (2, 2)
    @test render(withparams(cm, [4.0, 0.0]), 3.0) ≈ 0.75  # cache rebuilt: cos(0) / 4 * 3

    # The macros rebind their variable, so give each one a local to the @test_throws scope.
    @test_throws ArgumentError (m = cm; @fix m.t._cache_)
    @test_throws ArgumentError (m = cm; @tie m.t.sigma -> m.t._cache_)

    # d/dσ [cos(θ) x / σ] = -cos(θ) x / σ², with θ fixed
    fixedθ = @fix cm.t.theta
    g = ForwardDiff.gradient(p -> render(withparams(fixedθ, p), 3.0), [2.0])
    @test g[1] ≈ -cos(0.5) * 3.0 / 4.0

    @fix cm.t.sigma = 4.0
    @test render(cm, 3.0) ≈ cos(0.5) * 3.0 / 4.0  # _cache_ rebuilt from the new σ

    struct Misplaced{C, S <: Real} <: AstroFit.AbstractModel{1, 1}
        _cache_::C
        sigma::S
    end
    @test_throws "`_cache_` must be the last field" @model begin
        m = Misplaced(nothing, 1.0)
        m
    end
end

@testitem "zoo models with a `_cache_` differentiate through withparams" tags = [:zoo, :autodiff] setup = [Fixtures] begin
    using AstroFit, ForwardDiff

    voigt = @model begin
        v = Voigt1D(amplitude = 2.0, mean = 0.1, sigma = 0.8, gamma = 0.5)
        v
    end
    fv(p) = render(withparams(voigt, p), 0.7)
    @test ForwardDiff.gradient(fv, params(voigt)) ≈ Fixtures.fdgrad(fv, params(voigt)) rtol = 1.0e-6

    # q fixed to an Int: the cache mixes a fixed Int, Float64 and dual fields (ADR 0005).
    sersic = @model begin
        s = Sersic2D(amplitude = 2.0, x0 = 0.1, y0 = -0.2, r_eff = 1.5, n = 2.0, q = 1, theta = 0.3)
        s
    end
    @fix sersic.s.q
    fs(p) = render(withparams(sersic, p), (0.9, 0.4))
    @test ForwardDiff.gradient(fs, params(sersic)) ≈ Fixtures.fdgrad(fs, params(sersic)) rtol = 1.0e-6
end

@testitem "withparams honours a constructorof overload defined after loading" tags = [:authoring] begin
    using AstroFit

    struct Mirrored{A <: Real} <: AstroFit.AbstractModel{1, 1}
        amplitude::A
    end
    AstroFit.evaluate(m::Mirrored, x::Number) = m.amplitude
    AstroFit.constructorof(::Type{<:Mirrored}) = a -> Mirrored(abs(a))

    cm = @model begin
        m = Mirrored(1.0)
        m
    end
    @test render(withparams(cm, [-2.0]), 0.0) == 2.0
end
