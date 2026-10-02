@testitem "a `_cache_` field is derived state, not a parameter" tags = [:authoring] begin
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
    AstroFit.render(m::Tilted, x::Number) = m._cache_[1] * x

    cm = @model begin
        t = Tilted(2.0, 0.5)
        t
    end
    @test paramnames(cm) == [:t_sigma, :t_theta]
    @test params(cm) == [2.0, 0.5]
    @test length.(bounds(cm)) == (2, 2)
    @test render(withparams(cm, [4.0, 0.0]), 3.0) ≈ 0.75  # cache rebuilt: cos(0) / 4 * 3
    @test !occursin("_cache_", sprint(show, MIME"text/plain"(), cm))

    # The macros rebind their variable, so give each one a local to the @test_throws scope.
    @test_throws ArgumentError (m = cm; @fix m.t._cache_)
    @test_throws ArgumentError (m = cm; @tie m.t.sigma -> m.t._cache_)

    # d/dσ [cos(θ) x / σ] = -cos(θ) x / σ², with θ fixed
    fixedθ = @fix cm.t.theta
    g = ForwardDiff.gradient(p -> render(withparams(fixedθ, p), 3.0), [2.0])
    @test g[1] ≈ -cos(0.5) * 3.0 / 4.0

    struct Misplaced{C, S <: Real} <: AstroFit.AbstractModel{1, 1}
        _cache_::C
        sigma::S
    end
    @test_throws ArgumentError @model begin
        m = Misplaced(nothing, 1.0)
        m
    end

    @fix cm.t.sigma = 4.0
    @test render(cm, 3.0) ≈ cos(0.5) * 3.0 / 4.0  # _cache_ rebuilt from the new σ
end

@testitem "withparams honours a constructorof overload defined after loading" tags = [:authoring] begin
    using AstroFit

    struct Mirrored{A <: Real} <: AstroFit.AbstractModel{1, 1}
        amplitude::A
    end
    AstroFit.render(m::Mirrored, x::Number) = m.amplitude
    AstroFit.constructorof(::Type{<:Mirrored}) = a -> Mirrored(abs(a))

    cm = @model begin
        m = Mirrored(1.0)
        m
    end
    @test withparams(cm, [-2.0]).m.model.amplitude == 2.0
end
