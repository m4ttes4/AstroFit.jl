@testitem "composition rules follow the declared arities" tags = [:core] begin
    using AstroFit

    struct Rot{T <: Real} <: AbstractModel{2, 2}
        theta::T
    end
    g1, g2 = Gaussian1D(), Gaussian2D()

    @test_throws "cannot add Gaussian1D (1 => 1) and Gaussian2D (2 => 1)" g1 + g2
    @test_throws "cannot add Rot (2 => 2) and Rot (2 => 2)" Rot(0.1) + Rot(0.2)
    @test_throws "cannot multiply Rot (2 => 2) and Rot (2 => 2)" Rot(0.1) * Rot(0.2)
    @test_throws "Rot produces 2 value(s) per point, Gaussian1D expects 1" Rot(0.1) |> g1
    @test_throws "Rot produces 2 value(s) per point, Gaussian1D expects 1" g1 ∘ Rot(0.1)

    # Inside @model the operands are leaves: the message names them as the user did.
    @test_throws "cannot add g1 (1 => 1) and g2 (2 => 1)" @model begin
        g1 = Gaussian1D()
        g2 = Gaussian2D()
        g1 + g2
    end

    @test (g1 |> g1) isa AbstractModel{1, 1}  # arity says nothing about meaning
    @test (Rot(0.1) |> g2) isa AbstractModel{2, 1}
    @test (g2 ∘ Rot(0.1)) === (Rot(0.1) |> g2)
    @test (g2 - g2) isa AbstractModel{2, 1}

    # Rebuilds go through the unchecked outer constructors and still infer.
    cm = @model begin
        z = Redshift1D(z = 0.1)
        g = Gaussian1D()
        z |> g
    end
    @test @inferred(withparams(cm, params(cm))) isa CompiledModel

    @test_throws "can only subtype data types" @eval struct NoArity <: AbstractModel end
end

@testitem "no method ambiguities" tags = [:core] begin
    using AstroFit, Test
    @test isempty(Test.detect_ambiguities(AstroFit))
end
