@testitem "operators combine components pointwise, as written" tags = [:core] begin
    using AstroFit

    cm = @model begin
        line = Linear1D(slope = 2.0, intercept = 1.0)
        g = Gaussian1D(amplitude = 3.0, mean = 0.0, sigma = 1.0)
        c = Const1D(value = 0.5)
        k = Const1D(value = 4.0)
        d = Const1D(value = 2.0)
        (line + g - c) * k / d
    end
    # at x = 1: line = 3, g = 3e^{-1/2}
    @test render(cm, 1.0) ≈ (3 + 3exp(-1 / 2) - 0.5) * 4 / 2

    # A pipe feeds the left side's output to the right side as its coordinate:
    # a shared redshift moves both lines, observed 2λ is rest-frame λ at z = 1.
    spec = @model begin
        z = Redshift1D(z = 1.0)
        ha = Gaussian1D(amplitude = 1.0, mean = 6563.0, sigma = 2.0)
        hb = Gaussian1D(amplitude = 0.5, mean = 4861.0, sigma = 2.0)
        z |> (ha + hb)
    end
    @test render(spec, 2 * 6565.0) ≈ exp(-1 / 2)  # one σ off Hα; Hβ is ~850σ away
    @test render(spec, 2 * 4861.0) ≈ 0.5
end

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
end

@testitem "@model names parameters after its leaves, prefabs included" tags = [:authoring, :prefab] begin
    using AstroFit

    g1 = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
    cm = @model begin
        left = g1
        right = Gaussian1D(amplitude = 1.0, mean = 5.0, sigma = 2.0)
        left + right
    end
    @test cm.left.model === g1  # a caller's model is stored as is
    @test paramnames(cm) ==
        [:left_amplitude, :left_mean, :left_sigma, :right_amplitude, :right_mean, :right_sigma]

    # A CompiledModel used as a leaf keeps its constraints; its names get the leaf's prefix.
    pre = @model begin
        oiii = doublet(blue_center = 4959.0, red_center = 5007.0)
        cont = Const1D(value = 0.1)
        oiii + cont
    end
    @test paramnames(pre) == [:oiii_blue_amplitude, :oiii_blue_mean, :oiii_blue_sigma, :cont_value]
    @test bounds(pre) == ([0.0, 4957.0, 0.0, -Inf], [Inf, 4961.0, Inf, Inf])
end

@testitem "@model refuses malformed blocks with a message naming the problem" tags = [:authoring] begin
    using AstroFit

    g1, g2 = Gaussian1D(), Gaussian1D()
    @test_throws "@model expects a begin…end block" macroexpand(@__MODULE__, :(AstroFit.@model $g1 + $g2))
    @test_throws "leaf(s) used more than once: g" @model begin
        g = Gaussian1D()
        g + g
    end
    for reserved in (:tree, :priors)
        @test_throws "`$reserved` is a reserved CompiledModel field name" macroexpand(
            @__MODULE__, :(
                AstroFit.@model begin
                    $reserved = Gaussian1D()
                    $reserved
                end
            )
        )
    end
end

@testitem "kernels compose with models in any position" tags = [:kernel] begin
    using AstroFit

    struct Double <: AbstractKernel{1, 1} end
    AstroFit.evaluate(::Double, f::Number) = 2f

    # A subtree feeding a chain of kernels, then more composition on both sides.
    cm = @model begin
        cont = Const1D(value = 0.5)
        line = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        base = Const1D(value = 1.0)
        k1 = Double()
        k2 = Double()
        t = Const1D(value = 3.0)
        cont + (((line + base) |> k1 |> k2) * t)
    end
    x = [-1.0, 0.0, 2.0]
    @test render(cm, x) ≈ @. 0.5 + 4 * (2exp(-x^2 / 2) + 1) * 3
end

@testitem "kernel fields default Fixed, @free opts in" tags = [:kernel] begin
    using AstroFit

    struct Saturate{L <: Real} <: AbstractKernel{1, 1}
        level::L
    end
    AstroFit.evaluate(k::Saturate, f::Number) = min(f, k.level)

    cm = @model begin
        line = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        sat = Saturate(1.5)
        line |> sat
    end
    # the saturation level is a calibration input: no optimizer slot by default
    @test paramnames(cm) == [:line_amplitude, :line_mean, :line_sigma]
    @test render(cm, [0.0, 3.0]) ≈ [1.5, 2exp(-4.5)]  # clipped at the peak, untouched in the wing

    free = @free cm.sat.level
    @test paramnames(free) == [:line_amplitude, :line_mean, :line_sigma, :sat_level]
    @test params(free) isa Vector{Float64}  # not Vector{Real}
end

@testitem "no method ambiguities" tags = [:core] begin
    using AstroFit, Test

    # An author's untyped evaluate lives on a different function from render's
    # methods, so it cannot make them ambiguous. detect_ambiguities only looks at
    # methods defined in the modules it is given, hence both.
    struct Loose <: AbstractModel{1, 1} end
    AstroFit.evaluate(::Loose, x) = x
    @test isempty(Test.detect_ambiguities(AstroFit, @__MODULE__))
end
