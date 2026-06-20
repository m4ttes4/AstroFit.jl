@testitem "render!: base models write into the provided buffer" tags=[:core, :render] begin
    using AstroFit

    xs = collect(range(-2.0, 2.0; length=9))

    for model in (
        Gaussian1D(amplitude=2.0, mean=0.5, sigma=1.2),
        Const1D(value=3.0),
        Linear1D(slope=-0.5, intercept=1.0),
    )
        out = fill(NaN, length(xs))
        ret = render!(out, model, xs)

        @test ret === out
        @test out ≈ render(model, xs)
    end
end

@testitem "render!: compound and compiled models match allocating render" tags=[:core, :render] begin
    using AstroFit

    xs = collect(range(-3.0, 3.0; length=17))
    g = Gaussian1D(amplitude=2.0, mean=0.0, sigma=1.1)
    l = Linear1D(slope=0.2, intercept=0.7)
    c = Const1D(value=1.5)

    models = (
        l + g,
        l - g,
        g * c,
        (g + c) / Linear1D(slope=0.1, intercept=2.0),
        l |> g,
    )

    for model in models
        out = similar(xs)
        @test render!(out, model, xs) === out
        @test out ≈ render(model, xs)
    end

    cm = @model begin
        cont = Linear1D(slope=0.2, intercept=0.7)
        line = Gaussian1D(amplitude=2.0, mean=0.0, sigma=1.1)
        cont + line
    end

    out = similar(xs)
    @test render!(out, cm, xs) === out
    @test out ≈ render(cm, xs)
end

@testitem "render!: generic fallback supports custom multi-coordinate models" tags=[:core, :render] begin
    using AstroFit

    struct TestPlane2D{T<:Real} <: AbstractModel
        a::T
        b::T
        c::T
    end

    AstroFit.render(m::TestPlane2D, x::Number, y::Number) = m.a * x + m.b * y + m.c

    xs = reshape(collect(1.0:6.0), 2, 3)
    ys = xs ./ 10
    model = TestPlane2D(2.0, -3.0, 0.5)
    out = similar(xs)

    @test render!(out, model, xs, ys) === out
    @test out ≈ render(model, xs, ys)
end
