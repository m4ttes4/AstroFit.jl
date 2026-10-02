@testitem "render protocol: array-native models are transparent through wrappers" tags = [:core, :render] begin
    using AstroFit, Test, ForwardDiff

    struct ArrayScale{T} <: AbstractModel{1, 1}
        scale::T
    end
    AstroFit.evalstyle(::Type{<:ArrayScale}) = AstroFit.Domainwise()
    AstroFit.render(m::ArrayScale, xs::AbstractArray) = m.scale .* xs

    cm = @model begin
        a = ArrayScale(2.0)
        a
    end
    for xs in ([1.0, 2.0, 3.0], [1.0 2.0; 3.0 4.0])
        for m in (cm.a.model, cm.a, cm)
            @test render(m, xs) == 2 .* xs
            out = similar(xs)
            @test render!(out, m, xs) === out
            @test out == 2 .* xs
            @test AstroFit.chi2(m, (xs,), 2 .* xs, nothing) == 0
        end
        @test render(cm.a + cm.a, xs) == 4 .* xs
        @test render(cm.a.model + cm.a.model, xs) == 4 .* xs
        for (op, expected) in ((-, zero(xs)), (*, 4 .* xs .^ 2), (/, ones(size(xs))))
            @test render(op(cm.a.model, cm.a.model), xs) == expected
            @test render!(similar(xs), op(cm.a, cm.a), xs) == expected
        end
        @test render(cm.a |> Linear1D(slope = 3.0, intercept = 1.0), xs) == 6 .* xs .+ 1
    end
    xs = [1.0, 2.0, 3.0]
    objective = ObjectiveFunction(cm, xs, zeros(3))
    @test ForwardDiff.gradient(objective, [2.0]) ≈ [56.0]
    dual = withparams(cm, ForwardDiff.Dual.([2.0], 1.0))
    out = similar(xs, typeof(dual.a.model.scale))
    @test render!(out, dual, xs) == render(dual, xs)
    @test_throws ArgumentError render(cm.a.model, [1.0], [2.0])
    @test_throws MethodError render(cm.a.model, 1.0)
end

@testitem "render protocol: array-native coordinate models support flat and grid forms" tags = [:core, :render] begin
    using AstroFit, Test

    struct ArrayPlane{T} <: AbstractModel{2, 1}
        scale::T
    end
    AstroFit.evalstyle(::Type{<:ArrayPlane}) = Domainwise()
    AstroFit.render(m::ArrayPlane, x::AbstractArray, y::AbstractArray) = m.scale .* (x .+ 2 .* y)
    cm = @model begin
        p = ArrayPlane(1.0)
        p
    end
    col = [1.0, 2.0, 3.0]; row = reshape([4.0, 5.0], 1, :)
    expected = col .+ 2 .* row
    for m in (ArrayPlane(1.0), cm.p, cm)
        @test render(m, col, row) == expected
        @test render!(similar(expected), m, col, row) == expected
        @test render(m, repeat(col, 1, 2), repeat(row, 3, 1)) == expected
        @test render(m, vec(repeat(col, 1, 2)), vec(repeat(row, 3, 1))) == vec(expected)
    end
    @test ObjectiveFunction(cm, (col, row), expected)([1.0]) == 0
end

@testitem "render protocol: kernels work in-place bare and wrapped" tags = [:kernel, :render] begin
    using AstroFit, Test

    struct DoubleArray <: AbstractKernel{1, 1} end
    AstroFit.render(::DoubleArray, xs::AbstractArray) = 2 .* xs
    cm = @model begin
        k = DoubleArray()
        k
    end
    for xs in ([1.0, 2.0], [1.0 2.0; 3.0 4.0])
        for m in (cm.k.model, cm.k, cm)
            @test render!(similar(xs), m, xs) == render(m, xs)
            @test AstroFit.chi2(m, (xs,), 2 .* xs, nothing) == 0
        end
    end
    xs = collect(-2.0:0.5:2.0)
    k = GaussianPSF(sigma = 1.4)
    @test render!(similar(xs), k, xs) ≈ render(k, xs)
    @test_throws ArgumentError render!(zeros(2, 2), k, zeros(2, 2))
end

@testitem "render protocol: pointwise consumers after kernels stay lazy" tags = [:kernel, :render] begin
    using AstroFit, Test

    source = Gaussian1D() |> GaussianPSF(sigma = 1.4)
    model = source |> Linear1D(slope = 2.0, intercept = 1.0)
    xs = collect(range(-3.0, 3.0; length = 512))
    out = similar(xs)
    alloc(out, m, xs) = @allocated render!(out, m, xs)
    alloc(out, source, xs)
    alloc(out, model, xs)
    @test alloc(out, source, xs) == alloc(out, model, xs)
    @test render!(out, model, xs) ≈ 2 .* render(source, xs) .+ 1
end

@testitem "render protocol: matrix pipe outputs remain values" tags = [:kernel, :render] begin
    using AstroFit, Test

    struct Plane <: AbstractModel{2, 1} end
    AstroFit.render(::Plane, x::Number, y::Number) = x + 2y
    struct DoubleImage <: AbstractKernel{1, 1} end
    AstroFit.render(::DoubleImage, xs::AbstractMatrix) = 2 .* xs
    cm = @model begin
        source = Plane()
        k = DoubleImage()
        transform = Linear1D(slope = 3.0, intercept = 1.0)
        (source |> k) |> transform
    end
    col = [1.0, 2.0, 3.0]; row = reshape([4.0, 5.0], 1, :)
    expected = 6 .* (col .+ 2 .* row) .+ 1
    @test render(cm, col, row) == expected
    @test render!(similar(expected), cm, col, row) == expected
    @test render(cm, zeros(3, 2)) == 6 .* ((1:3) .+ 2 .* reshape(1:2, 1, :)) .+ 1
    @test render!(zeros(3, 2), cm) == render(cm, zeros(3, 2))
    @test_throws "(Plane |> DoubleImage) produces 1 value(s) per point, Plane expects 2" (Plane() |> DoubleImage()) |> Plane()
    @test render(Plane(), col, 4.0) == col .+ 8
    @test render!(similar(col), Plane(), col, 4.0) == col .+ 8
end

@testitem "render protocol: reject kernel shape changes before broadcasting" tags = [:kernel, :render] begin
    using AstroFit, Test

    struct SingletonKernel <: AbstractKernel{1, 1} end
    AstroFit.render(::SingletonKernel, xs::AbstractArray) = [sum(xs)]
    g = Gaussian1D()
    xs = [1.0, 2.0, 3.0]
    for m in (g |> SingletonKernel(), (g |> SingletonKernel()) + g,
            g + (g |> SingletonKernel()), g |> SingletonKernel() |> g)
        @test_throws DimensionMismatch render(m, xs)
        @test_throws DimensionMismatch render!(similar(xs), m, xs)
        @test_throws DimensionMismatch AstroFit.chi2(m, (xs,), zeros(3), nothing)
    end
end

@testitem "render protocol: destination axes and matrix templates agree" tags = [:core, :render] begin
    using AstroFit, Test

    xs = [1.0, 2.0, 3.0]
    @test_throws DimensionMismatch render!(zeros(3, 2), Gaussian1D(), xs)
    @test_throws DimensionMismatch render!(zeros(3, 2), Voigt1D(), xs)
    for m in (Gaussian2D(), Sersic2D(), Moffat2D(), Beta2D())
        @test_throws DimensionMismatch render!(zeros(3, 2), m, xs, xs)
    end
    @test_throws MethodError render(Voigt1D(), zeros(3, 2))
    @test_throws MethodError render!(zeros(3, 2), Voigt1D(), zeros(3, 2))
end
