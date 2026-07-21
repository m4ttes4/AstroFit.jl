@testitem "render!: base models write into the provided buffer" tags = [:core, :render] begin
    using AstroFit

    xs = collect(range(-2.0, 2.0; length = 9))

    for model in (
            Gaussian1D(amplitude = 2.0, mean = 0.5, sigma = 1.2),
            Const1D(value = 3.0),
            Linear1D(slope = -0.5, intercept = 1.0),
        )
        out = fill(NaN, length(xs))
        ret = render!(out, model, xs)

        @test ret === out
        @test out ≈ render(model, xs)
    end
end

@testitem "render!: compound and compiled models match allocating render" tags = [:core, :render] begin
    using AstroFit

    xs = collect(range(-3.0, 3.0; length = 17))
    g = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.1)
    l = Linear1D(slope = 0.2, intercept = 0.7)
    c = Const1D(value = 1.5)

    models = (
        l + g,
        l - g,
        g * c,
        (g + c) / Linear1D(slope = 0.1, intercept = 2.0),
        l |> g,
    )

    for model in models
        out = similar(xs)
        @test render!(out, model, xs) === out
        @test out ≈ render(model, xs)
    end

    cm = @model begin
        cont = Linear1D(slope = 0.2, intercept = 0.7)
        line = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.1)
        cont + line
    end

    out = similar(xs)
    @test render!(out, cm, xs) === out
    @test out ≈ render(cm, xs)
end

@testitem "render!: 2D models take grid-form coordinates without allocating" tags = [:core, :render] begin
    using AstroFit

    xa = collect(range(-2.0, 2.0; length = 7))
    ya = collect(range(-1.0, 1.0; length = 5))
    col, row = xa, reshape(ya, 1, :)                        # grid form: one axis per dimension
    Xm = repeat(xa, 1, length(ya))                          # the same grid as a co-shaped point list
    Ym = repeat(row, length(xa), 1)

    models = (
        Gaussian2D(amplitude = 2.0, x0 = 0.1, y0 = -0.2, sigma = 1.3, q = 0.8, theta = 0.3),
        Sersic2D(amplitude = 2.0, x0 = 0.1, y0 = -0.2, r_eff = 1.5, n = 2.0, q = 0.8, theta = 0.3),
        Moffat2D(amplitude = 2.0, x0 = 0.1, y0 = -0.2, alpha = 1.2, beta = 2.0, q = 0.8, theta = 0.3),
        Beta2D(amplitude = 2.0, x0 = 0.1, y0 = -0.2, r_core = 1.2, beta = 0.7, q = 0.8, theta = 0.3),
    )

    for m in models
        ref = render(m, Xm, Ym)
        @test size(ref) == (length(xa), length(ya))
        @test render(m, col, row) ≈ ref                     # both forms describe one image

        out = fill(NaN, size(ref))
        @test render!(out, m, col, row) === out
        @test out ≈ ref
        fill!(out, NaN)
        @test render!(out, m, Xm, Ym) ≈ ref
    end

    # The buffer is the whole story: what render! allocates must not grow with the
    # grid. Compared across two sizes so the testitem's global access can't skew it.
    g = models[1]
    bigcol = collect(range(-2.0, 2.0; length = 70))
    bigrow = reshape(collect(range(-1.0, 1.0; length = 50)), 1, :)
    outsmall, outbig = fill(NaN, 7, 5), fill(NaN, 70, 50)
    render!(outsmall, g, col, row)
    render!(outbig, g, bigcol, bigrow)

    a_small = @allocated render!(outsmall, g, col, row)
    a_big = @allocated render!(outbig, g, bigcol, bigrow)
    @test a_small == a_big
    @test a_big < 512

    # And the allocating render pays for the output array and nothing else —
    # compared against the bare array rather than against sizeof, since the
    # allocator rounds a 28000-byte array up to whole pages either way.
    allocrender(m, a, b) = @allocated render(m, a, b)
    allocarray(dims) = @allocated Array{Float64}(undef, dims)
    allocrender(g, bigcol, bigrow)
    allocarray((70, 50))
    @test allocrender(g, bigcol, bigrow) == allocarray((70, 50))
end

@testitem "render: a lone matrix is the index grid, not values" tags = [:core, :render] begin
    using AstroFit

    g = Gaussian2D(amplitude = 2.0, x0 = 5.0, y0 = 3.0, sigma = 1.5, q = 0.8, theta = 0.3)
    img = zeros(9, 7)

    r = render(g, img)
    @test size(r) == size(img)
    @test r == render(g, axes(img, 1), reshape(axes(img, 2), 1, :))
    @test r == render(g, 1:9, reshape(1:7, 1, :))
    @test render(g, fill(999.0, 9, 7)) == r        # the values are a template, nothing more

    out = fill(NaN, 9, 7)
    @test render!(out, g) === out
    # `≈`, not `==`: the zoo's render! hoists the divisions into reciprocals, so
    # it rounds differently from the scalar render broadcast above.
    @test out ≈ r

    # A single vector still means coordinate values — the index-grid reading is
    # for matrices only, or every 1D render would change meaning.
    l = Gaussian1D(amplitude = 1.0, mean = 0.0, sigma = 1.0)
    xs = collect(-2.0:0.5:2.0)
    @test render(l, xs) == render.(l, xs)

    allocbang(o, m) = @allocated render!(o, m)
    allocrender(m, i) = @allocated render(m, i)
    allocarray(d) = @allocated Array{Float64}(undef, d)
    big, bigout = zeros(70, 50), fill(NaN, 70, 50)
    allocbang(bigout, g)
    allocrender(g, big)
    allocarray((70, 50))
    @test allocbang(bigout, g) == 0
    @test allocrender(g, big) == allocarray((70, 50))
end

@testitem "render!: generic fallback supports custom multi-coordinate models" tags = [:core, :render] begin
    using AstroFit

    struct TestPlane2D{T <: Real} <: AbstractModel
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
