@testitem "render: points are laid out in axis order, first pixel at 1" tags = [:core, :render] begin
    using AstroFit

    # Independent references: a transposed or 0-based layout changes these values.
    struct Plane <: AbstractModel{2, 1} end
    AstroFit.evaluate(::Plane, (x, y)::NTuple{2, Number}) = x + 10y

    @test render(Plane(), CartesianIndices((2, 3))) == [i + 10j for i in 1:2, j in 1:3]
    x, y = [0.5, 1.5], [-1.0, 0.0, 2.0]
    @test render(Plane(), Coords(x, y)) == [xi + 10yj for xi in x, yj in y]
    @test render(Plane(), CartesianIndex(2, 3)) == 32
end

@testitem "render: array methods agree with the point broadcast for every container" tags = [:core, :render] begin
    using AstroFit

    line = Gaussian1D(amplitude = 2.0, mean = 0.5, sigma = 1.2)
    disk = Gaussian2D(amplitude = 2.0, x0 = 0.1, y0 = -0.2, sigma = 1.3, q = 0.8, theta = 0.3)
    λ = collect(range(-2.0, 2.0; length = 9))
    img = zeros(7, 5)
    mask = isodd.(LinearIndices(img))

    @test render(line, λ) == render.(line, λ)
    @test render(line, range(-2.0, 2.0; length = 9)) == render.(line, λ)
    @test render(line, CartesianIndices(λ)) == render.(line, 1:9)
    for A in (
            Coords(collect(1.0:7.0), collect(-2.0:2.0)),
            CartesianIndices(img),
            CartesianIndices(img)[mask],
            [(1, 2.0), (0.5, -1)],  # scattered points, mixed element types
            fill((0.0, 0.0)),  # 0-dimensional
        )
        @test render(disk, A) == render.(disk, A)
    end
    # A model broadcasts as a scalar: a 0-d input gives one value, not a 1-element vector.
    @test render.(line, fill(0.5)) == render(line, 0.5)

    cm = @model begin
        a = Gaussian2D(amplitude = 2.0, x0 = 3.0, y0 = 2.0, sigma = 1.3, q = 0.8, theta = 0.3)
        b = Gaussian2D(amplitude = 1.0, x0 = 5.0, y0 = 4.0, sigma = 2.0, q = 1.0, theta = 0.0)
        a + b
    end
    @test render.(cm, CartesianIndices(img)) == render(cm, CartesianIndices(img))
    @test render(cm, (3.0, 2.0)) ≈ 2.0 + exp(-(4 + 4) / 8)  # a's peak, b two pixels off in each axis
end

@testitem "render: a 2 => 2 transform pipes coordinates into a 2D model" tags = [:core, :render] begin
    using AstroFit

    struct Rot{T <: Real} <: AbstractModel{2, 2}
        theta::T
    end
    AstroFit.evaluate(r::Rot, (x, y)::NTuple{2, Number}) =
        (cos(r.theta) * x - sin(r.theta) * y, sin(r.theta) * x + cos(r.theta) * y)

    g = Gaussian2D(amplitude = 1.0, x0 = 1.0, y0 = 0.0, sigma = 0.5, q = 1.0, theta = 0.0)
    m = Rot(π / 2) |> g
    @test render(m, (0.0, -1.0)) ≈ 1.0  # a quarter turn takes (0, -1) to g's peak at (1, 0)
    x, y = [0.0, 1.0], [-1.0, 0.0]
    # a quarter turn maps (x, y) to (-y, x)
    @test render(m, Coords(x, y)) ≈ [exp(-((-yj - 1)^2 + xi^2) / (2 * 0.5^2)) for xi in x, yj in y]
end

@testitem "render: in-place broadcast allocates nothing" tags = [:core, :render] begin
    using AstroFit

    cm = @model begin
        a = Gaussian2D(amplitude = 2.0, x0 = 3.0, y0 = 2.0, sigma = 1.3, q = 0.8, theta = 0.3)
        b = Sersic2D(amplitude = 1.0, x0 = 5.0, y0 = 4.0, r_eff = 2.0, n = 1.5, q = 0.9, theta = 0.1)
        a + b
    end
    out = zeros(40, 30)
    fill_ci!(o, m) = @allocated o .= render.(m, CartesianIndices(o))
    fill_co!(o, m, A) = @allocated o .= render.(m, A)
    co = Coords(collect(1.0:40.0), collect(1.0:30.0))
    fill_ci!(out, cm); fill_co!(out, cm, co)
    @test fill_ci!(out, cm) == 0
    @test fill_co!(out, cm, co) == 0
end

@testitem "render: wrong inputs throw an ArgumentError that names the rule" tags = [:core, :render] begin
    using AstroFit

    line = Gaussian1D()
    disk = Gaussian2D()
    λ = [1.0, 2.0, 3.0]
    img = zeros(3, 4)

    arrays = "is not a set of points: use CartesianIndices(A)"
    @test_throws "an array of numbers with 2 dimension(s) $arrays" render(disk, img)
    @test_throws "an array of numbers with 2 dimension(s) $arrays" render(line, img)
    @test_throws "an array of numbers with 0 dimension(s) $arrays" render(line, fill(1.0))
    @test_throws "Gaussian2D takes 2 numbers per point, got a vector of numbers" render(disk, λ)
    @test_throws "Gaussian2D takes 2 number(s) per point, got an array of CartesianIndex{1}" render(disk, CartesianIndices(λ))
    @test_throws "Gaussian1D takes 1 number(s) per point, got an array of Tuple{Float64, Float64}" render(line, Coords(λ, λ))
    @test_throws "Gaussian2D takes 2 number(s) per point, got Float64" render(disk, 1.0)
    @test_throws "Gaussian1D takes 1 number(s) per point, got Tuple{Float64, Float64}" render(line, (1.0, 2.0))
    @test_throws "Gaussian1D takes 1 number per point: pass the number, not a 1-tuple" render(line, (1.0,))
    @test_throws "Gaussian2D takes 2 number(s) per point, got Float64" render.(disk, img)

    cm = @model begin
        g = Gaussian1D()
        g
    end
    @test_throws "g takes 1 number(s) per point, got String" render(cm, "x")  # the leaf, by name
end

@testitem "render: author mistakes are reported at the faulty component" tags = [:core, :render] begin
    using AstroFit

    struct WantsPair <: AbstractModel{2, 1} end
    AstroFit.evaluate(::WantsPair, x::Number) = x  # declares 2 inputs, implements 1
    struct TwoOut <: AbstractModel{1, 2} end
    AstroFit.evaluate(::TwoOut, x::Number) = x  # declares 2 outputs, returns 1
    struct OneOut <: AbstractModel{1, 1} end
    AstroFit.evaluate(::OneOut, x::Number) = (x, x)  # declares 1 output, returns 2

    @test_throws "WantsPair takes 2 number(s) per point, got Float64" render(WantsPair(), 1.0)
    @test_throws "TwoOut declares 2 output(s) per point, but evaluate returned a Float64" render(TwoOut(), 1.0)
    @test_throws "OneOut declares 1 output(s) per point, but evaluate returned a Tuple{Float64, Float64}" render(OneOut(), 1.0)

    # deep in a tree the message still names the component, not the root
    cm = @model begin
        g = Gaussian1D()
        bad = OneOut()
        g + bad
    end
    @test_throws "OneOut declares 1 output(s)" render(cm, [1.0, 2.0])
end
