@testitem "Coords is the lazy Cartesian product of its axes" tags = [:core] begin
    using AstroFit

    x = Float32[0.5, 1.5, 2.5]
    y = range(-1.0, 1.0; length = 4)
    z = [10, 20]
    g = Coords(x, y)
    @test g == [(xi, yj) for xi in x, yj in y]  # independent: a comprehension
    @test eltype(g) === Tuple{Float32, Float64}  # axis element types kept, not promoted
    @test size(Coords(x, y, z)) == (3, 4, 2)
    @test Coords(x, y, z)[2, 3, 1] === (x[2], y[3], z[1])

    @test_throws MethodError Coords((x,))  # one axis cannot be built
    @test_throws "CartesianIndices(img)" Coords(zeros(3, 4))
    @test_throws "CartesianIndices(img)" Coords(x)

    build(a, b) = @allocated Coords(a, b)
    build(x, y)
    @test build(x, y) == 0
    @test @inferred(Coords(x, y)) isa Coords{Tuple{Float32, Float64}, 2}
end
