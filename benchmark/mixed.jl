using ForwardDiff

# Exercise the consumer paths that a bare-formula benchmark cannot cover.
SUITE["mixed"] = BenchmarkGroup()
let
    pointwise = @model begin
        g = Gaussian1D(amplitude = 2.0, mean = 0.1, sigma = 1.2)
        c = Const1D(value = 0.3)
        g + c
    end
    convolved = @model begin
        g = Gaussian1D(amplitude = 2.0, mean = 0.1, sigma = 1.2)
        k = GaussianPSF(sigma = 1.4)
        c = Const1D(value = 0.3)
        (g |> k) + c
    end
    chained = @model begin
        g = Gaussian1D()
        k1 = GaussianPSF(sigma = 1.4)
        k2 = GaussianPSF(sigma = 1.4)
        g |> k1 |> k2
    end
    transformed = @model begin
        g = Gaussian1D()
        k = GaussianPSF(sigma = 1.4)
        t = Linear1D(slope = 2.0, intercept = 1.0)
        g |> k |> t
    end
    for n in (512, 4096)
        xs = collect(range(-5.0, 5.0; length = n))
        out = similar(xs)
        for (name, m) in (("pointwise", pointwise), ("convolved", convolved),
                ("chained", chained), ("transformed", transformed))
            group = SUITE["mixed"]["$name/$n"] = BenchmarkGroup()
            f = ObjectiveFunction(m, xs, render(m, xs))
            p = AstroFit.params(m)
            group["render"] = @benchmarkable render($m, $xs)
            group["render!"] = @benchmarkable render!($out, $m, $xs)
            group["objective"] = @benchmarkable $f($p)
            group["gradient"] = @benchmarkable ForwardDiff.gradient($f, $p)
        end
    end
end

SUITE["profiles"] = BenchmarkGroup()
let col = collect(range(-5.0, 5.0; length = 256))
    row = reshape(copy(col), 1, :)
    out = zeros(256, 256)
    for m in (Gaussian2D(theta = 0.3, q = 0.8), Sersic2D(theta = 0.3, q = 0.8))
        name = string(nameof(typeof(m)))
        sum_model = m + m
        SUITE["profiles"]["$name/grid!"] = @benchmarkable render!($out, $m, $col, $row)
        SUITE["profiles"]["$name/sum!"] = @benchmarkable render!($out, $sum_model, $col, $row)
    end
    xs = collect(range(-5.0, 5.0; length = 512))
    out1 = similar(xs)
    v = Voigt1D()
    sum_model = v + v
    SUITE["profiles"]["Voigt1D/render!"] = @benchmarkable render!($out1, $v, $xs)
    SUITE["profiles"]["Voigt1D/sum!"] = @benchmarkable render!($out1, $sum_model, $xs)
end
