using ForwardDiff
using Random: Xoshiro

# Exercise the consumer paths that a bare-formula benchmark cannot cover.
SUITE["mixed"] = BenchmarkGroup()
let
    pointwise = @model begin
        g = Gaussian1D(amplitude = 2.0, mean = 0.1, sigma = 1.2)
        c = Const1D(value = 0.3)
        g + c
    end
    for n in (512, 4096)
        xs = collect(range(-5.0, 5.0; length = n))
        out = similar(xs)
        for (name, m) in (("pointwise", pointwise),)
            group = SUITE["mixed"]["$name/$n"] = BenchmarkGroup()
            f = ObjectiveFunction(m, xs, render(m, xs))
            p = AstroFit.params(m)
            group["render"] = @benchmarkable render($m, $xs)
            group["render!"] = @benchmarkable $out .= render.($m, $xs)
            group["objective"] = @benchmarkable $f($p)
            group["gradient"] = @benchmarkable ForwardDiff.gradient($f, $p)
        end
    end
end

SUITE["profiles"] = BenchmarkGroup()
let col = collect(range(-5.0, 5.0; length = 256))
    pts = Coords(col, copy(col))
    out = zeros(256, 256)
    for m in (Gaussian2D(theta = 0.3, q = 0.8), Sersic2D(theta = 0.3, q = 0.8))
        name = string(nameof(typeof(m)))
        sum_model = m + m
        # the same sum as a real CompiledModel tree: Leaf and CompiledModel layers on the point path
        compiled = @model begin
            a = m
            b = m
            a + b
        end
        SUITE["profiles"]["$name/grid!"] = @benchmarkable $out .= render.($m, $pts)
        SUITE["profiles"]["$name/sum!"] = @benchmarkable $out .= render.($sum_model, $pts)
        SUITE["profiles"]["$name/compiled!"] = @benchmarkable $out .= render.($compiled, $pts)
    end
    xs = collect(range(-5.0, 5.0; length = 512))
    out1 = similar(xs)
    v = Voigt1D()
    sum_model = v + v
    SUITE["profiles"]["Voigt1D/render!"] = @benchmarkable $out1 .= render.($v, $xs)
    SUITE["profiles"]["Voigt1D/sum!"] = @benchmarkable $out1 .= render.($sum_model, $xs)
end

# The 2D fit path: chi2 over Coords against the same loop written by hand.
function hand_coords_chi2(p, xs, ys, y, err)
    A, x0, y0, σ, q, θ, c = p
    s, co = sincos(θ)
    acc = zero(eltype(p))
    @inbounds for j in eachindex(ys), i in eachindex(xs)
        dx, dy = xs[i] - x0, ys[j] - y0
        xr, yr = co * dx + s * dy, -s * dx + co * dy
        r = A * exp(-0.5 * (xr^2 + yr^2 / q^2) / σ^2) + c - y[i, j]
        acc += abs2(r / err[i, j])
    end
    return acc
end

let col = collect(range(-5.0, 5.0; length = 100))
    pts = Coords(col, copy(col))
    m = @model begin
        g = Gaussian2D(amplitude = 2.0, x0 = 0.3, y0 = -0.4, sigma = 1.2, q = 0.7, theta = 0.4)
        sky = Const2D(value = 0.1)
        g + sky
    end
    err = fill(0.05, size(pts))
    y = render(m, pts) .+ err .* randn(Xoshiro(42), size(pts))
    f = ObjectiveFunction(m, pts, y, err)
    hw(p) = hand_coords_chi2(p, pts.ax[1], pts.ax[2], y, err)
    p = AstroFit.params(m)
    @assert f(p) ≈ hw(p)
    group = SUITE["mixed"]["coords/100"] = BenchmarkGroup()
    group["objective"] = @benchmarkable $f($p)
    group["objective_hw"] = @benchmarkable $hw($p)
    group["gradient"] = @benchmarkable ForwardDiff.gradient($f, $p)
    group["gradient_hw"] = @benchmarkable ForwardDiff.gradient($hw, $p)
end
