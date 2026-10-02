@testitem "kernel: composes in any order with model semantics" tags = [:kernel] begin
    using AstroFit

    struct DoubleKernel <: AbstractKernel{1, 1} end
    AstroFit.evaluate(::DoubleKernel, f::Number) = 2f

    x = collect(-2.0:0.5:2.0)
    g(a) = Gaussian1D(amplitude = a, mean = 0.0, sigma = 1.0)
    gx(a) = render(g(a), x)

    # (model |> psf) + continuum — the convolved component summed with a bare one
    cm1 = @model begin
        line = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        cont = Const1D(value = 0.5)
        k = DoubleKernel()
        (line |> k) + cont
    end
    @test render(cm1, x) ≈ 2 .* gx(2.0) .+ 0.5

    # continuum on the LEFT of the sum — order must not matter
    cm2 = @model begin
        line = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        cont = Const1D(value = 0.5)
        k = DoubleKernel()
        cont + (line |> k)
    end
    @test render(cm2, x) ≈ 0.5 .+ 2 .* gx(2.0)

    # multiplied by a bare component
    cm3 = @model begin
        line = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        t = Const1D(value = 3.0)
        k = DoubleKernel()
        (line |> k) * t
    end
    @test render(cm3, x) ≈ (2 .* gx(2.0)) .* 3.0

    # chained kernels
    cm4 = @model begin
        line = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        k1 = DoubleKernel()
        k2 = DoubleKernel()
        line |> k1 |> k2
    end
    @test render(cm4, x) ≈ 4 .* gx(2.0)

    # a pointwise SUBTREE feeding the kernel, then more composition on top
    cm5 = @model begin
        a = Gaussian1D(amplitude = 1.0, mean = 0.0, sigma = 1.0)
        b = Const1D(value = 1.0)
        c = Const1D(value = 0.25)
        k = DoubleKernel()
        ((a + b) |> k) + c
    end
    @test render(cm5, x) ≈ 2 .* (gx(1.0) .+ 1.0) .+ 0.25

    # difference and quotient close the operator set
    cm6 = @model begin
        line = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        d = Const1D(value = 2.0)
        k = DoubleKernel()
        (line |> k) - d
    end
    @test render(cm6, x) ≈ 2 .* gx(2.0) .- 2.0

    cm7 = @model begin
        line = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        d = Const1D(value = 2.0)
        k = DoubleKernel()
        (line |> k) / d
    end
    @test render(cm7, x) ≈ (2 .* gx(2.0)) ./ 2.0
end

@testitem "kernel: pointwise chi2 stays allocation-free" tags = [:kernel] begin
    using AstroFit

    cm = @model begin
        g = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        c = Const1D(value = 0.5)
        g + c
    end
    x = collect(-5.0:0.05:5.0)
    y = render(cm, x)
    f = ObjectiveFunction(cm, x, y)
    p = params(cm)
    f(p)                                  # compile
    @test (@allocated f(p)) == 0          # the ≤1.0x-vs-handwritten guarantee
end

@testitem "kernel: fields default Fixed, @free opts in" tags = [:kernel] begin
    using AstroFit

    # A kernel is an ordinary scalar model whose fields default Fixed.
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
    @test nfree(cm) == 3
    @test paramnames(cm) == [:line_amplitude, :line_mean, :line_sigma]
    @test params(cm) isa Vector{Float64}     # not Vector{Real}
    @test render(cm, [0.0, 3.0]) ≈ [1.5, 2exp(-4.5)]  # clipped at the peak, untouched in the wing

    free = @free cm.sat.level
    @test nfree(free) == 4
    @test :sat_level in paramnames(free)
    @test params(free) isa Vector{Float64}
end

@testitem "kernel: array-valued fields (a measured instrumental PSF)" tags = [:kernel] begin
    using AstroFit
    using ForwardDiff

    # A measured PSF: the kernel IS data, not a parametric shape. The struct
    # therefore holds an array, and a scalar the user may want to fit.
    struct ScaledPSF{V <: AbstractVector, T <: Real} <: AbstractKernel{1, 1}
        kernel::V
        scale::T
    end
    AstroFit.evaluate(k::ScaledPSF, f::Number) = k.scale * f  # the stored kernel is calibration data

    kern = [0.1, 0.2, 0.4, 0.2, 0.1]
    x = collect(-5.0:0.25:5.0)
    cm = @model begin
        line = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        psf = ScaledPSF(kern, 1.0)
        line |> psf
    end
    y = render(cm, x)

    # both kernel fields are Fixed, so the array never reaches the parameter vector
    @test nfree(cm) == 3
    @test params(cm) isa Vector{Float64}
    @test withparams(cm, [3.0, 0.5, 1.2]).psf.model.kernel === kern

    # the hard case: a free scalar sitting next to the array field. withparams
    # must promote the numeric fields among themselves and leave the array be —
    # promoting all fields together throws on Vector-vs-Float64.
    cm = @free cm.psf.scale
    @test paramnames(cm) == [:line_amplitude, :line_mean, :line_sigma, :psf_scale]

    dual = withparams(cm, ForwardDiff.Dual.(params(cm), 1.0))
    @test dual.psf.model.kernel isa Vector{Float64}          # array untouched
    @test dual.psf.model.scale isa ForwardDiff.Dual          # scalar lifted

    f = ObjectiveFunction(cm, x, y)
    p = params(cm) .+ [0.3, 0.2, -0.15, 0.1]                 # away from the minimum
    g = ForwardDiff.gradient(f, p)
    @test all(isfinite, g)
    @test !all(iszero, g)
    h = 1.0e-6
    for i in eachindex(p)
        step = zeros(length(p)); step[i] = h
        @test g[i] ≈ (f(p .+ step) - f(p .- step)) / (2h) rtol = 1.0e-6
    end
end

@testitem "kernel: heterogeneous fields survive reconstruction" tags = [:kernel] begin
    using AstroFit
    using ForwardDiff

    # An edge policy is a Symbol: nothing to promote against a number.
    struct EdgePSF{T <: Real} <: AbstractKernel{1, 1}
        sigma::T
        edge::Symbol
    end
    AstroFit.evaluate(k::EdgePSF, f::Number) = k.sigma * f

    cm = @model begin
        l = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        psf = EdgePSF(1.5, :clamp)
        l |> psf
    end
    @test withparams(cm, [3.0, 0.5, 1.2]).psf.model.edge === :clamp

    free = @free cm.psf.sigma
    dual = withparams(free, ForwardDiff.Dual.(params(free), 1.0))
    @test dual.psf.model.edge === :clamp
    @test dual.psf.model.sigma isa ForwardDiff.Dual

    # A concrete Int beside a parametric Float: the Int must NOT be promoted,
    # or the constructor stops matching (and under AD it would become a Dual).
    struct IntPSF{T <: Real} <: AbstractKernel{1, 1}
        halfwidth::Int
        sigma::T
    end
    AstroFit.evaluate(k::IntPSF, f::Number) = k.sigma * f

    cmi = @model begin
        l = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        psf = IntPSF(3, 1.5)
        l |> psf
    end
    rebuilt = withparams(cmi, [3.0, 0.5, 1.2]).psf.model
    @test rebuilt.halfwidth === 3            # still an Int, still 3
    @test rebuilt.sigma === 1.5

    freei = @free cmi.psf.sigma
    duali = withparams(freei, ForwardDiff.Dual.(params(freei), 1.0)).psf.model
    @test duali.halfwidth === 3              # the Dual did not reach it
    @test duali.sigma isa ForwardDiff.Dual
end

@testitem "nested duals: re-parameterizing a model that already carries duals" tags = [:kernel] begin
    using AstroFit
    using ForwardDiff

    # Each field keeps its own type parameter, so `<: Real` (not `<: AbstractFloat`) is
    # the bound that admits a Dual — and a Dual of a Dual, which second-order AD produces.
    struct MixPSF{S <: Real, C <: Real} <: AbstractKernel{1, 1}
        sigma::S
        scale::C
        halfwidth::Int
        edge::Symbol
    end
    AstroFit.evaluate(k::MixPSF, f::Number) = k.scale * f

    cm = @model begin
        l = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        psf = MixPSF(1.5, 2.0, 3, :clamp)
        l |> psf
    end
    cm = @free cm.psf.sigma

    d1 = withparams(cm, ForwardDiff.Dual.(params(cm), 1.0))
    d2 = withparams(d1, ForwardDiff.Dual.(params(d1), 1.0))
    @test d2.psf.model.sigma isa ForwardDiff.Dual
    @test d2.psf.model.scale === 2.0                     # Fixed: stays a Float64
    @test d2.psf.model.halfwidth === 3                   # internal, untouched
    @test d2.psf.model.edge === :clamp
end

@testitem "kernel: many fields of many types" tags = [:kernel] begin
    using AstroFit
    using ForwardDiff

    # Every field is its own type parameter or its own concrete type; reconstruction
    # passes each one through untouched, whatever it is.
    struct BigPSF{S <: Real, C <: Real, V <: AbstractVector, M <: AbstractMatrix, F} <: AbstractKernel{1, 1}
        sigma::S
        scale::C
        taps::V
        weights::M
        apodize::F
        edge::Symbol
        normalize::Bool
        order::Int
        label::String
        span::Tuple{Int, Int}
    end
    AstroFit.evaluate(k::BigPSF, f::Number) = k.scale * f

    cm = @model begin
        l = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        psf = BigPSF(
            1.5, 2.0, [0.25, 0.5, 0.25], [1.0 0.0; 0.0 1.0],
            abs, :clamp, true, 2, "measured-2024", (1, 5)
        )
        l |> psf
    end

    @test nfree(cm) == 3                      # every kernel field is Fixed
    @test size(render(cm, collect(-2.0:0.5:2.0))) == (9,)

    m = withparams(cm, [3.0, 0.5, 1.2]).psf.model
    @test m.sigma === 1.5 && m.scale === 2.0
    @test m.taps == [0.25, 0.5, 0.25]
    @test m.weights == [1.0 0.0; 0.0 1.0]
    @test m.apodize === abs
    @test m.edge === :clamp
    @test m.normalize === true                # Bool <: Number, must stay a Bool
    @test m.order === 2                       # Int, must stay an Int
    @test m.label == "measured-2024"
    @test m.span === (1, 5)

    # free one field: only that one becomes a Dual, every other field is untouched
    free = @free cm.psf.sigma
    d = withparams(free, ForwardDiff.Dual.(params(free), 1.0)).psf.model
    @test d.sigma isa ForwardDiff.Dual
    @test d.scale === 2.0                      # Fixed, stays a Float64
    @test d.taps isa Vector{Float64}           # its own parameter
    @test d.order === 2
    @test d.normalize === true
end

@testitem "kernel: display" tags = [:kernel] begin
    using AstroFit

    struct Saturate{L <: Real} <: AbstractKernel{1, 1}
        level::L
    end

    cm = @model begin
        line = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        cont = Const1D(value = 0.5)
        psf = Saturate(1.5)
        (line |> psf) + cont
    end
    @test sprint(show, cm) == "(line |> psf) + cont"
    @test occursin("psf", sprint(show, MIME"text/plain"(), cm))
end

@testitem "kernel: array fields display compactly" tags = [:kernel] begin
    using AstroFit

    struct ImagePSF{M <: AbstractMatrix} <: AstroFit.AbstractKernel{1, 1}
        kernel::M
    end

    cm = @model begin
        line = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        psf = ImagePSF(fill(0.25, 4, 4))
        line |> psf
    end
    out = sprint(show, MIME"text/plain"(), cm)
    @test occursin("4×4 Matrix{Float64}", out)
    @test !occursin("0.25", out)   # no element dump
end
