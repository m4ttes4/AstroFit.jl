@testitem "kernel: user-defined kernel renders over a vector" tags = [:kernel] begin
    using AstroFit

    # A kernel is written like any other model — a struct plus one render method.
    # It defines the ARRAY render instead of the scalar one.
    struct ReverseKernel <: AbstractKernel end
    AstroFit.render(::ReverseKernel, ys::AbstractVector) = reverse(ys)

    k = ReverseKernel()
    @test render(k, [1.0, 2.0, 3.0]) == [3.0, 2.0, 1.0]
    @test AstroFit.evalstyle(k) === AstroFit.Domainwise()

    cm = @model begin
        g = Gaussian1D(amplitude = 2.0, mean = 1.0, sigma = 1.0)
        k = ReverseKernel()
        g |> k
    end
    x = collect(-3.0:0.5:3.0)
    @test render(cm, x) == reverse(render(cm.g, x))
    # navigating to the kernel and rendering it directly also works
    @test render(cm.k, [1.0, 2.0]) == [2.0, 1.0]
end

@testitem "kernel: composes in any order with model semantics" tags = [:kernel] begin
    using AstroFit

    struct DoubleKernel <: AbstractKernel end
    AstroFit.render(::DoubleKernel, ys::AbstractVector) = 2 .* ys

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

@testitem "kernel: evalstyle propagates, pointwise trees stay pointwise" tags = [:kernel] begin
    using AstroFit

    pointwise = @model begin
        g = Gaussian1D(amplitude = 1.0, mean = 0.0, sigma = 1.0)
        c = Const1D(value = 0.5)
        g + c
    end
    @test AstroFit.evalstyle(pointwise) === AstroFit.Pointwise()

    withkernel = @model begin
        g = Gaussian1D(amplitude = 1.0, mean = 0.0, sigma = 1.0)
        c = Const1D(value = 0.5)
        psf = GaussianPSF(sigma = 1.5)
        (g |> psf) + c
    end
    @test AstroFit.evalstyle(withkernel) === AstroFit.Domainwise()
    # the pointwise branch of a domainwise tree is still pointwise
    @test AstroFit.evalstyle(withkernel.c) === AstroFit.Pointwise()
    @test AstroFit.evalstyle(withkernel.psf) === AstroFit.Domainwise()
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

@testitem "kernel: render! works on both paths" tags = [:kernel] begin
    using AstroFit

    x = collect(-2.0:0.5:2.0)
    cm = @model begin
        g = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        psf = GaussianPSF(sigma = 1.0)
        g |> psf
    end
    out = similar(x)
    render!(out, cm, x)
    @test out ≈ render(cm, x)

    pw = @model begin
        g = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        g
    end
    out2 = similar(x)
    render!(out2, pw, x)
    @test out2 ≈ render(pw, x)
end

@testitem "kernel: GaussianPSF is size-preserving and conserves a flat level" tags = [:kernel] begin
    using AstroFit

    k = GaussianPSF(sigma = 2.0)
    ys = randn(50)
    @test size(render(k, ys)) == size(ys)

    # edge renormalization: a constant signal must come back unchanged, borders
    # included — otherwise truncation would darken the array ends.
    flat = fill(3.0, 40)
    @test render(k, flat) ≈ flat

    # convolution widens a narrow feature and conserves its peak position
    spike = zeros(41); spike[21] = 1.0
    sm = render(k, spike)
    @test argmax(sm) == 21
    @test sum(sm) ≈ 1.0 rtol = 1.0e-6
    @test maximum(sm) < 1.0

    @test_throws ArgumentError render(GaussianPSF(sigma = -1.0), ys)
end

@testitem "kernel: fields default Fixed, @free opts in" tags = [:kernel] begin
    using AstroFit

    cm = @model begin
        line = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        psf = GaussianPSF(sigma = 1.5)
        line |> psf
    end
    # the PSF width is a calibration input: no optimizer slot by default
    @test nfree(cm) == 3
    @test paramnames(cm) == [:line_amplitude, :line_mean, :line_sigma]
    @test params(cm) isa Vector{Float64}     # not Vector{Real}

    free = @free cm.psf.sigma
    @test nfree(free) == 4
    @test :psf_sigma in paramnames(free)
    @test params(free) isa Vector{Float64}
end

@testitem "kernel: mis-sized kernel output is caught, not silently misaligned" tags = [:kernel] begin
    using AstroFit

    struct TruncKernel <: AbstractKernel end
    AstroFit.render(::TruncKernel, ys::AbstractVector) = ys[1:(end - 1)]

    cm = @model begin
        g = Gaussian1D(amplitude = 1.0, mean = 0.0, sigma = 1.0)
        k = TruncKernel()
        g |> k
    end
    x = collect(-2.0:0.5:2.0)
    y = render(cm.g, x)
    f = ObjectiveFunction(cm, x, y)
    @test_throws DimensionMismatch f(params(cm))
end

@testitem "kernel: chi2 and ForwardDiff gradients agree with finite differences" tags = [:kernel] begin
    using AstroFit
    using ForwardDiff

    x = collect(-10.0:0.2:10.0)
    truth = @model begin
        line = Gaussian1D(amplitude = 2.0, mean = 1.0, sigma = 1.0)
        psf = GaussianPSF(sigma = 2.5)
        line |> psf
    end
    y = render(truth, x)

    cm = @model begin
        line = Gaussian1D(amplitude = 1.3, mean = 0.4, sigma = 1.2)
        psf = GaussianPSF(sigma = 1.7)
        line |> psf
    end
    cm = @free cm.psf.sigma          # the hard case: a Dual flowing into the kernel
    f = ObjectiveFunction(cm, x, y)
    p = params(cm)

    ad = ForwardDiff.gradient(f, p)
    @test all(isfinite, ad)
    h = 1.0e-6
    for i in eachindex(p)
        step = zeros(length(p)); step[i] = h
        fd = (f(p .+ step) - f(p .- step)) / (2h)
        @test ad[i] ≈ fd rtol = 1.0e-5
    end

    # χ² is exactly zero at the truth's parameters
    exact = @model begin
        line = Gaussian1D(amplitude = 2.0, mean = 1.0, sigma = 1.0)
        psf = GaussianPSF(sigma = 2.5)
        line |> psf
    end
    @test ObjectiveFunction(exact, x, y)(params(exact)) ≈ 0.0 atol = 1.0e-20
end

@testitem "kernel: fits through Optimization" tags = [:kernel] begin
    using AstroFit
    using Optimization, OptimizationOptimJL

    x = collect(-10.0:0.2:10.0)
    truth = @model begin
        line = Gaussian1D(amplitude = 2.0, mean = 1.0, sigma = 1.0)
        cont = Const1D(value = 0.5)
        psf = GaussianPSF(sigma = 2.0)
        (line |> psf) + cont
    end
    y = render(truth, x)

    start = @model begin
        line = Gaussian1D(amplitude = 1.0, mean = 0.0, sigma = 2.0)
        cont = Const1D(value = 0.1)
        psf = GaussianPSF(sigma = 2.0)
        (line |> psf) + cont
    end
    sol = solve(OptimizationProblem(start, x, y), Optim.LBFGS())
    @test sol.u ≈ [2.0, 1.0, 1.0, 0.5] rtol = 1.0e-4
end

@testitem "kernel: the Distributions path routes through the kernel chi2" tags = [:kernel] begin
    using AstroFit
    using Distributions

    x = collect(-5.0:0.2:5.0)
    cm = @model begin
        line = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        cont = Const1D(value = 0.5)
        psf = GaussianPSF(sigma = 1.5)
        (line |> psf) + cont
    end
    y = render(cm, x)
    err = fill(0.1, length(y))
    p = AstroFit.params(cm)

    # loglikelihood/logposterior are chi2-derived, so they inherit the split
    ll = ObjectiveFunction(cm, x, y, err; statistic = AstroFit.loglikelihood)
    @test isfinite(ll(p))
    @test ll(p) ≈ -0.5 * AstroFit.chi2(ObjectiveFunction(cm, x, y, err), p) + ll._loglike_const

    # the posterior path needs a prior on every free slot
    prior = @constrain cm begin
        line.amplitude ~ LogNormal(0.0, 1.0)
        line.mean ~ Normal(0.0, 1.0)
        line.sigma ~ truncated(Normal(1.0, 0.5); lower = 0.0)
        cont.value ~ Normal(0.5, 1.0)
    end
    lp = ObjectiveFunction(prior, x, y, err; statistic = AstroFit.neglogposterior)
    @test isfinite(lp(AstroFit.params(prior)))
end

@testitem "kernel: Bayesian targets and posterior gradients" tags = [:kernel] begin
    using AstroFit
    using Distributions, LogDensityProblems, ForwardDiff

    x = collect(-6.0:0.3:6.0)
    cm = @model begin
        line = Gaussian1D(amplitude = 2.0, mean = 0.5, sigma = 1.0)
        cont = Const1D(value = 0.4)
        psf = GaussianPSF(sigma = 1.4)      # off the 4σ∈ℤ truncation boundary
        (line |> psf) + cont
    end
    y = render(cm, x)
    err = fill(0.05, length(y))
    cm = @constrain cm begin
        line.amplitude ~ truncated(Normal(2.0, 1.0); lower = 0.0)
        line.mean ~ Normal(0.5, 1.0)
        line.sigma ~ truncated(Normal(1.0, 0.5); lower = 0.01)
        cont.value ~ Normal(0.4, 0.5)
    end

    f = ObjectiveFunction(cm, x, y, err; statistic = AstroFit.logposterior)
    p = AstroFit.params(cm)

    # the LogDensityProblems target routes through the kernel chi2
    @test isfinite(f(p))
    @test LogDensityProblems.logdensity(f, p) == f(p)
    @test LogDensityProblems.dimension(f) == 4     # the PSF width is not a slot

    # the posterior is differentiable through the convolution
    g = ForwardDiff.gradient(f, p)
    @test all(isfinite, g)
    h = 1.0e-6
    for i in eachindex(p)
        step = zeros(length(p)); step[i] = h
        @test g[i] ≈ (f(p .+ step) - f(p .- step)) / (2h) rtol = 1.0e-5
    end

    # and with the PSF width itself free — a Dual reaching the kernel's own math
    free = @free cm.psf.sigma
    free = @prior free.psf.sigma ~ truncated(Normal(1.4, 0.3); lower = 0.1)
    ff = ObjectiveFunction(free, x, y, err; statistic = AstroFit.logposterior)
    pf = AstroFit.params(free)
    gf = ForwardDiff.gradient(ff, pf)
    @test all(isfinite, gf)
    for i in eachindex(pf)
        step = zeros(length(pf)); step[i] = h
        @test gf[i] ≈ (ff(pf .+ step) - ff(pf .- step)) / (2h) rtol = 1.0e-4
    end
end

@testitem "kernel: Pigeons adapter accepts a kernel model" tags = [:kernel] begin
    using AstroFit
    using Distributions, Pigeons, Random

    x = collect(-6.0:0.5:6.0)
    cm = @model begin
        line = Gaussian1D(amplitude = 2.0, mean = 0.5, sigma = 1.0)
        cont = Const1D(value = 0.4)
        psf = GaussianPSF(sigma = 1.4)
        (line |> psf) + cont
    end
    y = render(cm, x)
    cm = @constrain cm begin
        line.amplitude ~ truncated(Normal(2.0, 1.0); lower = 0.0)
        line.mean ~ Normal(0.5, 1.0)
        line.sigma ~ truncated(Normal(1.0, 0.5); lower = 0.01)
        cont.value ~ Normal(0.4, 0.5)
    end
    f = ObjectiveFunction(cm, x, y, fill(0.05, length(y)); statistic = AstroFit.logposterior)

    init = Pigeons.initialization(f, Random.default_rng(), 1)
    @test length(init) == 4                  # the fixed PSF width contributes no slot
    @test all(isfinite, init)
    @test isfinite(f(init))                  # the drawn point evaluates through the kernel
    @test Pigeons.default_reference(f) isa Pigeons.DistributionLogPotential
end

@testitem "kernel: array-valued fields (a measured instrumental PSF)" tags = [:kernel] begin
    using AstroFit
    using ForwardDiff

    # A measured PSF: the kernel IS data, not a parametric shape. The struct
    # therefore holds an array, and a scalar the user may want to fit.
    struct ScaledPSF{V <: AbstractVector, T <: Real} <: AbstractKernel
        kernel::V
        scale::T
    end
    function AstroFit.render(k::ScaledPSF, ys::AbstractVector)
        w = k.kernel
        h = length(w) ÷ 2
        n = length(ys)
        out = similar(ys, promote_type(eltype(ys), eltype(w), typeof(k.scale)))
        for i in 1:n
            acc = zero(eltype(out))
            wsum = zero(eltype(w))
            for (j, d) in enumerate((-h):h)
                m = i + d
                1 <= m <= n || continue
                acc += w[j] * ys[m]
                wsum += w[j]
            end
            out[i] = k.scale * acc / wsum
        end
        return out
    end

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

@testitem "kernel: 2D PSF whose field is the intensity matrix" tags = [:kernel] begin
    using AstroFit
    using ForwardDiff

    struct ImagePSF{M <: AbstractMatrix} <: AbstractKernel
        psf::M
    end
    function AstroFit.render(k::ImagePSF, img::AbstractMatrix)
        w = k.psf
        hy, hx = size(w) .÷ 2
        ny, nx = size(img)
        out = similar(img, promote_type(eltype(img), eltype(w)))
        for j in 1:nx, i in 1:ny
            acc = zero(eltype(out))
            wsum = zero(eltype(w))
            for (bj, dj) in enumerate((-hx):hx), (bi, di) in enumerate((-hy):hy)
                ii, jj = i + di, j + dj
                (1 <= ii <= ny && 1 <= jj <= nx) || continue
                acc += w[bi, bj] * img[ii, jj]
                wsum += w[bi, bj]
            end
            out[i, j] = acc / wsum
        end
        return out
    end

    struct Gauss2D{T <: Real} <: AbstractModel
        amp::T
        x0::T
        y0::T
        s::T
    end
    AstroFit.render(m::Gauss2D, x::Number, y::Number) =
        m.amp * exp(-((x - m.x0)^2 + (y - m.y0)^2) / (2 * m.s^2))

    psfmat = [0.05 0.1 0.05; 0.1 0.4 0.1; 0.05 0.1 0.05]
    im = @model begin
        src = Gauss2D(3.0, 0.0, 0.0, 1.0)
        ipsf = ImagePSF(psfmat)
        src |> ipsf
    end

    # column × row coordinates produce the grid the kernel convolves
    X = collect(-3.0:0.5:3.0)
    Y = permutedims(collect(-3.0:0.5:3.0))
    out = render(im, X, Y)
    @test size(out) == (length(X), length(Y))
    @test nfree(im) == 4                       # the PSF matrix is not a parameter
    @test maximum(out) < maximum(render(im.src, X, Y))   # convolution smooths the peak

    obj(q) = sum(abs2, render(withparams(im, q), X, Y) .- out)
    q = params(im) .+ [0.4, 0.2, -0.3, 0.15]
    g = ForwardDiff.gradient(obj, q)
    @test all(isfinite, g)
    @test !all(iszero, g)
    h = 1.0e-6
    for i in eachindex(q)
        step = zeros(length(q)); step[i] = h
        @test g[i] ≈ (obj(q .+ step) - obj(q .- step)) / (2h) rtol = 1.0e-6
    end
end

@testitem "kernel: heterogeneous fields survive reconstruction" tags = [:kernel] begin
    using AstroFit
    using ForwardDiff

    # An edge policy is a Symbol: nothing to promote against a number.
    struct EdgePSF{T <: Real} <: AbstractKernel
        sigma::T
        edge::Symbol
    end
    AstroFit.render(k::EdgePSF, ys::AbstractVector) = k.sigma .* ys

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
    struct IntPSF{T <: Real} <: AbstractKernel
        halfwidth::Int
        sigma::T
    end
    AstroFit.render(k::IntPSF, ys::AbstractVector) = k.sigma .* ys

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
    struct MixPSF{S <: Real, C <: Real} <: AbstractKernel
        sigma::S
        scale::C
        halfwidth::Int
        edge::Symbol
    end
    AstroFit.render(k::MixPSF, ys::AbstractVector) = k.scale .* ys

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
    struct BigPSF{S <: Real, C <: Real, V <: AbstractVector, M <: AbstractMatrix, F} <: AbstractKernel
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
    AstroFit.render(k::BigPSF, ys::AbstractVector) = k.scale .* ys

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

@testitem "kernel: errors and display" tags = [:kernel] begin
    using AstroFit

    # a vector-only kernel handed a matrix reports what it needs
    k = GaussianPSF(sigma = 1.0)
    @test_throws ArgumentError render(k, rand(4, 4))

    cm = @model begin
        line = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        cont = Const1D(value = 0.5)
        psf = GaussianPSF(sigma = 1.5)
        (line |> psf) + cont
    end
    @test sprint(show, cm) == "(line |> psf) + cont"
    @test occursin("psf", sprint(show, MIME"text/plain"(), cm))
end

@testitem "kernel: array fields display compactly" tags = [:kernel] begin
    using AstroFit

    struct ImagePSF{M <: AbstractMatrix} <: AstroFit.AbstractKernel
        kernel::M
    end
    AstroFit.render(k::ImagePSF, img::AbstractMatrix) = img

    cm = @model begin
        line = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        psf = ImagePSF(fill(0.25, 4, 4))
        line |> psf
    end
    out = sprint(show, MIME"text/plain"(), cm)
    @test occursin("4×4 Matrix{Float64}", out)
    @test !occursin("0.25", out)   # no element dump
end
