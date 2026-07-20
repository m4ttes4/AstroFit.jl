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
