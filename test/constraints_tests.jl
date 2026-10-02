@testitem "only Free and Bounded fields get optimizer slots, in tree order" tags = [:core, :params] setup = [Fixtures] begin
    using AstroFit

    cm = Fixtures.constrained
    @test paramnames(cm) == [:cont_intercept, :a_amplitude, :a_mean, :a_sigma, :b_mean]
    @test params(cm) == [1.0, 2.0, 0.0, 1.0, 5.0]  # stored values, passed through
    @test bounds(cm) == ([-Inf, -Inf, -Inf, 0.1, -Inf], [Inf, Inf, Inf, 10.0, Inf])
end

@testitem "Fixed and Tied fields are filled in, both in withparams and in the stored model" tags = [:core, :tied] setup = [Fixtures] begin
    using AstroFit

    cm = Fixtures.constrained
    # slope fixed at 0.1; b's amplitude = 2 × a's, b's σ = a's
    spec(x, (c, A, μ, σ, μb)) = 0.1x + c + A * exp(-((x - μ) / σ)^2 / 2) + 2A * exp(-((x - μb) / σ)^2 / 2)

    x = [0.5, 7.0]  # a's peak, and one point that sees b's width
    p = [1.5, 3.0, 0.5, 2.0, 6.0]
    @test render(withparams(cm, p), x) ≈ spec.(x, Ref(p))
    # the edit already applied the constraints to the stored values
    @test render(cm, x) ≈ spec.(x, Ref(params(cm)))
end

@testitem "standalone macros make the same edits as a @constrain block" tags = [:authoring] setup = [Fixtures] begin
    using AstroFit

    cm = Fixtures.spectrum
    @fix cm.cont.slope = 0.1
    @bound cm.a.sigma in (0.1, 10.0)
    @tie cm.b.amplitude -> 2 * cm.a.amplitude
    @tie cm.b.sigma -> cm.a.sigma
    @test paramnames(cm) == paramnames(Fixtures.constrained)
    @test bounds(cm) == bounds(Fixtures.constrained)
    @test render(cm, [0.5, 7.0]) ≈ render(Fixtures.constrained, [0.5, 7.0])

    @free cm.a.sigma
    @test bounds(cm) == (fill(-Inf, 5), fill(Inf, 5))
end

@testitem "@constrain reads values from caller data and fixes bare paths at their current value" tags = [:authoring] begin
    using AstroFit

    cm = @model begin
        line = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        line
    end
    cfg = (factor = 2.5, lo = -3.0, hi = 4.0)
    @constrain cm begin
        line.amplitude = cfg.factor
        line.mean in (cfg.lo, cfg.hi)
        line.sigma
    end
    @test paramnames(cm) == [:line_mean]
    @test bounds(cm) == ([-3.0], [4.0])
    @test render(cm, 1.0) ≈ 2.5exp(-1 / 2)  # amplitude 2.5, σ kept at 1

    # A block states the full constraint set: what it does not mention goes back to its
    # default, unlike the standalone macros, which edit one field.
    edited = @fix cm.line.mean
    @test isempty(paramnames(edited))
    @constrain cm begin
        line.sigma in (0.5, 2.0)
    end
    @test paramnames(cm) == [:line_amplitude, :line_mean, :line_sigma]
    @test bounds(cm) == ([-Inf, -Inf, 0.5], [Inf, Inf, 2.0])
end

@testitem "tie masters must be free parameters that exist" tags = [:authoring, :tied] begin
    using AstroFit

    cm = @model begin
        a = Gaussian1D()
        b = Gaussian1D()
        c = Gaussian1D()
        a + b + c
    end
    notfree = "which is not a free parameter"
    @test_throws notfree (
        m -> @constrain m begin
            a.sigma = 1.0
            b.sigma -> a.sigma
        end
    )(cm)
    @test_throws notfree (
        m -> @constrain m begin  # no chained ties
            b.sigma -> a.sigma
            c.sigma -> b.sigma
        end
    )(cm)
    @test_throws notfree (
        m -> @constrain m begin
            a.sigma -> 2 * a.sigma
        end
    )(cm)
    @test_throws notfree (m -> @tie m.b.sigma -> m.a.nope)(cm)
    @test_throws "no component `d` in model" (
        m -> @constrain m begin
            d.sigma = 1.0
        end
    )(cm)

    # Ties are checked against the block's final state: here the master is a kernel
    # field (Fixed by default) freed later in the same block.
    struct Gain{G <: Real} <: AbstractKernel{1, 1}
        gain::G
    end
    ck = @model begin
        g = Gaussian1D(amplitude = 1.0, mean = 0.0, sigma = 1.0)
        k = Gain(3.0)
        g |> k
    end
    @constrain ck begin
        g.sigma -> k.gain
        @free k.gain
    end
    @test paramnames(ck) == [:g_amplitude, :g_mean, :k_gain]
    @test ck.g.model.sigma == 3.0
end

@testitem "@constrain rejects malformed blocks at expansion" tags = [:authoring] begin
    using AstroFit

    expand(blk) = macroexpand(@__MODULE__, :(AstroFit.@constrain cm $blk))
    @test_throws "`a.sigma` constrained twice" expand(
        :(
            begin
                a.sigma = 1.0
                a.sigma in (0, 1)
            end
        )
    )
    @test_throws "unrecognized expression" expand(
        :(
            begin
                @free a
            end
        )
    )
end

@testitem "setconstraint edits one field and names a missing one" tags = [:authoring] begin
    using AstroFit

    cm = @model begin
        line = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        line
    end
    cm2 = setconstraint(cm, :line, :sigma, Bounded(0.1, 10.0))
    @test bounds(cm2) == ([-Inf, -Inf, 0.1], [Inf, Inf, 10.0])
    @test bounds(cm) == (fill(-Inf, 3), fill(Inf, 3))  # the original is untouched

    @test_throws "no component `missing` in model" setconstraint(cm, :missing, :sigma, Free())
    @test_throws "no parameter `missing` in `line` (Gaussian1D)" setconstraint(cm, :line, :missing, Free())
end

@testitem "Bounded rejects empty and NaN intervals for every argument type" tags = [:core] begin
    using AstroFit

    # Same-typed arguments used to reach the default constructor and skip the checks.
    @test_throws ArgumentError Bounded(5.0, 1.0)
    @test_throws ArgumentError Bounded(1.0, 1.0)
    @test_throws ArgumentError Bounded(NaN, 1.0)
    @test_throws ArgumentError Bounded(5, 1.0)
    @test_throws ArgumentError Bounded{Float64}(5, 1)
    @test Bounded(0, Inf) === Bounded{Float64}(0.0, Inf)

    cm = @model begin
        g = Gaussian1D()
        g
    end
    # `@bound` rebinds its variable, so give it one local to the @test_throws scope.
    @test_throws ArgumentError (m = cm; @bound m.g.sigma in (5.0, 1.0))
end
