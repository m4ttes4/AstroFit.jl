@testmodule Fixtures begin
    using AstroFit

    # A continuum and two lines, shared by the items that need one so its withparams
    # and render specializations compile once per run.
    const spectrum = @model begin
        cont = Linear1D(slope = 0.0, intercept = 1.0)
        a = Gaussian1D(amplitude = 2.0, mean = 0.0, sigma = 1.0)
        b = Gaussian1D(amplitude = 1.0, mean = 5.0, sigma = 2.0)
        cont + a + b
    end

    # The same spectrum with every constraint kind: Fixed (at a value other than the
    # stored one), Bounded, two Tied fields, and Free.
    const constrained = let cm = spectrum
        @constrain cm begin
            cont.slope = 0.1
            a.sigma in (0.1, 10.0)
            b.amplitude -> 2 * a.amplitude
            b.sigma -> a.sigma
        end
        cm
    end

    # Central differences, an independent reference for ForwardDiff gradients:
    # truncation ~h² ≈ 1e-12 and rounding ~eps/h ≈ 1e-10 sit well inside rtol = 1e-6.
    function fdgrad(f, p; h = 1.0e-6)
        return map(eachindex(p)) do i
            e = zeros(length(p))
            e[i] = h
            (f(p .+ e) - f(p .- e)) / 2h
        end
    end
end
