# handwritten kernels — bare-loop floor for render! benchmarks
function hw_gauss1d!(out, A, μ, σ, xs)
    @inbounds for i in eachindex(out, xs)
        out[i] = A * exp(-((xs[i] - μ) / σ)^2 / 2)
    end
    out
end

function hw_lorentz1d!(out, A, μ, γ, xs)
    @inbounds for i in eachindex(out, xs)
        out[i] = A / (1 + ((xs[i] - μ) / γ)^2)
    end
    out
end

function hw_blackbody1d!(out, A, T, xs)
    @inbounds for i in eachindex(out, xs)
        x = xs[i]
        out[i] = A * x^3 / (exp(x / T) - 1)
    end
    out
end

# ---- render group ----
SUITE["render"] = BenchmarkGroup()

let out = similar(X1)
    let m = Gaussian1D(8.0, 5.0, 0.6)
        SUITE["render"]["Gaussian1D/astrofit"]    = @benchmarkable render!($out, $m, $X1)
        SUITE["render"]["Gaussian1D/handwritten"] = @benchmarkable hw_gauss1d!($out, 8.0, 5.0, 0.6, $X1)
    end

    let m = Lorentzian1D(8.0, 5.0, 0.6)
        SUITE["render"]["Lorentzian1D/astrofit"]    = @benchmarkable render!($out, $m, $X1)
        SUITE["render"]["Lorentzian1D/handwritten"] = @benchmarkable hw_lorentz1d!($out, 8.0, 5.0, 0.6, $X1)
    end

    let m = BlackBody1D(1.0, 3.0)
        SUITE["render"]["BlackBody1D/astrofit"]    = @benchmarkable render!($out, $m, $X1)
        SUITE["render"]["BlackBody1D/handwritten"] = @benchmarkable hw_blackbody1d!($out, 1.0, 3.0, $X1)
    end

    # Voigt1D — formula identica al sorgente, no handwritten analogue
    let m = Voigt1D(8.0, 5.0, 0.6, 0.4)
        SUITE["render"]["Voigt1D/astrofit"] = @benchmarkable render!($out, $m, $X1)
    end

    # PowerLaw1D — pow path
    let m = PowerLaw1D(1.0, 2.0, 1.5)
        SUITE["render"]["PowerLaw1D/astrofit"] = @benchmarkable render!($out, $m, $X1)
    end

    # BrokenPowerLaw1D — branch inside loop
    let m = BrokenPowerLaw1D(1.0, 5.0, 1.0, 2.0)
        SUITE["render"]["BrokenPowerLaw1D/astrofit"] = @benchmarkable render!($out, $m, $X1)
    end

    # Exponential1D
    let m = Exponential1D(1.0, 2.0)
        SUITE["render"]["Exponential1D/astrofit"] = @benchmarkable render!($out, $m, $X1)
    end
end
