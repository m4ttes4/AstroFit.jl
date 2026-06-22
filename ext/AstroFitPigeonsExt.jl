module AstroFitPigeonsExt

using AstroFit
using Pigeons: Pigeons
using LogDensityProblems: LogDensityProblems

Pigeons.initialization(lp::AstroFit.PosteriorTarget, _rng, _int) =
    AstroFit.params(lp.cm)

LogDensityProblems.dimension(lp::AstroFit.PosteriorTarget) = AstroFit.nfree(lp.cm)
LogDensityProblems.logdensity(lp::AstroFit.PosteriorTarget, p) = lp(p)
LogDensityProblems.capabilities(::Type{<:AstroFit.PosteriorTarget}) =
    LogDensityProblems.LogDensityOrder{0}()

function Pigeons.sample_names(state::AbstractVector, lp::AstroFit.PosteriorTarget)
    return [string.(lp.names); "log_density"]
end

end
