module AstroFitDistributionsExt

using AstroFit
using Distributions: logpdf

import AstroFit: logprior

function logprior(cm::AstroFit.CompiledModel)
    model = getfield(cm, :model)
    sum((logpdf(prior, optic(model)) for (optic, prior) in getfield(cm, :priors));
        init = 0.0)
end

logprior(cm::AstroFit.CompiledModel, p) = logprior(AstroFit.withparams(cm, p))

end
