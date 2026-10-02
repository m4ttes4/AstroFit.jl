# Shared fixtures for the AstroFit benchmark suite.
#
# Environment: `examples/` (it has AstroFit as a path dep plus the solver stack).
#   julia --project=examples --startup-file=no bench/bench_run.jl
# NOTE the stale `bench/Project.toml` belongs to the older scripts in this folder
# and is NOT used by this suite.
#
# Three cases, each a cut-down version of a real script in examples/main:
#
#   SPEC  Halpha + [NII] doublet over a linear continuum   — pointwise 1D, ties, 6 free
#   WIDE  13-leaf galaxy spectrum, tie-heavy               — wide pointwise tree, 12 free
#   IMG   two blended galaxies (bulge + disk)              — 2D, 100x100 image, 20 free
#
# Every case exposes the same fields, so a benchmark can loop over them:
#
#   cm      CompiledModel (constraints applied, validated)
#   coords  Tuple of coordinate arrays, as passed to ObjectiveFunction
#   y       data (noisy realization of the truth), err its 1-sigma vector
#   p       free-parameter vector (params(cm)) — the fit's starting point
#   f       ObjectiveFunction (chi2)
#   cfg     ForwardDiff.GradientConfig with FULL chunk, mirroring
#           `_fullchunk` in ext/AstroFitOptimizationExt.jl
#   g       preallocated gradient buffer
#   out     preallocated render! buffer
#
# Data is built with an explicit Xoshiro(42) — never the global RNG — so a
# `setup=` block that re-runs still sees identical numbers.

using AstroFit
# `params` collides with Distributions/StatsAPI *and* BenchmarkTools;
# `loglikelihood` collides with Distributions. Import explicitly.
using AstroFit: params, logposterior, loglikelihood, logprior
using Distributions
using ForwardDiff
using Random: Xoshiro

const RNG = Xoshiro(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Consume every field of a rebuilt tree. `withparams` is straight-line struct
# construction, so an unconsumed result is deleted by the compiler and the
# benchmark measures nothing (~1.5 ns). Folding the fields into one number keeps
# the constructors alive.
fieldsum(cm::CompiledModel) = _fieldsum(getfield(cm, :tree))
_fieldsum(l::AstroFit.Leaf) = _fieldsum(l.model)
_fieldsum(n::Union{AstroFit.Sum, AstroFit.Difference, AstroFit.Product, AstroFit.Quotient, AstroFit.Pipe}) =
    _fieldsum(n.left) + _fieldsum(n.right)
@generated function _fieldsum(m::AbstractModel)
    return Expr(:call, :+, (:(getfield(m, $i)) for i in 1:fieldcount(m))...)
end

# Full-chunk forward-mode config: seed all n free parameters in one sweep, which
# is what OptimizationFunction does through `AutoForwardDiff(chunksize = n)`.
fullchunk(f, p) = ForwardDiff.GradientConfig(f, p, ForwardDiff.Chunk{length(p)}())

_case(cm, coords, y, err, out) = (;
    cm, coords, y, err, out,
    p = params(cm),
    f = ObjectiveFunction(cm, length(coords) == 1 ? coords[1] : coords, y, err),
    cfg = fullchunk(ObjectiveFunction(cm, length(coords) == 1 ? coords[1] : coords, y, err), params(cm)),
    g = zeros(nfree(cm)),
)

# ---------------------------------------------------------------------------
# SPEC — Halpha + [NII] over a linear continuum. Pointwise, tie-heavy, small.
# ---------------------------------------------------------------------------

const L_HA = 6562.8
const L_NII_B = 6548.05
const L_NII_R = 6583.45

function spec_model(; slope = 0.0, intercept = 8.0, ha_amp = 8.0, ha_mean = L_HA, ha_sigma = 3.0, nii_amp = 0.8)
    cm = @model begin
        cont = Linear1D(slope = slope, intercept = intercept)
        ha = Gaussian1D(amplitude = ha_amp, mean = ha_mean, sigma = ha_sigma)
        nii_b = Gaussian1D(amplitude = nii_amp, mean = L_NII_B, sigma = ha_sigma)
        nii_r = Gaussian1D(amplitude = 3.06 * nii_amp, mean = L_NII_R, sigma = ha_sigma)
        cont + ha + nii_b + nii_r
    end
    @constrain cm begin
        cont.slope in (-1.0, 1.0)
        cont.intercept in (0.0, 50.0)
        ha.amplitude in (0.0, 100.0)
        ha.mean in (L_HA - 5.0, L_HA + 5.0)
        ha.sigma in (0.5, 15.0)
        nii_b.amplitude in (0.0, 50.0)
        nii_b.mean -> ha.mean + (L_NII_B - L_HA)      # atomic separation
        nii_b.sigma -> ha.sigma                       # same gas
        nii_r.amplitude -> 3.06 * nii_b.amplitude     # fixed line ratio
        nii_r.mean -> ha.mean + (L_NII_R - L_HA)
        nii_r.sigma -> ha.sigma
    end
    return cm
end

const SPEC_X = collect(range(6500.0, 6650.0; length = 1000))

let truth = spec_model(slope = -0.002, intercept = 9.5, ha_amp = 10.5, ha_mean = L_HA + 1.2, ha_sigma = 3.4, nii_amp = 0.95)
    y_true = render(truth, SPEC_X)
    err = fill(0.08, length(SPEC_X))
    y = y_true .+ err .* randn(RNG, length(SPEC_X))
    global const SPEC = _case(spec_model(), (SPEC_X,), y, err, similar(y))
end

# Handwritten baseline: the SPEC constraints (2 ties on mean, 2 on sigma, 1 line
# ratio) hardcoded into a plain function. This is the natural performance
# ceiling, and the ratio against it is the claim the package rests on.
function hand_spec_chi2(p, x, y, err)
    slope, intercept, ha_amp, ha_mean, ha_sigma, nii_amp = p
    nii_b_mean = ha_mean + (L_NII_B - L_HA)
    nii_r_mean = ha_mean + (L_NII_R - L_HA)
    nii_r_amp = 3.06 * nii_amp
    acc = zero(eltype(p))
    @inbounds for i in eachindex(y)
        xi = x[i]
        mu = slope * xi + intercept +
            ha_amp * exp(-((xi - ha_mean) / ha_sigma)^2 / 2) +
            nii_amp * exp(-((xi - nii_b_mean) / ha_sigma)^2 / 2) +
            nii_r_amp * exp(-((xi - nii_r_mean) / ha_sigma)^2 / 2)
        acc += abs2((mu - y[i]) / err[i])
    end
    return acc
end

# ---------------------------------------------------------------------------
# WIDE — 13 leaves, most of them tied to one master line. Stresses the tree
# depth of the generated `withparams` and of the fused broadcast in `render`.
# ---------------------------------------------------------------------------

const L_HD = 4101.73
const L_HG = 4340.47
const L_HEII = 4685.68
const L_HEI = 5875.62
const L_HB = 4861.33
const L_OIII_B = 4958.91
const L_OIII_R = 5006.84
const L_SII_B = 6716.44
const L_SII_R = 6730.82
const L_REF = 5500.0

function wide_model(; norm = 2.5, index = -1.2, ha_amp = 9.0, sigma = 4.5)
    cm = @model begin
        cont = PowerLaw1D(norm = norm, x_ref = L_REF, index = index)
        hd = Gaussian1D(amplitude = 0.256 * ha_amp / 2.86, mean = L_HD, sigma = sigma)
        hg = Gaussian1D(amplitude = 0.466 * ha_amp / 2.86, mean = L_HG, sigma = sigma)
        heii = Gaussian1D(amplitude = 1.1, mean = L_HEII, sigma = sigma)
        hb = Gaussian1D(amplitude = ha_amp / 2.86, mean = L_HB, sigma = sigma)
        oiii_b = Gaussian1D(amplitude = 1.8, mean = L_OIII_B, sigma = sigma)
        oiii_r = Gaussian1D(amplitude = 2.98 * 1.8, mean = L_OIII_R, sigma = sigma)
        hei = Gaussian1D(amplitude = 0.7, mean = L_HEI, sigma = sigma)
        ha = Gaussian1D(amplitude = ha_amp, mean = L_HA, sigma = sigma)
        nii_b = Gaussian1D(amplitude = 0.9, mean = L_NII_B, sigma = sigma)
        nii_r = Gaussian1D(amplitude = 3.06 * 0.9, mean = L_NII_R, sigma = sigma)
        sii_b = Gaussian1D(amplitude = 1.0, mean = L_SII_B, sigma = sigma)
        sii_r = Gaussian1D(amplitude = 0.8, mean = L_SII_R, sigma = sigma)
        cont + hd + hg + heii + hb + oiii_b + oiii_r + hei + ha + nii_b + nii_r + sii_b + sii_r
    end
    @constrain cm begin
        cont.norm in (0.0, 12.0)
        cont.x_ref
        cont.index in (-3.0, 0.0)

        hd.amplitude -> 0.256 * ha.amplitude / 2.86   # Balmer decrement
        hd.mean
        hd.sigma -> ha.sigma
        hg.amplitude -> 0.466 * ha.amplitude / 2.86
        hg.mean
        hg.sigma -> ha.sigma
        hb.amplitude -> ha.amplitude / 2.86
        hb.mean
        hb.sigma -> ha.sigma

        heii.amplitude in (0.0, 10.0)
        heii.mean
        heii.sigma -> ha.sigma
        hei.amplitude in (0.0, 10.0)
        hei.mean
        hei.sigma -> ha.sigma

        oiii_b.amplitude in (0.0, 20.0)
        oiii_b.mean
        oiii_b.sigma -> ha.sigma
        oiii_r.amplitude -> 2.98 * oiii_b.amplitude
        oiii_r.mean
        oiii_r.sigma -> ha.sigma

        ha.amplitude in (0.0, 40.0)
        ha.mean in (L_HA - 5.0, L_HA + 5.0)
        ha.sigma in (1.0, 12.0)

        nii_b.amplitude in (0.0, 15.0)
        nii_b.mean
        nii_b.sigma -> ha.sigma
        nii_r.amplitude -> 3.06 * nii_b.amplitude
        nii_r.mean
        nii_r.sigma -> ha.sigma

        sii_b.amplitude in (0.0, 15.0)
        sii_b.mean
        sii_b.sigma -> ha.sigma
        sii_r.amplitude in (0.0, 15.0)
        sii_r.mean
        sii_r.sigma -> ha.sigma
    end
    return cm
end

const WIDE_X = collect(range(4000.0, 7000.0; length = 2000))

let truth = wide_model(norm = 3.1, index = -1.6, ha_amp = 10.5, sigma = 4.3)
    y_true = render(truth, WIDE_X)
    err = 0.055 .+ 0.018 .* sqrt.(clamp.(y_true, 0.0, Inf))
    y = y_true .+ err .* randn(RNG, length(WIDE_X))
    global const WIDE = _case(wide_model(), (WIDE_X,), y, err, similar(y))
end

# ---------------------------------------------------------------------------
# IMG — two blended galaxies, each a Gaussian bulge on a Sersic disk, with the
# bulge center and position angle tied to its disk. 2D, 100x100.
# ---------------------------------------------------------------------------

function img_model(;
        a1 = 15.0, x1 = -3.5, y1 = 0.5, s1 = 1.5, r1 = 3.5, n1 = 1.5, q1 = 0.5, t1 = 0.3,
        a2 = 60.0, x2 = 4.5, y2 = 0.0, s2 = 1.0, r2 = 2.5, n2 = 2.0, q2 = 0.9, t2 = -0.2,
        ad1 = 35.0, ad2 = 8.0
    )
    cm = @model begin
        bulge1 = Gaussian2D(amplitude = a1, x0 = x1, y0 = y1, sigma = s1, q = 0.9, theta = t1)
        disk1 = Sersic2D(amplitude = ad1, x0 = x1, y0 = y1, r_eff = r1, n = n1, q = q1, theta = t1)
        bulge2 = Gaussian2D(amplitude = a2, x0 = x2, y0 = y2, sigma = s2, q = 0.9, theta = t2)
        disk2 = Sersic2D(amplitude = ad2, x0 = x2, y0 = y2, r_eff = r2, n = n2, q = q2, theta = t2)
        bulge1 + disk1 + bulge2 + disk2
    end
    @constrain cm begin
        bulge1.amplitude in (0.0, 500.0)
        bulge1.x0 -> disk1.x0                 # bulge shares its disk's center
        bulge1.y0 -> disk1.y0
        bulge1.sigma in (0.05, 10.0)
        bulge1.q in (0.05, 1.0)
        bulge1.theta -> disk1.theta
        disk1.amplitude in (0.0, 500.0)
        disk1.x0 in (-8.0, 8.0)
        disk1.y0 in (-8.0, 8.0)
        disk1.r_eff in (0.1, 10.0)
        disk1.n in (0.5, 6.0)
        disk1.q in (0.05, 1.0)
        disk1.theta in (-1.6, 1.6)
        bulge2.amplitude in (0.0, 500.0)
        bulge2.x0 -> disk2.x0
        bulge2.y0 -> disk2.y0
        bulge2.sigma in (0.05, 10.0)
        bulge2.q in (0.05, 1.0)
        bulge2.theta -> disk2.theta
        disk2.amplitude in (0.0, 500.0)
        disk2.x0 in (-8.0, 8.0)
        disk2.y0 in (-8.0, 8.0)
        disk2.r_eff in (0.1, 10.0)
        disk2.n in (0.5, 6.0)
        disk2.q in (0.05, 1.0)
        disk2.theta in (-1.6, 1.6)
    end
    return cm
end

const NPIX = 100
const IMG_COORD = range(-8.0, 8.0; length = NPIX)
# Two coordinate forms, both legal and NOT equally cheap (ADR-0006):
#   materialized — a full X and Y matrix per pixel, as examples/main writes it
#   grid form    — a column against a row, broadcast on the fly, no coord memory
const IMG_X = [x for x in IMG_COORD, _ in IMG_COORD]
const IMG_Y = [y for _ in IMG_COORD, y in IMG_COORD]
const IMG_XG = collect(IMG_COORD)
const IMG_YG = reshape(collect(IMG_COORD), 1, :)

let truth = img_model(
        a1 = 25.0, x1 = -2.0, y1 = -1.0, s1 = 0.8, r1 = 2.5, n1 = 1.0, q1 = 0.38, t1 = 0.8,
        a2 = 90.0, x2 = 3.0, y2 = 1.5, s2 = 1.4, r2 = 1.6, n2 = 3.5, q2 = 0.85, t2 = -0.5,
        ad1 = 50.0, ad2 = 15.0
    )
    img_true = render(truth, IMG_X, IMG_Y)
    err = fill(0.4, size(IMG_X))
    img = img_true .+ err .* randn(RNG, size(IMG_X))
    global const IMG = _case(img_model(), (IMG_X, IMG_Y), img, err, similar(img))
end

# Same scene rendered through the grid form — the shape a kernel would need.
const IMG_GRID_F = ObjectiveFunction(img_model(), (IMG_XG, IMG_YG), IMG.y, IMG.err)

# Handwritten baseline for the 2D render: the IMG scene with its ties resolved,
# written as one fused broadcast. Same 20 free parameters, same order.
function hand_img_render!(out, p, xs, ys)
    b1a, b1s, b1q, d1a, d1x, d1y, d1r, d1n, d1q, d1t,
        b2a, b2s, b2q, d2a, d2x, d2y, d2r, d2n, d2q, d2t = p
    c1, s1 = cos(d1t), sin(d1t)
    c2, s2 = cos(d2t), sin(d2t)
    bn1 = 2 * d1n - 1 / 3 + 4 / (405 * d1n)
    bn2 = 2 * d2n - 1 / 3 + 4 / (405 * d2n)
    @inbounds for j in axes(out, 2), i in axes(out, 1)
        x, y = xs[i], ys[j]
        dx1, dy1 = x - d1x, y - d1y
        xr1 = c1 * dx1 + s1 * dy1
        yr1 = -s1 * dx1 + c1 * dy1
        g1 = b1a * exp(-0.5 * (xr1^2 + (yr1 / b1q)^2) / b1s^2)
        r1 = sqrt(xr1^2 + (yr1 / d1q)^2)
        e1 = d1a * exp(-bn1 * ((r1 / d1r)^(1 / d1n) - 1))
        dx2, dy2 = x - d2x, y - d2y
        xr2 = c2 * dx2 + s2 * dy2
        yr2 = -s2 * dx2 + c2 * dy2
        g2 = b2a * exp(-0.5 * (xr2^2 + (yr2 / b2q)^2) / b2s^2)
        r2 = sqrt(xr2^2 + (yr2 / d2q)^2)
        e2 = d2a * exp(-bn2 * ((r2 / d2r)^(1 / d2n) - 1))
        out[i, j] = g1 + e1 + g2 + e2
    end
    return out
end

# ---------------------------------------------------------------------------
# POST — SPEC with a prior on every free parameter, for the Bayesian path.
# logposterior throws unless every free parameter has one.
# ---------------------------------------------------------------------------

function spec_prior_model()
    cm = spec_model()
    @prior cm.cont.slope ~ Normal(0.0, 0.05)
    @prior cm.cont.intercept ~ Normal(8.0, 2.0)
    @prior cm.ha.amplitude ~ LogNormal(2.0, 1.0)
    @prior cm.ha.mean ~ Normal(L_HA, 2.0)
    @prior cm.ha.sigma ~ LogNormal(1.0, 0.5)
    @prior cm.nii_b.amplitude ~ LogNormal(0.0, 1.0)
    return cm
end

const POST = let cm = spec_prior_model()
    f = ObjectiveFunction(cm, SPEC_X, SPEC.y, SPEC.err; statistic = logposterior)
    p = params(cm)
    (; cm, f, p, cfg = fullchunk(f, p), g = zeros(length(p)))
end

const CASES = (; SPEC, WIDE, IMG)
