const _WP_SIZES = [1, 2, 4, 8, 16, 32, 64]

for n in _WP_SIZES
    assignments = [:($(Symbol(:g, i)) = Gaussian1D(1.0, $(3.0*(i-1)), 1.0)) for i in 1:n]
    sum_expr    = foldl((a, b) -> :($a + $b), [Symbol(:g, i) for i in 1:n])
    block       = Expr(:block, assignments..., sum_expr)
    @eval const $(Symbol(:CM_WP_, n)) = @model $block
end

SUITE["withparams"] = BenchmarkGroup()

for n in _WP_SIZES
    cm = eval(Symbol(:CM_WP_, n))
    p  = AstroFit.params(cm)
    SUITE["withparams"]["$(n)G"] = @benchmarkable Base.donotdelete(withparams($cm, $p))
end
