# Base has no indexable lazy product: `Iterators.product` is not an AbstractArray and
# broadcasting over it collects the tuples first (16 bytes per pixel; a Coords is the
# axes only).
"""
    Coords(x, y, zs...)

The lazy Cartesian product of two or more axis vectors: an `AbstractArray` whose
element `Coords(x, y)[i, j]` is the point `(x[i], y[j])`. Use it for physical axes
(`Coords(ra, dec)`, `Coords(x, y, λ)`); for the pixels of an image use
`CartesianIndices(img)`. A 1D model takes its coordinate vector directly.

Each axis keeps its own element type, and the axes are not copied: range axes stay
ranges (about 20% slower to index than `collect`ed vectors).
"""
struct Coords{T, N, A <: NTuple{N, AbstractVector}} <: AbstractArray{T, N}
    ax::A
    # Two or more axes: a 1D input is the vector itself.
    Coords(ax::A) where {N, A <: Tuple{AbstractVector, AbstractVector, Vararg{AbstractVector, N}}} =
        new{Tuple{map(eltype, ax)...}, N + 2, A}(ax)
end
Coords(x::AbstractVector, y::AbstractVector, zs::AbstractVector...) = Coords((x, y, zs...))
Coords(img::AbstractArray) = throw(
    ArgumentError(
        "Coords takes two or more axis vectors, got one $(summary(img)): " *
            "for the pixels of an image use CartesianIndices(img); a 1D model takes the vector itself"
    )
)

Base.size(g::Coords) = map(length, g.ax)
Base.axes(g::Coords) = map(a -> axes(a, 1), g.ax)
Base.@propagate_inbounds Base.getindex(g::Coords{T, N}, I::Vararg{Int, N}) where {T, N} =
    map(getindex, g.ax, I)
