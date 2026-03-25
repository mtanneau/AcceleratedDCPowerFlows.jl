const DEFAULT_OPERATOR_CACHE_COLS = 1

import Base: resize!, view

"""
	OperatorCache

Backend-agnostic workspace cache used by PTDF/LODF operators.
"""
mutable struct OperatorCache{A}
    workspace::A
end

"""
	OperatorCache(backend, T, nrows; ncols=DEFAULT_OPERATOR_CACHE_COLS)

Create a cache backed by an array allocated on `backend`.
"""
function OperatorCache(
    backend::KA.Backend,
    ::Type{T},
    nrows::Integer;
    ncols::Integer=DEFAULT_OPERATOR_CACHE_COLS,
) where {T}
    nrows > 0 || throw(ArgumentError("cache row count must be strictly positive"))
    ncols > 0 || throw(ArgumentError("cache column count must be strictly positive"))
    workspace = KA.allocate(backend, T, (Int(nrows), Int(ncols)))
    return OperatorCache(workspace)
end

cache_size(cache::OperatorCache) = size(cache.workspace, 2)

function resize!(cache::OperatorCache, ncols::Integer; force=false)
    ncols >= 0 || throw(ArgumentError("cache column count must be non-negative"))
    ncols = Int(ncols)

    ncols_now = size(cache.workspace, 2)
    if (ncols_now == ncols) || (!force && (ncols_now > ncols))
        return nothing
    end

    backend = KA.get_backend(cache.workspace)
    nrows = size(cache.workspace, 1)
    cache.workspace = KA.allocate(backend, eltype(cache.workspace), (nrows, ncols))
    return nothing
end

function Base.view(cache::OperatorCache, inds...)
    return Base.view(cache.workspace, inds...)
end
