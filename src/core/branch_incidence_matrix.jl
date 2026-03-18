import Base.size
import LinearAlgebra.mul!
import SparseArrays: sparse

"""
    BranchIncidenceMatrix{V}

Efficient data structure for representing the branch indicence matrix of a power grid.

A[k, i] = +1 if branch k starts at bus i
        = -1 if branch k ends at bus j
        =  0 otherwise
"""
struct BranchIncidenceMatrix{V}
    N::Int
    E::Int

    # Two Vector{Int} of length `E`
    # listing each branch's "from" and "to" bus
    bus_fr::V
    bus_to::V
end

Base.size(A::BranchIncidenceMatrix) = (A.E, A.N)
function Base.size(A::BranchIncidenceMatrix, d::Integer)
    s = size(A)
    if d == 1 || d == 2
        return s[d]
    elseif d > 2
        return 1
    else
        error("arraysize: dimension $d out of range")
    end
end

KA.get_backend(A::BranchIncidenceMatrix) = KA.get_backend(A.bus_fr)

"""
    branch_incidence_matrix([backend], network::Network)

Build branch incidence matrix on the specified backend.
Defaults to cpu if no backend is provided.
"""
function branch_incidence_matrix(backend::KA.Backend, network::Network)
    N = num_buses(network)
    E = num_branches(network)

    # Build on host (CPU)...
    bus_fr_host = [br.bus_fr for br in network.branches]
    bus_to_host = [br.bus_to for br in network.branches]

    # ... then transfer to device (typically GPU)
    bus_fr_dev = KA.allocate(backend, eltype(bus_fr_host), (E,))
    bus_to_dev = KA.allocate(backend, eltype(bus_fr_host), (E,))
    copyto!(bus_fr_dev, bus_fr_host)
    copyto!(bus_to_dev, bus_to_host)

    return BranchIncidenceMatrix(N, E, bus_fr_dev, bus_to_dev)
end

function branch_incidence_matrix(network::Network)
    return branch_incidence_matrix(default_backend(), network)
end

SparseArrays.sparse(A::BranchIncidenceMatrix) = _sparse(KA.get_backend(A), A)

function _sparse(backend::KA.Backend, ::BranchIncidenceMatrix)
    return error(
        "Sparse conversion of branch incidence matrix is not supported on backend $(backend)",
    )
end

function _sparse(::KA.CPU, A::BranchIncidenceMatrix)
    Ti = eltype(A.bus_fr)
    E, N = size(A)
    Is = zeros(Ti, 2*E)
    Js = zeros(Ti, 2*E)
    Vs = zeros(Float64, 2*E)

    for e in 1:E
        i = A.bus_fr[e]
        j = A.bus_to[e]
        Is[2*e-1] = e
        Js[2*e-1] = i
        Vs[2*e-1] = +1.0
        Is[2*e+0] = e
        Js[2*e+0] = j
        Vs[2*e+0] = -1.0
    end
    return SparseArrays.sparse(Is, Js, Vs, E, N)
end

# The main entry point for matrix multiplication is `LinearAlgebra.mul!`
# We provide a generic kernel-based implementation which should work on any backend,
#   but may achieve sub-optimal performance in some cases.
# Specific implementations should extend the `_unsafe_mul!(backend, x, A, y)` function
#   (see example below for a CPU-specific implementation)
function LinearAlgebra.mul!(y::AbstractVecOrMat, A::BranchIncidenceMatrix, x::AbstractVecOrMat)
    E, N = size(A)
    K = size(y, 2)
    N == size(x, 1) || throw(DimensionMismatch("A has size $(size(A)), but x has size $(size(x))"))
    E == size(y, 1) || throw(DimensionMismatch("A has size $(size(A)), but y has size $(size(y))"))
    K == size(x, 2) || throw(DimensionMismatch("x has size $(size(x)), but y has size $(size(y))"))

    backend = KA.get_backend(A)
    if !(backend == KA.get_backend(x) == KA.get_backend(y))
        error("A, x, and y have different backends.")
    end

    _unsafe_mul!(backend, y, A, x)

    return y
end

# Fallback implementation using KA
function _unsafe_mul!(
    backend::KA.Backend,
    y::AbstractVecOrMat,
    A::BranchIncidenceMatrix,
    x::AbstractVecOrMat,
)
    backend === KA.get_backend(A) || error("backend ≠ KA.get_backend(A)")
    E = size(y, 1)
    K = size(y, 2)

    @kernel function mul_kernel!(y, @Const(bus_fr), @Const(bus_to), @Const(x))
        e, k = @index(Global, NTuple)
        @inbounds i = bus_fr[e]
        @inbounds j = bus_to[e]
        @inbounds y[e, k] = x[i, k] - x[j, k]
    end

    kernel! = mul_kernel!(backend)
    # `ndrange` will be (E, 1) if y is a Vector
    #                or (E, K) if y is a Matrix
    # This ensures that indexing is correct within the kernel 
    kernel!(y, A.bus_fr, A.bus_to, x; ndrange=(E, K))
    synchronize(backend)

    return y
end

# Specialized implementation on CPU, single threaded.
# This is ~2x faster than KA for single-threaded code
# The performance gap becomes smaller as `size(y, 2)` and the number of cores increases
function _unsafe_mul!(::KA.CPU, y::AbstractVecOrMat, A::BranchIncidenceMatrix, x::AbstractVecOrMat)
    E = size(y, 1)
    K = size(y, 2)

    @inbounds @simd for k in 1:K
        for e in 1:E
            i = A.bus_fr[e]
            j = A.bus_to[e]
            y[e, k] = x[i, k] - x[j, k]
        end
    end
    return y
end
