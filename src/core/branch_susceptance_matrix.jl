#=
    Mathematical notes:
    The branch susceptance matrix is M = B × Ab, where
        B = Diag(b) and Ab is the branch incidence matrix.
    
    Therefore, M and Ab have the same sparsity structure (2E non-zeros coeffs)
        and differ only in the absolute value of the numerical coeffs
        (+1 and -1 for Ab, +bₑ and -bₑ for M).

    As a consequence, much of the code is repeated boilerplate,
        and could (conceptually) be merged by representing Ab as a
        branch susceptance matrix where all branches have unit susceptance.
        This is left for future work 
=#

"""
    BranchSusceptanceMatrix{Vi,Vf}

Efficient data structure for representing the branch susceptance matrix of a power grid.

A[k, i] = +bₖ if branch k starts at bus i
        = -bₖ if branch k ends at bus j
        =  0 otherwise
"""
struct BranchSusceptanceMatrix{Vi,Vf}
    N::Int
    E::Int

    # Vi is a <:AbstractVector{<:Integer}
    # Vf is a <:AbstractVector{<:Real}
    bus_fr::Vi
    bus_to::Vi
    br_b::Vf
end

Base.size(A::BranchSusceptanceMatrix) = (A.E, A.N)
function Base.size(A::BranchSusceptanceMatrix, d::Integer)
    s = size(A)
    if d == 1 || d == 2
        return s[d]
    elseif d > 2
        return 1
    else
        error("arraysize: dimension $d out of range")
    end
end

KA.get_backend(A::BranchSusceptanceMatrix) = KA.get_backend(A.bus_fr)

"""
    branch_susceptance_matrix([backend], network::Network)

Build branch susceptance matrix on the specified backend.
Defaults to cpu if no backend is provided.
"""
function branch_susceptance_matrix(backend::KA.Backend, network::Network)
    N = num_buses(network)
    E = num_branches(network)

    # Build on host (CPU)
    bus_fr_host = [br.bus_fr for br in network.branches]
    bus_to_host = [br.bus_to for br in network.branches]
    br_b_host = [br.b for br in network.branches]

    # Transfer to device (typically GPU)
    bus_fr_dev = KA.allocate(backend, eltype(bus_fr_host), (E,))
    bus_to_dev = KA.allocate(backend, eltype(bus_to_host), (E,))
    br_b_dev = KA.allocate(backend, eltype(br_b_host), (E,))
    copyto!(bus_fr_dev, bus_fr_host)
    copyto!(bus_to_dev, bus_to_host)
    copyto!(br_b_dev, br_b_host)

    return BranchSusceptanceMatrix(N, E, bus_fr_dev, bus_to_dev, br_b_dev)
end

branch_susceptance_matrix(network::Network) = branch_susceptance_matrix(default_backend(), network)

function SparseArrays.sparse(A::BranchSusceptanceMatrix)
    # Sanity check: make sure we are on CPU
    backend = KA.get_backend(A)
    if !isa(backend, KA.CPU)
        error("Unsupported backend for building a sparse branch susceptance matrix: $(typeof(backend))")
    end

    E, N = size(A)
    Is = zeros(Int, 2*E)
    Js = zeros(Int, 2*E)
    Vs = zeros(Float64, 2*E)

    for e in 1:E
        i = A.bus_fr[e]
        j = A.bus_to[e]
        Is[2*e-1] = e
        Js[2*e-1] = i
        Vs[2*e-1] = +A.br_b[e]
        Is[2*e+0] = e
        Js[2*e+0] = j
        Vs[2*e+0] = -A.br_b[e]
    end
    return SparseArrays.sparse(Is, Js, Vs, E, N)
end

function LinearAlgebra.mul!(y::AbstractVecOrMat, A::BranchSusceptanceMatrix, x::AbstractVecOrMat)
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
function _unsafe_mul!(backend::KA.Backend, y::AbstractVecOrMat, A::BranchSusceptanceMatrix, x::AbstractVecOrMat)
    backend === KA.get_backend(A) || error("backend ≠ KA.get_backend(A)")
    E = size(y, 1)
    K = size(y, 2)

    @kernel function mul_kernel!(y, @Const(bus_fr), @Const(bus_to), @Const(br_b), @Const(x))
        e, k = @index(Global, NTuple)
        @inbounds i = bus_fr[e]
        @inbounds j = bus_to[e]
        @inbounds y[e, k] = br_b[e] * (x[i, k] - x[j, k])
    end

    kernel! = mul_kernel!(backend)
    # `ndrange` will be (E, 1) if y is a Vector
    #                or (E, K) if y is a Matrix
    # This ensures that indexing is correct within the kernel 
    kernel!(y, A.bus_fr, A.bus_to, A.br_b, x, ndrange=(E, K))
    synchronize(backend)

    return y
end

# Specialized implementation on CPU, single threaded.
function _unsafe_mul!(::KA.CPU, y::AbstractVecOrMat, A::BranchSusceptanceMatrix, x::AbstractVecOrMat)
    E = size(y, 1)
    K = size(y, 2)

    @inbounds @simd for k in 1:K
        for e in 1:E
            i = A.bus_fr[e]
            j = A.bus_to[e]
            y[e,k] = A.br_b[e] * (x[i,k] - x[j,k])
        end
    end
    return y
end
