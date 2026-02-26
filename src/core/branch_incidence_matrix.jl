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
function branch_incidence_matrix(::KA.CPU, network::Network)
    N = num_buses(network)
    E = num_branches(network)

    bus_fr = [br.bus_fr for br in network.branches]
    bus_to = [br.bus_to for br in network.branches]

    return BranchIncidenceMatrix(N, E, bus_fr, bus_to)
end

branch_incidence_matrix(network::Network) = branch_incidence_matrix(DefaultBackend(), network)

function SparseArrays.sparse(A::BranchIncidenceMatrix)
    # Sanity check: make sure we are on CPU
    backend = KA.get_backend(A)
    if !isa(backend, KA.CPU)
        error("Unsupported backend for building a sparse branch susceptance matrix: $(typeof(backend))")
    end

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

function LinearAlgebra.mul!(y::AbstractVecOrMat, A::BranchIncidenceMatrix, x::AbstractVecOrMat)
    E, N = size(A)
    K = size(y, 2)

    N == size(x, 1) || throw(DimensionMismatch("A has size $(size(A)), but x has size $(size(x))"))
    E == size(y, 1) || throw(DimensionMismatch("A has size $(size(A)), but y has size $(size(y))"))
    K == size(x, 2) || throw(DimensionMismatch("x has size $(size(x)), but y has size $(size(y))"))

    @inbounds @simd for k in 1:K
        for e in 1:E
            i = A.bus_fr[e]
            j = A.bus_to[e]
            y[e,k] = x[i,k] - x[j,k]
        end
    end
    return y
end