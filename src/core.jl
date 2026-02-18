"""
    BranchIncidenceMatrix

Efficient data structure for representing the branch indicence matrix of a power grid.

A[k, i] = +1 if branch k starts at bus i
        = -1 if branch k ends at bus j
        =  0 otherwise
"""
struct BranchIncidenceMatrix
    N::Int
    E::Int

    bus_fr::Vector{Int}
    bus_to::Vector{Int}
end

import Base.size
Base.size(A::BranchIncidenceMatrix) = (A.E, A.N)

import LinearAlgebra.mul!

"""
    mul!(y, A::BranchIncidenceMatrix, x)

Efficient implementation of Matrix-vector product with `A`.
This function assumes that 
"""
function LinearAlgebra.mul!(y::AbstractVector, A::BranchIncidenceMatrix, x::AbstractVector)
    N, E = A.N, A.E
    N == size(x, 1) || throw(DimensionMismatch("A has size $(size(A)), but x has size $(size(x))"))
    E == size(y, 1) || throw(DimensionMismatch("A has size $(size(A)), but y has size $(size(y))"))
    
    @inbounds @simd for e in 1:E
        i = A.bus_fr[e]
        j = A.bus_to[e]
        y[e] = x[i] - x[j]
    end
    return y
end

function LinearAlgebra.mul!(y::AbstractMatrix, A::BranchIncidenceMatrix, x::AbstractMatrix)
    N, E = A.N, A.E
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