module FastPowerFlow

using BenchmarkTools
using DataFrames
using LinearAlgebra
using SparseArrays
using Statistics
using SuiteSparse

# PowerModels
using PowerModels
const PM = PowerModels
using PGLib

# CUDA stuff
using CUDA
using CUDSS
using KLU

function __init__()
    PM.silence()
end

export FullPTDF, LazyPTDF
export FullLODF, LazyLODF
export compute_flow!

include("graphs.jl")
include("ptdf.jl")
include("lodf.jl")

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

function BranchIncidenceMatrix(data::Dict)
    N::Int = length(data["bus"])
    E::Int = length(data["branch"])

    bus_fr = Vector{Int}(undef, E)
    bus_to = Vector{Int}(undef, E)
    for e in 1:E
        bus_fr[e] = data["branch"]["$e"]["f_bus"]
        bus_to[e] = data["branch"]["$e"]["t_bus"]
    end
    
    # data checks
    imin, imax = extrema(bus_fr)
    1 <= imin <= imax <= N || throw(ArgumentError("bus_fr out of bounds"))
    jmin, jmax = extrema(bus_to)
    1 <= jmin <= jmax <= N || throw(ArgumentError("bus_to out of bounds"))

    return BranchIncidenceMatrix(N, E, bus_fr, bus_to)
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

# GPU code
struct BranchIncidenceMatrixGPU
    N::Int
    E::Int

    bus_fr::CUDA.CuVector{Int32}
    bus_to::CUDA.CuVector{Int32}
end

BranchIncidenceMatrixGPU(A::BranchIncidenceMatrix) = BranchIncidenceMatrixGPU(A.N, A.E, CuVector{Int32}(A.bus_fr), CuVector{Int32}(A.bus_to))
Base.size(A::BranchIncidenceMatrixGPU) = (A.E, A.N)

function _mul_kernel!(y, bus_fr, bus_to, x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    @inbounds for e in index:stride:length(y)
        i = bus_fr[e]
        j = bus_to[e]
        y[e] = x[i] - x[j]
    end

    return nothing
end

function LinearAlgebra.mul!(y::CuVector, A::BranchIncidenceMatrixGPU, x::CuVector)
    kernel = @cuda launch=false _mul_kernel!(y, A.bus_fr, A.bus_to, x)
    config = launch_configuration(kernel.fun)
    threads = min(length(y), config.threads)
    blocks = cld(length(y), threads)

    kernel(y, A.bus_fr, A.bus_to, x; threads, blocks)

    return y
end

end # module FastPowerFlow
