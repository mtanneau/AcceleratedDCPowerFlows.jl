module CUDAExt

import AcceleratedDCPowerFlows as APF

using LinearAlgebra

import KernelAbstractions as KA
using CUDA
using CUDSS

function APF.branch_incidence_matrix(::CUDA.CUDABackend, network::APF.Network)
    N = APF.num_buses(network)
    E = APF.num_branches(network)

    bus_fr = CuVector{Int32}([br.bus_fr for br in network.branches])
    bus_to = CuVector{Int32}([br.bus_to for br in network.branches])

    return APF.BranchIncidenceMatrix(N, E, bus_fr, bus_to)
end

function _mul_kernel!(y, bus_fr, bus_to, x)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if e <= size(y, 1) && k <= size(y, 2)
        @inbounds begin
            i = bus_fr[e]
            j = bus_to[e]
            y[e, k] = x[i, k] - x[j, k]
        end
    end

    return nothing
end

function LinearAlgebra.mul!(y::CuVecOrMat, A::APF.BranchIncidenceMatrix{CuVector{<:Integer}}, x::CuVecOrMat)
    E, N = A.E, A.N
    K = size(y, 2)
    size(y, 1) == E || throw(DimensionMismatch("A has size $(size(A)) but y has size $(size(y))"))
    size(x, 1) == N || throw(DimensionMismatch("A has size $(size(A)) but x has size $(size(y))"))
    size(x, 2) == K || throw(DimensionMismatch("x and y must have same number of columns"))

    kernel = @cuda launch=false _mul_kernel!(y, A.bus_fr, A.bus_to, x)
    # Set gridDim and block size
    threads = (16, min(16, max(1, nextpow(2, K))))
    blocks = (cld(E, threads[1]), cld(K, threads[2]))

    kernel(y, A.bus_fr, A.bus_to, x; threads, blocks)

    return y
end

end  # module