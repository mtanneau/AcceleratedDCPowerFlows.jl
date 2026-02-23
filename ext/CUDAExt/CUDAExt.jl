module CUDAExt

using AcceleratedDCPowerFlows
using AcceleratedDCPowerFlows: Network, BranchIncidenceMatrix

using LinearAlgebra

using CUDA
using CUDSS

# GPU code
struct BranchIncidenceMatrixGPU
    N::Int
    E::Int

    bus_fr::CUDA.CuVector{Int32}
    bus_to::CUDA.CuVector{Int32}
end

BranchIncidenceMatrixGPU(A::BranchIncidenceMatrix) = BranchIncidenceMatrixGPU(A.N, A.E, CuVector{Int32}(A.bus_fr), CuVector{Int32}(A.bus_to))
BranchIncidenceMatrixGPU(network::Network) = BranchIncidenceMatrixGPU(BranchIncidenceMatrix(network))
Base.size(A::BranchIncidenceMatrixGPU) = (A.E, A.N)

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

function LinearAlgebra.mul!(y::CuVecOrMat, A::BranchIncidenceMatrixGPU, x::CuVecOrMat)
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