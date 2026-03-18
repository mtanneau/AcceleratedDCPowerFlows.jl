"""
    _sparse(::CUDA.CUDABackend, A::APF.BranchSusceptanceMatrix)

Returns a sparse representation of `A` on a CUDA GPU.

This function returns a `CUDA.CUSPARSE.CuSparseMatrixCSR`, since
    most CUSPARSE operations work with CSR formatted matrices.
"""
function APF._sparse(::CUDA.CUDABackend, A::APF.BranchSusceptanceMatrix)
    if !isa(KA.get_backend(A), CUDA.CUDABackend)
        error("Trying to build CuSparseMatrixCSR but A is not on a CUDA device")
    end

    # For simplicity, we move to CPU, build a CSC on CPU, then back to GPU
    A_host =
        APF.BranchSusceptanceMatrix(A.N, A.E, collect(A.bus_fr), collect(A.bus_to), collect(A.br_b))
    A_host_csc = sparse(A_host)

    return CUDA.CUSPARSE.CuSparseMatrixCSR(A_host_csc)
end

function _mul_branchSusceptanceMatrix_kernel!(y, bus_fr, bus_to, br_b, x)
    e = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if e <= size(y, 1) && k <= size(y, 2)
        @inbounds begin
            i = bus_fr[e]
            j = bus_to[e]
            y[e, k] = br_b[e] * (x[i, k] - x[j, k])
        end
    end

    return nothing
end

function APF._unsafe_mul!(
    ::CUDA.CUDABackend,
    y::CUDA.CuVecOrMat,
    A::APF.BranchSusceptanceMatrix,
    x::CUDA.CuVecOrMat,
)
    E, N = size(A)
    K = size(y, 2)

    kernel =
        @cuda launch=false _mul_branchSusceptanceMatrix_kernel!(y, A.bus_fr, A.bus_to, A.br_b, x)
    # Set gridDim and block size
    threads = (16, min(16, max(1, nextpow(2, K))))
    blocks = (cld(E, threads[1]), cld(K, threads[2]))

    kernel(y, A.bus_fr, A.bus_to, A.br_b, x; threads, blocks)

    return y
end
