"""
    _sparse(::CUDA.CUDABackend, A::APF.NodalSusceptanceMatrix)

Returns a sparse representation of `A` on a CUDA GPU.

This function returns a `CUDA.CUSPARSE.CuSparseMatrixCSR`, since
    most CUSPARSE operations work with CSR formatted matrices.
"""
function APF._sparse(::CUDA.CUDABackend, A::APF.NodalSusceptanceMatrix)
    if !isa(KA.get_backend(A), CUDA.CUDABackend)
        error("Trying to build CuSparseMatrixCSR but A is not on a CUDA device")
    end

    # For simplicity, we move to CPU, build a CSC on CPU, then back to GPU
    # This could be improved (with no GPU -> CPU -> GPU movements) by either
    #   * building a CuSparseMatrixCOO then converting to CSR,
    #   * building the CSR matrix data directly
    A_host =
        APF.NodalSusceptanceMatrix(A.N, A.E, collect(A.bus_fr), collect(A.bus_to), collect(A.br_b))
    A_host_csc = sparse(A_host)

    return CUDA.CUSPARSE.CuSparseMatrixCSR(A_host_csc)
end

# TODO: better kernels
