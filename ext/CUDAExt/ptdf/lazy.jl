function APF.lazy_ptdf(backend::CUDA.CUDABackend, network::APF.Network; linear_solver=:auto)
    N = APF.num_buses(network)
    E = APF.num_branches(network)
    islack = network.slack_bus_index

    # Build incidence matrix on GPU
    A = APF.branch_incidence_matrix(backend, network)

    # Build branch susceptances on GPU (negated)
    b = CuVector([-br.b for br in network.branches])

    # Build modified nodal susceptance matrix on CPU, then move to GPU
    #   because it's easier to zero-out a row/column and reset a diagonal coeff there
    Y_cpu = -sparse(APF.nodal_susceptance_matrix(KA.CPU(), network))  # Negated!
    Y_cpu[islack, :] .= 0.0
    Y_cpu[:, islack] .= 0.0
    Y_cpu[islack, islack] = 1.0
    Y_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR(Y_cpu)

    if (linear_solver == :auto) || (linear_solver == :CUDSS)
        # All good, we'll use CUDSS
    else
        error("""Unsupported CUDA linear solver for lazy PTDF: $(linear_solver).
        Supported options are: `:CUDSS` and `:auto`""")
    end

    # Form factorization using CUDSS
    F = LinearAlgebra.ldlt(Y_gpu)

    return APF.LazyPTDF(N, E, islack, A, b, F)
end
