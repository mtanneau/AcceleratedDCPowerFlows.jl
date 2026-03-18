function APF.full_ptdf(backend::CUDA.CUDABackend, network::APF.Network; linear_solver=:auto)
    N = APF.num_buses(network)
    E = APF.num_branches(network)
    islack = network.slack_bus_index

    A = APF.branch_incidence_matrix(backend, network)
    B = APF.branch_susceptance_matrix(backend, network)
    # Build modified nodal susceptance matrix on CPU, then move to GPU
    #   because it's easier to zero-out a row/column and reset a diagonal coeff there
    # FIXME: have an option to build the modified nodal susceptance matrix
    Y_cpu = -sparse(APF.nodal_susceptance_matrix(KA.CPU(), network))  # Negated!
    Y_cpu[islack, :] .= 0.0
    Y_cpu[:, islack] .= 0.0
    Y_cpu[islack, islack] = 1.0
    Y_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR(Y_cpu)

    if (linear_solver == :auto) || (linear_solver == :CUDSS)
        # All good, we'll use CUDSS
    else
        error("""Unsupported CUDA linear solver for full PTDF: $(linear_solver).
        Supported options are: `:CUDSS` and `:auto`""")
    end

    # Form matrix inverse using CUDSS
    F = LinearAlgebra.ldlt(Y_gpu)
    I_gpu = CuMatrix(1.0I, N, N)
    Yinv = F \ I_gpu

    # Zero out slack bus row
    Yinv[islack, :] .= 0

    return APF.FullPTDF(N, E, Yinv, A, -B.br_b)
end
