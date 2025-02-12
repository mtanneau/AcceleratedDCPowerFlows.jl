abstract type AbstractPTDF end

function _linear_solver(s)
    if s == :lu
        return lu
    elseif s == :klu
        return KLU.klu
    elseif s == :ldlt
        return ldlt
    elseif s == :cholesky
        return cholesky
    else
        error("Invalid linear solver: only lu, klu, ldlt, cholesky are supported")
    end
end

struct FullPTDF{M} <: AbstractPTDF
    N::Int  # number of buses
    E::Int  # number of branches

    matrix::M  # PTDF matrix
end

function FullPTDF(network; gpu=false)
    N = length(network["bus"])
    E = length(network["branch"])
    A = Float64.(calc_basic_incidence_matrix(network))
    b = [
        calc_branch_y(network["branch"]["$e"])[2]
        for e in 1:E
    ]
    B = Diagonal(b)
    BA = B * A
    S = A' * BA
    ref_idx = reference_bus(network)["bus_i"]
    S[ref_idx, :] .= 0.0
    S[:, ref_idx] .= 0.0
    S[ref_idx, ref_idx] = -1.0;  # to enable cholesky
    S = -S

    opfact = ldlt  # FIXME

    Φ = if gpu
        S = CUDA.CUSPARSE.CuSparseMatrixCSR(S)
        F = opfact(S)
        cI = CuMatrix(1.0I, N, N)
        M = F \ cI
        M[ref_idx, :] .= 0
        BA = CUDA.CUSPARSE.CuSparseMatrixCSR(BA)
        BA * M
    else
        F = opfact(S)
        M = F \ Matrix(1.0I, N, N)
        M[ref_idx, :] .= 0
        BA * M
    end

    # TODO: droptol
    # TODO: lower precision

    return FullPTDF(N, E, Φ)
end

struct LazyPTDF{TF,V,SM} <: AbstractPTDF
    N::Int  # number of buses
    E::Int  # number of branches
    islack::Int  # Index of slack bus

    A::SM  # incidence matrix
    b::V  # branch susceptances
    BA::SM  # B*A
    AtBA::SM  # AᵀBA

    F::TF   # Factorization of -(AᵀBA). Must be able to solve linear systems with F \ p
            # We use a factorization of -(AᵀBA) to support cholesky factorization when possible

    # TODO: cache
end

function LazyPTDF(network; solver::Symbol=:ldlt, gpu=false)
    N = length(network["bus"])
    E = length(network["branch"])
    A = Float64.(calc_basic_incidence_matrix(network))
    b = [
        calc_branch_y(network["branch"]["$e"])[2]
        for e in 1:E
    ]
    B = Diagonal(b)
    BA = B * A
    S = AtBA = A' * BA
    ref_idx = reference_bus(network)["bus_i"]
    S[ref_idx, :] .= 0.0
    S[:, ref_idx] .= 0.0
    S[ref_idx, ref_idx] = -1.0;  # to enable cholesky
    S = -S

    if gpu
        A = CUDA.CUSPARSE.CuSparseMatrixCSR(A)
        b = CuArray(b)
        BA = CUDA.CUSPARSE.CuSparseMatrixCSR(BA)
        AtBA = CUDA.CUSPARSE.CuSparseMatrixCSR(AtBA)
        S = CUDA.CUSPARSE.CuSparseMatrixCSR(S)
    end

    if solver == :lu
        F = lu(S)
    elseif solver == :klu
        F = KLU.klu(S)
    elseif solver == :ldlt
        F = ldlt(S)
    elseif solver == :cholesky
        # If Cholesky is not possible, default to LDLᵀ
        F = cholesky(S)
    else
        error("Invalid linear solver: only cholesky, ldlt, lu, and klu (CPU-only) are supported")
    end

    return LazyPTDF(N, E, ref_idx, A, b, BA, AtBA, F)
end

"""
    compute_flow_lazy!(pf, pg, Φ::LazyPTDF)

Compute power flow `pf = Φ*pg` lazyly, without forming the PTDF matrix.

Namely, `pf` is computed as `pf = BA * (F \\ pg)`, where `F` is a factorization
    of (-AᵀBA), e.g., a cholesky / LDLᵀ / LU factorization.
"""
function compute_flow!(pf, pg, Φ::LazyPTDF)
    θ = Φ.F \ pg
    θ[Φ.islack, :] .= 0  # slack voltage angle is zero
    mul!(pf, Φ.BA, θ, -one(eltype(pf)), zero(eltype(pf)))
    return pf
end

"""
    compute_flow_direct!(pf, pg, Φ::FullPTDF)

Compute power flow `pf = Φ*pg` given PTDF matrix `Φ` and nodal injections `pg`.
"""
function compute_flow!(pf, pg, Φ::FullPTDF)
    mul!(pf, Φ.matrix, pg)
    return pf
end

function benchmark_ptdf(data;
    linear_solvers = [:cholesky, :ldlt, :lu, :klu],
    verbose=false,
    batch_size=96,
    gpu=false,
)
    df = DataFrame(
        :casename => String[],
        :num_bus => Int[],
        :num_branch => Int[],
        :device => String[],
        :cpu_cores => Int[],
        :solver => String[],
        :time_analyze => Float64[],
        :time_factorize => Float64[],
        :mem_analyze => Float64[],
        :mem_factorize => Float64[],
        :time_solve_1 => Float64[],
        :mem_solve_1 => Float64[],
        :batch_size => Int[],
        :time_solve_batch => Float64[],
        :mem_solve_batch => Float64[],
    )

    # Extract basic data
    N = length(data["bus"])
	E = length(data["branch"])
    
    # Pre-allocate data to compute power flows (lazy)
    pg_x1 = randn(Float64, N)
    pf_x1 = zeros(Float64, E)
    pg_batch = randn(Float64, N, batch_size)
    pf_batch = zeros(Float64, E, batch_size)
    if gpu
        pg_x1 = CuArray(pg_x1)
        pf_x1 = CuArray(pf_x1)
        pg_batch = CuArray(pg_batch)
        pf_batch = CuArray(pf_batch)
    end

    # Benchmark time to factorize matrix
    for solver in linear_solvers

        if gpu && solver == :klu
            @warn "KLU is not supported on GPU; skipping"
            continue
        end

        try
            LazyPTDF(data, solver=solver; gpu=gpu)
        catch err
            if isa(err, PosDefException)
                @warn "$solver failed with PosDefException; skipping"
                # TODO: record in dataframe but skip
                continue
            end
            rethrow(err)
        end

        Φ = LazyPTDF(data, solver=solver; gpu=gpu)

        # Benchmark only factorization step
        S = -Φ.AtBA  # this is the matrix that should be factorized
        op_fact = _linear_solver(solver)
        b_fact = if gpu
            cS = CUDA.CUSPARSE.CuSparseMatrixCSR(S)
            @benchmark CUDA.@sync $(op_fact)($cS)
        else
            @benchmark $(op_fact)($S)
        end
        verbose && println("$solver; factorize")
        verbose && display(b_fact)

        # One vector
        b_x1 = if gpu
            @benchmark CUDA.@sync compute_flow!($pf_x1, $pg_x1, $Φ)
        else
            @benchmark compute_flow!($pf_x1, $pg_x1, $Φ)
        end
        verbose && println("$solver; m=1")
        verbose && display(b_x1)
        
        # minibatch
        b_batch = if gpu
            @benchmark CUDA.@sync compute_flow!($pf_batch, $pg_batch, $Φ)
        else
            @benchmark compute_flow!($pf_batch, $pg_batch, $Φ)
        end
        verbose && println("$solver; m=$(batch_size)")
        verbose && display(b_batch)

        row = Dict(
            :casename => data["name"],
            :num_bus => N,
            :num_branch => E,
            :device => gpu ? name(CUDA.device()) : "CPU",
            :cpu_cores => BLAS.get_num_threads(),
            :solver => "$(solver)",
            :time_analyze => 0.0,
            :time_factorize => median(b_fact.times) / 1e9,
            :mem_analyze => 0,
            :mem_factorize => b_fact.memory,
            :time_solve_1 => median(b_x1.times) / 1e9,
            :mem_solve_1 => b_x1.memory,
            :time_solve_batch => median(b_batch.times) / 1e9,
            :mem_solve_batch => b_batch.memory,
            :batch_size => batch_size,
        )

        push!(df, row)
    end

    return df
end
