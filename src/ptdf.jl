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
