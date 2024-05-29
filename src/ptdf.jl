abstract type AbstractPTDF end

struct FullPTDF{T} <: AbstractPTDF
    N::Int  # number of buses
    E::Int  # number of branches

    matrix::Matrix{T}  # PTDF matrix
end

function FullPTDF(network)
    N = length(network["bus"])
    E = length(network["branch"])
    matrix = calc_basic_ptdf_matrix(network)

    return FullPTDF(N, E, matrix)
end

struct LazyPTDF{TF} <: AbstractPTDF
    N::Int  # number of buses
    E::Int  # number of branches
    islack::Int  # Index of slack bus

    A::SparseMatrixCSC{Float64,Int}  # incidence matrix
    B::SparseMatrixCSC{Float64,Int}  # branch susceptance matrix
    BA::SparseMatrixCSC{Float64,Int}  # B*A
    AtBA::SparseMatrixCSC{Float64,Int}  # AᵀBA

    F::TF   # Factorization of -(AᵀBA). Must be able to solve linear systems with F \ p
            # We use a factorization of -(AᵀBA) to support cholesky factorization when possible

    # TODO: cache
end

function LazyPTDF(network; solver::Symbol=:klu)
    N = length(network["bus"])
    E = length(network["branch"])
    A = Float64.(calc_basic_incidence_matrix(network))
    b = [
        calc_branch_y(network["branch"]["$e"])[2]
        for e in 1:E
    ]
    B = sparse(Diagonal(b))
    BA = B * A
    S = AtBA = A' * BA
    ref_idx = reference_bus(network)["bus_i"]
    S[ref_idx, :] .= 0.0
    S[:, ref_idx] .= 0.0
    S[ref_idx, ref_idx] = -1.0;  # to enable cholesky

    if solver == :lu
        F = lu(S)
    elseif solver == :klu
        F = KLU.klu(S)
    elseif solver == :ldlt
        F = ldlt(S)
    elseif solver == :cholesky
        # If Cholesky is not possible, default to LDLᵀ
        if maximum(b) < 0.0
            F = cholesky(S)
        else
            @warn "Some branches have positive susceptance, cannot use Cholesky; defaulting to LDLᵀ"
            F = ldlt(S)
        end
    else
        error("Invalid linear solver: only lu, klu, ldlt are supported")
    end

    return LazyPTDF(N, E, ref_idx, A, B, BA, AtBA, F)
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
    mul!(pf, Φ.BA, θ)
    return pf
end

"""
    compute_flow_direct!(pf, pg, Φ::PTDF)

Compute power flow `pf = Φ*pg` given PTDF matrix `Φ` and nodal injections `pg`.
"""
function compute_flow!(pf, pg, Φ::PTDF)
    mul!(pf, Φ.matrix, pg)
    return pf
end
