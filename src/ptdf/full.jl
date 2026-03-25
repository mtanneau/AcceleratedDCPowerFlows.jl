"""
    FullPTDF

Dense PTDF matrix data structure.
"""
struct FullPTDF{D,TA,V,C} <: AbstractPTDF
    N::Int  # number of buses
    E::Int  # number of branches

    Yinv::D  # Inverse of admittance matrix (dense)
    # Note: we actually store (-Y)⁻¹
    A::TA    # branch incidence matrix (either sparse array or specialized type)
    b::V     # branch susceptances (negated)
    cache::C
end

KA.get_backend(M::FullPTDF) = KA.get_backend(M.Yinv)

get_cache(Φ::FullPTDF) = Φ.cache

function full_ptdf(network::Network; kwargs...)
    return full_ptdf(default_backend(), network; kwargs...)
end

function full_ptdf(
    bkd::KA.CPU,
    network::Network;
    linear_solver=:auto,
    cache_size::Integer=DEFAULT_OPERATOR_CACHE_COLS,
)
    N = num_buses(network)
    E = num_branches(network)
    islack = network.slack_bus_index

    # Build nodal susceptance matrix
    # ⚠ susceptances are _negated_ so that AᵀBA is positive definite
    A = branch_incidence_matrix(bkd, network)
    b = [-br.b for br in network.branches]
    bmin = minimum(b)
    Y = -sparse(nodal_susceptance_matrix(bkd, network))  # Negated!
    Y[islack, :] .= 0.0
    Y[:, islack] .= 0.0
    Y[islack, islack] = 1.0

    # Which factorization should we use?
    opfact = if (linear_solver == :auto) || (linear_solver == :SuiteSparse)
        if bmin >= 0.0
            LinearAlgebra.cholesky
        else
            LinearAlgebra.ldlt
        end
    elseif linear_solver == :KLU
        KLU.klu
    else
        error("""Unsupported CPU linear solver for full PTDF: $(linear_solver).
        Supported options are: `:KLU` and `:SuiteSparse`""")
    end

    # Form matrix inverse
    F = opfact(Y)
    Yinv = F \ Matrix(1.0I, N, N)
    Yinv[islack, :] .= 0

    cache = OperatorCache(bkd, eltype(Yinv), N; ncols=cache_size)
    return FullPTDF(N, E, Yinv, A, b, cache)
end

# The following implementations are already backend- and shape- agnostic,
#   so should not need to be extended.

"""
    compute_flow!(pf, pg, Φ::FullPTDF)

Compute power flow `pf = Φ*pg` given PTDF matrix `Φ` and nodal injections `pg`.
"""
function compute_flow!(pf::AbstractVecOrMat, pg::AbstractVecOrMat, Φ::FullPTDF)
    K = size(pg, 2)

    cache = get_cache(Φ)
    resize!(cache, K; force=false)
    θ = view(cache, :, 1:K)

    # θ will be a matrix-shaped view, but pg may be Vector-shape
    # So we reshape `pg` to avoid any issues
    pg = reshape(pg, (size(pg, 1), size(pg, 2)))

    compute_flow!(pf, pg, Φ, θ)
    return pf
end

function compute_flow!(pf::AbstractVecOrMat, pg::AbstractVecOrMat, Φ::FullPTDF, θ::AbstractVecOrMat)
    # TODO: dimension checks
    mul!(θ, Φ.Yinv, pg)

    # Note: if `A` is stored as a SparseMatrix, then it's likely
    #      more efficient to store `(B*A)` directly
    # Separating the product as `B * (A * θ)` is faster with a specialized A*θ 
    mul!(pf, Φ.A, θ)
    pf .*= Φ.b  # we use broadcast instead of lmul!(Diagonal(Φ.b), pf) to avoid issues when running on GPU
    return pf
end
