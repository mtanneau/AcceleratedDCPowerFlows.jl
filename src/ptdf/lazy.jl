"""
    LazyPTDF

Lazy data structure for PTDF matrix.

Instead of forming the (dense) PTDF matrix, this approach
    only stores a sparse factorization of 
"""
struct LazyPTDF{TA,V,TF} <: AbstractPTDF
    N::Int  # number of buses
    E::Int  # number of branches
    islack::Int  # Index of slack bus

    A::TA   # incidence matrix
    b::V    # branch susceptances (negated)

    F::TF   # Factorization of Y. Must be able to solve linear systems with F \ p
            # ⚠ We use a factorization of -(AᵀBA) to support cholesky factorization when possible
            #    this is because branch susceptances are typically negative, hence AᵀBA is negative definite

    # TODO: cache
end

KA.get_backend(M::LazyPTDF) = KA.get_backend(M.A)

function lazy_ptdf(bkd::KA.CPU, network::Network; linear_solver=:auto)
    N = num_buses(network)
    E = num_branches(network)
    islack = network.slack_bus_index

    # Build nodal admittance matrix
    # TODO: build it directly
    A = branch_incidence_matrix(bkd, network)
    A_sparse = SparseArrays.sparse(A)  

    # ⚠ we negate the susceptance here
    #    so that AᵀBA is positive definite
    b = [-br.b for br in network.branches]
    bmin = minimum(b)
    B = Diagonal(b)
    BA = B * A_sparse
    Y = A_sparse' * BA
    Y[islack, :] .= 0.0
    Y[:, islack] .= 0.0
    Y[islack, islack] = 1.0

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

    F = opfact(Y)

    return LazyPTDF(N, E, islack, A, b, F)
end

lazy_ptdf(network::Network; kwargs...) = lazy_ptdf(DefaultBackend(), network; kwargs...)

"""
    compute_flow_lazy!(pf, pg, Φ::LazyPTDF)

Compute power flow `pf = Φ*pg` lazyly, without forming the PTDF matrix.

Namely, `pf` is computed as `pf = BA * (F \\ pg)`, where `F` is a factorization
    of (-AᵀBA), e.g., a cholesky / LDLᵀ / LU factorization.
"""
function compute_flow!(pf, pg, Φ::LazyPTDF)
    θ = Φ.F \ pg
    θ[Φ.islack, :] .= 0  # slack voltage angle is zero
    # Recall that Φ.F is negated, and Φ.B is also negated...
    #   .. so we are doing pf = (-B) * (A * (-Y⁻¹ * pg)) ..
    #   .. and the two negations cancel out
    # [perf] It is slightly faster to use (BA*θ) if A::SparseMatrix
    mul!(pf, Φ.A, θ)
    pf .*= Φ.b  # broadcast instead of lmul!(Diagonal(Φ.b), pf) to avoid issues when running on GPU
    return pf
end
