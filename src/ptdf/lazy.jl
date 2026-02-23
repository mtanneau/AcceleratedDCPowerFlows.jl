"""
    LazyPTDF

Lazy data structure for PTDF matrix.

Instead of forming the (dense) PTDF matrix, this approach
    only stores a sparse factorization of 
"""
struct LazyPTDF{TF,TA,V,SM} <: AbstractPTDF
    N::Int  # number of buses
    E::Int  # number of branches
    islack::Int  # Index of slack bus

    A::TA   # incidence matrix
    b::V    # branch susceptances (negated)
    BA::SM  # B*A (negated)
    Y::SM   # Y = A'BA (nodal admittance matrix, negated)

    F::TF   # Factorization of Y. Must be able to solve linear systems with F \ p
            # ⚠ We use a factorization of -(AᵀBA) to support cholesky factorization when possible
            #    this is because branch susceptances are typically negative, hence AᵀBA is negative definite

    # TODO: cache
end

function LazyPTDF(network::Network; solver::Symbol=:ldlt, gpu=false)
    N = num_buses(network)
    E = num_branches(network)
    A = sparse(BranchIncidenceMatrix(network))
    b = [-br.b for br in network.branches]
    # TODO: move to GPU here, instead of forming Y on CPU
    B = Diagonal(b)
    BA = B * A
    Y = A' * BA
    ref_idx = network.slack_bus_index
    Y[ref_idx, :] .= 0.0
    Y[:, ref_idx] .= 0.0
    Y[ref_idx, ref_idx] = 1.0

    if gpu
        A = CUDA.CUSPARSE.CuSparseMatrixCSR(A)
        b = CuArray(b)
        BA = CUDA.CUSPARSE.CuSparseMatrixCSR(BA)
        Y = CUDA.CUSPARSE.CuSparseMatrixCSR(Y)
    else
        A = BranchIncidenceMatrix(network)
    end

    gpu && (solver == :klu) && error("KLU is not supported on GPU")

    opfact = _linear_solver(solver)
    F = opfact(Y)

    return LazyPTDF(N, E, ref_idx, A, b, BA, Y, F)
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
    # Recall that Φ.F is negated, and Φ.B is also negated...
    #   .. so we are doing pf = (-B) * (A * (-Y⁻¹ * pg)) ..
    #   .. and the two negations cancel out
    # [perf] It is slightly faster to use (BA*θ) if A::SparseMatrix
    mul!(pf, Φ.A, θ)
    pf .*= Φ.b  # broadcast instead of lmul!(Diagonal(Φ.b), pf) to avoid issues when running on GPU
    return pf
end
