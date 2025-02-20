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
        error("Invalid linear solver: only lu, klu (CPU only), ldlt, cholesky are supported")
    end
end

struct FullPTDF{D,TA,V} <: AbstractPTDF
    N::Int  # number of buses
    E::Int  # number of branches

    Yinv::D  # Inverse of admittance matrix (dense)
             # Note: we actually store (-Y)⁻¹
    A::TA    # branch incidence matrix (either sparse array or specialized type)
    b::V     # branch susceptances (negated)
end

function FullPTDF(network; solver=:ldlt, gpu=false)
    N = length(network["bus"])
    E = length(network["branch"])
    A = Float64.(calc_basic_incidence_matrix(network))
    # ⚠ we negate the susceptance here
    #    so that AᵀBA is positive definite
    b = [
        -calc_branch_y(network["branch"]["$e"])[2]
        for e in 1:E
    ]
    B = Diagonal(b)
    BA = B * A
    Y = A' * BA
    ref_idx = reference_bus(network)["bus_i"]
    Y[ref_idx, :] .= 0.0
    Y[:, ref_idx] .= 0.0
    Y[ref_idx, ref_idx] = 1.0

    opfact = _linear_solver(solver)

    # TODO: droptol
    # TODO: allow lower precision

    if gpu
        Y = CUDA.CUSPARSE.CuSparseMatrixCSR(Y)

        # TODO: fast matvac product with `A` on GPU
        A = CUDA.CUSPARSE.CuSparseMatrixCSR(A)
        b = CuArray(b)

        F = opfact(Y)
        cI = CuMatrix(1.0I, N, N)
        Yinv = F \ cI
        Yinv[ref_idx, :] .= 0
        
        return FullPTDF(N, E, Yinv, A, b)
    else
        F = opfact(Y)
        Yinv = F \ Matrix(1.0I, N, N)
        Yinv[ref_idx, :] .= 0

        A = BranchIncidenceMatrix(network)
        
        return FullPTDF(N, E, Yinv, A, b)
    end
end

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

function LazyPTDF(network; solver::Symbol=:ldlt, gpu=false)
    N = length(network["bus"])
    E = length(network["branch"])
    A = Float64.(calc_basic_incidence_matrix(network))
    b = [
        -calc_branch_y(network["branch"]["$e"])[2]
        for e in 1:E
    ]
    # TODO: move to GPU here, instead of forming Y on CPU
    B = Diagonal(b)
    BA = B * A
    Y = A' * BA
    ref_idx = reference_bus(network)["bus_i"]
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

"""
    compute_flow_direct!(pf, pg, Φ::FullPTDF)

Compute power flow `pf = Φ*pg` given PTDF matrix `Φ` and nodal injections `pg`.
"""
function compute_flow!(pf, pg, Φ::FullPTDF)
    θ = similar(pg)
    compute_flow!(pf, pg, Φ, θ)
    return pf
end

function compute_flow!(pf, pg, Φ::FullPTDF, θ)
    # TODO: dimension checks
    mul!(θ, Φ.Yinv, pg)
    
    # Note: if `A` is stored as a SparseMatrix, then it's likely
    #      more efficient to store `(B*A)` directly
    # Separating the product as `B * (A * θ)` is faster with a specialized A*θ 
    mul!(pf, Φ.A, θ)
    pf .*= Φ.b  # we use broadcast instead of lmul!(Diagonal(Φ.b), pf) to avoid issues when running on GPU
    return pf
end
