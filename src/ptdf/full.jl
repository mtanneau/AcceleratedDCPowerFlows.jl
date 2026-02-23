"""
    FullPTDF

Dense PTDF matrix data structure.
"""
struct FullPTDF{D,TA,V} <: AbstractPTDF
    N::Int  # number of buses
    E::Int  # number of branches

    Yinv::D  # Inverse of admittance matrix (dense)
             # Note: we actually store (-Y)⁻¹
    A::TA    # branch incidence matrix (either sparse array or specialized type)
    b::V     # branch susceptances (negated)
end

function FullPTDF(network::Network; solver=:ldlt, gpu=false)
    N = num_buses(network)
    E = num_branches(network)
    A = sparse(BranchIncidenceMatrix(network))
    # ⚠ we negate the susceptance here
    #    so that AᵀBA is positive definite
    b = [-br.b for br in network.branches]
    B = Diagonal(b)
    BA = B * A
    Y = A' * BA
    ref_idx = network.slack_bus_index
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

"""
    compute_flow!(pf, pg, Φ::FullPTDF)

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
