abstract type AbstractInverseSusceptance end

# --- Private helpers for building and factorizing the negated nodal susceptance matrix ---

function _build_negated_nodal_susceptance(bkd::KA.CPU, network::Network)
    islack = network.slack_bus_index

    bmin = -maximum(br.b for br in network.branches)
    Y = -sparse(nodal_susceptance_matrix(bkd, network))
    Y[islack, :] .= 0.0
    Y[:, islack] .= 0.0
    Y[islack, islack] = 1.0
    return Y, islack, bmin
end

function _select_factorization(bmin, linear_solver)
    if (linear_solver == :auto) || (linear_solver == :SuiteSparse)
        if bmin >= 0.0
            return LinearAlgebra.cholesky
        else
            return LinearAlgebra.ldlt
        end
    elseif linear_solver == :KLU
        return KLU.klu
    else
        error("""Unsupported CPU linear solver: $(linear_solver).
        Supported options are: `:KLU` and `:SuiteSparse`""")
    end
end

function _factorize(Y, bmin; linear_solver=:auto)
    opfact = _select_factorization(bmin, linear_solver)
    F = opfact(Y)
    return F
end

# --- Subtypes ---
include("full.jl")
include("lazy.jl")

# --- Shared methods ---

function Base.:\(S::AbstractInverseSusceptance, p::AbstractVecOrMat)
    θ = similar(p)
    compute_angles!(θ, p, S)
    return θ
end
