#=
    Shared helpers for building and factorizing the negated nodal susceptance matrix.

    Convention: all quantities use the negated susceptance convention,
    i.e., we form `-Bn` (positive definite when all susceptances are negative).
    This enables Cholesky factorization for the common case.
=#

"""
    _build_negated_nodal_susceptance(bkd, network)

Build the negated nodal susceptance matrix `-Bn` with slack bus handling.

Returns `(Y, b)` where:
- `Y` is the sparse negated nodal susceptance matrix with slack row/column zeroed
- `b` is the vector of negated branch susceptances
"""
function _build_negated_nodal_susceptance(bkd::KA.CPU, network::Network)
    islack = network.slack_bus_index

    b = [-br.b for br in network.branches]
    Y = -sparse(nodal_susceptance_matrix(bkd, network))
    Y[islack, :] .= 0.0
    Y[:, islack] .= 0.0
    Y[islack, islack] = 1.0

    return Y, b
end

"""
    _select_factorization(bmin, linear_solver)

Select the factorization function based on the minimum negated susceptance and solver choice.
"""
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

"""
    _factorize(bkd, network; linear_solver)

Build and factorize the negated nodal susceptance matrix.

Returns `(F, b)` where `F` is the factorization of `-Bn` and `b` is the
vector of negated branch susceptances.
"""
function _factorize(bkd::KA.CPU, network::Network; linear_solver=:auto)
    Y, b = _build_negated_nodal_susceptance(bkd, network)
    bmin = minimum(b)
    opfact = _select_factorization(bmin, linear_solver)
    F = opfact(Y)
    return F, b
end
