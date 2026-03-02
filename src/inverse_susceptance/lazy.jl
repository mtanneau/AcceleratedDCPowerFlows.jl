"""
    LazyInverseSusceptance

Sparse factorization of the negated nodal susceptance matrix.

Stores a factorization of `-Bn` and solves linear systems on demand.
"""
struct LazyInverseSusceptance{TF} <: AbstractInverseSusceptance
    N::Int
    islack::Int
    F::TF  # Factorization of -Bn
end

function lazy_inverse_susceptance(bkd::KA.CPU, network::Network; linear_solver=:auto)
    islack = network.slack_bus_index
    N = num_buses(network)

    F, _ = _factorize(bkd, network; linear_solver)

    return LazyInverseSusceptance(N, islack, F)
end
