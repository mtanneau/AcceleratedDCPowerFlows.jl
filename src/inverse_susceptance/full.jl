"""
    FullInverseSusceptance

Dense inverse of the negated nodal susceptance matrix.

Stores `(-Bn)⁻¹` as a dense matrix with the slack row zeroed.
"""
struct FullInverseSusceptance{D} <: AbstractInverseSusceptance
    N::Int
    islack::Int
    Yinv::D  # (-Bn)⁻¹ with slack row zeroed
end

function full_inverse_susceptance(bkd::KA.CPU, network::Network; linear_solver=:auto)
    N = num_buses(network)
    islack = network.slack_bus_index

    F, _ = _factorize(bkd, network; linear_solver)
    Yinv = F \ Matrix(1.0I, N, N)
    Yinv[islack, :] .= 0

    return FullInverseSusceptance(N, islack, Yinv)
end
