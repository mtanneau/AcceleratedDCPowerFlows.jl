"""
    FullInverseSusceptance

Dense inverse of the negated nodal susceptance matrix.

Stores `(-Bn)⁻¹` as a dense matrix with the slack row zeroed,
plus the branch susceptance matrix for computing flows.
"""
struct FullInverseSusceptance{D, TBf} <: AbstractInverseSusceptance
    islack::Int
    Yinv::D    # (-Bn)⁻¹ with slack row zeroed
    Bf::TBf    # BranchSusceptanceMatrix
end

KA.get_backend(S::FullInverseSusceptance) = KA.get_backend(S.Yinv)

function full_inverse_susceptance(bkd::KA.CPU, network::Network; linear_solver=:auto)
    F, _ = _factorize(bkd, network; linear_solver)
    N = num_buses(network)
    Yinv = F \ Matrix(1.0I, N, N)
    Yinv[network.slack_bus_index, :] .= 0

    Bf = branch_susceptance_matrix(bkd, network)

    return FullInverseSusceptance(network.slack_bus_index, Yinv, Bf)
end

"""
    solve!(θ, p, S::FullInverseSusceptance)

Solve the DC power flow equation `Bθ = p` for voltage angles `θ`.
"""
function solve!(θ, p, S::FullInverseSusceptance)
    mul!(θ, S.Yinv, p)
    θ .*= -1  # Yinv is (-Bn)⁻¹, so negate to get Bn⁻¹ * p
    θ[S.islack, :] .= 0
    return θ
end
