"""
    FullInverseSusceptance

Dense inverse of the negated nodal susceptance matrix.

Stores `(-Bn)⁻¹` as a dense matrix with the slack row zeroed,
plus the branch susceptance matrix for computing flows.
"""
struct FullInverseSusceptance{D, TBf} <: AbstractInverseSusceptance
    N::Int
    E::Int
    islack::Int
    Yinv::D    # (-Bn)⁻¹ with slack row zeroed
    Bf::TBf    # BranchSusceptanceMatrix
end

KA.get_backend(S::FullInverseSusceptance) = KA.get_backend(S.Yinv)

full_inverse_susceptance(network::Network; kwargs...) = full_inverse_susceptance(default_backend(), network; kwargs...)

function full_inverse_susceptance(bkd::KA.CPU, network::Network; linear_solver=:auto)
    N = num_buses(network)
    E = num_branches(network)
    islack = network.slack_bus_index

    F, _ = _factorize(bkd, network; linear_solver)
    Yinv = F \ Matrix(1.0I, N, N)
    Yinv[islack, :] .= 0

    Bf = branch_susceptance_matrix(bkd, network)

    return FullInverseSusceptance(N, E, islack, Yinv, Bf)
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

"""
    compute_flow!(pf, p, S::FullInverseSusceptance)

Compute branch power flows from nodal injections.
"""
function compute_flow!(pf, p, S::FullInverseSusceptance)
    θ = similar(p)
    compute_flow!(pf, p, S, θ)
    return pf
end

"""
    compute_flow!(pf, p, S::FullInverseSusceptance, θ)

Compute branch power flows and voltage angles from nodal injections.
"""
function compute_flow!(pf, p, S::FullInverseSusceptance, θ)
    solve!(θ, p, S)
    mul!(pf, S.Bf, θ)
    return pf
end
