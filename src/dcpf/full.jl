"""
    FullDCPF

Dense DC power flow solver.

Stores a `FullInverseSusceptance` and the branch susceptance matrix.
Solves `Bθ = p` for voltage angles `θ`, then computes branch flows as `pf = Bf * θ`.
"""
struct FullDCPF{TS,TBf} <: AbstractDCPF
    N::Int   # number of buses
    E::Int   # number of branches

    S::TS    # FullInverseSusceptance
    Bf::TBf  # BranchSusceptanceMatrix (physical susceptances)
end

KA.get_backend(D::FullDCPF) = KA.get_backend(D.S.Yinv)

function full_dcpf(bkd::KA.CPU, network::Network; linear_solver=:auto)
    N = num_buses(network)
    E = num_branches(network)

    S = full_inverse_susceptance(bkd, network; linear_solver)
    Bf = branch_susceptance_matrix(bkd, network)

    return FullDCPF(N, E, S, Bf)
end

"""
    solve!(θ, p, D::FullDCPF)

Solve the DC power flow equation `Bθ = p` for voltage angles `θ`.

Since the inverse stores `(-Bn)⁻¹`, we negate the output to get physical angles.
"""
function solve!(θ, p, D::FullDCPF)
    mul!(θ, D.S.Yinv, p)
    θ .*= -1  # Yinv is (-Bn)⁻¹, so negate to get Bn⁻¹ * p
    θ[D.S.islack, :] .= 0
    return θ
end

"""
    compute_flow!(pf, p, D::FullDCPF)

Compute branch power flows from nodal injections using the DC power flow equations.
"""
function compute_flow!(pf, p, D::FullDCPF)
    θ = similar(p)
    compute_flow!(pf, p, D, θ)
    return pf
end

"""
    compute_flow!(pf, p, D::FullDCPF, θ)

Compute branch power flows and voltage angles from nodal injections.
"""
function compute_flow!(pf, p, D::FullDCPF, θ)
    solve!(θ, p, D)
    mul!(pf, D.Bf, θ)
    return pf
end
