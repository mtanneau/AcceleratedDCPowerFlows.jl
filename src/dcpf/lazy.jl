"""
    LazyDCPF

Lazy DC power flow solver.

Instead of forming a dense inverse, stores a `LazyInverseSusceptance`
(sparse factorization of the nodal susceptance matrix) and solves a linear system per query.
"""
struct LazyDCPF{TS,TBf} <: AbstractDCPF
    N::Int   # number of buses
    E::Int   # number of branches

    S::TS    # LazyInverseSusceptance
    Bf::TBf  # BranchSusceptanceMatrix (physical susceptances)
end

KA.get_backend(D::LazyDCPF) = KA.get_backend(D.Bf)

function lazy_dcpf(bkd::KA.CPU, network::Network; linear_solver=:auto)
    N = num_buses(network)
    E = num_branches(network)

    S = lazy_inverse_susceptance(bkd, network; linear_solver)
    Bf = branch_susceptance_matrix(bkd, network)

    return LazyDCPF(N, E, S, Bf)
end

"""
    solve!(θ, p, D::LazyDCPF)

Solve the DC power flow equation `Bθ = p` for voltage angles `θ`.
"""
function solve!(θ, p, D::LazyDCPF)
    θ .= D.S.F \ p
    θ .*= -1  # F is factorization of -Bn, so negate
    θ[D.S.islack, :] .= 0  # slack voltage angle is zero
    return θ
end

"""
    compute_flow!(pf, p, D::LazyDCPF)

Compute branch power flows from nodal injections using the DC power flow equations.
"""
function compute_flow!(pf, p, D::LazyDCPF)
    θ = similar(p)
    compute_flow!(pf, p, D, θ)
    return pf
end

"""
    compute_flow!(pf, p, D::LazyDCPF, θ)

Compute branch power flows and voltage angles from nodal injections.
"""
function compute_flow!(pf, p, D::LazyDCPF, θ)
    solve!(θ, p, D)
    mul!(pf, D.Bf, θ)
    return pf
end
