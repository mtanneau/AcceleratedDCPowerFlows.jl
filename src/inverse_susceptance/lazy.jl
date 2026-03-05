"""
    LazyInverseSusceptance

Sparse factorization of the negated nodal susceptance matrix.

Stores a factorization of `-Bn` and solves linear systems on demand,
plus the branch susceptance matrix for computing flows.
"""
struct LazyInverseSusceptance{TF, TBf} <: AbstractInverseSusceptance
    N::Int
    E::Int
    islack::Int
    F::TF    # Factorization of -Bn
    Bf::TBf  # BranchSusceptanceMatrix
end

KA.get_backend(S::LazyInverseSusceptance) = KA.get_backend(S.Bf)

lazy_inverse_susceptance(network::Network; kwargs...) = lazy_inverse_susceptance(default_backend(), network; kwargs...)

function lazy_inverse_susceptance(bkd::KA.CPU, network::Network; linear_solver=:auto)
    N = num_buses(network)
    E = num_branches(network)
    islack = network.slack_bus_index

    F, _ = _factorize(bkd, network; linear_solver)

    Bf = branch_susceptance_matrix(bkd, network)

    return LazyInverseSusceptance(N, E, islack, F, Bf)
end

"""
    solve!(θ, p, S::LazyInverseSusceptance)

Solve the DC power flow equation `Bθ = p` for voltage angles `θ`.
"""
function solve!(θ, p, S::LazyInverseSusceptance)
    θ .= S.F \ p
    θ .*= -1  # F is factorization of -Bn, so negate
    θ[S.islack, :] .= 0
    return θ
end

"""
    compute_flow!(pf, p, S::LazyInverseSusceptance)

Compute branch power flows from nodal injections.
"""
function compute_flow!(pf, p, S::LazyInverseSusceptance)
    θ = similar(p)
    compute_flow!(pf, p, S, θ)
    return pf
end

"""
    compute_flow!(pf, p, S::LazyInverseSusceptance, θ)

Compute branch power flows and voltage angles from nodal injections.
"""
function compute_flow!(pf, p, S::LazyInverseSusceptance, θ)
    solve!(θ, p, S)
    mul!(pf, S.Bf, θ)
    return pf
end
