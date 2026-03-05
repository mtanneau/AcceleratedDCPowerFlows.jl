"""
    LazyInverseSusceptance

Sparse factorization of the negated nodal susceptance matrix.

Stores a factorization of `-Bn` and solves linear systems on demand,
plus the branch susceptance matrix for computing flows.
"""
struct LazyInverseSusceptance{TF, TBf} <: AbstractInverseSusceptance
    islack::Int
    F::TF    # Factorization of -Bn
    Bf::TBf  # BranchSusceptanceMatrix
end

KA.get_backend(S::LazyInverseSusceptance) = KA.get_backend(S.Bf)

function lazy_inverse_susceptance(bkd::KA.CPU, network::Network; linear_solver=:auto)
    F, _ = _factorize(bkd, network; linear_solver)
    Bf = branch_susceptance_matrix(bkd, network)
    return LazyInverseSusceptance(network.slack_bus_index, F, Bf)
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
