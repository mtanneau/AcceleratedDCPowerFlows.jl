"""
    LazyInverseSusceptance

Sparse factorization of the negated nodal susceptance matrix.

Stores a factorization of `-Bn` and solves linear systems on demand.
"""
struct LazyInverseSusceptance{TF} <: AbstractInverseSusceptance
    islack::Int
    F::TF    # Factorization of -Bn
end

KA.get_backend(::LazyInverseSusceptance) = KA.CPU()

function lazy_inverse_susceptance(Y, islack, bmin; linear_solver=:auto)
    F = _factorize(Y, bmin; linear_solver)
    return LazyInverseSusceptance(islack, F)
end

"""
    compute_angles!(θ, p, S::LazyInverseSusceptance)

Solve the DC power flow equation `Bθ = p` for voltage angles `θ`.
"""
function compute_angles!(θ, p, S::LazyInverseSusceptance)
    θ .= S.F \ p
    θ .*= -1  # F is factorization of -Bn, so negate
    θ[S.islack, :] .= 0
    return θ
end

function Base.getindex(S::LazyInverseSusceptance, ::Colon, i::Int)
    eᵢ = zeros(size(S.F, 1))
    eᵢ[i] = 1.0
    col = -(S.F \ eᵢ)
    col[S.islack] = 0.0
    return col
end
