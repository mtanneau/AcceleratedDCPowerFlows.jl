"""
    FullInverseSusceptance

Dense inverse of the negated nodal susceptance matrix.

Stores `(-Bn)⁻¹` as a dense matrix with the slack row zeroed.
"""
struct FullInverseSusceptance{D} <: AbstractInverseSusceptance
    islack::Int
    Yinv::D    # (-Bn)⁻¹ with slack row zeroed
end

KA.get_backend(S::FullInverseSusceptance) = KA.get_backend(S.Yinv)

function full_inverse_susceptance(Y, islack, bmin; linear_solver=:auto)
    F = _factorize(Y, bmin; linear_solver)
    N = size(Y, 1)
    Yinv = F \ Matrix(1.0I, N, N)
    Yinv[islack, :] .= 0

    return FullInverseSusceptance(islack, Yinv)
end

"""
    compute_angles!(θ, p, S::FullInverseSusceptance)

Solve the DC power flow equation `Bθ = p` for voltage angles `θ`.
"""
function compute_angles!(θ, p, S::FullInverseSusceptance)
    mul!(θ, S.Yinv, p)
    θ .*= -1  # Yinv is (-Bn)⁻¹, so negate to get Bn⁻¹ * p
    θ[S.islack, :] .= 0
    return θ
end

function Base.getindex(S::FullInverseSusceptance, ::Colon, i::Int)
    col = -S.Yinv[:, i]
    col[S.islack] = 0.0
    return col
end
