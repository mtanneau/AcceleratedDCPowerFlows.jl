"""
    FullInverseSusceptance

Dense inverse of the nodal susceptance matrix.

Stores `Bn⁻¹` as a dense matrix. The slack bus row and column are zeroed
in the underlying nodal susceptance matrix (with the slack diagonal set to 1)
before inversion, so the slack bus angle is always zero.
"""
struct FullInverseSusceptance{D} <: AbstractInverseSusceptance
    islack::Int
    Yinv::D    # (-Bn)⁻¹ with slack row zeroed
end

KA.get_backend(S::FullInverseSusceptance) = KA.get_backend(S.Yinv)

function full_inverse_susceptance(backend::KA.CPU, Y, islack, bmin; linear_solver=:auto)
function full_inverse_susceptance(backend::KA.CPU, network::Network; linear_solver=:auto)
    Y, islack, bmin = _build_negated_nodal_susceptance(backend, network)
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
    backend = KA.get_backend(S)
    isa(backend, KA.CPU) || error("Only CPU is supported at this point")
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
