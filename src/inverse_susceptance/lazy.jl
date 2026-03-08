"""
    LazyInverseSusceptance

Sparse factorization-based inverse of the nodal susceptance matrix.

Stores a sparse factorization and solves linear systems on demand.
The slack bus row and column are zeroed in the underlying nodal susceptance
matrix (with the slack diagonal set to 1) before factorization, so the
slack bus angle is always zero.
"""
struct LazyInverseSusceptance{TF} <: AbstractInverseSusceptance
    islack::Int
    F::TF    # Factorization of -Bn
end

KA.get_backend(::LazyInverseSusceptance) = KA.CPU()

function lazy_inverse_susceptance(backend::KA.CPU, Y, islack, bmin; linear_solver=:auto)
    F = _factorize(Y, bmin; linear_solver)
    return LazyInverseSusceptance(islack, F)
end

"""
    compute_angles!(θ, p, S::LazyInverseSusceptance)

Solve the DC power flow equation `Bθ = p` for voltage angles `θ`.
"""
function compute_angles!(θ, p, S::LazyInverseSusceptance)
    backend = KA.get_backend(S)
    isa(backend, KA.CPU) || error("Only CPU is supported at this point")
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
