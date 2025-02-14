abstract type AbstractLODF end

struct FullLODF{M} <: AbstractLODF
    N::Int
    E::Int
    matrix::M
end

function FullLODF(network)
    Φ = LazyPTDF(network)
    N = length(network["bus"])
    i0 = reference_bus(network)["bus_i"]

    At = Matrix(Φ.A')
    _M = (Φ.F \ At)
    _M[i0, :] .= 0  # ⚠ need to zero-out slack bus angle
    M = Φ.BA * _M
    d = inv.(1 .- diag(M))
    d .*= (abs.(d) .<= 1e8)
    D = Diagonal(d)
    rmul!(M, D)

    E = Φ.E
    for i in 1:E
        M[i, i] = -1
    end

    return FullLODF(Φ.N, Φ.E, M)
end

struct LazyLODF{TF,V,SM} <: AbstractLODF
    N::Int
    E::Int
    
    Φ::LazyPTDF{TF,V,SM}
end

function LazyLODF(data; solver::Symbol=:klu)
    Φ = LazyPTDF(data; solver=solver)
    return LazyLODF(Φ.N, Φ.E, Φ)
end

function LazyLODF(Φ::LazyPTDF)
    return LazyLODF(Φ.N, Φ.E, Φ)
end

function compute_flow!(pf, p::Vector, L::LazyLODF, k::Int)
    Φ = L.Φ
    i0 = Φ.islack
    # TODO: give option to pass θ0 or pf0 as argument
    # This would save one linear system solve
    θ0 = -(Φ.F \ p)  # Lazy PTDF factorizes -(AᵀBA), so we need to negate
    θ0[i0] = 0       # slack bus has zero phase angle

    # Next, we need to compute (AᵀBA)⁻¹ aₖ, where aₖ is the kth row of A
    a = Vector(Φ.A[k, :])  # ⚠ most linear solvers wrappers do not support passing a sparse RHS here
    bk = Φ.b[k]
    ξ = -(Φ.F \ a)  # Lazy PTDF factorizes -(AᵀBA), so we need to negate
    ξ[i0] = 0       # slack bus has zero phase angle

    # Compute post-update phase angles via SWM formula
    θc = θ0 + (dot(a, θ0) * bk / (1 - bk*dot(a, ξ))) .* ξ
    θc[i0] = 0

    # Re-multiply by BA, and zero-out the kth element
    mul!(pf, Φ.BA, θc)
    pf[k] = 0

    return pf
end

function compute_all_flows!(pf, p, pf0, L::FullLODF; outages=Int[])
    N, E = L.N, L.E

    for (i, l) in enumerate(outages)
        pf[:, i] .= pf0 .+ (pf0[l] .* L.matrix[:, l])
    end

    return pf
end
