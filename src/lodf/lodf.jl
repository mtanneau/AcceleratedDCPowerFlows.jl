abstract type AbstractLODF end

struct FullLODF{M} <: AbstractLODF
    N::Int
    E::Int
    matrix::M
end

function FullLODF(network::Network)
    Φ = LazyPTDF(network)
    N = num_buses(network)
    i0 = network.slack_bus_index

    A = sparse(BranchIncidenceMatrix(network))
    At = Matrix(A')
    _M = (Φ.F \ At)
    _M[i0, :] .= 0  # ⚠ need to zero-out slack bus angle
    M = Φ.BA * _M
    d = inv.(1 .- diag(M))
    d .*= (abs.(d) .<= 1e8)
    D = Diagonal(d)
    rmul!(M, D)

    # Set diagonal elements to -1
    # --> this ensures that post-contingency flow on tripped branch is zero
    for i in 1:Φ.E
        M[i, i] = -1
    end

    return FullLODF(Φ.N, Φ.E, M)
end

struct LazyLODF{SM,PTDF} <: AbstractLODF
    N::Int
    E::Int

    # Some network info
    islack::Int
    A::SM
    b::Vector{Float64}
    
    Φ::PTDF
end

function LazyLODF(network::Network; ptdf_type=:lazy, kwargs...)
    islack = network.slack_bus_index
    A = sparse(BranchIncidenceMatrix(network))
    b = [-br.b for br in network.branches]
    if ptdf_type == :lazy
        Φ = LazyPTDF(network; kwargs...)
    elseif ptdf_type == :full
        Φ = FullPTDF(network; kwargs...)
    else
        throw(ErrorException("Invalid PTDF type: $ptdf_type; only :lazy and :full are supported"))
    end
    return LazyLODF(Φ.N, Φ.E, islack, A, b, Φ)
end

function LazyLODF(Φ::LazyPTDF)
    return LazyLODF(Φ.N, Φ.E, Φ.islack, Φ.A, b, Φ)
end

"""
    compute_flow!(pfc, p, pf0, L::FullLODF, c::Int)

Compute the power flow adjustments after a contingency.

# Arguments
- `pfc`: Post-contingency power flow vector (pre-allocated)
- `p`: Nodal power injections
- `pf0`: Pre-contingency power flow vector
- `L::FullLODF`: The Line Outage Distribution Factor (LODF) matrix.
- `c::Int`: The index of the contingency line.

# Description
This function updates the power flow vector `pf` to reflect the changes in power flow due to the outage of the line specified by the index `c`. 
    The LODF matrix `L` is used to compute the impact of the line outage on the power flows.
"""
function compute_flow!(pfc, pf0::Vector, L::FullLODF, br::Branch)
    c = br.index
    @views pfc .= pf0 .+ (pf0[c] .* L.matrix[:, c])
    return pfc
end

function compute_flow!(pfc, pf0::Vector, L::LazyLODF{SM,<:FullPTDF}, br::Branch) where{SM}
    i0 = L.islack
    Φ = L.Φ

    # Compute Y⁻¹ aₖ, where aₖ is the kth row of A
    # Since we have formed Y⁻¹, we just need to compute 
    #   the difference between columns `i` and `j` of Y⁻¹
    k = br.index
    i = br.bus_fr
    j = br.bus_to
    @views ξ = Φ.Yinv[:, j] .- Φ.Yinv[:, i]
    ξ[i0] = 0               # slack bus has zero phase angle

    # Compute post-update phase angles via SWM formula
    bk = -L.b[k]            # susceptance of tripped branch
    β = (1.0 - bk*(ξ[i] - ξ[j]))

    # Re-multiply by BA, and zero-out the kth element
    mul!(pfc, Φ.A, ξ)
    pfc .*= Φ.b
    pfc .*= (-pf0[k] / β)
    pfc .+= pf0
    pfc[k] = 0

    return pfc
end

function compute_flow!(pfc, pf0::Vector, L::LazyLODF{SM,<:LazyPTDF}, br::Branch) where{SM}
    i0 = L.islack
    Φ = L.Φ

    # Compute Y⁻¹ aₖ, where aₖ is the kth row of A
    # First, extract (i, j) indices, such that branch k = (i, j)
    k = br.index
    i = br.bus_fr
    j = br.bus_to
    ak = zeros(eltype(pfc), L.N)
    ak[i] = 1
    ak[j] = -1

    # Solve linear system
    ξ  = -(Φ.F \ ak)        # Lazy PTDF factorizes -(AᵀBA), so we need to negate
                            # TODO: since `a` only has two non-zeros, 
                            #   (Y⁻¹ * a) is just combining two columns of Y⁻¹.
    ξ[i0] = 0               # slack bus has zero phase angle

    # Compute post-update phase angles via SWM formula
    bk = -L.b[k]
    β = (1.0 - bk*(ξ[i] - ξ[j]))

    # Re-multiply by BA, and zero-out the kth element
    mul!(pfc, Φ.A, ξ)
    pfc .*= Φ.b
    pfc .*= (-pf0[k] / β)
    pfc .+= pf0
    pfc[k] = 0

    return pfc
end

function compute_all_flows!(pfc, pf0, L::FullLODF; outages=Int[])
    N, E = L.N, L.E

    for (i, l) in enumerate(outages)
        @views pfc[:, i] .= pf0 .+ (pf0[l] .* L.matrix[:, l])
    end

    return pfc
end
