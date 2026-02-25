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
    if ptdf_type == :lazy
        Φ = LazyPTDF(network; kwargs...)
    elseif ptdf_type == :full
        Φ = FullPTDF(network; kwargs...)
    else
        throw(ErrorException("Invalid PTDF type: $ptdf_type; only :lazy and :full are supported"))
    end

    return LazyLODF(network, Φ)
end

function LazyLODF(network::Network, Φ::AbstractPTDF)
    A = branch_incidence_matrix(KA.CPU(), network)
    b = [-br.b for br in network.branches]
    return LazyLODF(
        num_buses(network), 
        num_branches(network), 
        network.slack_bus_index, 
        A, 
        b, 
        Φ,
    )
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
