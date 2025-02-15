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

    # Set diagonal elements to -1
    # --> this ensures that post-contingency flow on tripped branch is zero
    E = Φ.E
    for i in 1:E
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

function LazyLODF(data; ptdf_type=:lazy, kwargs...)
    islack = reference_bus(data)["bus_i"]
    A = Float64.(calc_basic_incidence_matrix(data))
    b = [
        -calc_branch_y(data["branch"]["$e"])[2]
        for e in 1:length(data["branch"])
    ]
    if ptdf_type == :lazy
        Φ = LazyPTDF(data; kwargs...)
    elseif ptdf_type == :full
        Φ = FullPTDF(data; kwargs...)
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
function compute_flow!(pfc, p::Vector, pf0::Vector, L::FullLODF, c::Int)
    @views pfc .= pf0 .+ (pf0[c] .* L.matrix[:, c])
    return pfc
end

function compute_flow!(pfc, p::Vector, pf0::Vector, L::LazyLODF{SM,<:FullPTDF}, k::Int) where{SM}
    i0 = L.islack
    A = L.A
    Φ = L.Φ

    # Compute Y⁻¹ aₖ, where aₖ is the kth row of A
    ak = Vector(A[k, :])    # ⚠ most linear solvers wrappers do not support passing a sparse RHS here
    bk = -L.b[k]            # susceptance of tripped branch
    ξ = -(Φ.Yinv * ak)      # FullPTDF stores -Y⁻¹, so we need to negate
                            # TODO: since `a` only has two non-zeros, 
                            #   (Y⁻¹ * a) is just combining two columns of Y⁻¹.
    ξ[i0] = 0               # slack bus has zero phase angle

    # Compute post-update phase angles via SWM formula
    β = (1.0 - bk*dot(ak, ξ))
    ξ ./= β   # TODO: make sure there's no division by zero here
    ξ[i0] = 0

    # Re-multiply by BA, and zero-out the kth element
    pfc .= pf0
    mul!(pfc, Φ.BA, ξ, -pf0[k], 1.0)
    pfc[k] = 0

    return pfc
end

function compute_flow!(pfc, p::Vector, pf0::Vector, L::LazyLODF{SM,<:LazyPTDF}, k::Int) where{SM}
    i0 = L.islack
    A = L.A
    Φ = L.Φ

    # Compute Y⁻¹ aₖ, where aₖ is the kth row of A
    ak = Vector(A[k, :])    # ⚠ most linear solvers wrappers do not support passing a sparse RHS here
    bk = -L.b[k]
    ξ  = -(Φ.F \ ak)        # Lazy PTDF factorizes -(AᵀBA), so we need to negate
                            # TODO: since `a` only has two non-zeros, 
                            #   (Y⁻¹ * a) is just combining two columns of Y⁻¹.
    ξ[i0] = 0               # slack bus has zero phase angle

    # Compute post-update phase angles via SWM formula
    β = (1.0 - bk*dot(ak, ξ))
    ξ ./= β   # TODO: make sure there's no division by zero here
    ξ[i0] = 0

    # Re-multiply by BA, and zero-out the kth element
    pfc .= pf0
    mul!(pfc, Φ.BA, ξ, -pf0[k], 1.0)
    pfc[k] = 0

    return pfc
end



function compute_all_flows!(pf, p, pf0, L::FullLODF; outages=Int[])
    N, E = L.N, L.E

    for (i, l) in enumerate(outages)
        pf[:, i] .= pf0 .+ (pf0[l] .* L.matrix[:, l])
    end

    return pf
end
