struct FullLODF{M} <: AbstractLODF
    N::Int
    E::Int
    matrix::M
end

KA.get_backend(L::FullLODF) = KA.get_backend(L.matrix)

function full_lodf(bkd::KA.CPU, network::Network; kwargs...)
    # TODO: how should we handle bridges?

    N = num_buses(network)
    E = num_branches(network)
    i0 = network.slack_bus_index

    # Build a lazy PTDF to get factorization of nodal admittance matrix
    Φ = lazy_ptdf(bkd, network; linear_solver=:auto)

    A = sparse(branch_incidence_matrix(bkd, network))
    b = [-br.b for br in network.branches]
    At = Matrix(A')
    _M = (Φ.F \ At)
    _M[i0, :] .= 0  # ⚠ need to zero-out slack bus angle
    M = (Diagonal(b) * A) * _M
    d = inv.(1 .- diag(M))
    # zero-out entries of `d` corresponding to bridges
    is_bridge = find_bridges(network)
    d .*= (.!is_bridge)
    D = Diagonal(d)
    rmul!(M, D)  # M ← M * D

    # Set diagonal elements to -1
    # --> this ensures that post-contingency flow on tripped branch is zero
    @inbounds @simd for i in 1:E
        M[i, i] = -1
    end

    return FullLODF(N, E, M)
end

full_lodf(network::Network; kwargs...) = full_lodf(DefaultBackend(), network; kwargs...)

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

function compute_all_flows!(pfc, pf0, L::FullLODF; outages=Int[])
    N, E = L.N, L.E

    for (i, l) in enumerate(outages)
        @views pfc[:, i] .= pf0 .+ (pf0[l] .* L.matrix[:, l])
    end

    return pfc
end
