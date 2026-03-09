function APF.lazy_lodf(
    backend::CUDA.CUDABackend,
    network::APF.Network;
    ptdf_type=:lazy,
    linear_solver=:auto,
    kwargs...,
)
    if ptdf_type == :lazy
        Φ = APF.lazy_ptdf(backend, network; linear_solver=linear_solver, kwargs...)
    elseif ptdf_type == :full
        Φ = APF.full_ptdf(backend, network; linear_solver=linear_solver, kwargs...)
    else
        error("Invalid PTDF type: $(ptdf_type); only :lazy and :full are supported")
    end

    return APF.lazy_lodf(backend, network, Φ)
end

function APF.lazy_lodf(backend::CUDA.CUDABackend, network::APF.Network, Φ::APF.AbstractPTDF)
    A = APF.branch_incidence_matrix(backend, network)
    b = CuVector([-br.b for br in network.branches])
    return APF.LazyLODF(
        APF.num_buses(network),
        APF.num_branches(network),
        network.slack_bus_index,
        A,
        b,
        Φ,
    )
end

# GPU compute_flow! for LazyLODF with FullPTDF backend
function APF.compute_flow!(
    pfc::CUDA.CuVector,
    pf0::CUDA.CuVector,
    L::APF.LazyLODF{<:Any,<:Any,<:APF.FullPTDF},
    br::APF.Branch,
)
    i0 = L.islack
    Φ = L.Φ

    k = br.index
    i = br.bus_fr
    j = br.bus_to

    # Compute difference of columns in Yinv: (Y⁻¹)ⱼ - (Y⁻¹)ᵢ
    ξ = view(Φ.Yinv, :, j) .- view(Φ.Yinv, :, i)

    # Zero slack bus angle
    ξ[i0:i0] .= 0

    # Re-multiply by BA
    mul!(pfc, Φ.A, ξ)
    pfc .*= Φ.b
    β = 1.0 .+ view(pfc, k:k)
    pfc .*= (-view(pf0, k:k) ./ β)
    pfc .+= pf0
    view(pfc, k:k) .= 0

    return pfc
end

# GPU compute_flow! for LazyLODF with LazyPTDF backend
function APF.compute_flow!(
    pfc::CUDA.CuVector,
    pf0::CUDA.CuVector,
    L::APF.LazyLODF{<:Any,<:Any,<:APF.LazyPTDF},
    br::APF.Branch,
)
    i0 = L.islack
    Φ = L.Φ

    k = br.index
    i = br.bus_fr
    j = br.bus_to

    # Build vector with +1 at i, -1 at j, zeros elsewhere
    ak_host = zeros(eltype(pf0), L.N)
    ak_host[i] = 1
    ak_host[j] = -1
    ak = CuVector(ak_host)

    # Solve linear system: ξ = -(F \ ak)
    # where F factorizes -(AᵀBA)
    ξ = -(Φ.F \ ak)

    # Zero slack bus angle
    ξ[i0:i0] .= 0

    # Re-multiply by BA
    mul!(pfc, Φ.A, ξ)
    pfc .*= Φ.b
    β = 1.0 .+ view(pfc, k:k)
    pfc .*= (-view(pf0, k:k) ./ β)
    pfc .+= pf0
    view(pfc, k:k) .= 0

    return pfc
end
