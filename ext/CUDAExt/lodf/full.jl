function APF.full_lodf(
    backend::CUDA.CUDABackend,
    network::APF.Network;
    linear_solver=:auto,
    kwargs...,
)
    N = APF.num_buses(network)
    E = APF.num_branches(network)
    i0 = network.slack_bus_index

    # Reuse CUDA lazy PTDF to get GPU linear-system factorization
    Φ = APF.lazy_ptdf(backend, network; linear_solver=linear_solver, kwargs...)

    # Build Aᵀ as a dense matrix on GPU
    A = APF.branch_incidence_matrix(backend, network)
    At_dense = CuMatrix(sparse(sparse(A)'))

    _M = Φ.F \ At_dense
    _M[i0, :] .= 0

    M = CUDA.zeros(eltype(_M), E, E)
    mul!(M, A, _M)
    M .*= Φ.b

    d = inv.(1 .- diag(M))

    # Zero-out bridge contingencies
    is_bridge = APF.find_bridges(network)
    d .*= CuVector(eltype(d).(.!is_bridge))

    # M <- M * Diagonal(d) (scale columns)
    M .*= reshape(d, 1, E)

    # Ensure tripped branch has zero post-contingency flow
    M[diagind(M)] .= -1

    return APF.FullLODF(N, E, M)
end

function APF.compute_flow!(
    pfc::CUDA.CuVector,
    pf0::CUDA.CuVector,
    L::APF.FullLODF,
    br::APF.Branch,
)
    c = br.index
    copyto!(pfc, pf0)
    pfc .+= view(L.matrix, :, c) .* view(pf0, c:c)
    return pfc
end
