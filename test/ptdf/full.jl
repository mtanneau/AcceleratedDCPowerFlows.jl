function test_ptdf_full()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    network = AcceleratedDCPowerFlows.from_power_models(data)
    N = num_buses(network)
    E = num_branches(network)
    p = randn(N)
    f = zeros(E)

    # Reference power flows, computed with PowerModels
    Φ_pm = PM.calc_basic_ptdf_matrix(data)
    fpm = Φ_pm * p  # power flows, computed with PowerModels
    
    # Form PTDF and compute power flows
    Φ = FP.FullPTDF(network; gpu=false)
    FP.compute_flow!(f, p, Φ)  
    @test isapprox(f, fpm; atol=1e-6)

    # Check again with batched mode
    K = 4
    p = randn(N, K)
    f = zeros(E, K)
    fpm = Φ_pm * p
    FP.compute_flow!(f, p, Φ)
    @test isapprox(f, fpm; atol=1e-6)

    return nothing
end
