function test_ptdf_full()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    network = APF.from_power_models(data)
    N = APF.num_buses(network)
    E = APF.num_branches(network)
    p = randn(N)
    f = zeros(E)

    # Reference power flows, computed with PowerModels
    Φ_pm = PM.calc_basic_ptdf_matrix(data)
    fpm = Φ_pm * p  # power flows, computed with PowerModels
    
    # Form PTDF and compute power flows
    Φ = APF.FullPTDF(network; gpu=false)
    APF.compute_flow!(f, p, Φ)  
    @test isapprox(f, fpm; atol=1e-6)

    # Check again with batched mode
    K = 4
    p = randn(N, K)
    f = zeros(E, K)
    fpm = Φ_pm * p
    APF.compute_flow!(f, p, Φ)
    @test isapprox(f, fpm; atol=1e-6)

    return nothing
end
