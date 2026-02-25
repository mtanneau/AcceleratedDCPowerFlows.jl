function test_lazy_lodf()
    @testset "Full PTDF" _test_lazy_lodf(ptdf_type=:full)
    @testset "Lazy PTDF" _test_lazy_lodf(ptdf_type=:lazy)
end

function _test_lazy_lodf(; ptdf_type)
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    network = APF.from_power_models(data)
    N = length(data["bus"])
    E = length(data["branch"])
    p = real.(PM.calc_basic_bus_injection(data))

    is_bridge = APF.find_bridges(network)
    outages = collect(1:E)[.! is_bridge]
    K = length(outages)

    Φ = PM.calc_basic_ptdf_matrix(data)
    pf0 = Φ * p

    # Compute LODF
    L = APF.LazyLODF(network)
    pf_fp = zeros(E)

    for (i, k) in enumerate(outages)
        br = data["branch"]["$k"]
        _r = br["br_r"]

        # Set branch resistance to Inf --> will zero-out the flow
        br["br_r"] = Inf
        # Re-compute power flows
        Φk = PM.calc_basic_ptdf_matrix(data)
        pf_pm = Φk * p
        
        APF.compute_flow!(pf_fp, pf0, L, network.branches[k])

        @test isapprox(pf_pm, pf_fp, atol=1e-6)

        # reset branch resistance
        br["br_r"] = _r
    end

    return nothing
end
