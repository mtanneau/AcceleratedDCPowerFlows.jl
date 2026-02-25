function test_full_lodf()
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
    L = APF.FullLODF(network)

    pf_pm = zeros(E, K)
    pf_fp = zeros(E, K)

    APF.compute_all_flows!(pf_fp, pf0, L; outages=outages)

    for (i, k) in enumerate(outages)
        br = data["branch"]["$k"]
        _r = br["br_r"]

        # Set branch resistance to Inf --> will zero-out the flow
        br["br_r"] = Inf
        # Re-compute power flows
        Φk = PM.calc_basic_ptdf_matrix(data)
        @views mul!(pf_pm[:, i], Φk, p)

        @test isapprox(pf_pm[:, i], pf_fp[:, i], atol=1e-6)

        # reset branch resistance
        br["br_r"] = _r
    end

    return nothing
end
