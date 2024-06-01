function test_full_lodf()
    data = PM.make_basic_network(pglib("pglib_opf_case5_pjm"))
    N = length(data["bus"])
    E = length(data["branch"])
    p = real.(calc_basic_bus_injection(data))

    is_bridge = FP.find_bridges(data)
    outages = collect(1:E)[.! is_bridge]
    K = length(outages)

    Φ = calc_basic_ptdf_matrix(data)
    pf0 = Φ * p

    # Compute LODF
    L = FP.FullLODF(data)

    pf_pm = zeros(E, K)
    pf_fp = zeros(E, K)

    FP.compute_all_flows!(pf_fp, p, pf0, L; outages=outages)

    for (i, k) in enumerate(outages)
        br = data["branch"]["$k"]
        _r = br["br_r"]

        # Set branch resistance to Inf --> will zero-out the flow
        br["br_r"] = Inf
        # Re-compute power flows
        Φk = calc_basic_ptdf_matrix(data)
        @views mul!(pf_pm[:, i], Φk, p)

        @test isapprox(pf_pm[:, i], pf_fp[:, i], atol=1e-6)

        # reset branch resistance
        br["br_r"] = _r
    end

    return nothing
end

function test_lazy_lodf()
    data = PM.make_basic_network(pglib("pglib_opf_case5_pjm"))
    N = length(data["bus"])
    E = length(data["branch"])
    p = real.(calc_basic_bus_injection(data))

    is_bridge = FP.find_bridges(data)
    outages = collect(1:E)[.! is_bridge]
    K = length(outages)

    Φ = calc_basic_ptdf_matrix(data)

    # Compute LODF
    L = FP.LazyLODF(data)
    pf_fp = zeros(E)

    for (i, k) in enumerate(outages)
        br = data["branch"]["$k"]
        _r = br["br_r"]

        # Set branch resistance to Inf --> will zero-out the flow
        br["br_r"] = Inf
        # Re-compute power flows
        Φk = calc_basic_ptdf_matrix(data)
        pf_pm = Φk * p
        
        FP.compute_flow!(pf_fp, p, L, k)

        @test isapprox(pf_pm, pf_fp, atol=1e-6)

        # reset branch resistance
        br["br_r"] = _r
    end

    return nothing
end

@testset "LODF" begin
    @testset test_full_lodf()
    @testset test_lazy_lodf()
end
