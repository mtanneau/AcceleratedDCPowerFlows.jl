@testset "net_injection" begin
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    network = APF.from_power_models(data)

    p = APF.net_injection(network)

    # PM.calc_basic_bus_injection returns ComplexF64; compare real parts only (DC assumption)
    p_pm = real.(PM.calc_basic_bus_injection(data))
    @test isapprox(p, p_pm; atol=1e-10)
end

function _test_dcpf(data_pm; dcpf_type)
    network = APF.from_power_models(data_pm)
    N = APF.num_buses(network)
    E = APF.num_branches(network)
    islack = network.slack_bus_index
    p = randn(N)

    # Reference: PTDF-based flows from PowerModels
    Φ_pm = PM.calc_basic_ptdf_matrix(data_pm)
    fpm = Φ_pm * p

    @testset "$(linear_solver)" for linear_solver in [:auto, :SuiteSparse, :KLU]
        D = APF.dcpf(network; dcpf_type, linear_solver)

        # Test compute_flow! with θ output — checks both flows and angles
        θ = zeros(N)
        f = zeros(E)
        APF.compute_flow!(f, p, D, θ)
        @test isapprox(f, fpm; atol=1e-6)
        @test θ[islack] ≈ 0.0 atol=1e-12

        # Batched mode
        K = 4
        pb = randn(N, K)
        fb = zeros(E, K)
        fpm_b = Φ_pm * pb
        APF.compute_flow!(fb, pb, D)
        @test isapprox(fb, fpm_b; atol=1e-6)
    end
    return nothing
end

@testset "DCPF" begin
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))

    @testset "Full" begin
        _test_dcpf(data; dcpf_type=:full)
        @testset "Negative admittance" begin
            data["branch"]["1"]["br_x"] = -1.0
            _test_dcpf(data; dcpf_type=:full)
        end
    end

    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))

    @testset "Lazy" begin
        _test_dcpf(data; dcpf_type=:lazy)
        @testset "Negative admittance" begin
            data["branch"]["1"]["br_x"] = -1.0
            _test_dcpf(data; dcpf_type=:lazy)
        end
    end
end
