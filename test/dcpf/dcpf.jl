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

        # Test \ operator matches solve!
        θ_bs = D \ p
        θ_ref = zeros(N)
        APF.solve!(θ_ref, p, D)
        @test isapprox(θ_bs, θ_ref; atol=1e-12)
        @test θ_bs[islack] ≈ 0.0 atol=1e-12

        # Test batched \ operator
        θ_bs_b = D \ pb
        θ_ref_b = zeros(N, K)
        APF.solve!(θ_ref_b, pb, D)
        @test isapprox(θ_bs_b, θ_ref_b; atol=1e-12)

        # Test getindex — column access S[:, i]
        for i in [1, islack, N]
            col = D[:, i]
            eᵢ = zeros(N)
            eᵢ[i] = 1.0
            @test isapprox(col, D \ eᵢ; atol=1e-10)
            @test col[islack] ≈ 0.0 atol=1e-12
        end
    end
    return nothing
end

function test_dcpf_entry_points()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    network = APF.from_power_models(data)

    D = APF.dcpf(network; dcpf_type=:full)
    @test isa(D, APF.FullInverseSusceptance)
    @test KA.get_backend(D) == APF.default_backend()

    D = APF.dcpf(network; dcpf_type=:lazy)
    @test isa(D, APF.LazyInverseSusceptance)
    @test KA.get_backend(D) == APF.default_backend()

    @test_throws ErrorException APF.dcpf(network; dcpf_type=:other)

    @test_throws MethodError APF.full_dcpf(network; dcpf_type=:lazy)
    @test_throws MethodError APF.lazy_dcpf(network; dcpf_type=:full)

    @inferred APF.FullInverseSusceptance APF.full_dcpf(network)
    @inferred APF.LazyInverseSusceptance APF.lazy_dcpf(network)
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

    @testset test_dcpf_entry_points()
end
