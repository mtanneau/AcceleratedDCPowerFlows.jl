function _test_inverse_susceptance(data_pm; type)
    network = APF.from_power_models(data_pm)
    N = APF.num_buses(network)
    islack = network.slack_bus_index
    p = randn(N)

    Y, islack_built, bmin = APF._build_negated_nodal_susceptance(KA.CPU(), network)
    @test islack_built == islack

    @testset "$(linear_solver)" for linear_solver in [:auto, :SuiteSparse, :KLU]
        if type == :full
            S = APF.full_inverse_susceptance(KA.CPU(), Y, islack, bmin; linear_solver)
            @test isa(S, APF.FullInverseSusceptance)
        else
            S = APF.lazy_inverse_susceptance(KA.CPU(), Y, islack, bmin; linear_solver)
            @test isa(S, APF.LazyInverseSusceptance)
        end

        # Reference angles: Y θ = p, but Y is -Bn with slack fixed
        # So θ = Y \ p gives the raw solve; we negate and zero slack
        θ_ref = -(Matrix(Y) \ p)
        θ_ref[islack] = 0.0

        # Test compute_angles!
        θ = zeros(N)
        APF.compute_angles!(θ, p, S)
        @test isapprox(θ, θ_ref; atol=1e-10)
        @test θ[islack] ≈ 0.0 atol=1e-12

        # Test \ operator matches compute_angles!
        θ_bs = S \ p
        @test isapprox(θ_bs, θ_ref; atol=1e-10)
        @test θ_bs[islack] ≈ 0.0 atol=1e-12

        # Test getindex — column access S[:, i]
        for i in [1, islack, N]
            col = S[:, i]
            eᵢ = zeros(N)
            eᵢ[i] = 1.0
            @test isapprox(col, S \ eᵢ; atol=1e-10)
            @test col[islack] ≈ 0.0 atol=1e-12
        end

        # Test batched (matrix) RHS
        K = 4
        pb = randn(N, K)
        θ_b = S \ pb
        θ_ref_b = zeros(N, K)
        APF.compute_angles!(θ_ref_b, pb, S)
        @test isapprox(θ_b, θ_ref_b; atol=1e-10)
    end
    return nothing
end

@testset "InverseSusceptance" begin
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))

    @testset "Full" begin
        _test_inverse_susceptance(data; type=:full)
        @testset "Negative admittance" begin
            data["branch"]["1"]["br_x"] = -1.0
            _test_inverse_susceptance(data; type=:full)
        end
    end

    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))

    @testset "Lazy" begin
        _test_inverse_susceptance(data; type=:lazy)
        @testset "Negative admittance" begin
            data["branch"]["1"]["br_x"] = -1.0
            _test_inverse_susceptance(data; type=:lazy)
        end
    end
end
