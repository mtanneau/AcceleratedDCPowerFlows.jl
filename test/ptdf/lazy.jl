function test_ptdf_lazy()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))

    _test_ptdf_lazy(data)

    # One more test with a branch that has negative susceptance
    @testset "Negative admittance" begin
        data["branch"]["1"]["br_x"] = -1.0
        _test_ptdf_lazy(data)
    end

    return nothing
end

function _test_ptdf_lazy(data_pm)
    network = APF.from_power_models(data_pm)
    N = APF.num_buses(network)
    E = APF.num_branches(network)
    p = randn(N)
    f = zeros(E)

    # Reference power flows, computed with PowerModels
    Φ_pm = PM.calc_basic_ptdf_matrix(data_pm)
    fpm = Φ_pm * p  # power flows, computed with PowerModels

    @testset "$(linear_solver)" for linear_solver in [:auto, :SuiteSparse, :KLU]
        # Form PTDF and compute power flows
        Φ = APF.ptdf(network; ptdf_type=:lazy, linear_solver=linear_solver)
        APF.compute_flow!(f, p, Φ)  
        @test isapprox(f, fpm; atol=1e-6)

        # Check again with batched mode
        K = 4
        p = randn(N, K)
        f = zeros(E, K)
        fpm = Φ_pm * p
        APF.compute_flow!(f, p, Φ)
        @test isapprox(f, fpm; atol=1e-6)
    end
    return nothing
end