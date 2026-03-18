function test_ptdf_full()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))

    _test_ptdf_full(data)

    # One more test with a branch that has negative susceptance
    @testset "Negative admittance" begin
        data["branch"]["1"]["br_x"] = -1.0
        _test_ptdf_full(data)
    end

    return nothing
end

function _test_ptdf_full(data_pm)
    network = APF.from_power_models(data_pm)
    N = APF.num_buses(network)
    E = APF.num_branches(network)

    # Reference power flows, computed with PowerModels
    Φ_pm = PM.calc_basic_ptdf_matrix(data_pm)

    @testset "$(linear_solver)" for linear_solver in [:auto, :SuiteSparse, :KLU]
        Φ = APF.ptdf(network; ptdf_type=:full, linear_solver=linear_solver)
        @test isa(Φ, APF.FullPTDF)
        @test Φ.E == E
        @test Φ.N == N

        @test size(Φ.A) == (E, N)
        @test size(Φ.b) == (E,)

        @test KA.get_backend(Φ) == APF.default_backend()
        _test_ptdf_matrix(Φ, Φ_pm)
    end

    return nothing
end
