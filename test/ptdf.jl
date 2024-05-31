function test_ptdf_full()
    data = PM.make_basic_network(pglib("14_ieee"))
    N = length(data["bus"])
    E = length(data["branch"])
    Φ_pm = PM.calc_basic_ptdf_matrix(data)
    p = randn(N)
    f = zeros(E)
    
    Φ = FP.FullPTDF(data)
    fpm = Φ_pm * p
    FP.compute_flow!(f, p, Φ)
    @test isapprox(f, fpm; atol=1e-6)

    return nothing
end

function test_ptdf_lazy()
    data = PM.make_basic_network(pglib("14_ieee"))
    N = length(data["bus"])
    E = length(data["branch"])

    p = randn(N)
    f = zeros(E)
    Φ_pm = PM.calc_basic_ptdf_matrix(data)
    fpm = Φ_pm * p
    
    @testset "$solver" for solver in [:cholesky, :ldlt, :lu, :klu]
        Φ = FP.LazyPTDF(data; solver)

        # Check power flow computation
        FP.compute_flow!(f, p, Φ)
        @test isapprox(f, fpm; atol=1e-6)
    end

    return nothing
end

@testset "PTDF" begin
    @testset test_ptdf_full()
    @testset test_ptdf_lazy()
end
