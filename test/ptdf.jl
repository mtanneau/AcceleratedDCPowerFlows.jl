function test_ptdf_full()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    N = length(data["bus"])
    E = length(data["branch"])
    p = randn(N)
    f = zeros(E)

    # Reference power flows, computed with PowerModels
    Φ_pm = PM.calc_basic_ptdf_matrix(data)
    fpm = Φ_pm * p  # power flows, computed with PowerModels
    
    # Form PTDF and compute power flows
    Φ = FP.FullPTDF(data; gpu=false)
    FP.compute_flow!(f, p, Φ)  
    @test isapprox(f, fpm; atol=1e-6)

    # Check again with batched mode
    K = 4
    p = randn(N, K)
    f = zeros(E, K)
    fpm = Φ_pm * p
    FP.compute_flow!(f, p, Φ)
    @test isapprox(f, fpm; atol=1e-6)

    return nothing
end

function test_ptdf_lazy()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    N = length(data["bus"])
    E = length(data["branch"])

    p = randn(N)
    f = zeros(E)

    # Reference power flows, computed with PowerModels
    Φ_pm = PM.calc_basic_ptdf_matrix(data)
    fpm = Φ_pm * p

    K = 4
    p_batch = randn(N, K)
    f_batch = zeros(E, K)
    f_batch_pm = Φ_pm * p_batch
    
    @testset "$solver" for solver in [:cholesky, :ldlt, :lu, :klu]
        Φ = FP.LazyPTDF(data; solver, gpu=false)

        # Check our power flows against PowerModels
        FP.compute_flow!(f, p, Φ)
        @test isapprox(f, fpm; atol=1e-6)

        # Check batched mode
        FP.compute_flow!(f_batch, p_batch, Φ)
        @test isapprox(f_batch, f_batch_pm; atol=1e-6)
    end

    return nothing
end

@testset "PTDF" begin
    @testset test_ptdf_full()
    @testset test_ptdf_lazy()
end
