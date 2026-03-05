include("full.jl")
include("lazy.jl")

function test_ptdf_entry_points()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    network = APF.from_power_models(data)

    M = APF.ptdf(network; ptdf_type=:full)
    @test isa(M, APF.FullPTDF)
    @test KA.get_backend(M) == APF.default_backend()

    M = APF.ptdf(network; ptdf_type=:lazy)
    @test isa(M, APF.LazyPTDF)
    @test KA.get_backend(M) == APF.default_backend()

    @test_throws ErrorException APF.ptdf(network; ptdf_type=:other)

    @test_throws MethodError APF.full_ptdf(network; ptdf_type=:lazy)
    @test_throws MethodError APF.lazy_ptdf(network; ptdf_type=:full)

    # Additional type inference tests
    @inferred APF.FullPTDF APF.full_ptdf(network)
    @inferred APF.LazyPTDF APF.lazy_ptdf(network)
end

@testset "PTDF" begin
    @testset test_ptdf_full()
    @testset test_ptdf_lazy()
    @testset test_ptdf_entry_points()
end
