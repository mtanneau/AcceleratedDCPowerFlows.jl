include("full.jl")
include("lazy.jl")

function test_ptdf_entry_points()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    network = APF.from_power_models(data)

    M = APF.ptdf(network; ptdf_type=:full)
    @test isa(M, APF.FullPTDF)

    M = APF.ptdf(network; ptdf_type=:lazy)
    @test isa(M, APF.LazyPTDF)

    @test_throws ErrorException APF.ptdf(network; ptdf_type=:other)

    @test_throws MethodError APF.full_ptdf(network; ptdf_type=:lazy)
    @test_throws MethodError APF.lazy_ptdf(network; ptdf_type=:full) 
end

@testset "PTDF" begin
    @testset test_ptdf_full()
    @testset test_ptdf_lazy()
    @testset test_ptdf_entry_points()
end
