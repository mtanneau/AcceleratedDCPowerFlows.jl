include("full.jl")
include("lazy.jl")


function test_lodf_entry_points()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    network = APF.from_power_models(data)

    M = APF.lodf(network; lodf_type=:full)
    @test isa(M, APF.FullLODF)

    M = APF.lodf(network; lodf_type=:lazy)
    @test isa(M, APF.LazyLODF)

    @test_throws ErrorException APF.lodf(network; lodf_type=:other)
end

@testset "LODF" begin
    @testset test_full_lodf()
    @testset test_lazy_lodf()
    @testset test_lodf_entry_points()
end
