include("full.jl")
include("lazy.jl")


function test_lodf_entry_points()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    network = APF.from_power_models(data)

    M = APF.lodf(network; lodf_type=:full)
    @test isa(M, APF.FullLODF)
    @test KA.get_backend(M) == APF.default_backend()

    M = APF.lodf(network; lodf_type=:lazy)
    @test isa(M, APF.LazyLODF)
    @test KA.get_backend(M) == APF.default_backend()

    @test_throws ErrorException APF.lodf(network; lodf_type=:other)

    # Additional type inference tests
    @inferred APF.FullLODF APF.full_lodf(network)
    @inferred APF.LazyLODF APF.lazy_lodf(network)
end

@testset "LODF" begin
    @testset test_full_lodf()
    @testset test_lazy_lodf()
    @testset test_lodf_entry_points()
end
