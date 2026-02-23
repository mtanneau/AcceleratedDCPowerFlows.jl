include("full.jl")
include("lazy.jl")

@testset "PTDF" begin
    @testset test_ptdf_full()
    @testset test_ptdf_lazy()
end
