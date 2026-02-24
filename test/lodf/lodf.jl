include("full.jl")
include("lazy.jl")

@testset "LODF" begin
    @testset test_full_lodf()
    @testset test_lazy_lodf()
end
