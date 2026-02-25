function test_branch_incidence_matrix()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    network = APF.from_power_models(data)
    N = length(data["bus"])
    E = length(data["branch"])

    A = APF.branch_incidence_matrix(network)
    @test size(A) == (E, N)
    @test size(A, 1) == E
    @test size(A, 2) == N
    @test size(A, 3) == size(A, 4) == 1
    @test_throws ErrorException size(A, 0)

    # Reference implementation
    A_pm = PM.calc_basic_incidence_matrix(data)
    x = rand(N)
    y_pm = A_pm * x
    y = zeros(E)
    LinearAlgebra.mul!(y, A, x)
    @test y ≈ y_pm

    x = rand(N, 3)
    y_pm = A_pm * x
    y = zeros(E, 3)
    LinearAlgebra.mul!(y, A, x)
    @test y ≈ y_pm

    return nothing
end

@testset test_branch_incidence_matrix()