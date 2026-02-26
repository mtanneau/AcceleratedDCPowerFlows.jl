function test_nodal_susceptance_matrix()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    network = APF.from_power_models(data)
    N = length(data["bus"])
    E = length(data["branch"])

    A = APF.nodal_susceptance_matrix(network)
    @test A.N == N
    @test A.E == E
    @test size(A) == (N, N)
    @test size(A, 1) == N
    @test size(A, 2) == N
    @test size(A, 3) == size(A, 4) == 1
    @test_throws ErrorException size(A, 0)

    # Reference implementation
    A_pm = PM.calc_basic_susceptance_matrix(data)
    @test sparse(A) ≈ A_pm

    # Check matvec and matmat products
    x = rand(N)
    y_pm = A_pm * x
    y = zeros(N)
    LinearAlgebra.mul!(y, A, x)
    @test y ≈ y_pm

    x = rand(N, 3)
    y_pm = A_pm * x
    y = zeros(N, 3)
    LinearAlgebra.mul!(y, A, x)
    @test y ≈ y_pm

    return nothing
end

@testset test_nodal_susceptance_matrix()