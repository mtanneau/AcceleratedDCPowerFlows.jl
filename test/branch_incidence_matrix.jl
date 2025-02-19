function test_branch_incidence_matrix()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    N = length(data["bus"])
    E = length(data["branch"])

    A = FP.BranchIncidenceMatrix(data)
    @test A.N == N
    @test A.E == E

    # Reference implementation
    A_pm = calc_basic_incidence_matrix(data)
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