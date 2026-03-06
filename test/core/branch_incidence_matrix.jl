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
    @test sparse(A) ≈ A_pm

    # Check matvec and matmat products
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

# This test is designed to trigger the generic KA kernels
# by using a dummy backend for which no specialized code exists
function test_branch_incidence_matrix_kernel()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    network = APF.from_power_models(data)
    N = length(data["bus"])
    E = length(data["branch"])

    A_dev = APF.branch_incidence_matrix(MyBackend(), network)
    @test KA.get_backend(A_dev) == MyBackend()

    # Reference implementation
    A_pm = PM.calc_basic_incidence_matrix(data)

    # Check matvec and matmat products
    x = rand(N)
    y_pm = A_pm * x
    x_dev = MyArray(copy(x))
    y_dev = MyArray(zeros(E))
    LinearAlgebra.mul!(y_dev, A_dev, x_dev)
    @test y_dev ≈ y_pm

    x = rand(N, 3)
    y_pm = A_pm * x
    x_dev = MyArray(copy(x))
    y_dev = MyArray(zeros(E, 3))
    LinearAlgebra.mul!(y_dev, A_dev, x_dev)
    @test y_dev ≈ y_pm

    return nothing
end

@testset test_branch_incidence_matrix()
@testset test_branch_incidence_matrix_kernel()