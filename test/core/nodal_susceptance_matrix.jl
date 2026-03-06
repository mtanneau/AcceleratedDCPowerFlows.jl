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
    backend = KA.get_backend(A)
    @test isa(backend, typeof(APF.default_backend()))

    # Reference implementation
    A_pm = PM.calc_basic_susceptance_matrix(data)
    @test sparse(A) ≈ A_pm

    # Check matvec and matmat products
    # These will dispatch to a CPU-specific implementation
    x = rand(N)
    y_pm = A_pm * x
    y = rand(N)
    LinearAlgebra.mul!(y, A, x)
    @test y ≈ y_pm
    # Trigger backend-agnostic KA kernels
    y = rand(N)
    invoke(APF._unsafe_mul!, Tuple{KA.Backend,AbstractVecOrMat,APF.NodalSusceptanceMatrix,AbstractVecOrMat}, backend, y, A, x)
    @test y ≈ y_pm 

    x = rand(N, 2)
    y_pm = A_pm * x
    y = rand(N, 2)
    LinearAlgebra.mul!(y, A, x)
    @test y ≈ y_pm
    # Trigger backend-agnostic KA kernels
    y = rand(N, 2)
    invoke(APF._unsafe_mul!, Tuple{KA.Backend,AbstractVecOrMat,APF.NodalSusceptanceMatrix,AbstractVecOrMat}, backend, y, A, x)
    @test y ≈ y_pm 

    return nothing
end

@testset test_nodal_susceptance_matrix()
