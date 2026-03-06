# Check if CUDA is available on this machine
# if yes, run the test suite
# if not, skip and just keep a dummy test
module TestCUDAExt

using LinearAlgebra
using SparseArrays
using Test

import PowerModels as PM
using PGLib

import AcceleratedDCPowerFlows as APF
import KernelAbstractions as KA

using CUDA
using CUDSS

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return nothing
end

function test_branch_incidence_matrix()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    network = APF.from_power_models(data)
    N = length(data["bus"])
    E = length(data["branch"])

    A = APF.branch_incidence_matrix(CUDA.CUDABackend(), network)
    @test A.N == N
    @test A.E == E
    backend = KA.get_backend(A)
    @test isa(backend, CUDA.CUDABackend)

    A_csr = sparse(A)
    @test isa(A_csr, CUDA.CUSPARSE.CuSparseMatrixCSR)

    # Reference implementation
    A_pm = PM.calc_basic_incidence_matrix(data)

    # Test specialized implementation (will default to KA if none exists)
    # These tests should never error
    x_dev = CUDA.rand(Float64, N)
    y_dev = CUDA.rand(Float64, E)
    LinearAlgebra.mul!(y_dev, A, x_dev)
    y_pm = A_pm * collect(x_dev)
    @test collect(y_dev) ≈ y_pm

    x_dev = CUDA.rand(Float64, (N, 2))
    y_dev = CUDA.rand(Float64, (E, 2))
    LinearAlgebra.mul!(y_dev, A, x_dev)
    y_pm = A_pm * collect(x_dev)
    @test collect(y_dev) ≈ y_pm

    # Make sure we also check KA kernels
    # These tests ensure that the KA kernels
    #   execute properly on a CUDA backend
    x_dev = CUDA.rand(Float64, N)
    y_dev = CUDA.rand(Float64, E)
    invoke(
        APF._unsafe_mul!, 
        Tuple{KA.Backend,AbstractVecOrMat,APF.BranchIncidenceMatrix,AbstractVecOrMat}, 
        backend, 
        y_dev, 
        A, 
        x_dev,
    )
    y_pm = A_pm * collect(x_dev)
    @test collect(y_dev) ≈ y_pm

    x_dev = CUDA.rand(Float64, (N, 2))
    y_dev = CUDA.rand(Float64, (E, 2))
    invoke(
        APF._unsafe_mul!, 
        Tuple{KA.Backend,AbstractVecOrMat,APF.BranchIncidenceMatrix,AbstractVecOrMat}, 
        backend, 
        y_dev, 
        A, 
        x_dev,
    )
    y_pm = A_pm * collect(x_dev)
    @test collect(y_dev) ≈ y_pm

    return nothing
end

function test_branch_susceptance_matrix()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    network = APF.from_power_models(data)
    N = length(data["bus"])
    E = length(data["branch"])

    A = APF.branch_susceptance_matrix(CUDA.CUDABackend(), network)
    @test size(A) == (E, N)
    @test size(A, 1) == E
    @test size(A, 2) == N
    @test size(A, 3) == size(A, 4) == 1
    @test_throws ErrorException size(A, 0)
    backend = KA.get_backend(A)
    @test isa(backend, CUDA.CUDABackend)

    A_csr = sparse(A)
    @test isa(A_csr, CUDA.CUSPARSE.CuSparseMatrixCSR)

    # Reference implementation
    A_pm = PM.calc_basic_branch_susceptance_matrix(data)

    # Check matvec and matmat products
    x_dev = CUDA.rand(Float64, N)
    y_dev = CUDA.rand(Float64, E)
    LinearAlgebra.mul!(y_dev, A, x_dev)
    y_pm = A_pm * collect(x_dev)
    @test collect(y_dev) ≈ y_pm

    x_dev = CUDA.rand(Float64, (N, 2))
    y_dev = CUDA.rand(Float64, (E, 2))
    LinearAlgebra.mul!(y_dev, A, x_dev)
    y_pm = A_pm * collect(x_dev)
    @test collect(y_dev) ≈ y_pm

        # Make sure we also check KA kernels
    # These tests ensure that the KA kernels
    #   execute properly on a CUDA backend
    x_dev = CUDA.rand(Float64, N)
    y_dev = CUDA.rand(Float64, E)
    invoke(
        APF._unsafe_mul!, 
        Tuple{KA.Backend,AbstractVecOrMat,APF.BranchSusceptanceMatrix,AbstractVecOrMat}, 
        backend, 
        y_dev, 
        A, 
        x_dev,
    )
    y_pm = A_pm * collect(x_dev)
    @test collect(y_dev) ≈ y_pm

    x_dev = CUDA.rand(Float64, (N, 2))
    y_dev = CUDA.rand(Float64, (E, 2))
    invoke(
        APF._unsafe_mul!, 
        Tuple{KA.Backend,AbstractVecOrMat,APF.BranchSusceptanceMatrix,AbstractVecOrMat}, 
        backend, 
        y_dev, 
        A, 
        x_dev,
    )
    y_pm = A_pm * collect(x_dev)
    @test collect(y_dev) ≈ y_pm

    return nothing
end

function test_nodal_susceptance_matrix()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    network = APF.from_power_models(data)
    N = length(data["bus"])
    E = length(data["branch"])

    A = APF.nodal_susceptance_matrix(CUDA.CUDABackend(), network)
    @test A.N == N
    @test A.E == E
    @test size(A) == (N, N)
    @test size(A, 1) == N
    @test size(A, 2) == N
    @test size(A, 3) == size(A, 4) == 1
    @test_throws ErrorException size(A, 0)

    A_csr = sparse(A)
    @test isa(A_csr, CUDA.CUSPARSE.CuSparseMatrixCSR)

    # Reference implementation
    A_pm = PM.calc_basic_susceptance_matrix(data)

    # Check matvec and matmat products
    x = rand(N)
    y_pm = A_pm * x
    x_dev = CUDA.CuArray(x)
    y_dev = CUDA.CuArray(zeros(N))
    LinearAlgebra.mul!(y_dev, A, x_dev)
    @test collect(y_dev) ≈ y_pm

    x = rand(N, 3)
    y_pm = A_pm * x
    x_dev = CUDA.CuArray(x)
    y_dev = CUDA.CuArray(zeros(N, 3))
    LinearAlgebra.mul!(y_dev, A, x_dev)
    @test collect(y_dev) ≈ y_pm

    return nothing
end

include("ptdf.jl")
include("lodf.jl")

end

if !isinteractive()
    TestCUDAExt.runtests()
end
