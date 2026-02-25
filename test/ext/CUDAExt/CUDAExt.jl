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

using CUDA
using CUDSS

function runtests()
    if !CUDA.functional()
        @info "Non-CUDA machine, skipping CUDAExt tests"
        @test_skip true
        return nothing
    end
    
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

    # Reference implementation
    A_pm = PM.calc_basic_incidence_matrix(data)
    x = rand(N)
    y_pm = A_pm * x
    y_dev = CuArray(zeros(E))
    x_dev = CuArray(x)
    LinearAlgebra.mul!(y_dev, A, x_dev)
    @test collect(y_dev) ≈ y_pm

    x = rand(N, 3)
    y_pm = A_pm * x
    y_dev = CuArray(zeros(E, 3))
    x_dev = CuArray(x)
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
