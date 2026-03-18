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

# Load test utilities
using ..TestMatrixUtilities
using ..TestPTDFUtilities

function runtests()
    for name in names(@__MODULE__; all=true)
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
    A_ref = PM.calc_basic_incidence_matrix(data)

    # Test with CUDA backend
    _test_branch_incidence_matrix(CUDA.CUDABackend(), network, A_ref)

    return nothing
end

function test_branch_susceptance_matrix()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    network = APF.from_power_models(data)
    A_ref = PM.calc_basic_branch_susceptance_matrix(data)

    # Test with CUDA backend
    _test_branch_susceptance_matrix(CUDA.CUDABackend(), network, A_ref)

    return nothing
end

function test_nodal_susceptance_matrix()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    network = APF.from_power_models(data)
    A_ref = PM.calc_basic_susceptance_matrix(data)

    # Test with CUDA backend
    _test_nodal_susceptance_matrix(CUDA.CUDABackend(), network, A_ref)

    return nothing
end

include("ptdf.jl")
include("lodf.jl")

end

if !isinteractive()
    TestCUDAExt.runtests()
end
