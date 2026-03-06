using LinearAlgebra
using Random
using SparseArrays
using Test

import PowerModels as PM
PM.silence()
using PGLib

import AcceleratedDCPowerFlows as APF
import KernelAbstractions as KA

@testset "AcceleratedDCPowerFlows" begin
    @testset "core" begin
        include("core/network.jl")
        include("core/branch_incidence_matrix.jl")
        include("core/branch_susceptance_matrix.jl")
        include("core/nodal_susceptance_matrix.jl")
    end

    @testset "graph" begin
        include("graph/bridges.jl")
    end

    @testset "PTDF" begin
        include("ptdf/ptdf.jl")
    end

    @testset "LODF" begin
        include("lodf/lodf.jl")
    end
end

@testset "Extensions" begin
    @testset "CUDAExt" begin
        run_cuda_tests = try
            using CUDA
            CUDA.functional()
        catch err
            # Something went wrong, skip the tests
            false
        end

        if run_cuda_tests
            include("ext/CUDAExt/CUDAExt.jl")
        else
            @info "CUDA not functional, skipping tests"
            @test_skip true
        end
    end
end
