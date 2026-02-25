using LinearAlgebra
using Random
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
        include("ext/CUDAExt/CUDAExt.jl")
    end
end
