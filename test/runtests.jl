using LinearAlgebra
using Random
using Test

using PowerModels
const PM = PowerModels
using PGLib

using AcceleratedDCPowerFlows
const FP = AcceleratedDCPowerFlows

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
