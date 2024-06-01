using LinearAlgebra
using Random
using Test

using PowerModels
const PM = PowerModels
using PGLib

using FastPowerFlow
const FP = FastPowerFlow

@testset "FastPowerFlow" begin
    include("ptdf.jl")
    include("lodf.jl")
end
