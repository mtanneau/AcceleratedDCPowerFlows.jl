module FastPowerFlow

using BenchmarkTools
using DataFrames
using LinearAlgebra
using SparseArrays
using Statistics
using SuiteSparse

# PowerModels
using PowerModels
const PM = PowerModels
using PGLib

# CUDA stuff
using CUDA
using CUDSS
using KLU

function __init__()
    PM.silence()
end

include("graphs.jl")
include("ptdf.jl")

end # module FastPowerFlow
