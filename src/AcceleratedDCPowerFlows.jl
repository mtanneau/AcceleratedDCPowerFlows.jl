module AcceleratedDCPowerFlows

using LinearAlgebra
using SparseArrays
using SuiteSparse

using Graphs

import KernelAbstractions as KA
using KernelAbstractions: get_backend

using KLU

export Network
export BranchIncidenceMatrix
export num_buses, num_branches
export from_power_models
export ptdf, full_ptdf, lazy_ptdf
export lodf, full_lodf, lazy_lodf
export compute_flow!

# Some global definitions
DefaultBackend() = KA.CPU()

include("core/network.jl")
include("core/branch_incidence_matrix.jl")
include("graph/bridges.jl")
include("ptdf/ptdf.jl")
include("lodf/lodf.jl")

end  # module
