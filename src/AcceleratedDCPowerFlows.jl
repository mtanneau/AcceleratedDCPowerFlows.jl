module AcceleratedDCPowerFlows

import Base.size
using LinearAlgebra
import LinearAlgebra.mul!
using SparseArrays
import SparseArrays: sparse
using SuiteSparse

using Graphs

import KernelAbstractions as KA
using KernelAbstractions: get_backend

using KLU

export Network
export num_buses, num_branches
export branch_incidence_matrix
export branch_susceptance_matrix
export from_power_models
export ptdf, full_ptdf, lazy_ptdf
export lodf, full_lodf, lazy_lodf
export dcpf, full_dcpf, lazy_dcpf
export compute_flow!, solve!

# Some global definitions
"""
    default_backend()

Default backend, currently equivalent to `KernelAbstractions.CPU()`.
"""
default_backend() = KA.CPU()

include("core/network.jl")
include("graph/bridges.jl")
include("inverse_susceptance/inverse_susceptance.jl")
include("ptdf/ptdf.jl")
include("lodf/lodf.jl")

end  # module
