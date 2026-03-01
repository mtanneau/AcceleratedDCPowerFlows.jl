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
export net_injection

# Some global definitions
DefaultBackend() = KA.CPU()

include("core/network.jl")
include("graph/bridges.jl")
include("ptdf/ptdf.jl")
include("lodf/lodf.jl")
include("dcpf/dcpf.jl")

end  # module
