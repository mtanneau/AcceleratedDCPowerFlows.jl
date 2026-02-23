module AcceleratedDCPowerFlows

using LinearAlgebra
using SparseArrays
using SuiteSparse

using KLU

export Network
export BranchIncidenceMatrix
export num_buses, num_branches
export from_power_models
export FullPTDF, LazyPTDF
export FullLODF, LazyLODF
export compute_flow!

include("core/network.jl")
include("core/branch_incidence_matrix.jl")
include("graphs.jl")
include("ptdf/ptdf.jl")
include("lodf/lodf.jl")

function BranchIncidenceMatrix(data::Dict)
    N::Int = length(data["bus"])
    E::Int = length(data["branch"])

    bus_fr = Vector{Int}(undef, E)
    bus_to = Vector{Int}(undef, E)
    for e in 1:E
        bus_fr[e] = data["branch"]["$e"]["f_bus"]
        bus_to[e] = data["branch"]["$e"]["t_bus"]
    end
    
    # data checks
    imin, imax = extrema(bus_fr)
    1 <= imin <= imax <= N || throw(ArgumentError("bus_fr out of bounds"))
    jmin, jmax = extrema(bus_to)
    1 <= jmin <= jmax <= N || throw(ArgumentError("bus_to out of bounds"))

    return BranchIncidenceMatrix(N, E, bus_fr, bus_to)
end

end  # module
