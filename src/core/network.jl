"""
    Bus

Represents a bus (i.e. a node) in the network.
"""
mutable struct Bus
    index::Int
    status::Bool

    pd::Float64
end

"""
    Branch

Represents a branch, i.e., either a line or transformer in the network.
"""
mutable struct Branch
    index::Int
    status::Bool

    b::Float64
    pmax::Float64
    bus_fr::Int
    bus_to::Int
end

struct Network
    buses::Vector{Bus}
    slack_bus_index::Int  # index of slack bus

    branches::Vector{Branch}
end

num_buses(network::Network) = length(network.buses)
num_branches(network::Network) = length(network.branches)

"""
    from_power_models(pmdata::Dict)

Build `Network` struct from PowerModels data dictionary.
"""
function from_power_models(pmdata::Dict)
    N = length(pmdata["bus"])
    E = length(pmdata["branch"])

    # Start by extracting buses
    buses = Bus[]
    for pm_bus in values(pmdata["bus"])
        i::Int = pm_bus["index"]
        bus = Bus(i, true, 0.0)
        push!(buses, bus)
    end
    iref_pm::Int = PowerModels.reference_bus(pmdata)["index"]
    sort!(buses, by=bus->bus.index)
    pmidx2bus = Dict(bus.index => bus for bus in buses)
    # Now, go through loads and generators to grab injections
    for pmload in values(pmdata["load"])
        i::Int = pmload["load_bus"]
        st::Bool = pmload["status"]
        pd::Float64 = pmload["pd"]

        bus = pmidx2bus[i]
        bus.pd += st * pd
    end
    for pmgen in values(pmdata["gen"])
        i::Int = pmgen["gen_bus"]
        st::Bool = pmgen["gen_status"]
        pg::Float64 = pmgen["pg"]

        bus = pmidx2bus[i]
        bus.pd -= st * pg
    end
    # Re-index buses from 1 to N
    # (note that pmidx2bus remains valid)
    for (i, bus) in enumerate(buses)
        bus.index = i
    end

    # Next, we extract branches
    branches = Branch[]
    for pm_branch in values(pmdata["branch"])
        k::Int = pm_branch["index"]
        st::Bool = pm_branch["br_status"]
        b::Float64 = PM.calc_branch_y(pm_branch)[2]
        pmax::Float64 = pm_branch["rate_a"]

        # ⚠️ we convert PM bus indices before creating the branch
        i_pm::Int = pm_branch["f_bus"]
        j_pm::Int = pm_branch["t_bus"]
        i = pmidx2bus[i_pm].index
        j = pmidx2bus[j_pm].index

        br = Branch(k, st, b, pmax, i, j)
        push!(branches, br)
    end
    # Sort branches by index
    sort!(branches, by=br->br.index)
    pmidx2branch = Dict(br.index => br for br in branches)
    for (k, br) in enumerate(branches)
        br.index = k
    end

    slack_bus_index = pmidx2bus[iref_pm].index

    network = Network(buses, slack_bus_index, branches)
    return network
end
