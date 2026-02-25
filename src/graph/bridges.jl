"""
    find_bridges(network::Network)

Identify branches that are bridges in the network.

A branch is a bridge if removing it would increase the number of connected components.

!!! info
    Bridge computation is agnostic to branch orientation.

# Returns
* `b::Vector{Bool}` a vector of size `E`, where `E` is the number of branches,
    such that `b[e]` is `true` if and only if branch `e` is a bridge.
"""
function find_bridges(network::Network)
    N = num_buses(network)
    E = num_branches(network)

    # Build graph and identify multi-edges
    G = Graph(N)
    edge2branches = Dict{Tuple{Int,Int},Set{Int}}()
    for (k, br) in enumerate(network.branches)
        i = br.bus_fr
        j = br.bus_to
        ks = get!(edge2branches, (i, j), Set{Int}())
        push!(ks, k)

        # Graphs.jl will not add duplicate edges,
        #   so we can safely add every branch here
        add_edge!(G, i, j)
    end

    # Identify bridges
    is_bridge = zeros(Bool, E)
    for edge in bridges(G)
        i, j = (edge.src, edge.dst)
        ks = edge2branches[(i, j)]
        for k in ks
            is_bridge[k] = (length(ks) == 1)
        end
    end

    return is_bridge
end
