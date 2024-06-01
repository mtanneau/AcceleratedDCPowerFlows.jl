using Graphs

function find_bridges(data)
    N = length(data["bus"])
    E = length(data["branch"])
    # Identify bridges
    G = Graph(N)
    edge2id = Dict(
        (data["branch"]["$k"]["f_bus"], data["branch"]["$k"]["t_bus"]) => k
        for k in 1:E
    )
    for ((i, j), k) in edge2id
        add_edge!(G, i, j)
    end

    is_bridge = falses(E)
    for edge in bridges(G)
        k = edge2id[edge.src, edge.dst]
        is_bridge[k] = true
    end

    return is_bridge
end
