using Graphs

function find_bridges(data)
    N = length(data["bus"])
    E = length(data["branch"])

    # Identify all parallel branches
    edge2branches = Dict{Tuple{Int,Int},Set{Int}}()
    branch2edge   = Dict{Int,Tuple{Int,Int}}()
    for k in 1:E
        br = data["branch"]["$k"]
        i, j = extrema((br["f_bus"], br["t_bus"]))

        s = get!(edge2branches, (i, j), Set{Int}())
        push!(s, k)
        branch2edge[k] = (i, j)
    end

    # Identify bridges
    G = Graph(N)
    for ((i, j), k) in edge2branches
        add_edge!(G, i, j)
    end
    
    is_bridge = zeros(Bool, E)
    for edge in bridges(G)
        i, j = (edge.src, edge.dst)
        s = edge2branches[(i, j)]
        for k in s
            is_bridge[k] = (length(s) == 1)
        end
    end

    return is_bridge
end
