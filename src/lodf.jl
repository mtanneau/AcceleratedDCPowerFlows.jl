abstract type AbstractLODF end

struct FullLODF <: AbstractLODF
    N::Int
    E::Int
    matrix::Matrix{Float64}
end

function FullLODF(network)
    Φ = LazyPTDF(network)
    N = length(network["bus"])
    i0 = reference_bus(network)["bus_i"]

    At = Matrix(Φ.A')
    _M = (Φ.F \ At)
    _M[i0, :] .= 0  # ⚠ need to zero-out slack bus angle
    M = -Φ.BA * _M
    d = inv.(1 .- diag(M))
    d .*= (abs.(d) .<= 1e8)
    D = Diagonal(d)
    rmul!(M, D)

    E = Φ.E
    for i in 1:E
        M[i, i] = -1
    end

    return FullLODF(Φ.N, Φ.E, M)
end

