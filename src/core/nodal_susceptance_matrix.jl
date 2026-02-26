#=
    Mathematical notes:
    The nodal susceptance matrix is M = Aᵀ × B × A, where
        B = Diag(b) and A is the branch incidence matrix.
        Its size is N × N, where N is number of buses.

    The implementation below uses the identity M = Σₑ bₑ × (AₑᵀAₑ)
        where Aₑ is the e-th row of `A`.
        Note that (Aₑ)ᵢ = +1 and (Aₑ)ⱼ = -1 where branch `e = (i, j)`.
    
    This strategy is simple, but not optimal when parallel branches are present.
        (it is optimal if there are no parallel branches)
=#

"""
    NodalSusceptanceMatrix{Vi,Vf}

Efficient data structure for representing the nodal susceptance matrix of a power grid.

The nodal susceptance matrix is `M = Abᵀ × B × Ab`, where
* `Ab` is the branch incidence matrix, of size `E×N`
* `B` is a diagonal matrix of size `E×E` whose `e`-th entry is `bₑ`
"""
struct NodalSusceptanceMatrix{Vi,Vf}
    N::Int
    E::Int

    # Vi is a <:AbstractVector{<:Integer}
    # Vf is a <:AbstractVector{<:Real}
    bus_fr::Vi
    bus_to::Vi
    br_b::Vf
end

Base.size(A::NodalSusceptanceMatrix) = (A.N, A.N)
function Base.size(A::NodalSusceptanceMatrix, d::Integer)
    s = size(A)
    if d == 1 || d == 2
        return s[d]
    elseif d > 2
        return 1
    else
        error("arraysize: dimension $d out of range")
    end
end

KA.get_backend(A::NodalSusceptanceMatrix) = KA.get_backend(A.bus_fr)

"""
    nodal_susceptance_matrix([backend], network::Network)

Build branch susceptance matrix on the specified backend.
Defaults to cpu if no backend is provided.
"""
function nodal_susceptance_matrix(::KA.CPU, network::Network)
    N = num_buses(network)
    E = num_branches(network)

    bus_fr = [br.bus_fr for br in network.branches]
    bus_to = [br.bus_to for br in network.branches]
    br_b = [br.b for br in network.branches]

    return NodalSusceptanceMatrix(N, E, bus_fr, bus_to, br_b)
end

nodal_susceptance_matrix(network::Network) = nodal_susceptance_matrix(DefaultBackend(), network)

function SparseArrays.sparse(A::NodalSusceptanceMatrix)
    # Sanity check: make sure we are on CPU
    backend = KA.get_backend(A)
    if !isa(backend, KA.CPU)
        error("Unsupported backend for building a sparse nodal susceptance matrix: $(typeof(backend))")
    end

    # Grab integer and float types
    Ti = eltype(A.bus_fr)
    Tf = eltype(A.br_b)

    N = A.N
    E = A.E
    Is = zeros(Ti, 4*E)
    Js = zeros(Ti, 4*E)
    Vs = zeros(Tf, 4*E)

    for e in 1:E
        i = A.bus_fr[e]
        j = A.bus_to[e]

        Is[4*(e-1)+1] = i
        Js[4*(e-1)+1] = j
        Vs[4*(e-1)+1] = -A.br_b[e]

        Is[4*(e-1)+2] = j
        Js[4*(e-1)+2] = i
        Vs[4*(e-1)+2] = -A.br_b[e]

        Is[4*(e-1)+3] = i
        Js[4*(e-1)+3] = i
        Vs[4*(e-1)+3] = A.br_b[e]

        Is[4*(e-1)+4] = j
        Js[4*(e-1)+4] = j
        Vs[4*(e-1)+4] = A.br_b[e]
    end
    return SparseArrays.sparse(Is, Js, Vs, N, N)
end

function LinearAlgebra.mul!(y::AbstractVecOrMat, A::NodalSusceptanceMatrix, x::AbstractVecOrMat)
    N = A.N
    E = A.E
    K = size(y, 2)

    N == size(x, 1) || throw(DimensionMismatch("A has size $(size(A)), but x has size $(size(x))"))
    N == size(y, 1) || throw(DimensionMismatch("A has size $(size(A)), but y has size $(size(y))"))
    K == size(x, 2) || throw(DimensionMismatch("x has size $(size(x)), but y has size $(size(y))"))

    y .= zero(eltype(y))

    #=
        To compute the matrix-vector product, we use the relation
        * Compute `zₑ = bₑ * Aₑx = bₑ×(xᵢ - xⱼ) ∈ R`
        * yᵢ += zₑ
        * yⱼ -= zₑ

        Because we update entries of `yᵢ` multiple times 
            (as many times as the degree Δᵢ), this implementation
            cannot be @simd-ed (but it can be parallelized along 
            the second dimension of `y`)
    =#

    @inbounds for k in 1:K
        for e in 1:E
            i = A.bus_fr[e]
            j = A.bus_to[e]
            z = A.br_b[e] * (x[i,k] - x[j,k])
            y[i, k] += z
            y[j, k] -= z
        end
    end
    return y
end
