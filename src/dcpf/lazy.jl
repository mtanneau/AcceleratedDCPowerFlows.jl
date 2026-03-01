"""
    LazyDCPF

Lazy DC power flow solver.

Instead of forming a dense inverse, stores a sparse factorization
of the nodal susceptance matrix and solves a linear system per query.
"""
struct LazyDCPF{TBf,TF} <: AbstractDCPF
    N::Int   # number of buses
    E::Int   # number of branches
    islack::Int

    Bf::TBf  # BranchSusceptanceMatrix (physical susceptances)
    F::TF    # Sparse factorization of -Bn
end

KA.get_backend(D::LazyDCPF) = KA.get_backend(D.Bf)

function lazy_dcpf(bkd::KA.CPU, network::Network; linear_solver=:auto)
    N = num_buses(network)
    E = num_branches(network)
    islack = network.slack_bus_index

    # Build nodal susceptance matrix
    # ⚠ susceptances are _negated_ so that AᵀBA is positive definite
    b = [-br.b for br in network.branches]
    bmin = minimum(b)
    Y = -sparse(nodal_susceptance_matrix(bkd, network))
    Y[islack, :] .= 0.0
    Y[:, islack] .= 0.0
    Y[islack, islack] = 1.0

    opfact = if (linear_solver == :auto) || (linear_solver == :SuiteSparse)
        if bmin >= 0.0
            LinearAlgebra.cholesky
        else
            LinearAlgebra.ldlt
        end
    elseif linear_solver == :KLU
        KLU.klu
    else
        error("""Unsupported CPU linear solver for lazy DCPF: $(linear_solver).
        Supported options are: `:KLU` and `:SuiteSparse`""")
    end

    F = opfact(Y)

    # Build branch susceptance matrix with physical (non-negated) susceptances
    Bf = branch_susceptance_matrix(bkd, network)

    return LazyDCPF(N, E, islack, Bf, F)
end

"""
    solve!(θ, p, D::LazyDCPF)

Solve the DC power flow equation `Bθ = p` for voltage angles `θ`.
"""
function solve!(θ, p, D::LazyDCPF)
    θ .= D.F \ p
    θ .*= -1  # F is factorization of -Bn, so negate
    θ[D.islack, :] .= 0  # slack voltage angle is zero
    return θ
end

"""
    compute_flow!(pf, p, D::LazyDCPF)

Compute branch power flows from nodal injections using the DC power flow equations.
"""
function compute_flow!(pf, p, D::LazyDCPF)
    θ = similar(p)
    compute_flow!(pf, p, D, θ)
    return pf
end

"""
    compute_flow!(pf, p, D::LazyDCPF, θ)

Compute branch power flows and voltage angles from nodal injections.
"""
function compute_flow!(pf, p, D::LazyDCPF, θ)
    solve!(θ, p, D)
    mul!(pf, D.Bf, θ)
    return pf
end
