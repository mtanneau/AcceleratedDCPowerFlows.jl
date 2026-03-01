"""
    FullDCPF

Dense DC power flow solver.

Stores the dense inverse of the nodal susceptance matrix and the branch susceptance matrix.
Solves `Bθ = p` for voltage angles `θ`, then computes branch flows as `pf = Bf * θ`.
"""
struct FullDCPF{D,TBf} <: AbstractDCPF
    N::Int   # number of buses
    E::Int   # number of branches
    islack::Int

    Yinv::D    # Dense inverse of -Bn (internally negated for Cholesky)
    Bf::TBf    # BranchSusceptanceMatrix (physical susceptances)
end

KA.get_backend(D::FullDCPF) = KA.get_backend(D.Yinv)

function full_dcpf(bkd::KA.CPU, network::Network; linear_solver=:auto)
    N = num_buses(network)
    E = num_branches(network)
    islack = network.slack_bus_index

    # Build nodal susceptance matrix
    # ⚠ susceptances are _negated_ so that AᵀBA is positive definite
    b = [-br.b for br in network.branches]
    bmin = minimum(b)
    Y = -sparse(nodal_susceptance_matrix(bkd, network))  # Negated!
    Y[islack, :] .= 0.0
    Y[:, islack] .= 0.0
    Y[islack, islack] = 1.0

    # Which factorization should we use?
    opfact = if (linear_solver == :auto) || (linear_solver == :SuiteSparse)
        if bmin >= 0.0
            LinearAlgebra.cholesky
        else
            LinearAlgebra.ldlt
        end
    elseif linear_solver == :KLU
        KLU.klu
    else
        error("""Unsupported CPU linear solver for full DCPF: $(linear_solver).
        Supported options are: `:KLU` and `:SuiteSparse`""")
    end

    # Form matrix inverse
    F = opfact(Y)
    Yinv = F \ Matrix(1.0I, N, N)
    Yinv[islack, :] .= 0

    # Build branch susceptance matrix with physical (non-negated) susceptances
    Bf = branch_susceptance_matrix(bkd, network)

    return FullDCPF(N, E, islack, Yinv, Bf)
end

"""
    solve!(θ, p, D::FullDCPF)

Solve the DC power flow equation `Bθ = p` for voltage angles `θ`.

Since `Yinv` stores `(-Bn)⁻¹`, we negate the output to get physical angles.
The slack row of `Yinv` is zero by construction, but we zero `θ[islack]`
explicitly for robustness.
"""
function solve!(θ, p, D::FullDCPF)
    mul!(θ, D.Yinv, p)
    θ .*= -1  # Yinv is (-Bn)⁻¹, so negate to get Bn⁻¹ * p
    θ[D.islack, :] .= 0
    return θ
end

"""
    compute_flow!(pf, p, D::FullDCPF)

Compute branch power flows from nodal injections using the DC power flow equations.
"""
function compute_flow!(pf, p, D::FullDCPF)
    θ = similar(p)
    compute_flow!(pf, p, D, θ)
    return pf
end

"""
    compute_flow!(pf, p, D::FullDCPF, θ)

Compute branch power flows and voltage angles from nodal injections.
"""
function compute_flow!(pf, p, D::FullDCPF, θ)
    solve!(θ, p, D)
    mul!(pf, D.Bf, θ)
    return pf
end
