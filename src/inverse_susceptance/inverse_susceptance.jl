abstract type AbstractInverseSusceptance end

# --- Private helpers for building and factorizing the negated nodal susceptance matrix ---

function _build_negated_nodal_susceptance(bkd::KA.CPU, network::Network)
    islack = network.slack_bus_index

    b = [-br.b for br in network.branches]
    Y = -sparse(nodal_susceptance_matrix(bkd, network))
    Y[islack, :] .= 0.0
    Y[:, islack] .= 0.0
    Y[islack, islack] = 1.0

    return Y, b
end

function _select_factorization(bmin, linear_solver)
    if (linear_solver == :auto) || (linear_solver == :SuiteSparse)
        if bmin >= 0.0
            return LinearAlgebra.cholesky
        else
            return LinearAlgebra.ldlt
        end
    elseif linear_solver == :KLU
        return KLU.klu
    else
        error("""Unsupported CPU linear solver: $(linear_solver).
        Supported options are: `:KLU` and `:SuiteSparse`""")
    end
end

function _factorize(bkd::KA.CPU, network::Network; linear_solver=:auto)
    Y, b = _build_negated_nodal_susceptance(bkd, network)
    bmin = minimum(b)
    opfact = _select_factorization(bmin, linear_solver)
    F = opfact(Y)
    return F, b
end

# --- Subtypes ---
include("full.jl")
include("lazy.jl")

# --- Shared methods ---

"""
    compute_flow!(pf, p, S::AbstractInverseSusceptance)

Compute branch power flows from nodal injections.
"""
function compute_flow!(pf, p, S::AbstractInverseSusceptance)
    θ = similar(p)
    compute_flow!(pf, p, S, θ)
    return pf
end

"""
    compute_flow!(pf, p, S::AbstractInverseSusceptance, θ)

Compute branch power flows and voltage angles from nodal injections.
"""
function compute_flow!(pf, p, S::AbstractInverseSusceptance, θ)
    solve!(θ, p, S)
    mul!(pf, S.Bf, θ)
    return pf
end

# --- DCPF entry points ---

"""
    dcpf([backend::Backend], network::Network; dcpf_type::Symbol, kwargs...)

Build a DC power flow solver.

Returns a [`FullInverseSusceptance`](@ref) or [`LazyInverseSusceptance`](@ref),
which support [`solve!`](@ref) and [`compute_flow!`](@ref).

# Arguments
* `backend::Backend`: `KernelAbstractions` backend, defaults to [`default_backend`](@ref).
* `network::Network`: the power network data structure

# Key-word arguments
* `dcpf_type::Symbol`: the type of DCPF to be built.
    Only `:full` and `:lazy` are supported.
* Other key-word arguments are passed through to the underlying constructor.
"""
function dcpf(backend::KA.Backend, network::Network;
    dcpf_type=:lazy,
    kwargs...
)
    if dcpf_type == :full
        return full_dcpf(backend, network; kwargs...)
    elseif dcpf_type == :lazy
        return lazy_dcpf(backend, network; kwargs...)
    else
        error("Unsupported DCPF type: $(dcpf_type). Only :full and :lazy are supported.")
    end
end

dcpf(network::Network; kwargs...) = dcpf(default_backend(), network; kwargs...)

full_dcpf(network::Network; kwargs...) = full_dcpf(default_backend(), network; kwargs...)
full_dcpf(bkd::KA.Backend, network::Network; kwargs...) = full_inverse_susceptance(bkd, network; kwargs...)

lazy_dcpf(network::Network; kwargs...) = lazy_dcpf(default_backend(), network; kwargs...)
lazy_dcpf(bkd::KA.Backend, network::Network; kwargs...) = lazy_inverse_susceptance(bkd, network; kwargs...)
