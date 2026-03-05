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
