abstract type AbstractDCPF end

function dcpf(network::Network;
    backend=DefaultBackend(),
    linear_solver=:auto,
    dcpf_type=:lazy,
)
    if dcpf_type == :full
        return full_dcpf(backend, network; linear_solver)
    elseif dcpf_type == :lazy
        return lazy_dcpf(backend, network; linear_solver)
    else
        error("Unsupported DCPF type: $(dcpf_type). Only :full and :lazy are supported.")
    end
end

# This should be the only place where we don't specify a backend
full_dcpf(network::Network; kwargs...) = dcpf(network; dcpf_type=:full, kwargs...)
lazy_dcpf(network::Network; kwargs...) = dcpf(network; dcpf_type=:lazy, kwargs...)

include("full.jl")
include("lazy.jl")
