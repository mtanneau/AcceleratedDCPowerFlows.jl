abstract type AbstractPTDF end

"""
    ptdf([backend::Backend], network::Network; ptdf_type::Symbol, kwargs...)

# Arguments
* `backend::Backend`: `KernelAbstractions` backend, defaults to [`default_backend`](@ref).
* `network::Network`: the power network data structure

# Key-word arguments
* `ptdf_type::Symbol`: the type of PTDF to be built. 
    Only `:full` and `:lazy` are supported.
* Other key-word arguments are passed through to the underlying PTDF constructor.
"""
function ptdf(backend::KA.Backend, network::Network; ptdf_type=:lazy, kwargs...)
    if ptdf_type == :full
        return full_ptdf(backend, network; kwargs...)
    elseif ptdf_type == :lazy
        return lazy_ptdf(backend, network; kwargs...)
    else
        error("Unsupported PTDF type: $(ptdf_type). Only :full and :lazy are supported.")
    end
end

ptdf(network::Network; kwargs...) = ptdf(default_backend(), network; kwargs...)

include("full.jl")
include("lazy.jl")
