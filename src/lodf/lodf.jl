abstract type AbstractLODF end

function lodf(network::Network;
    backend=default_backend(),
    lodf_type=:lazy,
    kwargs...
)
    if lodf_type == :full
        return full_lodf(backend, network; kwargs...)
    elseif lodf_type == :lazy
        return lazy_lodf(backend, network; kwargs...)
    else
        error("Unsupported LODF type: $(lodf_type). Only :full and :lazy are supported.")
    end
end

include("full.jl")
include("lazy.jl")
