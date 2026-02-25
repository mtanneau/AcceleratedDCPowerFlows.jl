abstract type AbstractLODF end

function lodf(network::Network;
    backend=DefaultBackend(),
    linear_solver=:auto,
    lodf_type=:lazy,
)
    if lodf_type == :full
        return full_lodf(backend, network; linear_solver)
    elseif lodf_type == :lazy
        return lazy_lodf(backend, network; linear_solver)
    else
        error("Unsupported LODF type: $(lodf_type). Only :full and :lazy are supported.")
    end
end

include("full.jl")
include("lazy.jl")
