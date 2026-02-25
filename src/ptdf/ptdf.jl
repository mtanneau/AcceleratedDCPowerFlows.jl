abstract type AbstractPTDF end

function ptdf(network::Network;
    backend=DefaultBackend(),
    linear_solver=:auto,
    ptdf_type=:lazy,
)
    if ptdf_type == :full
        return full_ptdf(backend, network; linear_solver)
    elseif ptdf_type == :lazy
        return lazy_ptdf(backend, network; linear_solver)
    else
        error("Unsupported PTDF type: $(ptdf_type). Only :full and :lazy are supported.")
    end
end

include("full.jl")
include("lazy.jl")
