abstract type AbstractPTDF end

function _linear_solver(s)
    if s == :lu
        return lu
    elseif s == :klu
        return KLU.klu
    elseif s == :ldlt
        return ldlt
    elseif s == :cholesky
        return cholesky
    else
        error("Invalid linear solver: only lu, klu (CPU only), ldlt, cholesky are supported")
    end
end

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
