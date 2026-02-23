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

include("full.jl")
include("lazy.jl")
