# ============================================================================
# Backend-agnostic test functions for PTDF objects
# ============================================================================
module TestPTDFUtilities

using Test

import AcceleratedDCPowerFlows as APF
import KernelAbstractions as KA

export _test_ptdf_matrix

"""
    _test_ptdf_matrix(Φ, Φ_ref; atol=1e-6, batch_size=4)

Backend-agnostic test for the public `AbstractPTDF` API.

This checks `compute_flow!` on vector and matrix inputs against a reference PTDF
matrix computed independently.
"""
function _test_ptdf_matrix(Φ::APF.AbstractPTDF, Φ_ref; atol=1e-6, batch_size=4)
    backend = KA.get_backend(Φ)

    E, N = size(Φ_ref)

    @testset "vector input" begin
        p = randn(N)
        f_ref = Φ_ref * p

        p_dev = KA.allocate(backend, Float64, (N,))
        copyto!(p_dev, p)
        f_dev = KA.allocate(backend, Float64, (E,))
        copyto!(f_dev, randn(E))

        f_res = APF.compute_flow!(f_dev, p_dev, Φ)
        @test f_res === f_dev
        @test isapprox(collect(f_dev), f_ref; atol=atol)
    end

    @testset "matrix input" begin
        P = randn(N, batch_size)
        F_ref = Φ_ref * P

        P_dev = KA.allocate(backend, Float64, (N, batch_size))
        copyto!(P_dev, P)
        F_dev = KA.allocate(backend, Float64, (E, batch_size))
        copyto!(F_dev, randn(E))

        F_res = APF.compute_flow!(F_dev, P_dev, Φ)
        @test F_res === F_dev
        @test isapprox(collect(F_dev), F_ref; atol=atol)
    end

    return nothing
end

end
