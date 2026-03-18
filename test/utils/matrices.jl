# ============================================================================
# Backend-agnostic test functions for matrix types
# ============================================================================
module TestMatrixUtilities

using LinearAlgebra
using SparseArrays
using Test

import AcceleratedDCPowerFlows as APF
import KernelAbstractions as KA
import PowerModels as PM

export _test_branch_incidence_matrix
export _test_branch_susceptance_matrix
export _test_nodal_susceptance_matrix
export _test_matrix_equivalence

function _rand_array!(x)
    copyto!(x, randn(size(x)...))
    return x
end

"""
    _test_branch_incidence_matrix(backend, network, A_ref)

Backend-agnostic test for BranchIncidenceMatrix.
"""
function _test_branch_incidence_matrix(backend, network, A_ref)
    N = APF.num_buses(network)
    E = APF.num_branches(network)
    A = APF.branch_incidence_matrix(backend, network)

    @testset "Custom struct" begin
        @test A.N == N
        @test A.E == E
        @test isa(KA.get_backend(A), typeof(backend))
        _test_matrix_equivalence(A, A_ref; test_generic_kernel=true)
    end

    @testset "Sparse" begin
        A_sparse = sparse(A)
        @test isa(KA.get_backend(A_sparse), typeof(backend))
        _test_matrix_equivalence(A_sparse, A_ref; test_generic_kernel=false)
    end

    return nothing
end

"""
    _test_branch_susceptance_matrix(backend, network, A_ref)

Backend-agnostic test for BranchSusceptanceMatrix.
"""
function _test_branch_susceptance_matrix(backend, network, A_ref)
    N = APF.num_buses(network)
    E = APF.num_branches(network)
    A = APF.branch_susceptance_matrix(backend, network)

    @testset "Custom struct" begin
        @test A.N == N
        @test A.E == E
        @test isa(KA.get_backend(A), typeof(backend))
        _test_matrix_equivalence(A, A_ref; test_generic_kernel=true)
    end

    @testset "Sparse" begin
        A_sparse = sparse(A)
        @test isa(KA.get_backend(A_sparse), typeof(backend))
        _test_matrix_equivalence(A_sparse, A_ref; test_generic_kernel=false)
    end

    return nothing
end

"""
    _test_nodal_susceptance_matrix(backend, network, A_ref)

Backend-agnostic test for NodalSusceptanceMatrix.
"""
function _test_nodal_susceptance_matrix(backend, network, A_ref)
    N = APF.num_buses(network)
    E = APF.num_branches(network)
    A = APF.nodal_susceptance_matrix(backend, network)

    @testset "Custom struct" begin
        @test A.N == N
        @test A.E == E
        @test isa(KA.get_backend(A), typeof(backend))
        _test_matrix_equivalence(A, A_ref; test_generic_kernel=true)
    end

    @testset "Sparse" begin
        A_sparse = sparse(A)
        @test isa(KA.get_backend(A_sparse), typeof(backend))
        _test_matrix_equivalence(A_sparse, A_ref; test_generic_kernel=false)
    end

    return nothing
end

function _test_matrix_equivalence(A, A_ref; test_generic_kernel=false)
    m, n = size(A_ref)
    backend = KA.get_backend(A)

    @test size(A) == (m, n)
    @test size(A, 1) == m
    @test size(A, 2) == n
    @test size(A, 3) == size(A, 4) == 1

    x = _rand_array!(KA.allocate(backend, Float64, (n,)))
    X = _rand_array!(KA.allocate(backend, Float64, (n, 3)))
    y = _rand_array!(KA.allocate(backend, Float64, (m,)))
    Y = _rand_array!(KA.allocate(backend, Float64, (m, 3)))

    y_ref = A_ref * collect(x)
    Y_ref = A_ref * collect(X)

    @testset "matvec" begin
        LinearAlgebra.mul!(y, A, x)
        @test collect(y) ≈ y_ref
    end

    @testset "matmat" begin
        LinearAlgebra.mul!(Y, A, X)
        @test collect(Y) ≈ Y_ref
    end

    if test_generic_kernel
        @testset "generic kernel" begin
            _rand_array!(y)
            _rand_array!(Y)

            @testset "matvec" begin
                invoke(
                    APF._unsafe_mul!,
                    Tuple{KA.Backend,AbstractVecOrMat,typeof(A),AbstractVecOrMat},
                    backend,
                    y,
                    A,
                    x,
                )
                @test collect(y) ≈ y_ref
            end

            @testset "matmat" begin
                invoke(
                    APF._unsafe_mul!,
                    Tuple{KA.Backend,AbstractVecOrMat,typeof(A),AbstractVecOrMat},
                    backend,
                    Y,
                    A,
                    X,
                )
                @test collect(Y) ≈ Y_ref
            end
        end
    end

    return nothing
end

end
