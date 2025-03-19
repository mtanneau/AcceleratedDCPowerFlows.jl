isinteractive() && (using Revise)

using LinearAlgebra
using SparseArrays
using Statistics
using CUDA

using PowerModels
const PM = PowerModels
PM.silence()
using PGLib

using FastPowerFlow

using BenchmarkTools
using CSV
using DataFrames

using MKL
BLAS.set_num_threads(1)

function bamul!(f, A, b, θ)
    mul!(f, A, θ)
    # lmul!(Diagonal(b), f) is not supported on GPU,
    # so we use broadcasting instead
    f .*= b
    return f
end

function benchmark_BA(data; K=16)
    N = length(data["bus"])
    E = length(data["branch"])

    A = Float64.(calc_basic_incidence_matrix(data))
    b = [
        calc_branch_y(data["branch"]["$e"])[2]
        for e in 1:E
    ]
    B = Diagonal(b)
    BA = B * A

    θ = rand(N, K)
    f = zeros(E, K)

    # Using Sparse arrays (CPU)
    res1 = @benchmark mul!($f, $BA, $θ)
    println("case: $(data["name"]) ; K=$K ; sparse BA")
    display(res1)
    println()

    # Using B*(A*θ) implementation
    res2 = @benchmark bamul!($f, $A, $b, $θ)
    println("case: $(data["name"]) ; K=$K ; sparse B / sparse A")
    display(res2)
    println()

    # Fast implementation
    Af = FastPowerFlow.BranchIncidenceMatrix(data)
    res3 = @benchmark bamul!($f, $Af, $b, $θ)
    println("case: $(data["name"]) ; K=$K ; fast A")
    display(res3)
    println()

    # GPU implementations
    A_d = CUDA.CUSPARSE.CuSparseMatrixCSR(A)
    BA_d = CUDA.CUSPARSE.CuSparseMatrixCSR(BA)
    b_d = CuArray(b)
    θ_d = CuArray(θ)
    f_d = CuArray(f)
    
    # matvec with BA
    res4 = @benchmark CUDA.@sync mul!($f_d, $BA_d, $θ_d)
    println("case: $(data["name"]) ; K=$K ; sparse BA (GPU)")
    display(res4)
    println()

    # matvec with A + element-wise product
    res5 = @benchmark CUDA.@sync bamul!($f_d, $A_d, $b_d, $θ_d)
    println("case: $(data["name"]) ; K=$K ; sparse B / sparse A (gpu)")
    display(res5)
    println()

    # Specialized matvec product on GPU
    Af_d = FastPowerFlow.BranchIncidenceMatrixGPU(Af)
    res6 = @benchmark CUDA.@sync bamul!($f_d, $Af_d, $b_d, $θ_d)
    println("case: $(data["name"]) ; K=$K ; fast A (gpu)")
    display(res6)
    println()

    return (res1, res2, res3, res4, res5, res6)
end
    
function main()
    df = DataFrame(
        :case => String[],
        :num_bus => Int[],
        :num_branch => Int[],
        :minimatch => Int[],
        :device => String[],
        :cpu_cores => Int[],
        :time_BA_sparse => Float64[],
        :time_BA_sparse_2 => Float64[],
        :time_BA_optimized => Float64[],
    )

    CASES = ["6470_rte", "9241_pegase", "13659_pegase", "30000_goc", "78484_epigrids"]

    for casename in CASES
        data = make_basic_network(pglib("pglib_opf_case" * casename))
        N = length(data["bus"])
        E = length(data["branch"])
        for K in [1, 24, 96]
            res1, res2, res3, res4, res5, res6 = benchmark_BA(data; K=K)
            row = Dict(
                :case => casename,
                :num_bus => N,
                :num_branch => E,
                :minimatch => K,
                :device => "CPU",
                :cpu_cores => BLAS.get_num_threads(),
                :time_BA_sparse => median(res1.times) / 1e9,
                :time_BA_sparse_2 => median(res2.times) / 1e9,
                :time_BA_optimized => median(res3.times) / 1e9,
            )
            push!(df, row)
            row = Dict(
                :case => casename,
                :num_bus => N,
                :num_branch => E,
                :minimatch => K,
                :device => "GPU",
                :cpu_cores => BLAS.get_num_threads(),
                :time_BA_sparse => median(res4.times) / 1e9,
                :time_BA_sparse_2 => median(res5.times) / 1e9,
                :time_BA_optimized => median(res6.times) / 1e9,
            )
            push!(df, row)
        end
    end

    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    df = main()
    CSV.write(joinpath(@__DIR__, "benchmark_BA.csv"), df)
end
