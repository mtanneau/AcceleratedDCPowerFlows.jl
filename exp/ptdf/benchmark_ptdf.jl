using LinearAlgebra
using SparseArrays
using Statistics

using PowerModels
const PM = PowerModels
PM.silence()
using PGLib

using CUDA
using FastPowerFlow

using BenchmarkTools
using CSV
using DataFrames

using MKL
BLAS.set_num_threads(Base.Threads.nthreads())

"""
    is_on_gpu(Φ)

Detect whether PTDF matrix Φ is on GPU or not.
"""
function is_on_gpu(Φ)
    return isa(Φ.b, CuArray)
end

function form_ptdf(data, PTDF::Symbol, solver::Symbol, gpu::Bool)
    if PTDF == :full
        return FullPTDF(data; solver=solver, gpu=gpu)
    elseif PTDF == :lazy
        return LazyPTDF(data; solver=solver, gpu=gpu)
    else
        throw(ArgumentError("PTDF must be either :full or :lazy"))
    end
end

# Since forming a PTDF matrix may be (very!) costly,
#   we conduct two benchmarks: one to form the matrix, the other to compute the flow
function benchmark_ptdf(data, PTDF; solver, gpu, Ks=[1, 24, 96])
    N = length(data["bus"])
    E = length(data["branch"])
    Φ = form_ptdf(data, PTDF, solver, gpu)

    # 
    gpu = is_on_gpu(Φ)

    # Give more time for large systems
    # (Full PTDF on 78k system takes ~3min)
    n_seconds = (N >= 10_000) ? 300 : BenchmarkTools.DEFAULT_PARAMETERS.seconds

    # Benchmark PTDF matrix computation
    res_matrix = if gpu
        @benchmark begin CUDA.@sync form_ptdf($data, $PTDF, $solver, $gpu) end seconds=n_seconds
    else
        @benchmark form_ptdf($data, $PTDF, $solver, $gpu) seconds=n_seconds
    end
    println(typeof(Φ))
    display(res_matrix)

    # 
    res = Dict(
        "matrix" => res_matrix,
        "matvec" => Dict()
    )

    for K in Ks
        # TODO: support GPU here
        #  (this means casting p/pf as GPU arrays as needed)
        p = rand(N, K)
        pf = zeros(E, K)

        res_matvec = if gpu
            p_d = CuArray(p)
            pf_d = CuArray(pf)

            @benchmark CUDA.@sync compute_flow!($pf_d, $p_d, $Φ)
        else
            @benchmark compute_flow!($pf, $p, $Φ)
        end

        println("PTDF matvec, K=$K")
        display(res_matvec)

        res["matvec"][K] = res_matvec
    end

    return res
end
    
function main_ptdf(data)
    df = DataFrame(
        :casename => String[],
        :num_bus => Int[],
        :num_branch => Int[],
        :ptdf_type => String[],
        :solver => String[],
        :device => String[],
        :cpu_cores => Int[],
        :time_matrix => Float64[],
        :time_matvec_1 => Float64[],
        :time_matvec_24 => Float64[],
        :time_matvec_96 => Float64[],
    )
    
    N = length(data["bus"])
    E = length(data["branch"])

    for ptdf_type in [:full, :lazy], solver in [:klu, :ldlt], gpu in [false, true]

        gpu && solver == :klu && continue  # KLU is not supported on GPU
        (N > 30_000) && gpu && ptdf_type == :full && continue  # Avoid OOM on GPU

        GC.gc()
        res = benchmark_ptdf(data, ptdf_type; solver=solver, gpu=gpu, Ks=[1, 24, 96])

        row = Dict(
            :casename => data["name"],
            :num_bus => N,
            :num_branch => E,
            :ptdf_type => string(ptdf_type),
            :solver => string(solver),
            :device => gpu ? name(CUDA.device()) : (Sys.cpu_info()[1].model),
            :cpu_cores => BLAS.get_num_threads(),
            :time_matrix => median(res["matrix"].times) / 1e9,
            :time_matvec_1 => median(res["matvec"][1].times) / 1e9,
            :time_matvec_24 => median(res["matvec"][24].times) / 1e9,
            :time_matvec_96 => median(res["matvec"][96].times) / 1e9,
        )
        push!(df, row)
    end

    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    CASES = ["1354_pegase", "2869_pegase", "6470_rte", "9241_pegase", "13659_pegase", "30000_goc", "78484_epigrids"]

    CASE_IDX = parse(Int, ARGS[1])

    casename = CASES[CASE_IDX]

    resfile = joinpath(@__DIR__, "benchmark_PTDF_$(casename).csv")

    data = make_basic_network(pglib("pglib_opf_case" * casename))

    r = @benchmark calc_basic_incidence_matrix($data)
    println("Branch incidence matrix computation")
    display(r)
    println("\n\n")

    df = main_ptdf(data)

    CSV.write(resfile, df)

    exit(0)
end
