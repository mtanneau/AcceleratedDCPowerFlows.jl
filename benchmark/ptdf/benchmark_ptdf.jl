using BenchmarkTools
using CSV
using DataFrames
using Dates
using LinearAlgebra
using Statistics

BLAS.set_num_threads(Base.Threads.nthreads())

using CUDA
using CUDSS

import AcceleratedDCPowerFlows as APF
import KernelAbstractions as KA

using PGLib
import PowerModels as PM
PM.silence()

include(joinpath(@__DIR__, "..", "commons.jl"))

const PTDF_TYPES = [:full, :lazy]
const LINEAR_SOLVERS = Dict("cpu" => [:SuiteSparse, :KLU], "cuda" => [:CUDSS])
const RHS_WIDTHS = [96]

"""
    benchmark_ptdf_constructor(backend, network, ptdf_type, linear_solver)

Benchmark PTDF construction for one `(backend, ptdf_type, linear_solver)`
configuration on `network`.

Returns a `BenchmarkTools.Trial`.
"""
function benchmark_ptdf_constructor(
    backend::KA.Backend,
    network::APF.Network,
    ptdf_type::Symbol,
    linear_solver::Symbol,
)
    bres = @benchmark begin
        APF.ptdf($backend, $network; ptdf_type=($(ptdf_type)), linear_solver=($(linear_solver)));
        KA.synchronize($backend);
    end

    return bres
end

"""
    benchmark_ptdf_matprod(P, k)

Benchmark `compute_flow!` using PTDF object `P` and an `N x k` injection batch,
where `N` is the number of buses in `P`.

Returns a `BenchmarkTools.Trial`.
"""
function benchmark_ptdf_matprod(P::APF.AbstractPTDF, k::Int)
    backend = KA.get_backend(P)
    N = P.N
    E = P.E

    p = KA.allocate(backend, Float64, (N, k))
    f = KA.allocate(backend, Float64, (E, k))

    bres = @benchmark begin
        APF.compute_flow!($f, $p, $P)
        KA.synchronize($backend)
    end

    return bres
end

"""
    new_results_table()

Create an empty `DataFrame` with the benchmark CSV schema.
Timing values are recorded in milliseconds and memory in kilobytes.
"""
function new_results_table()
    return DataFrame(;
        timestamp=String[],
        case_name=String[],
        num_bus=Int[],
        num_branch=Int[],
        backend=String[],
        device=String[],
        blas_threads=Int[],
        ptdf_type=String[],
        linear_solver=String[],
        operation=String[],
        rhs_width=Int[],
        min_time_ms=Float64[],
        median_time_ms=Float64[],
        mean_time_ms=Float64[],
        std_time_ms=Float64[],
        memory_kb=Float64[],
    )
end

function push_trial_row!(
    df::DataFrame,
    network::APF.Network,
    backend::KA.Backend,
    ptdf_type::Symbol,
    solver::Symbol,
    operation::String,
    rhs_width::Int,
    trial,
)
    return push!(
        df,
        (
            string(Dates.now()),
            APF.case_name(network),
            APF.num_buses(network),
            APF.num_branches(network),
            backend_name(backend),
            device_name(backend),
            BLAS.get_num_threads(),
            string(ptdf_type),
            string(solver),
            operation,
            rhs_width,
            Float64(minimum(trial.times)) / 1e6,
            Float64(median(trial.times)) / 1e6,
            Float64(mean(trial.times)) / 1e6,
            Float64(std(trial.times)) / 1e6,
            Float64(trial.memory) / 1024,
        ),
    )
end

"""
    benchmark_ptdf(network; backends=DEFAULT_BACKENDS)

Run PTDF construction and `compute_flow!` benchmarks for all configured PTDF
types and linear solvers on `network` and return the result table.
"""
function benchmark_ptdf(
    network::APF.Network;
    backends=DEFAULT_BACKENDS,
    memory_limit_gb=_max_memory_estimate_gb(),
)
    println(
        "Running case=$(APF.case_name(network)) N=$(APF.num_buses(network)) E=$(APF.num_branches(network))",
    )

    df = new_results_table()

    memory_warning_printed = false

    for backend in backends
        bname = backend_name(backend)

        # Check if CUDA is functional, skip if not
        if isa(backend, CUDA.CUDABackend) && !CUDA.functional()
            println("CUDA is not functional; skipping CUDA backend benchmarks.")
            continue
        end

        for ptdf_type in PTDF_TYPES
            # Skip full PTDF if size is too big
            if ptdf_type == :full
                N = APF.num_buses(network)
                E = APF.num_branches(network)
                _full_ptdf_mem_estimate_gb = N*N*sizeof(one(Float64)) / (1024^3)
                _sys_ram_gb = round(Sys.total_memory() / (1024^3); digits=1)
                if _full_ptdf_mem_estimate_gb > memory_limit_gb
                    if memory_warning_printed
                        println("Skipping full PTDF benchmark for case $(network.case_name).")
                    else
                        println(
                            """Skipping full PTDF benchmark for case $(network.case_name).
                    An N×N matrix would require ~$(round(_full_ptdf_mem_estimate_gb, digits=1))GB of memory,
                    which exceeds the current limit of $(memory_limit_gb)GB.

                    To increase this tolerance, set `memory_limit_gb` to a higher threshold.
                    FYI, your system has $(_sys_ram_gb)GB of RAM, and it's recommended to not exceed 50% of that threshold.
                    """,
                        )
                        memory_warning_printed = true
                    end
                    continue
                end
            end

            for solver in LINEAR_SOLVERS[bname]
                println("  backend=$(bname) ptdf_type=$(ptdf_type) solver=$(solver)")

                constructor_trial = benchmark_ptdf_constructor(backend, network, ptdf_type, solver)
                push_trial_row!(
                    df,
                    network,
                    backend,
                    ptdf_type,
                    solver,
                    "construct",
                    0,
                    constructor_trial,
                )

                phi = APF.ptdf(backend, network; ptdf_type=ptdf_type, linear_solver=solver)

                for k in RHS_WIDTHS
                    trial = benchmark_ptdf_matprod(phi, k)
                    push_trial_row!(
                        df,
                        network,
                        backend,
                        ptdf_type,
                        solver,
                        "compute_flow",
                        k,
                        trial,
                    )
                end
            end
        end
    end

    return df
end

"""
    run_benchmark(; force=false, cases=PGLIB_BENCHMARK_CASES, export_path="")

Run PTDF benchmarks across all `cases` and return one concatenated result
`DataFrame`.

Notes:
- `force` and `export_path` are currently reserved for future export/overwrite
  behavior and are not yet used.
"""
function run_benchmark(;
    force=false,
    cases=PGLIB_BENCHMARK_CASES,
    export_path=joinpath(@__DIR__, "res"),
)
    skip_export = isempty(export_path)
    if !skip_export
        mkpath(export_path)
    end

    D = []
    for case in cases
        # If result file already exists, skip benchmark unless `force` is set to true
        fcsv_path = joinpath(export_path, case * ".csv")
        if isfile(fcsv_path) && !force
            # Load existing CSV and move to next case
            println("Reading existing CSV result file for case $(case)")
            df = CSV.read(fcsv_path, DataFrame)
            push!(D, df)
            continue
        end

        # Load casefile, skip if any issue is encountered
        network = try
            data = PM.make_basic_network(pglib(case))
            APF.from_power_models(data)
        catch err
            println("Error trying to load case $(case); skipping")
            continue
        end

        # Run the actual benchmark
        df = benchmark_ptdf(network)
        push!(D, df)

        if !skip_export
            CSV.write(fcsv_path, df)
        end
    end

    # Concatenate all results
    df_all = reduce(vcat, D)

    return df_all
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
