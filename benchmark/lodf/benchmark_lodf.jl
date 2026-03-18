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

const LODF_TYPES = [:full, :lazy]
const PTDF_TYPES = [:full, :lazy]
const LINEAR_SOLVERS = Dict("cpu" => [:KLU], "cuda" => [:CUDSS])
const CONTINGENCY_SAMPLES = 96

function build_lodf(
    backend::KA.Backend,
    network::APF.Network,
    lodf_type::Symbol,
    ptdf_type::Symbol,
    linear_solver::Symbol,
)
    if lodf_type == :lazy
        return APF.lodf(
            network;
            backend=backend,
            lodf_type=lodf_type,
            ptdf_type=ptdf_type,
            linear_solver=linear_solver,
        )
    end

    return APF.lodf(
        network;
        backend=backend,
        lodf_type=lodf_type,
        linear_solver=linear_solver,
    )
end

"""
    benchmark_lodf_constructor(backend, network, lodf_type, ptdf_type, linear_solver)

Benchmark LODF construction for one `(backend, lodf_type, ptdf_type, linear_solver)`
configuration on `network`.

Returns a `BenchmarkTools.Trial`.
"""
function benchmark_lodf_constructor(
    backend::KA.Backend,
    network::APF.Network,
    lodf_type::Symbol,
    ptdf_type::Symbol,
    linear_solver::Symbol,
)
    bres = @benchmark begin
        build_lodf($backend, $network, $lodf_type, $ptdf_type, $linear_solver)
        KA.synchronize($backend)
    end

    return bres
end

function _branch_samples(network::APF.Network, num_samples::Int)
    is_bridge = APF.find_bridges(network)
    candidates = [network.branches[i] for i in eachindex(network.branches) if !is_bridge[i]]

    if isempty(candidates)
        return APF.Branch[]
    end

    n = length(candidates)
    return [candidates[mod1(i, n)] for i in 1:num_samples]
end

"""
    benchmark_lodf_single_contingency(L, pf0, samples)

Benchmark post-contingency flow computation for the pre-built LODF object `L`
using a list of outage branches `samples`.

Returns a `BenchmarkTools.Trial`.
"""
function benchmark_lodf_single_contingency(
    L::APF.AbstractLODF,
    pf0::AbstractVector,
    samples::Vector{APF.Branch},
)
    backend = KA.get_backend(L)
    pfc = KA.allocate(backend, eltype(pf0), length(pf0))

    bres = @benchmark begin
        for br in $samples
            APF.compute_flow!($pfc, $pf0, $L, br)
        end
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
        lodf_type=String[],
        ptdf_type=String[],
        linear_solver=String[],
        operation=String[],
        contingency_samples=Int[],
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
    lodf_type::Symbol,
    ptdf_type::Symbol,
    solver::Symbol,
    operation::String,
    contingency_samples::Int,
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
            string(lodf_type),
            string(ptdf_type),
            string(solver),
            operation,
            contingency_samples,
            Float64(minimum(trial.times)) / 1e6,
            Float64(median(trial.times)) / 1e6,
            Float64(mean(trial.times)) / 1e6,
            Float64(std(trial.times)) / 1e6,
            Float64(trial.memory) / 1024,
        ),
    )
end

"""
    benchmark_lodf(network; backends=DEFAULT_BACKENDS)

Run LODF construction and single-contingency `compute_flow!` benchmarks for all
configured LODF types, PTDF types, and linear solvers on `network`.
"""
function benchmark_lodf(
    network::APF.Network;
    backends=DEFAULT_BACKENDS,
    memory_limit_gb=_max_memory_estimate_gb(),
)
    println(
        "Running case=$(APF.case_name(network)) N=$(APF.num_buses(network)) E=$(APF.num_branches(network))",
    )

    df = new_results_table()
    E = APF.num_branches(network)

    memory_warning_printed = false

    for backend in backends
        bname = backend_name(backend)

        if isa(backend, CUDA.CUDABackend) && !CUDA.functional()
            println("CUDA is not functional; skipping CUDA backend benchmarks.")
            continue
        end

        for lodf_type in LODF_TYPES
            if lodf_type == :full
                _full_lodf_mem_estimate_gb = E * E * sizeof(one(Float64)) / (1024^3)
                _sys_ram_gb = round(Sys.total_memory() / (1024^3); digits=1)
                if _full_lodf_mem_estimate_gb > memory_limit_gb
                    if memory_warning_printed
                        println("Skipping full LODF benchmark for case $(network.case_name).")
                    else
                        println(
                            """Skipping full LODF benchmark for case $(network.case_name).
                    An E×E matrix would require ~$(round(_full_lodf_mem_estimate_gb, digits=1))GB of memory,
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

            ptdf_types = lodf_type == :lazy ? PTDF_TYPES : [:none]
            samples = _branch_samples(network, CONTINGENCY_SAMPLES)

            for ptdf_type in ptdf_types
                for solver in LINEAR_SOLVERS[bname]
                    println(
                        "  backend=$(bname) lodf_type=$(lodf_type) ptdf_type=$(ptdf_type) solver=$(solver)",
                    )

                    constructor_trial =
                        benchmark_lodf_constructor(backend, network, lodf_type, ptdf_type, solver)

                    push_trial_row!(
                        df,
                        network,
                        backend,
                        lodf_type,
                        ptdf_type,
                        solver,
                        "construct",
                        0,
                        constructor_trial,
                    )

                    if isempty(samples)
                        println(
                            "No non-bridge branches are available in case $(network.case_name); skipping single-contingency benchmarks.",
                        )
                        continue
                    end

                    L = build_lodf(backend, network, lodf_type, ptdf_type, solver)
                    pf0 = KA.allocate(backend, Float64, E)
                    fill!(pf0, one(eltype(pf0)))

                    contingency_trial = benchmark_lodf_single_contingency(L, pf0, samples)

                    push_trial_row!(
                        df,
                        network,
                        backend,
                        lodf_type,
                        ptdf_type,
                        solver,
                        "single_contingency",
                        length(samples),
                        contingency_trial,
                    )
                end
            end
        end
    end

    return df
end

"""
    run_benchmark(; force=false, cases=PGLIB_BENCHMARK_CASES, export_path="")

Run LODF benchmarks across all `cases` and return one concatenated result
`DataFrame`.

Notes:
- `force` and `export_path` have identical behavior to the PTDF benchmark driver.
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
        fcsv_path = joinpath(export_path, case * ".csv")
        if isfile(fcsv_path) && !force
            println("Reading existing CSV result file for case $(case)")
            df = CSV.read(fcsv_path, DataFrame)
            push!(D, df)
            continue
        end

        network = try
            data = PM.make_basic_network(pglib(case))
            APF.from_power_models(data)
        catch err
            println("Error trying to load case $(case); skipping")
            continue
        end

        df = benchmark_lodf(network)
        push!(D, df)

        if !skip_export
            CSV.write(fcsv_path, df)
        end
    end

    if isempty(D)
        return new_results_table()
    end

    df_all = reduce(vcat, D)
    return df_all
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
