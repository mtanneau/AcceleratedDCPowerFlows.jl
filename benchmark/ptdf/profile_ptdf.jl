using ArgParse
using LinearAlgebra
using Profile
using ProfileSVG

BLAS.set_num_threads(Base.Threads.nthreads())

using CUDA
using CUDSS

import AcceleratedDCPowerFlows as APF
import KernelAbstractions as KA

using PGLib
import PowerModels as PM
PM.silence()

const FLOW_BATCH_SIZE = 96

include(joinpath(@__DIR__, "..", "commons.jl"))

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--case_name"
            help = "PGLib case name (example: pglib_opf_case2869_pegase)"
            arg_type = String
            default="pglib_opf_case9241_pegase"
        "--backend"
            help = "Backend (i.e. device) to use. Should be CPU or CUDA"
            arg_type = String
            default = "cpu"
        range_tester = (x -> lowercase(x) in ("cpu", "cuda"))
        "--ptdf_type"
            help = "PTDF type: full or lazy"
            arg_type = String
            default="lazy"
            range_tester = (x -> lowercase(x) in ("full", "lazy"))
        "--linear_solver"
            help = "Linear solver name (e.g., KLU, SuiteSparse, CUDSS). Leave unset to use default"
            arg_type = String
            default="auto"
    end

    return parse_args(settings)
end

function output_path(
    matrix_type::String,
    lodf_type::Symbol,
    ptdf_type::Symbol,
    backend_name_str::String,
    linear_solver::Symbol,
    operation::String,
)
    root = joinpath(@__DIR__, "prof")
    if matrix_type == "PTDF"
        fname = "$(ptdf_type)_$(backend_name_str)_$(linear_solver)_$(operation).svg"
        return joinpath(root, fname)
    end

    fname = "$(lodf_type)_$(ptdf_type)_$(backend_name_str)_$(linear_solver)_$(operation).svg"
    return joinpath(root, fname)
end

function profile_ptdf(
    backend::KA.Backend,
    network::APF.Network,
    ptdf_type::Symbol,
    linear_solver::Symbol,
    k::Int=FLOW_BATCH_SIZE,
)
    # Compilation run
    Φ = APF.ptdf(backend, network; ptdf_type=ptdf_type, linear_solver=linear_solver)
    p = KA.allocate(backend, Float64, (Φ.N, k))
    f = KA.allocate(backend, Float64, (Φ.E, k))

    function _build(trials=1)
        for _ in 1:trials
            APF.ptdf(backend, network; ptdf_type=ptdf_type, linear_solver=linear_solver)
            # KA.synchronize(backend)
        end
    end

    function _matprod(trials=1)
        for _ in 1:trials
            APF.compute_flow!(f, p, Φ)
        end
    end

    _build()
    _matprod()
    tbuild = @elapsed _build()
    tprod = @elapsed _matprod()

    # Profile run
    ntrials = max(1, ceil(1 / tbuild))
    @info "Running $(ntrials) trials for constructor"
    Profile.clear()
    Profile.init(; n=10^7, delay=1e-3)
    @profile _build(ntrials)
    ProfileSVG.save(
        output_path("PTDF", :none, ptdf_type, backend_name(backend), linear_solver, "build"),
    )

    ntrials = max(1, ceil(1 / tprod))
    @info "Running $(ntrials) trials for matprod"
    Profile.clear()
    Profile.init(; n=10^7, delay=1e-3)
    @profile _matprod(ntrials)
    ProfileSVG.save(
        output_path("PTDF", :none, ptdf_type, backend_name(backend), linear_solver, "matprod"),
    )

    return nothing
end

function main_profile_ptdf()
    parsed_args = parse_commandline()

    case_name = parsed_args["case_name"]
    backend_name_str = lowercase(parsed_args["backend"])
    linear_solver = Symbol(parsed_args["linear_solver"])
    ptdf_type = Symbol(lowercase(parsed_args["ptdf_type"]))

    backend = select_backend(backend_name_str)
    network = load_network(case_name)

    println(
        "Running profile for following options:
        * case=$(case_name)
        * backend=$(backend_name_str)
        * linear solver=$(linear_solver)",
    )

    profile_ptdf(backend, network, ptdf_type, linear_solver)
end

if abspath(PROGRAM_FILE) == @__FILE__
    mkpath(joinpath(@__DIR__, "prof"))
    main_profile_ptdf()
end
