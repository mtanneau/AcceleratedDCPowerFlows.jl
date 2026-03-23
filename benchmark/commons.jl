const DEFAULT_BACKENDS = [KA.CPU(), CUDA.CUDABackend()]

const PGLIB_BENCHMARK_CASES = [
    "pglib_opf_case2869_pegase",
    "pglib_opf_case6470_rte",
    "pglib_opf_case9241_pegase",
    "pglib_opf_case13659_pegase",
    "pglib_opf_case20758_epigrids",
    "pglib_opf_case24464_goc",
    "pglib_opf_case30000_goc",
    "pglib_opf_case78484_epigrids",
]

"""
    load_network(case_name::String)

Convenience constructor for PGLib cases.

`case_name` must be a valid PGLib case name, e.g., `300_ieee` or `9241_pegase`.
"""
function load_network(case_name::String)
    data = PM.make_basic_network(pglib(case_name))
    return APF.from_power_models(data)
end

backend_name(::KA.CPU) = "cpu"
backend_name(::CUDA.CUDABackend) = "cuda"

"""
    select_backend(backend_name::String)

Select `KernelAbstractions` backend. Only supports `"cpu"` and `"cuda"` as input
"""
function select_backend(backend_name::String)
    if lowercase(backend_name) == "cpu"
        return KA.CPU()
    elseif lowercase(backend_name) == "cuda"
        if CUDA.functional()
            return CUDA.CUDABackend()
        else
            error("CUDA backend requested but CUDA is not functional")
        end
    end

    return error("Unsupported backend: $(backend_name)")
end

device_name(::KA.CPU) = begin
    info = Sys.cpu_info()
    isempty(info) ? "unknown-cpu" : info[1].model
end

device_name(::CUDA.CUDABackend) = name(CUDA.device())

function _max_memory_estimate_gb()
    return (2 ^ (floor(log2(Sys.total_memory() / (1024^3))) - 1))
end

function _check_memory(nnz, T; memory_limit_gb=_max_memory_estimate_gb())
    mem_estimate_gb = nnz * sizeof(one(T)) / (1024^3)
    return mem_estimate_gb <= memory_limit_gb
end
