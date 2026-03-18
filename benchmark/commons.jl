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

backend_name(::KA.CPU) = "cpu"
backend_name(::CUDA.CUDABackend) = "cuda"

device_name(::KA.CPU) = begin
    info = Sys.cpu_info()
    isempty(info) ? "unknown-cpu" : info[1].model
end

device_name(::CUDA.CUDABackend) = name(CUDA.device())

function _max_memory_estimate_gb()
    return (2 ^ (floor(log2(Sys.total_memory() / (1024^3))) - 1))
end
