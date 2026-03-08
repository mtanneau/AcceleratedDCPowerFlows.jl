function test_full_ptdf()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    network = APF.from_power_models(data)
    N = APF.num_buses(network)
    E = APF.num_branches(network)

    # Reference power flows, computed with PowerModels on CPU
    Φ_pm = PM.calc_basic_ptdf_matrix(data)
    p = randn(N)
    fpm = Φ_pm * p

    # Test CUDA implementation
    Φ_gpu = APF.ptdf(CUDA.CUDABackend(), network; ptdf_type=:full, linear_solver=:CUDSS)
    @test isa(Φ_gpu, APF.FullPTDF)
    @test KA.get_backend(Φ_gpu) isa CUDA.CUDABackend

    # Compute power flows on GPU
    p_gpu = CuArray(p)
    f_gpu = CUDA.zeros(Float64, E)
    APF.compute_flow!(f_gpu, p_gpu, Φ_gpu)
    
    # Compare with CPU reference
    f = collect(f_gpu)
    @test isapprox(f, fpm; atol=1e-6)

    # Test with batched mode (multiple power injection scenarios)
    K = 4
    p_batch = randn(N, K)
    fpm_batch = Φ_pm * p_batch
    
    p_gpu_batch = CuArray(p_batch)
    f_gpu_batch = CUDA.zeros(Float64, E, K)
    APF.compute_flow!(f_gpu_batch, p_gpu_batch, Φ_gpu)
    
    f_batch = collect(f_gpu_batch)
    @test isapprox(f_batch, fpm_batch; atol=1e-6)

    return nothing
end

function test_full_ptdf_auto_solver()
    # Test that :auto solver option works
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    network = APF.from_power_models(data)
    
    Φ_gpu = APF.ptdf(CUDA.CUDABackend(), network; ptdf_type=:full, linear_solver=:auto)
    @test isa(Φ_gpu, APF.FullPTDF)
    @test KA.get_backend(Φ_gpu) isa CUDA.CUDABackend

    return nothing
end

function test_lazy_ptdf()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    network = APF.from_power_models(data)
    N = APF.num_buses(network)
    E = APF.num_branches(network)

    # Reference power flows, computed with PowerModels on CPU
    Φ_pm = PM.calc_basic_ptdf_matrix(data)
    p = randn(N)
    fpm = Φ_pm * p

    # Test CUDA implementation
    Φ_gpu = APF.ptdf(CUDA.CUDABackend(), network; ptdf_type=:lazy)
    @test isa(Φ_gpu, APF.LazyPTDF)
    @test KA.get_backend(Φ_gpu) isa CUDA.CUDABackend

    # Compute power flows on GPU
    p_gpu = CuArray(p)
    f_gpu = CUDA.zeros(Float64, E)
    APF.compute_flow!(f_gpu, p_gpu, Φ_gpu)

    # Compare with CPU reference
    f = collect(f_gpu)
    @test isapprox(f, fpm; atol=1e-6)

    # Test with batched mode (multiple power injection scenarios)
    K = 4
    p_batch = randn(N, K)
    fpm_batch = Φ_pm * p_batch

    p_gpu_batch = CuArray(p_batch)
    f_gpu_batch = CUDA.zeros(Float64, E, K)
    APF.compute_flow!(f_gpu_batch, p_gpu_batch, Φ_gpu)

    f_batch = collect(f_gpu_batch)
    @test isapprox(f_batch, fpm_batch; atol=1e-6)

    return nothing
end