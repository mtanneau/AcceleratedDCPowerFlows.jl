function test_full_ptdf()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    network = APF.from_power_models(data)

    # Reference power flows, computed with PowerModels on CPU
    Φ_pm = PM.calc_basic_ptdf_matrix(data)

    # Test CUDA implementation
    Φ_gpu = APF.ptdf(CUDA.CUDABackend(), network; ptdf_type=:full, linear_solver=:CUDSS)
    @test isa(Φ_gpu, APF.FullPTDF)
    @test KA.get_backend(Φ_gpu) isa CUDA.CUDABackend

    _test_ptdf_matrix(Φ_gpu, Φ_pm)

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

    # Reference power flows, computed with PowerModels on CPU
    Φ_pm = PM.calc_basic_ptdf_matrix(data)

    # Test CUDA implementation
    Φ_gpu = APF.ptdf(CUDA.CUDABackend(), network; ptdf_type=:lazy)
    @test isa(Φ_gpu, APF.LazyPTDF)
    @test KA.get_backend(Φ_gpu) isa CUDA.CUDABackend

    _test_ptdf_matrix(Φ_gpu, Φ_pm)

    return nothing
end
