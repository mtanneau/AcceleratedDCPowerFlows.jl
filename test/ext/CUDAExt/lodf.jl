function test_full_lodf()
	data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
	network = APF.from_power_models(data)
	E = APF.num_branches(network)

	# Build reference LODF on CPU and CUDA implementation on GPU
	L_cpu = APF.lodf(network; lodf_type=:full)
	L_gpu = APF.lodf(CUDA.CUDABackend(), network; lodf_type=:full, linear_solver=:CUDSS)

	@test isa(L_gpu, APF.FullLODF)
	@test KA.get_backend(L_gpu) isa CUDA.CUDABackend
	@test isapprox(collect(L_gpu.matrix), L_cpu.matrix; atol=1e-6)

	# Validate contingency flow computation on GPU
	p = real.(PM.calc_basic_bus_injection(data))
	Φ_pm = PM.calc_basic_ptdf_matrix(data)
	pf0 = Φ_pm * p

	outages = findall(.!APF.find_bridges(network))
	@test !isempty(outages)
	k = first(outages)
	br = network.branches[k]

	pfc_ref = similar(pf0)
	APF.compute_flow!(pfc_ref, pf0, L_cpu, br)

	pf0_gpu = CuArray(pf0)
	pfc_gpu = CUDA.zeros(Float64, E)
	APF.compute_flow!(pfc_gpu, pf0_gpu, L_gpu, br)

	@test isapprox(collect(pfc_gpu), pfc_ref; atol=1e-6)

	return nothing
end

function test_full_lodf_auto_solver()
	data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
	network = APF.from_power_models(data)

	L_gpu = APF.lodf(CUDA.CUDABackend(), network; lodf_type=:full, linear_solver=:auto)
	@test isa(L_gpu, APF.FullLODF)
	@test KA.get_backend(L_gpu) isa CUDA.CUDABackend

	return nothing
end
