function test_branch_incidence_matrix()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    network = APF.from_power_models(data)
    A_ref = PM.calc_basic_incidence_matrix(data)

    # Test with default backend
    _test_branch_incidence_matrix(APF.default_backend(), network, A_ref)

    return nothing
end

@testset test_branch_incidence_matrix()
