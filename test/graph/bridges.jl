function test_bridges()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    network = APF.from_power_models(data)

    b = APF.find_bridges(network)

    # There is a single bridge in IEEE14
    @test sum(b) == 1

    # ... corresponding to branch (7 => 8)
    k = argmax(b)
    @test k == 14

    br = network.branches[k]
    @test br.bus_fr == 7
    @test br.bus_to == 8

    return nothing
end

function test_bridges_reverse_orientation()
    data = PM.make_basic_network(pglib("pglib_opf_case14_ieee"))
    network = APF.from_power_models(data)
    b = APF.find_bridges(network)

    # Now reverse all branches
    for br in network.branches
        br.bus_fr, br.bus_to = br.bus_to, br.bus_fr
    end
    b_rev = APF.find_bridges(network)

    @test b == b_rev
end

test_bridges()
test_bridges_reverse_orientation()
