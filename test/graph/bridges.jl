function test_bridges()
    data = PM.make_basic_network(pglib("14_ieee"))
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

test_bridges()