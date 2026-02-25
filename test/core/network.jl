function test_network_conversion_from_power_models()
    pm = pglib("118_ieee")

    # Check
    network = APF.from_power_models(pm)

    @test APF.num_buses(network) == length(network.buses) == 118
    @test APF.num_branches(network) == length(network.branches) == 186

    # check that elements are indexed from 1 to N
    @test map(b->b.index, network.buses) == 1:length(network.buses)
    @test map(b->b.index, network.branches) == 1:length(network.branches)
end

@testset test_network_conversion_from_power_models()