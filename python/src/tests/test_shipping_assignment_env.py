from envs import shipping_assignment_env
from shipping_allocation import PhysicalNetwork


def test_sample():
    # Given
    physical_network = PhysicalNetwork(3, 5, 2, 100, 50)
    space = shipping_assignment_env.ShippingAssignmentSpace(physical_network)
    customer_node_id = 4
    customer_id = physical_network.get_customer_id(customer_node_id)
    # When
    print(physical_network.dcs_per_customer_array)
    print(physical_network.dcs_per_customer_array[customer_id, :])
    sampled_action = space.sample(customer_node_id)
    print(sampled_action)

    # Then
    assert physical_network.is_valid_arc(sampled_action, customer_node_id)
