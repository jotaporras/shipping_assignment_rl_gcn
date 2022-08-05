import torch
from envs import shipping_assignment_state
from envs.shipping_assignment_state import ShippingAssignmentState
from network.physical_network import Node
from shipping_allocation import PhysicalNetwork
import numpy as np
from experiment_utils.Order import Order


def test_state_to_demand_per_warehouse_commodity():
    # Given
    physical_network = PhysicalNetwork(3, 5, 2, 100, 50, num_commodities=3)
    dummy_customer = Node(4, 100, 0, 0, "dc")
    dc_0 = Node(0, 100, 0, 0, "dc")
    dc_1 = Node(1, 100, 0, 0, "dc")
    fixed_orders = [
        # A total 105,60,60 to DC 0
        Order(
            np.array([50.0, 30.0, 30.0]),
            dc_0,
            dummy_customer,
            0,
            "someord",
        ),
        Order(
            np.array([55.0, 30.0, 30.0]),
            dc_1,
            dummy_customer,
            0,
            "someord",
        ),
    ]

    state = ShippingAssignmentState(
        0,
        physical_network,
        fixed=fixed_orders,
        open=[],
        inventory=[],
        state_vector=None,
        big_m_counter_per_commodity=0,
        optimization_cost=0,
        big_m_units_per_commodity=0,
    )
    # When
    demand_per_warehouse_commodity = (
        shipping_assignment_state.state_to_demand_per_warehouse_commodity(state)
    )

    # Then
    assert (
        demand_per_warehouse_commodity
        == np.array([50.0, 30.0, 30.0, 55.0, 30.0, 30.0, 0.0, 0.0, 0.0])
    ).all()


# Todo test that one commodity still works.
