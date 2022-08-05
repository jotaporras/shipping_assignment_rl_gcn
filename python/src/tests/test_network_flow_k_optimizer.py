from network.physical_network import Node, PhysicalNetwork
import numpy as np

from experiment_utils import network_flow_k_optimizer
from experiment_utils.Order import Order


def test_optimize__one_commodity():
    # Given
    physical_network = PhysicalNetwork(
        num_dcs=2,
        num_customers=1,
        dcs_per_customer=2,
        demand_mean=100,
        demand_var=25,
        big_m_factor=10000,
        num_commodities=1,
        planning_horizon=3,
    )
    # (num_dcs,num_commodities)
    inventory = np.array([[1.0], [0.0]])
    open = []
    customer_node = physical_network.customers[0]
    dc_0_node = physical_network.dcs[0]
    fixed = [
        Order(
            demand=np.array([1.0]),
            shipping_point=dc_0_node,
            customer=customer_node,
            delivery_time=1,
            name="o1",
        )
    ]
    current_t = 0
    # When
    (
        total_cost,
        transport_matrix,
        all_movements,
        big_m_per_commodity,
        big_m_per_commodity_units,
    ) = network_flow_k_optimizer.optimize(
        physical_network, inventory, fixed, open, current_t, report_interplant_cost=True
    )
    print("--")
    print("total_cost\n", total_cost)
    print("transport_matrix\n", transport_matrix)
    print("all_movements\n", all_movements)
    print("big_m_per_commodity\n", big_m_per_commodity)
    print("--")
    # Then
    print("DONE")
    assert total_cost == 11.0  


def test_optimize_what_happens_negative_order(): 
    # Given
    physical_network = PhysicalNetwork(
        num_dcs=2,
        num_customers=1,
        dcs_per_customer=2,
        demand_mean=100,
        demand_var=25,
        big_m_factor=10000,
        num_commodities=1,
        planning_horizon=4,
    )
    # (num_dcs,num_commodities)
    inventory = np.array([[1.0], [0.0]])
    open = []
    customer_node = physical_network.customers[0]
    dc_0_node = physical_network.dcs[0]
    fixed = [
        Order(
            demand=np.array([2.0]),
            shipping_point=dc_0_node,
            customer=customer_node,
            delivery_time=3,
            name="o1",
        ),
        Order(
            demand=np.array([-1.0]),
            shipping_point=dc_0_node,
            customer=customer_node,
            delivery_time=2,
            name="o1",
        ),
    ]
    current_t = 0
    # When
    (
        total_cost,
        transport_matrix,
        all_movements,
        big_m_per_commodity,
        big_m_per_commodity_units,
    ) = network_flow_k_optimizer.optimize(
        physical_network, inventory, fixed, open, current_t
    )
    print("--")
    print("total_cost\n", total_cost)
    print("transport_matrix\n", transport_matrix)
    print("all_movements\n", all_movements)
    print("big_m_per_commodity\n", big_m_per_commodity)
    print("--")
    # Then
    print("DONE")
    assert total_cost == 11.0  # TODO double check that this cost is correct.
