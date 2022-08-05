from network.physical_network import Node, PhysicalNetwork
from experiment_utils import Orders
from experiment_utils.Order import Order
import numpy as np


def fixture():
    dummy_customer = Node(4, 100, 0, 0, "dc")
    dc_0 = Node(0, 100, 0, 0, "dc")
    dc_1 = Node(1, 100, 0, 0, "dc")
    fixed_orders = [
        # A total 105,60 to DC 0,10,10 to DC 1, And one order outside of horizon.
        Order(
            np.array([50.0, 30.0, 30.0]),
            dc_0,
            dummy_customer,
            0,
            "someord",
        ),
        Order(
            np.array([55.0, 30.0, 30.0]),
            dc_0,
            dummy_customer,
            1,
            "someord",
        ),
        Order(
            np.array([10.0, 10.0, 10.0]),
            dc_1,
            dummy_customer,
            1,
            "someord",
        ),
        Order(
            np.array([10.0, 10.0, 10.0]),
            dc_1,
            dummy_customer,
            4,
            "someord",
        ),
    ]
    return fixed_orders


def test_summarize_order_demand_per_dc():
    # Given
    # physical_network = PhysicalNetwork(3, 5, 2, 100, 50, num_commodities=3)
    dummy_customer = Node(4, 100, 0, 0, "dc")
    dc_0 = Node(0, 100, 0, 0, "dc")
    dc_1 = Node(1, 100, 0, 0, "dc")
    fixed_orders = [
        # A total 105,60 to DC 0,10,10 to DC 1, And one order outside of horizon.
        Order(
            np.array([50.0, 30.0, 30.0]),
            dc_0,
            dummy_customer,
            0,
            "someord",
        ),
        Order(
            np.array([55.0, 30.0, 30.0]),
            dc_0,
            dummy_customer,
            1,
            "someord",
        ),
        Order(
            np.array([10.0, 10.0, 10.0]),
            dc_1,
            dummy_customer,
            1,
            "someord",
        ),
        Order(
            np.array([10.0, 10.0, 10.0]),
            dc_1,
            dummy_customer,
            4,
            "someord",
        ),
    ]

    # When
    demand_per_dc = Orders.summarize_order_demand_per_dc(
        fixed_orders, start=0, end=3, num_dcs=2, num_commodities=3
    )

    # Then
    assert (
        demand_per_dc
        == np.array(
            [
                [105.0, 60.0, 60.0],
                [10.0, 10.0, 10.0],
            ]
        )
    ).all()


def test_summarize_demand_per_customer_in_horizon():
    # Given
    physical_network = PhysicalNetwork(
        num_dcs=3,
        num_customers=5,
        dcs_per_customer=2,
        demand_mean=100,
        demand_var=50,
        num_commodities=3,
    )
    fixed_orders = fixture()

    c0 = Node(3, 0, 0, 0, "dc", name="c0")
    c1 = Node(4, 0, 0, 0, "dc", name="c1")
    c3 = Node(6, 0, 0, 0, "dc", name="c3")
    dc_0 = Node(0, 0, 0, 0, "dc", name="dc0")
    dc_1 = Node(1, 0, 0, 0, "dc", name="dc1")
    fixed_orders = [
        # A total 105,60 to DC 0,10,10 to DC 1, And one order outside of horizon.
        Order(
            np.array([50.0, 30.0, 30.0]),
            dc_0,
            c0,
            0,
            "someord",
        ),
        Order(
            np.array([55.0, 30.0, 30.0]),
            dc_0,
            c1,
            1,
            "someord",
        ),
        Order(
            np.array([10.0, 10.0, 10.0]),
            dc_1,
            c3,
            1,
            "someord",
        ),
        Order(
            np.array([10.0, 10.0, 10.0]),
            dc_1,
            c3,
            4,
            "someord",
        ),
        Order(
            np.array([666.0, 666.0, 666.0]),
            dc_1,
            c0,
            5,
            "someord",
        ),
    ]

    # When
    demand_summary = Orders.summarize_demand_per_customer_in_horizon(
        fixed_orders,
        start=0,
        end=4,
        num_customers=5,
        num_commodities=3,
        physical_network=physical_network,
    )

    # Then there should be demand on rows 0,1,3. Third row is the sum of two orders.
    assert (
        demand_summary
        == np.array(
            [
                [-50.0, -30.0, -30.0],
                [-55.0, -30.0, -30.0],
                [-0.0, -0.0, -0.0],
                [-20.0, -20.0, -20.0],
                [-0.0, -0.0, -0.0],
            ]
        )
    ).all()
