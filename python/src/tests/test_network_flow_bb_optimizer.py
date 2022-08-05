import datetime
import logging

import logging
import time

import numpy as np
from envs import rewards
from envs.inventory_generators import DirichletInventoryGenerator
from envs.order_generators import NormalOrderGenerator
from network.ExtendedNetwork import ExtendedNetwork
from network.physical_network import PhysicalNetwork

from experiment_utils import network_flow_bb_optimizer
from experiment_utils.Order import Order


def test_optimize_bb_one_order():
    # Given
    logging.root.setLevel(logging.DEBUG)
    physical_network = PhysicalNetwork(
        num_dcs=3,
        num_customers=1,
        dcs_per_customer=2,
        demand_mean=100,
        demand_var=25,
        big_m_factor=10000,
        num_commodities=2,
        planning_horizon=4,
    )
    # (num_dcs,num_commodities)
    # fmt: off
    inventory = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )
    # fmt: on

    customer_node = physical_network.customers[0]
    dc_0_node = physical_network.dcs[0]
    open = [
        Order(
            demand=np.array([2.0, 1.0]),
            shipping_point=dc_0_node,
            customer=customer_node,
            delivery_time=3,
            name="o1",
        )
    ]
    fixed = []
    current_t = 1
    (
        extended_nodes,
        extended_arcs_with_flow,
    ) = network_flow_bb_optimizer.optimize_branch_and_bound(
        physical_network, inventory, fixed, open, current_t
    )

    next_actions = network_flow_bb_optimizer.bb_solution_to_agent_action(
        open, extended_arcs_with_flow
    )
    print("Soulitionr")
    print(next_actions)


def test_optimize_bb__scalability_test():
    logging.root.setLevel(logging.DEBUG)
    pn = PhysicalNetwork(
        num_dcs=5,
        num_customers=250,
        dcs_per_customer=2,
        demand_mean=100,
        demand_var=25,
        big_m_factor=10000,
        num_commodities=5,
        planning_horizon=5,
    )
    orders_per_day = 5
    order_generator = NormalOrderGenerator(pn, orders_per_day)
    inventory_generator = DirichletInventoryGenerator(pn)

    # reward_function = rewards.reward_chooser("negative_log_cost_minus_log_big_m_units")
    # ShippingAssignmentEnvironment(
    #   physical_network,
    #   order_generator,
    #   inventory_generator,
    #   reward_function,
    #   num_steps=3,
    # )
    current_t = 1
    open = order_generator.generate_orders(current_t)
    fixed = []
    inventory = inventory_generator.generate_new_inventory(pn, open)

    start = time.process_time()
    (
        extended_nodes,
        extended_arcs_with_flow,
    ) = network_flow_bb_optimizer.optimize_branch_and_bound(
        pn, inventory, fixed, open, current_t
    )

    next_actions = network_flow_bb_optimizer.bb_solution_to_agent_action(
        open, extended_arcs_with_flow
    )
    end = time.process_time()
    elapsed = datetime.timedelta(seconds=end - start)
    size = f"{pn.num_dcs}W{pn.num_customers}C{orders_per_day}D{pn.planning_horizon}PH"
    print(f"Optimization of size {size} took {str(elapsed)}")
    print("Soulitionr")
    print(
        next_actions
    ) 
