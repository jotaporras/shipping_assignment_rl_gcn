"""
This module contains the implementations of optimization-based  agents, leveraging the Ortools based linear opt or branch & bound
"""
import datetime
import itertools

import numpy as np
from envs.shipping_assignment_env import ShippingAssignmentEnvironment
from envs.shipping_assignment_state import (
    ShippingAssignmentState,
    state_to_fixed_demand,
)

from agents.Agent import Agent
from experiment_utils import network_flow_k_optimizer
import copy
import logging
from experiment_utils import network_flow_bb_optimizer
from experiment_utils.Order import Order
import time

logger = logging.getLogger(__name__)


class LookaheadAgent(Agent):
    """a
    Attempt of creating an agent that does a lookahead on the cost of assigning to each center.
    """

    env: ShippingAssignmentEnvironment

    def __init__(self, env):
        self.env = env  # todo not sure if needed

    def get_action(self, state: ShippingAssignmentState):
        current_open_order: Order = copy.deepcopy(state.open[0])

        customer_id = current_open_order.customer.node_id
        (valid_dcs_for_customer,) = state.physical_network.get_valid_dcs(customer_id)
        logger.debug(f"Evaluating valid DCS {valid_dcs_for_customer}")

        # unpacking the tuple due to the behavior of np where. TODO fix

        # Try every DC and choocted 2, got 1)se the lowest optimization cost.
        total_cost_acc = []
        for valid_dc in valid_dcs_for_customer:
            current_open_order: Order = copy.deepcopy(state.open[0])
            current_open_order.shipping_point = state.physical_network.dcs[valid_dc]

            (
                total_cost,
                transport_matrix,
                all_movements,
                big_m_per_commodity,
                big_m_units_per_commodity,
            ) = network_flow_k_optimizer.optimize(
                state.physical_network,
                state.inventory,
                state.fixed + [current_open_order],
                # TODO: as of Aug 24 2021, open orders are not considered at all on optimization.
                state.open[1:],
                state.current_t,
            )
            total_cost_acc.append(total_cost)
            logger.debug(
                f"Tried shipping point {valid_dc} for order {current_open_order}"
            )
            logger.debug(f"Got cost {total_cost}")
        best_action_idx = min(enumerate(total_cost_acc), key=lambda x: x[1])[0]
        best_dc = valid_dcs_for_customer[best_action_idx]
        logger.debug("For debugging")
        cost_per_dc = np.zeros(state.physical_network.num_dcs).reshape(
            -1, 1
        )  # only for debugging
        for i, dci in enumerate(valid_dcs_for_customer):
            cost_per_dc[dci] = total_cost_acc[i]
        logger.debug(f"Considering optimization costs {total_cost_acc}")
        logger.debug(
            f"Best action in lookahead is {best_dc} with cost {total_cost_acc[best_action_idx]}"
        )
        logger.debug("For manual debug of lookahead")
        logger.debug("Inventories (DC,Commodity)")
        logger.debug(f"\n{state.inventory}")
        logger.debug("Cost per choice (DC)")
        logger.debug(f"\n{cost_per_dc}")
        logger.debug("Latest Demand (Commodity)")
        logger.debug(f"\n{state.open[0].demand}")
        logger.debug("Fixed agg demand ")
        logger.debug(f"\n{state_to_fixed_demand(state)}")

        return best_dc

    def train(self, experience):
        pass


class TreeSearchAgent(Agent):
    """Evaluate cost of combinations of orders"""

    env: ShippingAssignmentEnvironment

    def __init__(self, env):
        self.env = env  # todo not sure if needed

    def get_action(self, state):
        # Assumes all customers have same number of DCs
        open_orders = state.open

        num_orders = len(open_orders)

        num_warehouses = state.physical_network.dcs_per_customer
        # Find choices for each warehouse.
        choices_per_order = []
        for o in open_orders:
            choices_per_order.append(
                state.physical_network.get_valid_dcs(o.customer.node_id)[
                    0
                ]  # The zero is to unset the tuple format that results from the np where
            )
        cost_array_dims = [num_warehouses] * num_orders
        cost_array = np.zeros(cost_array_dims)
        choice_per_order_indices = [list(range(num_warehouses))] * num_orders
        order_warehouse_idx_combinations = itertools.product(
            *choice_per_order_indices
        )  # the index of the warehouse choice to extrct of choices_per_order
        # logger.debug(f"Exploring {len(order_warehouse_idx_combinations)}")
        logger.debug(order_warehouse_idx_combinations)
        for warehouses_per_order in order_warehouse_idx_combinations:
            open_orders_copy = copy.deepcopy(open_orders)
            for o_i, w_i in enumerate(warehouses_per_order):
                # Creating
                open_orders_copy[o_i].shipping_point = state.physical_network.dcs[
                    choices_per_order[o_i][w_i]
                ]  # notice the index translation
            # Running optimization for this combination.
            (total_cost, _, _, _, _) = network_flow_k_optimizer.optimize(
                state.physical_network,
                state.inventory,
                state.fixed + open_orders_copy,
                [],
                state.current_t,
            )

            cost_array[warehouses_per_order] = total_cost

        dims_best = np.where(
            cost_array == cost_array.min()
        )  # todo what if multiple mins.
        best_action_idx = dims_best[0][
            0
        ]  # Get the first instance of the min for the first order.
        best_action = choices_per_order[0][best_action_idx]

        logger.debug("Choosing between the following costs:")
        logger.debug(cost_array)
        logger.debug(
            f"Chose action {best_action} with cost {cost_array[dims_best]} position {dims_best}"
        )

        return best_action

    def train(self, experience):
        pass


class BranchAndBoundAgent(Agent):
    """The world's best deterministic optimizer! Solves a MILP using B&BB"""


    open_order_actions = {}

    def __init__(self, env, time_limit_milliseconds=240 * 1000):
        super().__init__(env)
        self.env = env
        self.network = env.physical_network
        self.time_limit_milliseconds = time_limit_milliseconds
        self.lookahead_backup = LookaheadAgent(env)

    def get_action(self, state: ShippingAssignmentState):
        # If no actions are cached
        latest_open_order_key = state.open[0].order_key()
        logger.debug(f"Running with time limit: {self.time_limit_milliseconds}")
        if latest_open_order_key not in self.open_order_actions:
            # logger.debug("Running B&B optimization")
            start = time.process_time()
            try:
                (
                    extended_nodes,
                    extended_arcs_with_flow,
                ) = network_flow_bb_optimizer.optimize_branch_and_bound(
                    state.physical_network,
                    state.inventory,
                    state.fixed,
                    state.open,
                    state.current_t,
                )
                self.open_order_actions = (
                    network_flow_bb_optimizer.bb_solution_to_agent_action(
                        state.open, extended_arcs_with_flow
                    )
                )
            except Exception as e:
                logger.error(e)
                logger.error(
                    "FATAL, GOT ERROR WHEN BNB OPTIMIZING. DEFAULTING TO LOOKAHEAD"
                )
                return self.lookahead_backup.get_action(state)
            end = time.process_time()
            elapsed = datetime.timedelta(seconds=end - start)
            # logger.debug(f"B&B optimization took, took {elapsed}")

        return self.open_order_actions[latest_open_order_key]
