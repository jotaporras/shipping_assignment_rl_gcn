import gym
from collections import namedtuple
from functools import reduce
from shipping_allocation.envs.rewards import RewardFunction
from shipping_allocation.envs.shipping_assignment_state import ShippingAssignmentState

from typing import List

import typing
import numpy as np
import copy

# Environment and agent
import gym
from envs.inventory_generators import InventoryGenerator
from envs.order_generators import OrderGenerator
from gym import spaces

import logging

from experiment_utils import network_flow_k_optimizer, report_generator, Orders
from network import physical_network
from experiment_utils.Order import Order
import time
import datetime

# TODO delete
DEBUG = True


class ShippingAssignmentEnvironment(gym.Env):
    """Next generation (July  2021) Shipping Assignment Environment,
    AKA env version v2.

    The environment has no awareness of the episodes. A structure above it must manage
    any multi episode logic&metrics.
    """

    metadata = {"render.modes": ["human"]}

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        physical_network: physical_network,
        order_generator: OrderGenerator,
        inventory_generator: InventoryGenerator,
        reward_function: RewardFunction,
        num_steps: int,
    ):
        super(ShippingAssignmentEnvironment, self).__init__()
        self.physical_network = physical_network
        self.num_steps = num_steps
        self.order_generator = order_generator
        self.inventory_generator = inventory_generator
        self.reward_function = reward_function

        # Mutable state
        self.all_movements_history = []
        self.fixed_orders = []  # Orders already fixed for delivery
        self.open_orders = []  # Orders not yet decided upon.
        self.previous_state = None
        self.current_state = None
        self.state_history = []
        self.current_t = 0
        # self.action_space = spaces.Discrete(
        #     self.physical_network.num_dcs
        # )  # The action space is choosing a DC for the current order.

        self.action_space = ShippingAssignmentSpace(
            self.physical_network
        )  # New action space with custom sample
        self.inventory = np.zeros(
            (
                self.physical_network.num_dcs,
                self.physical_network.num_commodities,
            )
        )  # Matrix of inventory for each dc-k.

        # Transports accumulator for the current optimization, to update inventory movements. not tested.
        self.transports_acc = np.zeros(self.inventory.shape)

        # =====OBSERVATION SPEC=======
        # Current observation spec:
        # inventory per dc plus one K size vector of the current order
        # + one K size vector of other demand in horizon
        # + 4 metadata neurons
        dcs = self.physical_network.num_dcs
        commodities = self.physical_network.num_commodities
        shape = (1, dcs * commodities + 2 * commodities + 4)
        self.observation_space = spaces.Box(0, 1000000, shape=shape)
        # =====OBSERVATION SPEC=======

        # Debug vars
        self.approx_transport_mvmt_list = []
        self.total_costs = []
        self.total_rewards = []
        self.info = {}

        self.logger.debug("Calling init on the ShippingFacilityEnvironment")

    # Taking a step forward after the agent selects an action for the current state.
    def step(self, action):
        # Choose the shipping point for the selected order and fix the order.
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            self.logger.debug("=============================================")
            self.logger.debug(
                f"===============> STARTING ENVIRONMENT STEP {self.current_t}"
            )
            self.logger.debug("***Demand summary start of step***")
            self._check_if_demand_equals_supply()
            self.logger.debug("=============================================")
            # self.logger.debug(f"Received action {action}")
            # self.logger.debug("Pre env.step render:")
            # self.logger.debug("Current state: ",self.current_state)
            # self.render()

        # "Setting shipping point",action," for order ",self.open_orders[0])
        new_shipping_point = self.physical_network.dcs[action]
        self.open_orders[0].shipping_point = new_shipping_point
        self.fixed_orders = self.fixed_orders + [self.open_orders[0]]
        # self.current_state["fixed"] = self.fixed_orders  # TODO cleanup state update

        # Remove it from open locations
        self.open_orders = self.open_orders[1:]
        # self.current_state["open"] = self.open_orders  # TODO cleanup state update

        (
            total_cost,
            transports,
            all_movements,
            big_m_counter_per_commodity,
            big_m_units_per_commodity,
        ) = network_flow_k_optimizer.optimize(
            self.physical_network,
            self.inventory,
            self.fixed_orders,  # TODO: consider? + self.open_orders,
            [],
            self.current_t,
        )  # todo if we ever create another environment, we'll need the return of this to be an object.
        self.logger.debug(
            f"Optimization cost {total_cost}, big M count {big_m_counter_per_commodity}, big M units {big_m_units_per_commodity}"
        )
        self.transports_acc = transports

        self.assign_next_state_and_step_t(
            big_m_counter_per_commodity, total_cost, big_m_units_per_commodity
        )

        reward = self.reward_function.calculate_reward(
            self.previous_state, self.current_state
        )

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Got reward: {reward} for action {action}.")
            latest_order = self.fixed_orders[-1]
            valid_dcs = self.physical_network.get_valid_dcs(
                latest_order.customer.node_id
            )
            self.logger.debug(f"Latest order: {latest_order}")
            self.logger.debug(
                f"Action valid for last order {np.isin(action,valid_dcs)}"
            )

            self.logger.debug(f"Latest order's valid DCs: {valid_dcs}")
            self.logger.debug(
                f"Previous state Big M: {sum(self.previous_state.big_m_counter_per_commodity)}"
            )
            self.logger.debug(
                f"Current state Big M: {sum(self.current_state.big_m_counter_per_commodity)}"
            )

            self.logger.debug(f"All fixed: {self.fixed_orders}")
            self.logger.debug(f"Open left: {self.open_orders}")

            self.logger.debug("***Demand summary end of step***")
            self._check_if_demand_equals_supply()
            # self.logger.debug("Current open order demand:")
            # self.logger.debug(self.open_orders[0].demand)
            # self.logger.debug("Current total fixed demand:")
            # total_fixed_demand = reduce(
            #     lambda a, b: a + b, map(lambda o: o.demand, self.fixed_orders)
            # )
            # self.logger.debug(total_fixed_demand)
            # self.logger.debug("Current inventories: ")
            # self.logger.debug(self.inventory)
            # self.logger.debug("(Sanity) Sum of total demand")
            # demand_sum = sum(self.open_orders[0].demand + total_fixed_demand)
            # self.logger.debug(demand_sum)
            # self.logger.debug("(Sanity) Sum of total inventory")
            # inventory_sum = self.inventory.sum(axis=0)
            # self.logger.debug(inventory_sum)
            # demand_equals_inventory = np.isclose(demand_sum, inventory_sum)
            # self.logger.debug(f"Inventory equals demand: {demand_equals_inventory}")

        # Adding approximate transport cost by taking transport matrices where transports are greater than zero.
        # Assumes Customer transport == DC transport cost.
        # Only append movements after open orders have been depleted, meaning once per day.
        if len(self.open_orders) == 1:  # it was the last order
            self.all_movements_history.append(all_movements)
        self.approximate_transport_cost = (
            transports[np.where(transports > 0)].sum()
            * self.physical_network.default_customer_transport_cost
        )
        self.total_costs.append(total_cost)
        self.total_rewards.append(reward)
        self.approx_transport_mvmt_list.append(self.approximate_transport_cost)

        # Done when the current_t is equal to the number of steps per episode.
        done = self.current_t == self.num_steps + 1

        if done:
            # Appending final values to info object.
            final_ords: List[Order] = self.current_state.fixed

            movement_detail_report = report_generator.generate_movement_detail_report(
                self.all_movements_history,
                self.physical_network.big_m_cost,
            )
            summary_movement_report = report_generator.generate_summary_movement_report(
                movement_detail_report
            )

            served_demand = sum([sum(o.demand) for o in final_ords])
            approximate_to_customer_cost = (
                served_demand * self.physical_network.default_customer_transport_cost
            )

            # Stopping episode timer
            episode_end_time = time.process_time_ns()
            time_elapsed_ns = episode_end_time - self.episode_process_start_time
            time_elapsed_s = float(time_elapsed_ns) / 1e9
            time_pretty = datetime.timedelta(seconds=time_elapsed_s)
            self.logger.debug(f"Episode took: {time_elapsed_s} seconds.")
            self.logger.info(f"Episode took: {time_pretty}")

            self.info = {
                "final_orders": final_ords,  # The orders, with their final shipping point destinations.
                "total_costs": self.total_costs,  # Total costs per stepwise optimization
                "approximate_transport_movement_list": self.total_costs,  # Total costs per stepwise optimization
                "approximate_to_customer_cost": approximate_to_customer_cost,  # Approximate cost of shipping to customers: total demand multiplied by the default customer transport cost. If the cost is different, this is worthless.
                "movement_detail_report": movement_detail_report,  # DataFrame with all the movements that were made.
                "summary_movement_report": summary_movement_report,  # DataFrame with movements summary per day.
                "episode_process_time_ns": time_elapsed_ns,
                "episode_process_time_s": time_elapsed_s,
            }
        else:
            self.info = (
                {}
            )  # not done yet. #to do consider yielding this on every step for rendering purposes.
        return copy.copy(self.current_state), reward, done, self.info

    def generate_final_statistics(self):
        pass

    def reset(self):
        self.logger.debug("RESETTING ENVIRONMENT")
        # Reset the state of the environment to an initial state
        # self.logger.debug("Physical network for new env: ")
        # self.logger.debug(self.network)
        # self.logger.debug("Reseting environment")
        self.inventory = np.zeros(
            (
                self.physical_network.num_dcs,
                self.physical_network.num_commodities,
            )
        )  # Matrix of inventory for each dc-k.
        self.all_movements_history = []
        self.fixed_orders = []  # Orders already fixed for delivery
        self.open_orders = []

        self.current_t = 0

        # immediately update t, so first valid T is 1.
        self.assign_next_state_and_step_t()

        # debug var
        self.approx_transport_mvmt_list = []
        self.total_costs = []
        self.total_rewards = []

        # For calculating episode time elapsed.
        self.episode_process_start_time = time.process_time_ns()

        return copy.copy(self.current_state)

    def assign_next_state_and_step_t(
        self,
        big_m_counter_per_commodity=[0],
        total_cost=0,
        big_m_units_per_commodity=[0],
    ) -> None:
        """
        This function triggers the update of current_t, inventory, generation of new open orders
        and updates the current_state and previous_satate variables. Should be called
        in two places: in the environment's reset and at the end of the environment's step.
        Returns:

        """
        if self.previous_state:
            self.state_history.append(self.previous_state)
        self.previous_state = self.current_state

        self._step_t_update_inventory_and_orders()
        state_vector = self.create_state_vector()

        self.current_state = ShippingAssignmentState(
            self.current_t,
            self.physical_network,
            self.fixed_orders,
            self.open_orders,
            self.inventory,
            state_vector,
            big_m_counter_per_commodity,
            total_cost,
            big_m_units_per_commodity,
        )

    def _step_t_update_inventory_and_orders(self):
        # Create new orders if necessary
        if len(self.open_orders) == 0:
            # if self.current_t != 0: #Dont update the T if it's the start of the run/ #TODO VALIDATE THIS MIGHT BE AN ISSUE!!!!!!!

            consumed_inventory = self._calculate_consumed_inventory()
            self.current_t += 1
            new_orders = self._generate_orders()
            self.open_orders = self.open_orders + new_orders
            self.inventory = self._generate_updated_inventory(consumed_inventory)

            if (self.transports_acc > 0).any():
                # self.logger.debug("Applying transports!!! Transports:***")
                # self.logger.debug(self.transports_acc)
                self.inventory += self.transports_acc
                # self.logger.debug("New inventory after transports")
                # self.logger.debug(self.inventory)
                # self.logger.debug("setting all to zero again")
                self.transports_acc[:, :] = 0
                # self.logger.debug(self.transports_acc)

            if (self.inventory < 0).any():
                self.logger.error("THIS SHOULDNT HAPPEN!!!!! NEGATIVE INVENTORY")
                self.logger.error(self.inventory)
                raise Exception("THIS SHOULDNT HAPPEN!!!!! NEGATIVE INVENTORY")
        # else:
        #     self.inventory = self._generate_updated_inventory(0)
        generated_state = {
            "physical_network": self.physical_network,
            "inventory": self.inventory.copy(),
            "open": [copy.deepcopy(o) for o in self.open_orders],
            "fixed": [copy.deepcopy(o) for o in self.fixed_orders],
            "current_t": self.current_t,
        }

        return generated_state

    def _generate_orders(self) -> typing.List[Order]:
        # self.logger.debug(f"Calling order generator for t={self.current_t}")
        return self.order_generator.generate_orders(self.current_t)

    def _generate_updated_inventory(self, consumed_inventory):
        new_inventory = self.inventory_generator.generate_new_inventory(
            self.physical_network,
            # TODO: hack, we're just going to replenish the first order.!!!!!
            self.open_orders[0:],
        )  # must keep shape
        return self.inventory + new_inventory - consumed_inventory

    def create_state_vector(self):  # copied from first dqn agent #TODO TEST
        # TODO extract function from env. Agents should be responsible of calculating this.d
        # Inventory stack
        stacked_inventory = self.inventory.reshape(-1, 1)

        # latest open order demand
        latest_open_order = self.open_orders[0]
        reshaped_demand = latest_open_order.demand.reshape(-1, 1) * -1

        # Calculating stacked demand in horizon.
        horizon = self.current_t + self.physical_network.planning_horizon - 1
        stacked_demand_in_horizon = (
            Orders.summarize_order_demand(
                self.fixed_orders, self.current_t, horizon, reshaped_demand.shape
            )
            * -1
        )

        # 4 extra metadata neurons.
        ship_id = latest_open_order.shipping_point.node_id
        customer_id = latest_open_order.customer.node_id
        current_t = self.current_t
        delivery_t = latest_open_order.due_timestep
        metadata = np.array([[ship_id, customer_id, current_t, delivery_t]]).transpose()

        # State vector
        state_vector = np.concatenate(
            [stacked_inventory, reshaped_demand, stacked_demand_in_horizon, metadata]
        )

        return (
            state_vector.transpose()
        )  # np.array((1,num_dcs*num_commodities + num_commodities))

    def _calculate_consumed_inventory(self):
        """
        Consumed inventory is the inventory that will disappear this timelapse when the orders at current_t are delivered
        :return:
        """
        # self.logger.debug("Calculating consumed inventory")
        consumed = np.zeros(self.inventory.shape)
        for order in self.fixed_orders:
            if order.due_timestep == self.current_t:
                # self.logger.debug("Order",order.name,"is getting consumed on timelapse ",self.current_t," from ",order.shipping_point)
                consumed[order.shipping_point.node_id, :] += order.demand
        # self.logger.debug("Consumed inventory: ")telegram
        # self.logger.debug(consumed)
        return consumed

    def _render_state(self):
        # TODo fix it's rendering ugly
        if DEBUG:
            self.logger.debug("Rendering mutable part of the state")
            self.logger.debug("fixed: ", self.current_state.fixed)
            self.logger.debug("open: ", self.current_state.open)
            self.logger.debug("inventory: ", self.current_state.inventory)
            self.logger.debug(f"current_t: {self.current_state.current_t}")

    def render(self, mode="human", close=False):
        pass
        # self._render_state()

    ## debug methods
    def _calculate_total_inventory_per_k(self):
        return self.inventory.sum(axis=0)

    def _calculate_demand_summary_per_k(self, order_set):
        return reduce(
            lambda a, b: a + b,
            map(lambda o: o.demand, order_set),
            np.zeros(self.physical_network.num_commodities),
        )

    def _check_if_demand_equals_supply(self):
        inventory = self._calculate_total_inventory_per_k()
        open_sum = self._calculate_demand_summary_per_k(self.open_orders)
        horizon = self.current_t + self.physical_network.planning_horizon - 1
        fixed_sum = Orders.summarize_order_demand(  # todo redundant code watchout
            self.fixed_orders,
            self.current_t,
            horizon,
            self.physical_network.num_commodities,
        )

        total_demand = open_sum + fixed_sum

        inventory_eq_demand = np.isclose(inventory, total_demand)

        self.logger.debug("Open order sum")
        self.logger.debug(open_sum)
        self.logger.debug("Fixed order sum")
        self.logger.debug(fixed_sum)
        self.logger.debug("Total demand")
        self.logger.debug(total_demand)
        self.logger.debug("Inventory sum")
        self.logger.debug(inventory)
        self.logger.debug(f"Inventory matches demand: {inventory_eq_demand}")


class ShippingAssignmentSpace(gym.Space):
    # passing the whole physical network may be tech debt but ok.
    def __init__(self, physical_network):
        self.physical_network = physical_network  # (dc,customer)
        self.dcs_per_customer_array = physical_network.dcs_per_customer_array
        super(ShippingAssignmentSpace, self).__init__((), np.int64)

    def sample(self, customer_node_id):
        (valid_dcs,) = self.physical_network.get_valid_dcs(customer_node_id)
        return np.random.choice(valid_dcs)

    def contains(self, x):
        return True  # TODO a more meaningful representation if necessary.

    def __repr__(self):
        return "ShippingAssignmentSpace"  # TODO a more meaningful representation if necessary.

    def __eq__(self, other):
        return (
            isinstance(other, ShippingAssignmentSpace)
            and (self.dcs_per_customer_array == other.dcs_per_customer_array).all()
        )
