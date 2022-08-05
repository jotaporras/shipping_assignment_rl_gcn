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

# tf.disable_v2_behavior()

# Custome
from experiment_utils import network_flow_k_optimizer, report_generator, Orders
from network import physical_network
from experiment_utils.Order import Order

DEBUG = False


# All the immutable things that affect environment behavior, maybe needs more parameters?
class EnvironmentParameters:
    def __init__(
        self,
        network: physical_network,
        order_generator: OrderGenerator,
        inventory_generator: InventoryGenerator,
        num_steps: int,
    ):
        self.network = network
        self.num_steps = num_steps
        self.order_generator = order_generator
        self.inventory_generator = inventory_generator


class ShippingFacilityEnvironment(gym.Env):
    """Custom Environment that follows gym interface. Deprecated, See ShippingAssignmentEnvironment"""

    open_orders: List[Order]
    metadata = {"render.modes": ["human"]}

    def __init__(self, environment_parameters: EnvironmentParameters):
        super(ShippingFacilityEnvironment, self).__init__()

        # For compatibility with the new way of accessing env stuff.
        self.physical_network = environment_parameters.network
        self.order_generator = environment_parameters.order_generator
        self.inventory_generator = environment_parameters.inventory_generator
        self.num_steps = environment_parameters.num_steps

        self.environment_parameters = environment_parameters
        self.all_movements_history = []
        self.fixed_orders = []  # Orders already fixed for delivery
        self.open_orders = []  # Orders not yet decided upon.
        self.current_state = {}
        self.current_t = 0
        self.action_space = spaces.Discrete(
            self.environment_parameters.network.num_dcs
        )  # The action space is choosing a DC for the current order.
        self.inventory = np.zeros(
            (
                self.environment_parameters.network.num_dcs,
                self.environment_parameters.network.num_commodities,
            )
        )  # Matrix of inventory for each dc-k.
        self.transports_acc = np.zeros(self.inventory.shape)

        # =====OBSERVATION SPEC=======
        # Current observation spec:
        # inventory per dc plus one K size vector of the current order
        # + one K size vector of other demand in horizon
        # + 4 metadata neurons
        dcs = self.environment_parameters.network.num_dcs
        commodities = self.environment_parameters.network.num_commodities
        shape = (1, dcs * commodities + 2 * commodities + 4)
        self.observation_space = spaces.Box(0, 1000000, shape=shape)
        # =====OBSERVATION SPEC=======

        self.last_cost = 0

        # Debug vars
        self.approx_transport_mvmt_list = []
        self.total_costs = []
        self.total_rewards = []
        self.info = {}

        logging.info("Calling init on the ShippingFacilityEnvironment")
        raise RuntimeError(
            "Didn't expect to use this env, change to the v2 or comment out this exception!"
        )

    # Taking a step forward after the agent selects an action for the current state.
    def step(self, action):
        # Choose the shipping point for the selected order and fix the order.
        if DEBUG:
            logging.info("\n=============================================")
            logging.info(f"===============> STARTING ENVIRONMENT STEP {self.current_t}")
            logging.info("=============================================")
            logging.info(f"Received action {action}")
            # logging.info("Pre env.step render:")
            # logging.info("Current state: ",self.current_state)
            # self.render()

        # "Setting shipping point",action," for order ",self.open_orders[0])
        new_shipping_point = self.environment_parameters.network.dcs[action]
        self.open_orders[0].shipping_point = new_shipping_point
        # logging.info("Order after seting action: ",self.open_orders[0])
        self.fixed_orders = self.fixed_orders + [self.open_orders[0]]
        self.current_state["fixed"] = self.fixed_orders  # TODO cleanup state update

        # Remove it from open locations
        self.open_orders = self.open_orders[1:]
        self.current_state["open"] = self.open_orders  # TODO cleanup state update

        (
            cost,
            transports,
            all_movements,
            big_m_per_commodity,
            big_m_units_per_commodity,
        ) = (
            self._run_simulation()
        )  # todo if we ever create another environment, we'll need the return of this to be an object.

        # calculating if the current order was a Big M.
        big_m_counter = sum(big_m_per_commodity)
        ##### REWARD FUNCTION #####
        reward = -1 * big_m_counter  # Big M Counter based reward 0.3.
        # reward = (self.last_cost - cost)*-1.0 # diff reward function 0.2
        # reward = cost * -1 # old reward function. 0.1
        ##### END REWARD FUNCTION #####

        self.last_cost = cost  # Necessary for reward 0.2

        # Adding approximate transport cost by taking transport matrices where transports are greater than zero. Assumes Customer transport == DC transport cost.
        # Only append movements after open orders have been depleted, meaning once per day.
        if len(self.open_orders) == 0:
            # logging.info("Finished the day, adding all movements")
            self.all_movements_history.append(all_movements)
        # else:
        #     logging.info("Still are open orders left, no report added")
        #     logging.info(f"Number of orders left {len(self.open_orders)}")
        self.approximate_transport_cost = (
            transports[np.where(transports > 0)].sum()
            * self.environment_parameters.network.default_customer_transport_cost
        )
        self.total_costs.append(cost)
        self.total_rewards.append(reward)
        self.approx_transport_mvmt_list.append(self.approximate_transport_cost)
        # logging.info("self.approx_transport_cost", self.approx_transport_cost)
        # logging.info("Total transports (customer+dc) transports")

        self.transports_acc = transports

        # update timestep and generate new locations if needed
        self.current_state = self._next_observation()

        # Done when the number of timestep generations is the number of episodes.
        done = self.current_t == self.environment_parameters.num_steps + 1

        if done:
            # Appending final values to info object.
            final_ords: List[Order] = self.current_state["fixed"]

            movement_detail_report = report_generator.generate_movement_detail_report(
                self.all_movements_history,
                self.environment_parameters.network.big_m_cost,
            )
            summary_movement_report = report_generator.generate_summary_movement_report(
                movement_detail_report
            )

            served_demand = sum([sum(o.demand) for o in final_ords])
            approximate_to_customer_cost = (
                served_demand
                * self.environment_parameters.network.default_customer_transport_cost
            )
            self.info = {
                "final_orders": final_ords,  # The orders, with their final shipping point destinations.
                "total_costs": self.total_costs,  # Total costs per stepwise optimization
                "approximate_transport_movement_list": self.total_costs,  # Total costs per stepwise optimization
                "approximate_to_customer_cost": approximate_to_customer_cost,  # Approximate cost of shipping to customers: total demand multiplied by the default customer transport cost. If the cost is different, this is worthless.
                "movement_detail_report": movement_detail_report,  # DataFrame with all the movements that were made.
                "summary_movement_report": summary_movement_report,  # DataFrame with movements summary per day.
            }
            # logging.info("==== Copy and paste this into a notebook ====")
            # logging.info("Total costs per stepwise optimization", self.total_costs)
            # logging.info("Total cost list associated with all transport movements", self.approx_transport_mvmt_list) #approximate because they're intermixed.
            # logging.info("Removing approx to customer cost", sum(self.approx_transport_mvmt_list)-approximate_to_customer_cost)
        else:
            self.info = (
                {}
            )  # not done yet. #to do consider yielding this on every step for rendering purposes.
        # logging.info(f"Stepping with action {action}")
        # obs = random.randint(0, 10)
        # reward = random.randint(0, 100)
        # done = np.random.choice([True, False])
        return copy.copy(self.current_state), reward, done, self.info

    # def observation_space(self):
    #     dcs = self.environment_parameters.network.num_dcs
    #     commodities = self.environment_parameters.network.num_commodities
    #     shape = (dcs * commodities+num_commodities, 1)
    #     return spaces.Box(0,1000000,shape=shape)

    def generate_final_statistics(self):
        pass

    def reset(self):
        # Reset the state of the environment to an initial state
        # logging.info("Physical network for new env: ")
        # logging.info(self.environment_parameters.network)
        # logging.info("Reseting environment")
        self.inventory = np.zeros(
            (
                self.environment_parameters.network.num_dcs,
                self.environment_parameters.network.num_commodities,
            )
        )  # Matrix of inventory for each dc-k.
        self.all_movements_history = []
        self.fixed_orders = []  # Orders already fixed for delivery
        self.open_orders = []

        self.current_t = 0
        self.current_state = self._next_observation()

        # debug var
        self.approx_transport_mvmt_list = []
        self.total_costs = []
        self.total_rewards = []

        return copy.copy(self.current_state)

    def render(self, mode="human", close=False):
        pass  # todo refactor this, it's too much noise
        # Render the environment to the screen
        # if mode == "human" and DEBUG:
        #     logging.info("\n\n======== RENDERING ======")
        #     logging.info("Current t",self.current_t)
        #     logging.info(f"fixed_orders ({len(self.fixed_orders)})", self.fixed_orders)
        #     logging.info(
        #         f"Demand fixed orders: {sum(map(lambda o:o.demand, self.fixed_orders))}"
        #     )  # TODO Do for all commodities
        #     logging.info(f"open_orders ({len(self.open_orders)})", self.open_orders)
        #     logging.info(
        #         f"Demand open orders: {sum(map(lambda o:o.demand, self.open_orders))}"
        #     )  # TODO Do for all commodities
        #     logging.info("inventory\n", self.inventory)
        #     logging.info("Current State:")
        #     self._render_state()
        #     logging.info("======== END RENDERING ======\n\n")

    def _render_state(self):
        if DEBUG:
            logging.info("Rendering mutable part of the state")
            logging.info("fixed: ", self.current_state["fixed"])
            logging.info("open: ", self.current_state["open"])
            logging.info("inventory: ", self.current_state["inventory"])
            logging.info("current_t: ", self.current_state["current_t"])

    def _next_observation(self):
        if len(self.open_orders) == 0:  # Create new locations if necessary
            # if self.current_t != 0: #Dont update the T if it's the start of the run/ #TODO VALIDATE THIS MIGHT BE AN ISSUE!!!!!!!

            consumed_inventory = self._calculate_consumed_inventory()
            self.current_t += 1
            new_orders = self._generate_orders()
            self.open_orders = self.open_orders + new_orders

            # logging.info("Updating inventory with orders")
            # logging.info("Before update: ")
            # logging.info(self.inventory)

            self.inventory = self._generate_updated_inventory(consumed_inventory)

            # logging.info("inventory after orders before transports")
            # logging.info(self.inventory)

            if (self.transports_acc > 0).any():
                # logging.info("Applying transports!!! Transports:***")
                # logging.info(self.transports_acc)
                self.inventory += self.transports_acc
                # logging.info("New inventory after transports")
                # logging.info(self.inventory)
                # logging.info("setting all to zero again")
                self.transports_acc[:, :] = 0
                # logging.info(self.transports_acc)

            if (self.inventory < 0).any():
                logging.info("THIS SHOULDNT HAPPEN!!!!! NEGATIVE INVENTORY")
                logging.info(self.inventory)
                raise Exception("THIS SHOULDNT HAPPEN!!!!! NEGATIVE INVENTORY")
        # else:
        #     self.inventory = self._generate_updated_inventory(0)
        generated_state = {
            "physical_network": self.environment_parameters.network,
            "inventory": self.inventory.copy(),
            "open": [copy.deepcopy(o) for o in self.open_orders],
            "fixed": [copy.deepcopy(o) for o in self.fixed_orders],
            "current_t": self.current_t,
        }
        generated_state["state_vector"] = self.convert_state_to_vector(
            generated_state
        )  # todo refactor to something less nasty.
        return generated_state

    def convert_state_to_vector(self, state):  # copied from first dqn agent
        # Inventory stack
        inventory = state["inventory"]
        stacked_inventory = inventory.reshape(-1, 1)

        # latest open order demand
        latest_open_order = state["open"][0]
        reshaped_demand = latest_open_order.demand.reshape(-1, 1) * -1

        # Calculating stacked demand in horizon.
        fixed_orders: List[Order] = state["fixed"]
        current_t = state["current_t"]
        network: physical_network = state["physical_network"]
        horizon = current_t + network.planning_horizon - 1
        stacked_demand_in_horizon = (
            Orders.summarize_order_demand(
                fixed_orders, current_t, horizon, reshaped_demand.shape
            )
            * -1
        )

        # 4 extra metadata neurons.
        ship_id = latest_open_order.shipping_point.node_id
        customer_id = latest_open_order.customer.node_id
        current_t = state["current_t"]
        delivery_t = latest_open_order.due_timestep
        metadata = np.array([[ship_id, customer_id, current_t, delivery_t]]).transpose()

        # State vector
        state_vector = np.concatenate(
            [stacked_inventory, reshaped_demand, stacked_demand_in_horizon, metadata]
        )

        return (
            state_vector.transpose()
        )  # np.array((1,num_dcs*num_commodities + num_commodities))

    def _generate_orders(self) -> typing.List[Order]:
        # logging.info(f"Calling order generator for t={self.current_t}")
        return self.environment_parameters.order_generator.generate_orders(
            self.current_t
        )

    def _generate_updated_inventory(self, consumed_inventory):
        new_inventory = (
            self.environment_parameters.inventory_generator.generate_new_inventory(
                self.environment_parameters.network, self.open_orders
            )
        )  # must keep shape
        return self.inventory + new_inventory - consumed_inventory

    def _calculate_consumed_inventory(self):
        """
        Consumed inventory is the inventory that will disappear this timelapse when the orders at current_t are delivered
        :return:
        """
        # logging.info("Calculating consumed inventory")
        consumed = np.zeros(self.inventory.shape)
        for order in self.fixed_orders:
            if order.due_timestep == self.current_t:
                # logging.info("Order",order.name,"is getting consumed on timelapse ",self.current_t," from ",order.shipping_point)
                consumed[order.shipping_point.node_id, :] += order.demand
        # logging.info("Consumed inventory: ")
        # logging.info(consumed)
        return consumed

    def _run_simulation(self):
        # todo refactored from just passing state but not tested (legacy)
        return network_flow_k_optimizer.optimize(
            self.current_state["physical_network"],
            self.current_state["inventory"],
            self.current_state["fixed"],
            self.current_state["open"],
            self.current_state["current_t"],
        )


# Naive implementations of inventory and order generators to illustrate.


# if __name__ == "__main__":
#     num_dcs = 2
#     num_customers = 1
#     num_commodities = 3
#     orders_per_day = 1
#     dcs_per_customer = 1
#     demand_mean = 100
#     demand_var = 20
#
#     num_episodes = 5
#
#     physical_network = PhysicalNetwork(num_dcs, num_customers, dcs_per_customer,demand_mean,demand_var,num_commodities)
#     # order_generator = NaiveOrderGenerator(num_dcs, num_customers, orders_per_day)
#     order_generator = ActualOrderGenerator(physical_network, orders_per_day)
#     generator = NaiveInventoryGenerator()
#     environment_parameters = EnvironmentParameters(
#         physical_network, num_episodes, order_generator, generator
#     )
#
#     env = ShippingFacilityEnvironment(environment_parameters)
#     agent = QNAgent(env)
#
#     state = env.reset()
#     reward = 0
#     done = False
#     logging.info("=========== starting episode loop ===========")
#     logging.info("Initial environment: ")
#     env.render()
#     while not done:
#         action = agent.get_action((state, reward))
#         logging.info(f"Agent is taking action: {action}")
#         # the agent observes the first state and chooses an action
#         # environment steps with the agent's action and returns new state and reward
#         next_state, reward, done, info = env.step(action)
#         logging.info(f"Got reward {reward} done {done}")
#
#         agent.train((state,action,next_state,reward,done))
#
#         state = next_state
#         # Render the current state of the environment
#         env.render()
#
#         if done:
#             logging.info("===========Environment says we are DONE ===========")


# TODO aqui quede, works on first step pero no tengo la ventana de tiempo lista todavía, además está faltando un arco para la orden vieja.
# EJ:
# mcf.SetNodeSupply(0,int(114.0))
# mcf.SetNodeSupply(1,int(0))
# mcf.SetNodeSupply(2,int(0))
# mcf.SetNodeSupply(3,int(114.0))
# mcf.SetNodeSupply(4,int(0))
# mcf.SetNodeSupply(5,int(0))
# mcf.SetNodeSupply(6,int(-114.0))
# mcf.SetNodeSupply(7,int(-114.0))
# mcf.AddArcWithCapacityAndUnitCost(0, 1, 9000000, 1)
# mcf.AddArcWithCapacityAndUnitCost(1, 2, 9000000, 1)
# mcf.AddArcWithCapacityAndUnitCost(3, 4, 9000000, 1)
# mcf.AddArcWithCapacityAndUnitCost(4, 5, 9000000, 1)
# mcf.AddArcWithCapacityAndUnitCost(0, 3, 9000000, 10)
# mcf.AddArcWithCapacityAndUnitCost(1, 4, 9000000, 10)
# mcf.AddArcWithCapacityAndUnitCost(2, 5, 9000000, 10)
# mcf.AddArcWithCapacityAndUnitCost(3, 0, 9000000, 10)
# mcf.AddArcWithCapacityAndUnitCost(4, 1, 9000000, 10)
# mcf.AddArcWithCapacityAndUnitCost(5, 2, 9000000, 10)
# mcf.AddArcWithCapacityAndUnitCost(5, 7, 9000000, 10)
# Running optimization