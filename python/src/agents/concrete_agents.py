"""
This module contains the implementations of the most trivial agents, as well as a selector function to switch between all available agents (pytorch-based, optimizer-based, and trivials)
"""
from __future__ import annotations

import datetime
import logging
import time

import numpy as np

# from network import physical_network

# from shipping_allocation.envs.network_flow_env import (
#     ShippingFacilityEnvironment,
# )

# import experiments_seminar_2.agents.optimizer_agents
from envs import inventory_generators
from envs.shipping_assignment_state import ShippingAssignmentState

from agents import optimizer_agents, pytorch_agents
from agents.Agent import Agent

# Environment and agent


DEBUG = False

logger = logging.getLogger(__name__)


def get_agent(env, environment_config, hparams, agent_name: str):
    """
    Factory for creating agents for the given env.
    """
    logger.info("NN agents are using eps_start only.")
    num_customers = environment_config["num_customers"]
    num_dcs = environment_config["num_dcs"]
    num_commodities = environment_config["num_commodities"]
    epsilon = hparams["eps_start"]
    gnn_hidden_units = hparams["gnn_hidden_units"]
    time_limit_milliseconds = hparams.get("time_limit_milliseconds", 240 * 1000)
    if agent_name == "random":
        return RandomAgent(env)
    elif agent_name == "always_zero":
        return AlwaysZeroAgent(env)
    elif agent_name == "best_fit":
        return BestFitAgent(env)
    elif agent_name == "random_valid":
        return RandomValid(env)
    elif agent_name == "do_nothing":
        return DoNothingAgent(env)
    elif agent_name == "agent_highest":
        return AgentHighest(env)
    elif agent_name == "lookahead":
        return optimizer_agents.LookaheadAgent(env)
    elif agent_name == "tree_search":
        return optimizer_agents.TreeSearchAgent(env)
    elif agent_name == "nn_customer_onehot":
        customer_dqn = pytorch_agents.CustomerOnehotDQN(num_customers, num_dcs)
        return pytorch_agents.CustomerDQNAgent(env, customer_dqn, epsilon)
    elif agent_name == "nn_warehouse_mask":
        customer_dqn = pytorch_agents.MaskedMLPDQN(num_dcs)
        return pytorch_agents.MaskedMLPDQNAgent(env, customer_dqn, epsilon)
    elif agent_name == "nn_mask_plus_customer_onehot":
        mask_cust_dqn = pytorch_agents.MaskedPlusOneHotDQN(num_customers, num_dcs)
        return pytorch_agents.MaskedPlusOneHotDQNAgent(env, mask_cust_dqn, epsilon)
    elif agent_name == "nn_full_mlp":
        full_mlp = pytorch_agents.FullMLPDQN(num_customers, num_dcs, num_commodities)
        return pytorch_agents.FullMLPDQNAgent(env, full_mlp, epsilon)
    elif agent_name == "nn_debug_mlp_cheat":
        cheat_mlp = pytorch_agents.DebugMaskedMLPCheatDQN(num_dcs)
        return pytorch_agents.MaskedMLPWithCheatDQNAgent(env, cheat_mlp, epsilon)
    elif agent_name == "nn_mask_plus_consumption":
        mask_cons_mlp = pytorch_agents.MaskPlusConsumptionMLP(
            num_customers, num_dcs, num_commodities
        )
        return pytorch_agents.MaskPlusConsumptionMLPAgent(env, mask_cons_mlp, epsilon)
    elif agent_name == "gnn_physnet_aggdemand":
        gcn_module = pytorch_agents.PhysnetAggDemandGCN(
            num_commodities, num_dcs, num_customers, gnn_hidden_units
        )
        return pytorch_agents.PhysnetAggDemandGCNAgent(env, gcn_module, epsilon)
    elif agent_name == "gnn_physnet_ntype_aggdemand":
        # Node type and agg demand features
        gcn_module = pytorch_agents.AggDemandAndNodeTypeGraphlevelGCN(
            num_commodities, num_dcs, num_customers, gnn_hidden_units
        )
        return pytorch_agents.AggDemandAndNodeTypeGraphlevelGCNAgent(
            env, gcn_module, epsilon
        )
    elif agent_name == "lower_bound_zerocost":
        return LowerBoundAgent(env)
    elif agent_name == "lower_bound_oracle_inventory":
        return OracleInventoryAgent(env)
    elif agent_name == "branch_and_bound_milp":
        return optimizer_agents.BranchAndBoundAgent(env, time_limit_milliseconds)
    else:
        raise NotImplementedError(f"Agent {agent_name} not implemented.")


class RandomAgent(Agent):
    """The world's simplest agent!"""

    def train(self, experience):
        pass  # do nothing


class AlwaysZeroAgent(Agent):
    """The world's dumbest agent!"""

    def get_action(self, state):
        return 0

    def train(self, experience):
        pass  # do nothing


class BestFitAgent(Agent):
    """The world's most conservative agent!"""

    env: ShippingFacilityEnvironment
    network: physical_network

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.network = env.physical_network

    def get_action(self, state):
        # TODO: make this backwards compatible again.
        # inventory = state["inventory"]
        inventory = state.inventory
        # order = state["open"][0]
        order = state.open[0]
        customer = order.customer
        cid = customer.node_id - self.network.num_dcs
        cust_dcs = np.argwhere(self.network.dcs_per_customer_array[cid, :] > 0)[:, 0]
        allowed_dc_invs = inventory[cust_dcs, :]
        demand = order.demand
        remaining = np.sum(allowed_dc_invs - demand, axis=1)
        chosen_dc_index = np.argmax(remaining)
        chosen_dc_id = cust_dcs[chosen_dc_index]

        if DEBUG:
            print("Bestfit chose: ", chosen_dc_id)
            print("Inventories: ", inventory)
            print("Allowed DCs:", cust_dcs)

            if self.network.dcs_per_customer_array[cid, chosen_dc_id] == 1:
                print("Chose allowed DC:", cid, chosen_dc_index)
            else:
                print("Chose ILLEGAL OH NO DC:", cid, chosen_dc_index)
            if np.argwhere(cust_dcs == chosen_dc_id).size == 0:
                print(
                    "BESTFIT CHOSE ILLEGAL MOVEMENT. THIS SHOULD NOT HAPPEN. Illegal for customer ",
                    customer,
                    "DC",
                    chosen_dc_id,
                )
            else:
                print("Bestfit chose the legal move", chosen_dc_id)

        return chosen_dc_id  # todo test this.

    def train(self, experience):
        pass  # do nothing


class RandomValid(Agent):
    """The world's least screwup random agent!"""

    env: ShippingFacilityEnvironment
    network: physical_network

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.network = env.physical_network

    def get_action(self, state):
        inventory = state.inventory
        order = state.open[0]
        customer = order.customer
        cid = customer.node_id - self.network.num_dcs
        cust_dcs = np.argwhere(self.network.dcs_per_customer_array[cid, :] > 0)[:, 0]
        chosen_dc_id = np.random.choice(cust_dcs)

        if DEBUG:
            logging.debug(f"RandomValid chose:  {chosen_dc_id}")
            logging.debug(f"Inventories:  {inventory}")
            logging.debug(f"Allowed DCs: {cust_dcs}")
            logging.debug(
                f"Chose allowed DC {chosen_dc_id} for customer {cid}: {self.network.dcs_per_customer_array[cid, chosen_dc_id] == 1}"
            )

        return chosen_dc_id  # todo test this.

    def train(self, experience):
        pass  # do nothing


class DoNothingAgent(Agent):
    """The world's least screwup random agent!"""

    env: ShippingFacilityEnvironment
    network: physical_network

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.network = env.physical_network

    def get_action(self, state):
        order = state.open[0]
        dc = order.shipping_point

        return dc.node_id

    def train(self, experience):
        pass  # do nothing


class AgentHighest(Agent):
    """The world's debugging agent"""

    env: ShippingFacilityEnvironment
    network: physical_network

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.network = env.physical_network

    def get_action(self, state):
        order = state["open"][0]
        customer = order.customer
        cid = self.network.num_dcs - customer.node_id
        cust_dcs = np.argwhere(self.network.dcs_per_customer_array[cid, :] > 0)[:, 0]

        return cust_dcs[-1]  # choose the last

    def train(self, experience):
        pass  # do nothing


class LowerBoundAgent(RandomValid):
    """The world's nastiest lower bound agent. It modifies the actual physnet costs
    so that it looks like it's doing really low on costs."""

    env: ShippingFacilityEnvironment
    network: physical_network

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.network = env.physical_network
        self.network.default_storage_cost = 1  # TODO HARDWIRED CONSTANTS
        # self.network.default_dc_transport_cost = 10 #TODO HARDWIRED CONSTANTS
        self.network.default_dc_transport_cost = 1  # TODO HARDWIRED CONSTANTS
        self.network.default_customer_transport_cost = 1  # TODO HARDWIRED CONSTANTS


class OracleInventoryAgent(DoNothingAgent):
    """The world's  sneakiest agent. Instead of acting, it changes reality!
    Inventory generator generates exactly where new orders arrive, and this agent does nothing.
    NOTE IT INHERITS DONOTHINGAGENT
    """

    env: ShippingFacilityEnvironment
    network: physical_network

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.network = env.physical_network
        self.env.inventory_generator = inventory_generators.OracleGenerator(
            env.physical_network
        )
