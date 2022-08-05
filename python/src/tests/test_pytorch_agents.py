from collections import namedtuple

import numpy
import torch
from envs.inventory_generators import InventoryGenerator
from envs.shipping_assignment_env import ShippingAssignmentEnvironment
from envs.shipping_assignment_state import ShippingAssignmentState
from gym.vector.utils import spaces
from network import physical_network
from network.physical_network import Node

from agents import pytorch_agents
from experiment_utils.Order import Order

Mockenv = namedtuple("mockenv", ["action_space"])


def __fixture():
    env = Mockenv(spaces.Discrete(3))
    num_dcs = 3
    num_customers = 5
    num_commodities = 2
    physnet = physical_network.PhysicalNetwork(
        num_dcs,
        num_customers,
        2,
        100,
        50,
        big_m_factor=10000,
        num_commodities=num_commodities,
        planning_horizon=5,
    )
    # (num_dcs, num_commodities)
    inventory = numpy.array(
        [
            [10.0, 10.0],
            [50.0, 50.0],
            [20.0, 20.0],
        ]
    )
    state = ShippingAssignmentState(
        0,
        physnet,
        fixed=[
            Order(
                numpy.array([50.0, 30.0]),
                Node(0, 100, 0, 0, "dc"),
                Node(4, 100, 0, 0, "dc"),
                0,
                "someord",
            ),
            Order(
                numpy.array([50.0, 30.0]),
                Node(0, 100, 0, 0, "dc"),
                Node(4, 100, 0, 0, "dc"),
                0,
                "someord",
            ),
            Order(
                numpy.array([50.0, 30.0]),
                Node(0, 100, 0, 0, "dc"),
                Node(4, 100, 0, 0, "dc"),
                0,
                "someord",
            ),
        ],
        open=[
            Order(
                numpy.array([50.0, 30.0]),
                Node(0, 100, 0, 0, "dc"),
                Node(4, 100, 0, 0, "dc"),
                0,
                "someord",
            ),
            Order(
                numpy.array([50.0, 30.0]),
                Node(0, 100, 0, 0, "dc"),
                Node(4, 100, 0, 0, "dc"),
                0,
                "someord",
            ),
        ],
        inventory=inventory,
        state_vector=None,
        big_m_counter_per_commodity=0,
        optimization_cost=0,
        big_m_units_per_commodity=0,
    )
    return physnet, state, env


def test_mlp_dqn():
    physnet = physical_network.PhysicalNetwork(
        3, 5, 2, 100, 50, big_m_factor=10000, num_commodities=2, planning_horizon=5
    )
    print(physnet.dcs_per_customer_array)


def test_masked_plus_onehot():
    physnet, state, env = __fixture()
    net = pytorch_agents.MaskedPlusOneHotDQN(physnet.num_customers, physnet.num_dcs)
    agent = pytorch_agents.MaskedPlusOneHotDQNAgent(env, net, 0.01)
    state = ShippingAssignmentState(
        0,
        physnet,
        fixed=[],
        open=[
            Order(100, Node(0, 100, 0, 0, "dc"), Node(4, 100, 0, 0, "dc"), 0, "someord")
        ],
        inventory=None,
        state_vector=None,
        big_m_counter_per_commodity=0,
        optimization_cost=0,
        big_m_units_per_commodity=0,
    )
    example_input_tensor = agent.get_state_vector(state)
    print(example_input_tensor)
    print(net(torch.tensor(example_input_tensor)))


def test_state_to_inventory_vector():
    physnet, state, env = __fixture()
    flattened_inventory = pytorch_agents.state_to_inventory_vector(state)
    print(flattened_inventory)


def test_full_mlp_dqn():
    physnet, state, env = __fixture()
    net = pytorch_agents.FullMLPDQN(
        physnet.num_customers, physnet.num_dcs, physnet.num_commodities
    )
    agent = pytorch_agents.FullMLPDQNAgent(env, net, epsilon=0.01)
    input_vector = agent.get_state_vector(state)
    network_output = net(torch.tensor(input_vector).reshape(1, -1))
    print(input_vector)
    print(network_output)
