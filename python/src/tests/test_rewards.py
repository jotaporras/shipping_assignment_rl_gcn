import math

import numpy
import pytest
from envs.rewards import (
    BigMDiffRewardFunction,
    SimpleBigMRewardFunction,
    SimpleNegativeCostRewardFunction,
    NegativeCostDiffRewardFunction,
    BigMLastOrderRewardFunction,
    NegativeLogCostRewardFunction,
    NegativeLogCostMinusBigMUnitsRewardFunction,
)
from envs.shipping_assignment_env import ShippingAssignmentState
from network import physical_network
from network.physical_network import Node

from experiment_utils.Order import Order


def test_big_m_diff_reward():
    reward_function = BigMDiffRewardFunction()

    previous_state = ShippingAssignmentState(0, None, None, None, None, None, [3], 0, 0)
    current_state = ShippingAssignmentState(0, None, None, None, None, None, [5], 0, 0)

    reward = reward_function.calculate_reward(previous_state, current_state)

    assert reward == -1000.0  # 2 big ms*500.0


dcs_vs_expected_reward = [
    (numpy.array([[1.0, 0.0], [1.0, 0.0]]), 0),
    (numpy.array([[0.0, 1.0], [1.0, 0.0]]), -500.0),
]


@pytest.mark.parametrize(
    "dcs_per_customer_array,expected_reward",
    dcs_vs_expected_reward,
    ids=["valid", "invalid"],
)
def test_big_m_last_order(dcs_per_customer_array, expected_reward):
    reward_function = BigMLastOrderRewardFunction()

    phys_net = physical_network.PhysicalNetwork(2, 2, 1, 100, 100)
    numpy.random.seed(0)
    phys_net.dcs_per_customer_array = dcs_per_customer_array

    fixed = [Order(100, Node(0, 100, 100, 0, 0), Node(2, 100, 100, 0, 0), 1, "a")]
    previous_state = ShippingAssignmentState(0, None, None, None, None, None, [3], 0, 0)
    current_state = ShippingAssignmentState(
        0, phys_net, fixed, None, None, None, [5], 0
    )
    reward_1 = reward_function.calculate_reward(previous_state, current_state)

    assert reward_1 == expected_reward


def test_simple_neg_cost_reward():
    reward_function = SimpleNegativeCostRewardFunction()

    previous_state = ShippingAssignmentState(
        0, None, None, None, None, None, [0], 500, 0
    )
    current_state = ShippingAssignmentState(
        0, None, None, None, None, None, [0], 200, 0
    )

    reward = reward_function.calculate_reward(previous_state, current_state)

    assert reward == -200


def test_neg_cost_diff_reward():
    reward_function = NegativeCostDiffRewardFunction()

    previous_state = ShippingAssignmentState(
        0, None, None, None, None, None, [0], 300.0
    )
    current_state = ShippingAssignmentState(
        0, None, None, None, None, None, [0], 500.0, 0
    )

    reward = reward_function.calculate_reward(previous_state, current_state)

    assert reward == -200.0


def test_log_cost_reward():
    reward_function = NegativeLogCostRewardFunction()

    previous_state = ShippingAssignmentState(
        0,
        None,
        None,
        None,
        None,
        None,
        [0],
        optimization_cost=123.0,
        big_m_units_per_commodity=0,
    )
    current_state = ShippingAssignmentState(
        0,
        None,
        None,
        None,
        None,
        None,
        [0],
        optimization_cost=300.0,
        big_m_units_per_commodity=0,
    )

    print(math.log(300.0))
    print(math.log(123.0))

    reward = reward_function.calculate_reward(previous_state, current_state)

    assert math.isclose(reward, -5.70378247466)


def test_negative_log_cost_minus_big_m_uinits_reward():
    reward_function = NegativeLogCostMinusBigMUnitsRewardFunction()

    previous_state = None
    current_state = ShippingAssignmentState(
        0,
        None,
        None,
        None,
        None,
        None,
        [0],
        optimization_cost=300.0,
        big_m_units_per_commodity=[2, 3],
    )

    reward = reward_function.calculate_reward(previous_state, current_state)

    assert math.isclose(reward, -math.log(300.0 + 1e-9) - math.log(5.0 + 1e-9))
