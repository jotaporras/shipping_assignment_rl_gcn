import logging
from abc import ABC
import abc
import math
import numpy as np
from shipping_allocation.envs.shipping_assignment_state import ShippingAssignmentState

from experiment_utils.Order import Order


def reward_chooser(reward_type: str):
    """
    Returns the reward function required.
    Args:
      reward_type:

    Returns:

    """
    if reward_type == "simple_big_m":
        return SimpleBigMRewardFunction()
    elif reward_type == "big_m_diff":
        return BigMDiffRewardFunction()
    elif reward_type == "big_m_last_order":
        return BigMLastOrderRewardFunction()
    elif reward_type == "negative_cost":
        return SimpleNegativeCostRewardFunction()
    elif reward_type == "negative_cost_diff":
        return NegativeCostDiffRewardFunction()
    elif reward_type == "negative_log_cost":
        return NegativeLogCostRewardFunction()
    elif reward_type == "negative_log_cost_minus_log_big_m_units":
        return NegativeLogCostMinusBigMUnitsRewardFunction()  # the luis func
    elif reward_type == "negative_log_cost_minus_log_big_m_units_exp":
        return NegativeLogCostMinusBigMUnitsExpRewardFunction()  # the luis func
    elif reward_type == "negative_log_cost_or_bigm_penalty":
        return NegativeLogCostOrBigMPenaltyRewardFunction()  # the javi func
    elif reward_type == "negative_if_bigm_else_denom_cost":
        return NegIfBigMElseDenomCost()
    else:
        raise ValueError(f"Reward type {reward_type} not implemented.")


class DictStateHandler:
    """
    Tracks the current and next state for components that need to be aware of state transitions.
    """

    previous_state: dict = None
    current_state: dict = None
    state_history = []

    def update_state(self, current_state):
        state_history.append(previous_state)
        previous_state = current_state
        current_state = current_state


class RewardFunction(ABC):
    @abc.abstractmethod
    def calculate_reward(self, previous_state, current_state) -> float:
        pass


class SimpleBigMRewardFunction(RewardFunction):
    """
    Calculates the reward based on the state.
    It might be confusing because the penalty for assigning a Big M is persistent across time.
    """

    BIG_M_PENALTY = 500.0

    def calculate_reward(self, previous_state, current_state):
        return (
            -1.0 * self.BIG_M_PENALTY * sum(current_state.big_m_counter_per_commodity)
        )


class BigMDiffRewardFunction(RewardFunction):
    """
    Calculates the reward based on the difference between the Big M Count
    at S_t vs S_{t-1}.

    Solves the issue with the Simple Big M, as only the current order is responsible for the change

    Update Aug 14: The problem with this function is that you get a reward when an order leaves the horizon, and it has nothing to do with you.
    Update Aug 15 : but it still works for q learning discrete and big m.
    """

    BIG_M_PENALTY = 500.0

    def calculate_reward(
        self,
        previous_state: ShippingAssignmentState,
        current_state: ShippingAssignmentState,
    ):

        if previous_state:
            return (
                -1.0
                * self.BIG_M_PENALTY
                * (
                    sum(current_state.big_m_counter_per_commodity)
                    - sum(previous_state.big_m_counter_per_commodity)
                )
            )
        else:
            return (
                -1.0
                * self.BIG_M_PENALTY
                * sum(current_state.big_m_counter_per_commodity)
            )


class BigMLastOrderRewardFunction(RewardFunction):
    """
    Calculates the reward based on whether the last order was assigned to a valid DC or not. Attempts
    to solve the problem with analizing the sheer number of Big Ms sliding through the horizon.
    """

    def calculate_reward(
        self,
        previous_state: ShippingAssignmentState,
        current_state: ShippingAssignmentState,
    ):
        last_order: Order = current_state.fixed[-1]
        physical_network: PhysicalNetwork = current_state.physical_network
        is_valid = physical_network.is_valid_arc(
            last_order.shipping_point.node_id, last_order.customer.node_id
        )
        return (int(is_valid) - 1) * 500.0


class SimpleNegativeCostRewardFunction(RewardFunction):
    """
    Calculates the rward based on the difference between optimization cost.
    """

    def calculate_reward(self, previous_state, current_state):
        return -1.0 * current_state.optimization_cost


class NegativeCostDiffRewardFunction(RewardFunction):
    """
    Calculates the reward based on the difference between the optimization cost
    at S_t vs S_{t-1}.

    Solves the issue with the simple optimization cost, as only the current order is responsible for the change
    """

    def calculate_reward(
        self,
        previous_state: ShippingAssignmentState,
        current_state: ShippingAssignmentState,
    ):

        if previous_state:
            return -1.0 * (
                current_state.optimization_cost - previous_state.optimization_cost
            )
        else:
            return -1.0 * current_state.optimization_cost


class NegativeLogCostRewardFunction(RewardFunction):
    """
    Returns the negative log of the cost for numerical stability.
    """

    def calculate_reward(self, previous_state, current_state) -> float:
        return -1.0 * math.log(current_state.optimization_cost)


class NegativeLogCostMinusBigMUnitsRewardFunction(RewardFunction):
    """
    Returns the negative log of the cost minus the log of Big M, for numerical stability
    and in hopes of attenuating the impact of big M.
    AKA LLP Reward
    """

    epsilon = 1e-9

    def calculate_reward(
        self,
        previous_state: ShippingAssignmentState,
        current_state: ShippingAssignmentState,
    ) -> float:
        return -math.log(current_state.optimization_cost + self.epsilon) - math.log(
            sum(current_state.big_m_units_per_commodity) + self.epsilon
        )


class NegativeLogCostMinusBigMUnitsExpRewardFunction(RewardFunction):
    """
    Exp of LLP reward
    """

    llp_reward = NegativeLogCostMinusBigMUnitsRewardFunction()

    def calculate_reward(
        self,
        previous_state: ShippingAssignmentState,
        current_state: ShippingAssignmentState,
    ) -> float:
        return np.exp(self.llp_reward.calculate_reward(previous_state, current_state))


class NegativeLogCostOrBigMPenaltyRewardFunction(RewardFunction):
    """
    Returns negative log of cost minus Big M penalty if no big M, otherwise returns a really low Big M penalty.
    """

    epsilon = 1e-9

    def calculate_reward(
        self,
        previous_state: ShippingAssignmentState,
        current_state: ShippingAssignmentState,
    ) -> float:
        # TODO should it be units or units*cost?
        big_m_in_cost = (
            sum(current_state.big_m_units_per_commodity)
            * current_state.physical_network.big_m_cost
        )

        last_order: Order = current_state.fixed[-1]
        physical_network: PhysicalNetwork = current_state.physical_network
        is_last_order_valid = physical_network.is_valid_arc(
            last_order.shipping_point.node_id, last_order.customer.node_id
        )
        if is_last_order_valid:
            reward_without_big_ms = -math.log(
                current_state.optimization_cost - big_m_in_cost + self.epsilon
            )
            logging.debug(
                f"Valid action on last order {last_order}, so reward {reward_without_big_ms}"
            )
            difference = (
                -math.log(current_state.optimization_cost) - reward_without_big_ms
            )
            logging.debug(
                f"Cost with big M {current_state.optimization_cost} logged {math.log(current_state.optimization_cost)}"
            )
            logging.debug(
                f"Cost without big Ms {current_state.optimization_cost - big_m_in_cost} logged and epsiloned {-reward_without_big_ms}"
            )
            logging.debug(
                f"Removed from cost to amortize Big M: {big_m_in_cost+self.epsilon} (logged {difference})"
            )

            return (
                reward_without_big_ms  # Amortized cost if you didn't do anything wrong.
            )
        else:
            big_m_penalty_reward = -(current_state.optimization_cost)
            logging.debug(
                f"Big M on last order {last_order}, so reward {big_m_penalty_reward}"
            )
            return big_m_penalty_reward  # the full cost.


class NegIfBigMElseDenomCost(RewardFunction):
    """
    Returns the negative log of the cost minus the log of Big M, for numerical stability
    and in hopes of attenuating the impact of big M.
    AKA LLP Reward
    """

    epsilon = 1e-9

    def calculate_reward(
        self,
        previous_state: ShippingAssignmentState,
        current_state: ShippingAssignmentState,
    ) -> float:
        numerator = (
            -sum(current_state.big_m_counter_per_commodity)
            if sum(current_state.big_m_counter_per_commodity) > 0
            else 1.0
        )
        denom = current_state.optimization_cost
        return numerator / denom
