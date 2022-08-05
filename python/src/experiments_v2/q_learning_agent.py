import logging

import numpy as np

# from agents import Agent #todo wssee why not work
import wandb


class ShippingEnvQLearningAgent:  # (Agent):
    """
    A Q Learning agent with Dynamic Programming. Simple Q Learning (not neural net)
    """

    num_states: int
    num_actions: int
    q_func_values: np.array  # (num_states,num_actions)

    def __init__(
        self,
        num_states,
        num_actions,
        learning_rate,
        discount_factor,
        epsilon,
        env,
        init_val: float = 100.0,
    ):

        self.num_actions = num_actions
        self.num_states = num_states
        self.q_func_values = np.full((num_states, num_actions), float(init_val))

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        self.customer_metadata_neuron = -3  # todo hardcoded

        self.env = env
        initial_state = self.env.reset()

        # running for state
        self.old_customer_id = -1
        self.old_action = -1

        self.next_customer_id = self._extract_customer_id(initial_state)
        self.next_action = np.random.choice(self.num_actions, size=1)

    def _extract_customer_id(self, state):
        location_id = int(state.state_vector[0][self.customer_metadata_neuron])
        customer_id = (
            location_id - state.physical_network.num_dcs
        )  # ToDO this is the only place where physical net is used in state. consider removing.
        if customer_id < 0 or customer_id > self.num_states:
            raise RuntimeError("Something went wrong in calculating the customer id")
        return customer_id

    def get_action(self, state):
        logging.debug("Q Learning Agent: Call get action")
        customer_id = self._extract_customer_id(state)
        q_values = self.q_func_values[customer_id, :]

        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(q_values)

        self.next_customer_id = customer_id
        self.next_action = action

        return action

    def train(self, experience):
        logging.debug("Q Learning Agent: Call train")
        state, action, next_state, reward, done = experience

        next_customer = self._extract_customer_id(next_state)

        self.old_customer_id = self.next_customer_id
        self.old_action = self.next_action

        self.next_customer_id = next_customer
        self.next_action = np.argmax(self.q_func_values[next_customer, :])

        old_state = self.q_func_values[self.old_customer_id, self.old_action]
        new_state = self.q_func_values[self.next_customer_id, self.next_action]

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("Q Values before update")
            logging.debug(self.q_func_values)
            logging.debug(f"Reward to update {reward}")
            logging.debug(
                f"Updating C={self.old_customer_id} A={self.old_action} that contains value {old_state}"
            )
            logging.debug(
                f"(R+G*Q'-Q) {reward + self.discount_factor * new_state - old_state}"
            )
            logging.debug(
                f"Total update: {self.learning_rate*(reward + self.discount_factor * new_state - old_state)}"
            )

        # todo double check this formula
        self.q_func_values[
            self.old_customer_id, self.old_action
        ] = old_state + self.learning_rate * (
            reward + self.discount_factor * new_state - old_state
        )

        self.log_to_wandb()

    def log_to_wandb(self):
        x_labels = np.arange(self.num_actions)
        y_labels = np.arange(self.num_states)

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("Q Agent logging discrete q values")
            logging.debug(self.q_func_values)

        logging.info("Logging with commit false")
        wandb.log(
            {
                "q_func_values": wandb.plots.HeatMap(
                    x_labels,
                    y_labels,
                    self.q_func_values,  # show_text=True
                )
            },
            commit=False,
        )
