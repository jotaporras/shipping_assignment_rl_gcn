import logging
import random

import gym
import numpy as np


class Agent:
    """
    Class that represents an agent for the simulation
    """
    logger = logging.getLogger(__name__)

    def __init__(self, env):
        self.is_discrete = type(env.action_space) == gym.spaces.discrete.Discrete

    def get_action(self, state):
        pass

    def train(self, experience):
        pass

    def get_state_vector(self, state):
        """
        How to generate the state vector based on the state named tuple. The default implementation is for dummy agents
        that don't use neural networks, therefore don't need this.
        """
        return np.array([1.0])
