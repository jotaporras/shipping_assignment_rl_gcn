"""Deprecated, use base_ptl_nn_agent_runner instead. This one's just used in the legacy one hot dqn."""
import argparse
import logging
import random
from typing import List
from typing import Tuple

import gym
import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
import wandb
from envs import ShippingFacilityEnvironment, network_flow_env_builder
from envs.shipping_assignment_env import ShippingAssignmentEnvironment
from pytorch_lightning.loggers import WandbLogger
from shipping_allocation import EnvironmentParameters
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

# Named tuple for storing experience steps gathered in training
from dqn.dqn_common import Experience, RLDataset, ReplayBuffer
from experiment_utils import report_generator


class DQN(nn.Module):
    """
    Deprecated.
    Simple MLP network.


    Args:
        obs_size: observation/state size of the environment
        n_actions: number of discrete actions available in the environment
        hidden_size: size of hidden layers
    """

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x.float())


class Agent:
    """
    Deprecated.
    Base Agent class handling the interaction with the environment
    Deprecated, was copied a long time ago from some PTL example.
    Args:
        env: training environment
        replay_buffer: replay buffer storing experiences
    """

    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state = (
            self.env.reset()
        )  # todo bug double reset (but this agent is deprecatedf)

    def reset(self) -> None:
        """ Resents the environment and updates the state"""
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """
        Using the given network, decide what action to carry out
        using an epsilon-greedy policy

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor([self.state.state_vector.reshape(-1)])

            if device not in ["cpu"]:
                state = state.cuda(device)

            q_values = net(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(
        self, net: nn.Module, epsilon: float = 0.0, device: str = "cpu"
    ) -> Tuple[float, bool, dict]:
        """
        Carries out a single interaction step between the agent and the environment

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """

        action = self.get_action(net, epsilon, device)

        # do step in the environment

        new_state, reward, done, info = self.env.step(action)

        current_state_vector = self.state.state_vector.reshape(-1)
        next_state_vector = new_state.state_vector.reshape(-1)

        logging.info(f"Current state vector {current_state_vector}")
        logging.info(f"Next state vector {next_state_vector}")

        exp = Experience(
            current_state_vector,
            action,
            reward,
            done,
            next_state_vector,
        )

        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done, info


class DQNLightning(pl.LightningModule):
    """ Deprecated. Basic DQN Model """

    def __init__(
        self, hparams: argparse.Namespace, env: ShippingAssignmentEnvironment
    ) -> None:
        super().__init__()
        self.hparam_conf = hparams
        # v2
        self.env = env

        obs_size = self.env.observation_space.shape[1]
        n_actions = self.env.action_space.n

        self.net = DQN(obs_size, n_actions)
        self.target_net = DQN(obs_size, n_actions)

        self.buffer = ReplayBuffer(self.hparam_conf["replay_size"])
        self.agent = Agent(self.env, self.buffer)

        # to have an optimization metric
        self.episode_loss = 0.0
        self.running_loss = 0.0

        self.episode_reward = 0.0
        self.running_reward = 0.0

        # Initialize episode information for debugging.
        self.episodes_info = []
        self.episode_counter = 0
        self.populate(self.hparam_conf["warm_start_steps"])

    def populate(self, steps: int = 1000) -> None:
        """
        Carries out several random steps through the environment to initially fill
        up the replay buffer with experiences

        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            _, done, info = self.agent.play_step(self.net, epsilon=1.0)
            if done:
                self.store_episode_info(info)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes in a state x through the network and gets the q_values of each action as an output

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_number):
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved.
        The batch tuple comes from the Dataset?

        Args:
            batch: current mini batch of replay data
            batch_number: batch number

        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        epsilon = max(
            self.hparam_conf["eps_end"],
            self.hparam_conf["eps_start"]
            - self.global_step
            + 1 / self.hparam_conf["eps_last_frame"],
        )

        # step through environment with agent
        reward, done, info = self.agent.play_step(
            self.net, epsilon, device
        )  # refill the buffer every time.
        self.episode_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        # if self.trainer.use_dp or self.trainer.use_ddp2:
        #     loss = loss.unsqueeze(0t
        self.episode_loss += loss.detach().item()

        # Soft update of target network
        if self.global_step % self.hparam_conf["sync_rate"] == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        if done:
            self.store_episode_info(info)

        # result = pl.TrainResult(minimize=loss)

        logging.info("Logging current steps metrics")

        self.log("loss", loss)
        self.log("reward", reward)
        self.log("episode_reward", self.episode_reward)
        self.log("episodes", self.episode_counter)

        return loss

    def dqn_mse_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Calculates the mse loss using a mini batch from the replay buffer

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = (
            self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        )

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = (
            next_state_values * self.hparam_conf["gamma"] + rewards
        )

        logging.info("rewards")
        logging.info(rewards)
        logging.info("state_action_values")
        logging.info(state_action_values)
        logging.info("next_state_values")
        logging.info(next_state_values)
        logging.info("actions")
        logging.info(actions)

        return nn.MSELoss()(state_action_values, expected_state_action_values)
        # return nn.L1Loss()(state_action_values, expected_state_action_values)

    def store_episode_info(self, info):
        logging.info("Finished episode, storing information")
        self.episodes_info.append(info)

        self.episode_counter += (
            1  # First episode is 1. If you're going to index episodes_info, subtract 1
        )

        # Update running metrics
        self.running_reward += self.episode_reward
        self.running_loss += self.episode_loss

        self._wandb_custom_metrics(info)

        self.episode_reward = 0.0
        self.episode_loss = 0.0

    def _wandb_custom_metrics(self, info):
        report_metrics = report_generator.convert_info_into_metrics_summary_dict(info)

        trainer_metrics = {
            # "big_m_count": big_m_episode_count,
            # "incoming_interplants": incoming_interplants,
            # "total_interplants": total_interplants,
            # "deliveries_per_shipping_point_units": deliveries_per_shipping_point_units,
            # "deliveries_per_shipping_point_orders": deliveries_per_shipping_point_orders,
            # "mean_dcs_per_customer": mean_dcs_per_customer,
            "episode_loss": self.episode_loss,
            "episode_reward": self.episode_reward,
            # Running
            "running_loss": self.running_loss,
            "running_reward": self.running_reward,
            # Running mean
            "mean_loss_per_ep": self.running_loss / (self.episode_counter),
            "mean_reward_per_ep": self.running_reward / self.episode_counter,
        }
        wandb_metrics = {**report_metrics, **trainer_metrics}
        logging.info(
            f"Episode {self.episode_counter} had {wandb_metrics['big_m_count']} BigMs"
        )
        logging.info("Finished episode, logging metrics to wandb:")
        logging.info(wandb_metrics)
        logging.info("Logging with commit false (base_ptl_agent_runner)")
        wandb.log(
            wandb_metrics,
            commit=False,
        )

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer = optim.Adam(self.net.parameters(), lr=self.hparam_conf["lr"])
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDataset(self.buffer, self.hparam_conf["episode_length"])
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparam_conf["batch_size"],
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0].device.index if self.on_gpu else "cpu"
