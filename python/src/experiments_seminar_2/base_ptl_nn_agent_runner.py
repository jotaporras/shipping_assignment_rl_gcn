import time
from copy import deepcopy
import logging
from copy import deepcopy
from typing import List
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
import wandb
from envs.shipping_assignment_state import ShippingAssignmentState
from pytorch_lightning import Callback
from torch import nn
from torch.optim import Optimizer
import torch.utils.data

# Named tuple for storing experience steps gathered in training
from dqn.dqn_common import Experience, RLDataset, ReplayBuffer
from experiment_utils import report_generator


# Named tuple for storing experience steps gathered in training

# Default big-ish

# TODO implement all the RL metrics here.
from agents.pytorch_agents import CustomerOnehotDQN


class MetricsAccumulatorCallback(Callback):
    logger = logging.getLogger(__name__)

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.logger.debug("calling train batch start")


class ShippingAssignmentRunner(pl.LightningModule):
    """
    This is the Seminar II official runner.
    """

    logger = logging.getLogger(__name__)

    def __init__(self, agent, env, hparams, collate_fn=None, *args, **kwargs):
        """IMPORTANT: IF YOU ADD ANY NN specific parameters, make sure to not make them required for greedy agents."""
        super().__init__(*args, **kwargs)
        self.collate_fn = collate_fn
        self.hparam_conf = hparams
        self.env = env
        # Initializing the environment to get the first step.
        # TODo maybe would be cleaner on "on training start".
        self.state = self.env.reset()
        self.agent = agent

        # Add an example, registered tensor for use in agent

        # Thanks for nothing, pytorch lightning garbage
        if self.on_gpu or torch.cuda.device_count() > 0:
            device = torch.device("cuda")
            print("training on gpu")
            self.register_buffer("ref_tensor", torch.ones(5).to(device))
        else:
            print("not training on gpu")
            self.register_buffer("ref_tensor", torch.ones(5).to(self.device))
        print("Check device on ref tensor")
        print(self.ref_tensor.device)
        print("PTL self.on_gpu")
        print(self.on_gpu)
        print("PTL self.device")
        print(self.device)

        if hasattr(agent, "net"):
            self.net = agent.net
            self.is_nn_agent = True
            # setting ref tensor for devices purposes
            self.agent.ref_tensor = self.ref_tensor
            print(self.parameters())
            # wandb.watch(
            #     self.net, log_freq=5, log="all"
            # )  # TODO make this parameterizable.
        else:
            self.net = None
            self.is_nn_agent = False
        self.logger.info(f"is nn agent: {self.is_nn_agent}")
        self.target_net = deepcopy(self.net)

        # TODo deecide if this is the best way to circumvent greedy behavior.
        # For greedy agents if replay size not set, use ep length.
        self.replay_buffer = ReplayBuffer(
            self.hparam_conf.get("replay_size", self.hparam_conf["episode_length"])
        )

        self.episodes_info = []

        self.episode_loss = 0.0
        self.running_loss = 0.0

        self.episode_reward = 0.0
        self.running_reward = 0.0

        self.episode_counter = 0

        # If warm start step not used run one episode as warmup.
        # Ideally I wouldn't have warmup for any of the greedy agents
        self.populate(
            self.hparam_conf.get("warm_start_steps", self.hparam_conf["episode_length"])
        )

        # Define wandb metrics that should be summary-averaged
        wandb.define_metric("total_interplants", summary="mean")
        wandb.define_metric("average_cost_ep", summary="mean")
        wandb.define_metric("mean_dcs_per_customer", summary="mean")
        wandb.define_metric("reward", summary="mean")
        wandb.define_metric("episode_process_time_ns", summary="mean")
        wandb.define_metric("episode_process_time_s", summary="mean")
        wandb.define_metric("action_process_time_ns", summary="mean")
        wandb.define_metric("action_process_time_s", summary="mean")

    def step_env_and_store_experience(self, action):
        next_state, reward, done, info = self.env.step(action)

        # Deprecated useless call to train. Reinsert if it's supposed to be used. by some agent.
        # self.agent.train((self.state, action, next_state, reward, done))

        # Store current vectors to experience.
        # TODO: it could be that this is too limited for GCN? If not just a vector?
        current_state_vector = self.agent.get_state_vector(self.state)
        next_state_vector = self.agent.get_state_vector(next_state)
        self.logger.debug(f"Current state vector {current_state_vector}")
        self.logger.debug(f"Next state vector {next_state_vector}")
        exp = Experience(
            current_state_vector,
            action,
            reward,
            done,
            next_state_vector,
        )
        self.replay_buffer.append(exp)

        # Current state is now the next.
        self.state = next_state

        return next_state, reward, done, info

    def populate(self, steps: int = 1000) -> None:
        """
        Carries out several random steps through the environment to initially fill
        up the replay buffer with experiences

        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            # Todo duplicate from the training step except calling train on the agent.
            # populate with random actions
            current_customer = self.state.open[
                0
            ].customer.node_id  # TODO megahack to sample valid actions.
            action = self.env.action_space.sample(current_customer)
            next_state, reward, done, info = self.step_env_and_store_experience(action)
            if done:
                self._finish_episode(info)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_nn_agent:
            return self.net(x)
        else:
            return torch.ones(1, requires_grad=True).detach()

    # TODO see RLDataLoader for a signature of the batch.
    def training_step(self, batch, num_batch):
        """
        If batch is GNN, it's a list with batches of [G,E] tensors
        Args:
            batch:
            num_batch:

        Returns:

        """
        self.logger.debug(f"Training step. Global step is {self.global_step}")
        start = time.process_time_ns()
        with torch.no_grad():
            action = self.agent.get_action(self.state)
        end = time.process_time_ns()
        action_time_ns = end - start
        action_time_s = action_time_ns / 1e9
        self.logger.debug(f"Action took {action_time_s}s")
        self.logger.debug(f"Got action  {action}")

        # the agent observes the first state and chooses an action
        # environment steps with the agent's action and returns new state and reward

        next_state, reward, done, info = self.step_env_and_store_experience(action)
        loss = self.dqn_mse_loss(batch)  # Todo maybe this function can be generalized.

        # if self.trainer.use_dp or self.trainer.use_ddp2:
        #     loss = loss.unsqueeze(0)

        # Soft update of target network
        if self.is_nn_agent and self.global_step % self.hparam_conf["sync_rate"] == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.episode_loss += loss.detach().item()
        self.episode_reward += reward

        if done:
            self._finish_episode(info)
        self.logger.debug("Logging to wandb using the PTL Integration")
        self.log("loss", loss)
        self.log("reward", reward)
        self.log("action", action)
        self.log("action_process_time_s", action_time_s)
        self.log("action_process_time_ns", action_time_ns)
        self.log("episode_reward", self.episode_reward)  # Todo probably wrong anyways
        self.log("episodes", self.episode_counter)

        return loss

    def dqn_mse_loss(self, batch) -> torch.Tensor:
        """
        Calculates the mse loss using a mini batch from the replay buffer

        Args:
            batch: current mini batch of replay data. It comes from experience tuple, but state changes for GNN.

        Returns:
            loss
        """
        # hacky sack for non NN stuff (da shim)

        if not self.is_nn_agent:
            return (
                torch.zeros(2, 2, requires_grad=True) - 1
            ).sum()  # a dummy operation to trick ptl

        ####
        # Below this will only execute if it's an NN agent.
        states, actions, rewards, dones, next_states = batch

        q_values = self.net(states)
        # The gather selects the q values for the action that was taken (the argmax)
        # Then we squeeze to remove the extra dimension.
        state_action_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            # Here instead of using actions to choose from the array we do max over that dimension.
            next_state_values = self.target_net(next_states).max(1)[0]
            # next_state_values[dones] = 0.0 # TODO: Honestly IDK if we need this.
            next_state_values = next_state_values.detach()
            # Uncomment for reward normalizastion (didnt see much impact)
            rewards = (rewards - rewards.mean()) / rewards.std()
            # rewards = torch.nn.functional.normalize(rewards.reshape(1, -1)).reshape(
            #     -1
            # )

        expected_state_action_values = (
            next_state_values * self.hparam_conf["gamma"] + rewards
        )

        self.logger.debug("rewards on batch")
        self.logger.debug(rewards)
        self.logger.debug("state_action_values on batch")
        self.logger.debug(state_action_values)
        self.logger.debug("next_state_values on batch")
        self.logger.debug(next_state_values)
        self.logger.debug("actions on batch")
        self.logger.debug(actions)

        # return nn.MSELoss()(state_action_values, expected_state_action_values)
        return nn.HuberLoss()(state_action_values, expected_state_action_values)

    def _finish_episode(self, info):
        """Stores the episode info, logs to wandb, resets env."""
        self.logger.info("Finished episode, storing information, resetting env.")
        self.episodes_info.append(info)

        # Update running metrics
        # First episode is 1. If you're going to index episodes_info, subtract 1
        self.episode_counter += 1
        self.running_reward += self.episode_reward
        self.running_loss += self.episode_loss

        # Store episode metrics and reports.
        self._log_wandb_custom_episode_metrics(info)

        # Reset episode wide metrics
        self.episode_reward = 0.0
        self.episode_loss = 0.0
        self.state = self.env.reset()
        self.logger.info(f"==== ***FINISHED EPISODE {self.episode_counter}*** ====")

    def _log_wandb_custom_episode_metrics(self, info):
        wandb_metrics = report_generator.convert_info_into_metrics_summary_dict(info)

        self.logger.info(
            f"Episode {self.episode_counter} had {wandb_metrics['big_m_count']} BigMs"
        )
        self.logger.info("Logging metrics to wandb:")
        self.logger.info(wandb_metrics)

        wandb.log(
            wandb_metrics,
            commit=False,
        )

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        adam_lr = self.hparam_conf["lr"]
        lambda_lr_scheduler_discount = self.hparam_conf["lambda_lr_sched_discount"]
        if self.is_nn_agent:
            optimizer = optim.Adam(self.net.parameters(), lr=self.hparam_conf["lr"])
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: lambda_lr_scheduler_discount ** epoch
            )
            return [optimizer], [scheduler]
        else:
            optimizer = optim.Adam([torch.ones(2, 2, requires_grad=True)])
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: lambda_lr_scheduler_discount ** epoch
            )
            return [optimizer], [scheduler]

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        # The sample size is set by a parameter. For legacy, default to episodes, or just get one.
        if "replay_sample_size" not in self.hparam_conf.keys():
            self.logger.info(
                "replay_sample_size not set, defaulting to episodes_length"
            )
        sample_size = self.hparam_conf.get(
            "replay_sample_size", self.hparam_conf["episode_length"]
        )
        # TODO parameterize more intelligently:
        dataset = RLDataset(self.replay_buffer, sample_size=sample_size)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            # todo is this a sensible default? Or maybe should be higher to reduce overhead
            batch_size=self.hparam_conf.get("batch_size", 1),
            collate_fn=self.collate_fn,
        )
        return dataloader

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0].device.index if self.on_gpu else "cpu"
