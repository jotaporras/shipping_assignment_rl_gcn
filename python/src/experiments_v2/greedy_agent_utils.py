import logging
import random
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from envs import network_flow_env_builder
from pytorch_lightning.loggers import WandbLogger
from shipping_allocation.envs.network_flow_env import (
    EnvironmentParameters,
    ShippingFacilityEnvironment,
)
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

import agents
from agents import Agent
from dqn.dqn_common import ShippingFacilityEpisodesDataset
from experiment_utils import report_generator
from experiments_v2.ptl_callbacks import (
    MyPrintingCallback,
    WandbDataUploader,
    ShippingFacilityEnvironmentStorageCallback,
)

logger = logging.getLogger(__name__)

# Num epochs == num EPs.
class GreedyAgentRLModel(pl.LightningModule):
    """
    This runner is used for greedy agents or agents that
    don't need to use the PTL functions for updating a neural network.
    """

    environment_parameters: EnvironmentParameters
    agent: Agent

    DEBUG = False

    def __init__(
        self,
        agent,
        env,  # TODO#: ShippingAssignmentEnvironment,
        experiment_name="",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.agent = agent
        self.env = env
        self.physical_network = self.env.physical_network
        self.experiment_name = experiment_name

        # Running values
        self.state = self.env.reset()
        self.done = False

        self.episode_counter = 0
        # Metrics
        self.episode_reward = 0.0
        self.running_reward = 0.0
        self.actions = []
        self.episode_rewards = []
        self.info = {}
        self.episodes_info = []

        # debug var for env reset
        self.was_reset = True

    def forward(self, *args, **kwargs):
        pass  # do nothing related to NNs.

    def training_step(self, step_info: Tuple[int, int, int], num_batch):
        """
        A step of simulation. Step_info is a tuple of three integers,
        see ShippingFacilityEpisodesDataset for the specification
        Args:
            step_info: (step, num_order, ep_start)
            num_batch:

        Returns:

        """
        step, order, ep_start = step_info

        logger.debug("Getting into training step")

        if ep_start:
            logger.info(f"Starting episode {self.episode_counter}")
            if not self.was_reset:
                logger.error("ERROR!!! EXPECTED ENV TO BE RESET.")
            else:
                self.was_reset = False

        action = self.agent.get_action(self.state)

        # the agent observes the first state and chooses an action
        # environment steps with the agent's action and returns new state and reward
        next_state, reward, done, info = self.env.step(action)

        # print(f"Got reward {reward} done {done}")
        self.agent.train((self.state, action, next_state, reward, done))

        self.state = next_state
        self.episode_reward += reward

        if done:
            # update the info to store the reports
            self.info = info

        # Render the current state of the environment
        self.env.render()
        self.actions.append(action)
        self.episode_rewards.append(reward)

        shim = (
            torch.ones(2, 2, requires_grad=True) - 1
        ).sum()  # a dummy operation to trick ptl

        # result = pl.TrainResult(
        #     minimize=shim
        # )  # use the train result just for logging purposes.
        self.log("reward", reward)
        self.log("episode_reward", self.episode_reward)
        self.log("episodes", self.episode_counter)

        return shim

    def training_epoch_end(self, outputs):
        """
        This is triggered when the greedy dataset reaches the end of an episode.
        Args:
            outputs:

        Returns:

        """
        logger.info(f"Finishing episode {self.episode_counter}")
        # Finished one episode, store reports
        logger.info("Finished episode, storing information")
        self.episodes_info.append(self.info)

        self._wandb_custom_metrics(self.info)

        self.episode_counter += 1

        self._reset_env_and_metrics()
        # return outputs

    def _reset_env_and_metrics(self):
        logger.info(
            f"=========== starting episode {self.episode_counter} loop ==========="
        )
        logger.debug("Initial environment: ")
        self.env.render()
        self.state = self.env.reset()
        self.done = False
        self.episode_reward = 0.0
        self.actions = []
        self.episode_rewards = []
        self.info = {}
        self.was_reset = True  # Making sure PTL is doing its job.

    def train_dataloader(self) -> DataLoader:
        """
        This custom dataloader forces to run one step at a time (batching doesn't make sense here.)
        it's just a fancy iterator.
        """
        return DataLoader(
            dataset=ShippingFacilityEpisodesDataset(
                num_steps=self.env.num_steps,
                orders_per_day=self.env.order_generator.orders_per_day,
            ),
            batch_size=1,
            shuffle=False,
        )

    def _wandb_custom_metrics(self, info):
        wandb_metrics = report_generator.convert_info_into_metrics_summary_dict(info)

        logger.info(
            f"Episode {self.episode_counter} had {wandb_metrics['big_m_count']} BigMs"
        )
        logger.info("Finished episode with greedy runner, logging metrics to wandb:")
        logger.info(wandb_metrics)

        self.logger.info(
            "Logging with commit false (greedy_agent_utils(Lightning module))"
        )
        wandb.log(
            wandb_metrics,
            commit=False,
        )

    def configure_optimizers(self):
        # return [
        #     Adam([torch.ones(2, 2, requires_grad=True)])
        # ]  # shouldn't use it at all.
        return Adam([torch.ones(2, 2, requires_grad=True)])

    def backward(self, trainer, loss: Tensor, optimizer: Optimizer) -> None:
        return


def main():
    config_dict = {
        "env": {
            "num_dcs": 3,
            "num_customers": 5,
            "num_commodities": 3,
            "orders_per_day": 2,
            "dcs_per_customer": 2,
            "demand_mean": 500,
            "demand_var": 150,
            "num_steps": 10,  # steps per episode
            "big_m_factor": 10000,  # how many times the customer cost is the big m.
        },
        "hps": {
            "env": "shipping-v0",  # openai env ID.
            "episode_length": 30,  # todo isn't this an env thing?
            "max_episodes": 5,  # to do is this num episodes, is it being used?
            "batch_size": 30,
            "sync_rate": 2,  # Rate to sync the target and learning network.
        },
        "seed": 0,
        "agent": "best_fit"
        # "agent": "random_valid"
    }

    torch.manual_seed(config_dict["seed"])
    np.random.seed(config_dict["seed"])
    random.seed(config_dict["seed"])  # not sure if actually used
    np.random.seed(config_dict["seed"])

    run = wandb.init(  # todo debugging why wrong project and experiment
        config=config_dict,
        project="rl_warehouse_assignment",
        name="best_fit_few_warehouses_debugreward",
    )

    config = wandb.config
    environment_config = config.env
    hparams = config.hps

    experiment_name = f"gr_{config.agent}_few_warehouses_debugreward"
    wandb_logger = WandbLogger(
        project="rl_warehouse_assignment",
        name=experiment_name,
        tags=[
            # "debug"
            # "experiment"
            "local_debug"
        ],
        log_model=False,
    )

    wandb_logger.log_hyperparams(dict(config))

    environment_parameters = network_flow_env_builder.build_network_flow_env_parameters(
        environment_config, hparams["episode_length"], order_gen="biased"
    )

    env = ShippingFacilityEnvironment(environment_parameters)
    agent = agents.get_agent(env, environment_config, hparams, config.agent)

    model = GreedyAgentRLModel(agent, env, experiment_name=experiment_name)

    trainer = pl.Trainer(
        max_epochs=hparams["max_episodes"],
        # early_stop_callback=False,
        val_check_interval=100,
        logger=wandb_logger,
        # log_save_interval=1,
        # row_log_interval=1,  # the default of this may leave info behind.
        callbacks=[
            MyPrintingCallback(),
            ShippingFacilityEnvironmentStorageCallback(
                experiment_name,
                base="data/results/",
                experiment_uploader=WandbDataUploader(),
            ),
        ],
    )

    trainer.fit(model)


if __name__ == "__main__":
    # logging.root.level = logging.INFO
    logging.root.level = logging.DEBUG
    main()
