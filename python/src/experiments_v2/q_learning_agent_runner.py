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
from torch.utils.data import DataLoader, IterableDataset

import agents
from agents.concrete_agents import RandomAgent, Agent
from experiments_v2 import q_learning_agent
from experiments_v2.greedy_agent_utils import GreedyAgentRLModel
from experiments_v2.ptl_callbacks import ShippingFacilityEnvironmentStorageCallback
from experiments_v2.ptl_callbacks import MyPrintingCallback, WandbDataUploader


def main():
    config_dict = {
        "env": {
            "num_dcs": 4,  # 3
            "num_customers": 100,
            "num_commodities": 35,
            "orders_per_day": int(100 * 0.05),
            "dcs_per_customer": 2,
            "demand_mean": 500,
            "demand_var": 150,
            "num_steps": 30,  # steps per episode
            "big_m_factor": 10000,  # how many times the customer cost is the big m.
        },
        "hps": {
            "env": "shipping-v0",  # openai env ID.
            "episode_length": 30,
            "max_episodes": 35,  # to do is this num episodes, is it being used?
            "lr": 1e-4,
            "discount": 0.95,
            "epsilon": 0.01,
            # "batch_size": 30,
            # "sync_rate": 2, # Rate to sync the target and learning network.
        },
        "seed": 0,
        "agent": "q_learning"
        # "agent": "random_valid"
    }

    torch.manual_seed(config_dict["seed"])
    np.random.seed(config_dict["seed"])
    random.seed(config_dict["seed"])  # not sure if actually used
    np.random.seed(config_dict["seed"])

    run = wandb.init(  # todo debugging why wrong project and experiment
        config=config_dict,
        project="rl_warehouse_assignment",
        name="q_learning_few_warehouses_debugreward",
    )  # todo why not saving config???

    config = wandb.config
    environment_config = config.env
    hparams = config.hps

    experiment_name = f"q_{config.agent}_few_warehouses_debugreward"
    wandb_logger = WandbLogger(
        project="rl_warehouse_assignment",
        name=experiment_name,
        tags=[
            "debug"
            # "experiment"
        ],
        log_model=False,
    )

    wandb_logger.log_hyperparams(dict(config))

    environment_parameters = network_flow_env_builder.build_network_flow_env_parameters(
        environment_config, hparams["episode_length"], order_gen="biased"
    )

    env = ShippingFacilityEnvironment(environment_parameters)
    agent = q_learning_agent.ShippingEnvQLearningAgent(
        environment_config["num_customers"],
        environment_config["num_dcs"],
        hparams["lr"],
        hparams["discount"],
        hparams["epsilon"],
        env,
    )

    model = GreedyAgentRLModel(agent, env, experiment_name=experiment_name)

    trainer = pl.Trainer(
        max_epochs=hparams["max_episodes"],
        # early_stop_callback=False,
        # val_check_interval=100,
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
