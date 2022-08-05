import argparse
import logging
import random

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from envs import network_flow_env_builder
from pytorch_lightning.loggers import WandbLogger

# Default big-ish
from agents.pytorch_agents import CustomerOnehotDQN
from experiments_v2.base_ptl_agent_runner import DQNLightning
from experiments_v2.ptl_callbacks import (
    MyPrintingCallback,
    ShippingFacilityEnvironmentStorageCallback,
    WandbDataUploader,
)

# Named tuple for storing experience steps gathered in training

# new config Sep 2021, for testing.
config_dict = {  # default if no hyperparams set for sweep.
    "env": {
        "num_dcs": 3,
        "num_customers": 5,
        "num_commodities": 1,
        "orders_per_day": 1,  # start with one, and then play with this.
        "dcs_per_customer": 2,
        "demand_mean": 500,
        "demand_var": 150,
        "num_steps": 30,  # steps per episode
        "big_m_factor": 10000,  # how many times the customer cost is the big m.
        # New parameters 2021
        "version": "v2",
        "order_generator": "biased",
        "reward_function": "negative_log_cost_minus_log_big_m_units",
    },
    "hps": {
        "env": "shipping-v0",  # openai env ID.
        "replay_size": 150,
        "warm_start_steps": 150,  # apparently has to be smaller than batch size
        "max_episodes": 50,  # to do is this num episodes, is it being used?
        "episode_length": 30,  # todo isn't this an env thing?
        "batch_size": 30,
        "gamma": 0.8,
        "hidden_size": 12,
        "lr": 1e-3,
        "eps_end": 0.01,  # todo consider keeping constant to start.
        "eps_start": 0.01,  # todo consider keeping constant to start.
        "eps_last_frame": 1000,  # todo maybe drop
        "sync_rate": 2,  # Rate to sync the target and learning network.
    },
    "seed": 0,
    "agent": "dqn_onehot_test",
}

experiment_name = "dqn_onehot_v2"


class DQNLightningOneHot(DQNLightning):
    """
    DQN But with a network that does one hot encoding.
    This was the first experiment that reduced BigMs, damn son! I'm keeping it for legacy/sentimental reasons.
    """

    def __init__(self, hparams: argparse.Namespace, env) -> None:
        super(DQNLightningOneHot, self).__init__(hparams, env)

        # Observation space for this network is the number of customers (onehot).
        num_customers = self.env.physical_network.num_customers
        num_dcs = self.env.action_space.n

        self.net = CustomerOnehotDQN(num_customers, num_dcs)
        self.target_net = CustomerOnehotDQN(num_customers, num_dcs)


def main() -> None:
    torch.manual_seed(config_dict["seed"])
    np.random.seed(config_dict["seed"])
    random.seed(config_dict["seed"])  # not sure if actually used
    np.random.seed(config_dict["seed"])

    run = wandb.init(
        config=config_dict,
        project="rl_warehouse_assignment",
        name=experiment_name,
        tags=["experiment"],
    )  # todo why not saving config???

    # The config was set at the top of the file in case I'm not running a sweep.
    config = wandb.config
    environment_config = config.env
    hparams = config.hps

    # TODO Hotfix because wandb doesn't support sweeps.
    if "lr" in config:
        hparams["lr"] = config.lr
        hparams["gamma"] = config.gamma

    logging.warning("CONFIG CHECK FOR SWEEP")
    logging.warning(
        hparams["lr"]
    )  
    logging.warning(hparams["gamma"])

    wandb_logger = WandbLogger(
        project="rl_warehouse_assignment",
        name=experiment_name,
        tags=[
            "debug"
            # "experiment"
        ],
        log_model=False,  # todo sett this to true if you need the checkpoint models at some point
    )

    wandb_logger.log_hyperparams(dict(config))

    environment_instance = (
        network_flow_env_builder.build_next_gen_network_flow_environment(
            environment_config,
            hparams["episode_length"],
            order_gen=environment_config["order_generator"],
            reward_function_name=environment_config["reward_function"],
        )
    )

    model = DQNLightningOneHot(hparams, environment_instance)
    wandb.watch(model.net, log_freq=5, log="all")

    trainer = pl.Trainer(
        track_grad_norm=5,
        max_epochs=hparams["max_episodes"] * hparams["replay_size"],
        val_check_interval=100,
        logger=wandb_logger,
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
    # os.environ["WANDB_MODE"] = "offline"
    logging.root.level = logging.DEBUG
    main()
