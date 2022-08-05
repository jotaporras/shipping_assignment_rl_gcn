"""
Creates a PTL Runner based on some config, with WandB set up.
Before July 2021, this was duplicated in all agents.
"""
import logging
import os
import random
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import wandb

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from envs import network_flow_env_builder, ShippingFacilityEnvironment
import agents
from dqn import dqn_common
from experiments_seminar_2 import base_ptl_nn_agent_runner

from experiments_v2 import q_learning_agent
from experiments_v2.greedy_agent_utils import GreedyAgentRLModel
from experiments_v2.ptl_callbacks import (
    MyPrintingCallback,
    ShippingFacilityEnvironmentStorageCallback,
    WandbDataUploader,
)
import re

_VALID_RUN_MODES = {"debug", "experiment", "local_debug"}
_VALID_ENV_VERSIONS = {"v1", "v2"}
_logger = logging.getLogger(__name__)


def conf_to_env_codename(cd):
    """
    Args:
        cd: Config dict with the env.

    Returns:
    """
    return f"{cd['env']['num_customers']}C{cd['env']['num_dcs']}W{cd['env']['num_commodities']}K{cd['env']['orders_per_day']}F{cd['env']['dcs_per_customer']}V"


def codename_to_env_conf(codename: str) -> dict:
    """
    Converts a codename to a subdictionary of env config. Tested manually on ipytjhon.
    Args:
        codename: a string of format
        f"{cd['env']['num_customers']}C{cd['env']['num_dcs']}W{cd['env']['num_commodities']}K{cd['env']['orders_per_day']}F{cd['env']['dcs_per_customer']}V"
        e.g.
    Returns:
        A dict with those stuff.

    """
    _logger.info(f"Parsing {codename}")
    codename_template = "(\d+)C(\d+)W(\d+)K(\d+)F(\d+)V"
    compiled_expression = re.compile(codename_template)
    matcher = compiled_expression.match(codename)

    if matcher is None:
        raise ValueError(
            f"Illegal format for {codename}, should match expression {codename_template}"
        )

    env_dict = {
        "num_customers": int(matcher.group(1)),
        "num_dcs": int(matcher.group(2)),
        "num_commodities": int(matcher.group(3)),
        "orders_per_day": int(matcher.group(4)),
        "dcs_per_customer": int(matcher.group(5)),
    }
    return env_dict


def create_ptl_shipping_allocation_rl_runner(
    config_dict: dict,
    experiment_name="",
    run_mode: str = "debug",
    project_name="rl_warehouse_assignment",
    extra_tags: List = None,
):
    """
    TODO: this function is in progress, needs to support both Deep learning and greedy agents.
    Consider passing in the agents builder in the parameters.
    Args:
        config_dict:
        run_mode:
        project_name:

    Returns:
        trainer, model

    """
    if extra_tags is None:
        extra_tags = []
    if config_dict["seed"] is not None:
        _logger.info(f"Hard pinning RNG seeds to {config_dict['seed']}")
        _init_random_seeds(config_dict["seed"])
    else:
        _logger.debug("No seed set, seed is random")

    # Check required keys:
    if not all(k in config_dict for k in ["agent", "env", "hps"]):
        raise ValueError("Missing some of the required keys")
    if run_mode not in _VALID_RUN_MODES:
        raise ValueError("Wrong run mode, should be one of " + str(_VALID_RUN_MODES))

    agent_name = config_dict["agent"]

    if run_mode == "local_debug":
        logging.info("Disabling wandb sync for local debugging.")
        os.environ["WANDB_MODE"] = "offline"

    # todo debugging why wrong project and experiment
    # Make sure we're setting up wandb the correct way
    run = wandb.init(
        config=config_dict,
        project=project_name,
        name=experiment_name,
        tags=[run_mode] + extra_tags,
    )  # todo why not saving config???

    config = wandb.config
    environment_config = config.env
    hparams = config.hps

    # Validating config
    if "version" not in environment_config:
        raise ValueError("Missing new env parameter 'version'")
    if environment_config["version"] not in _VALID_ENV_VERSIONS:
        raise ValueError(
            "Wrong env version mode, should be one of " + str(_VALID_RUN_MODES)
        )

    # Backfilling parameters added after October
    if "lambda_lr_sched_discount" not in hparams.keys():
        _logger.info(
            "comfig.hps['lambda_lr_sched_discount'] not found, defaulting to 1"
        )
        hparams["lambda_lr_sched_discount"] = 1.0

    wandb_logger = WandbLogger(
        project=project_name,
        name=experiment_name,
        tags=[run_mode],
        log_model=False,
    )

    wandb_logger.log_hyperparams(dict(config))

    if environment_config["version"] == "v1":
        environment_parameters = (
            network_flow_env_builder.build_network_flow_env_parameters(
                environment_config, hparams["episode_length"], order_gen="biased"
            )
        )
        environment_instance = ShippingFacilityEnvironment(environment_parameters)
    elif environment_config["version"] == "v2":
        environment_instance = (
            network_flow_env_builder.build_next_gen_network_flow_environment(
                environment_config,
                hparams["episode_length"],
                order_gen=environment_config["order_generator"],
                reward_function_name=environment_config["reward_function"],
            )
        )
    else:
        raise ValueError("Illegal env version.")

    # TODO: mega hack to avoid extra code sins. This env-agent-config mess needs to be refactored.
    # TODO also documnt which agents are valid.
    if agent_name == "q_learning":
        agent = q_learning_agent.ShippingEnvQLearningAgent(
            environment_config["num_customers"],
            environment_config["num_dcs"],
            hparams["lr"],
            hparams["discount"],
            hparams["epsilon"],
            environment_instance,
            init_val=hparams.get("init_state_value", 100.0),
        )
    else:  # Using a greedy agent, use the builder
        agent = agents.get_agent(
            environment_instance, environment_config, hparams, config.agent
        )
    collate_fn = None
    if agent_name.startswith(
        "gnn"
    ):  # Todo not tested yet. also careful with agent names
        # GNN Agents need a special collate function to collate PyG graphs + experience.
        collate_fn = dqn_common.gnn_rl_collate_fn

    model = base_ptl_nn_agent_runner.ShippingAssignmentRunner(
        agent, environment_instance, hparams, collate_fn=collate_fn
    )
    # TODO keeping those commented parameters because they were once there,
    # In case I need them back, but unlikely.

    count = torch.cuda.device_count()
    print(f"Running with cuda device count {count}")
    one_gpu = 1 if count > 0 else None
    print(f"Triggering pl trainer with {one_gpu} (should show 1)")
    trainer = pl.Trainer(
        # max_epochs=hparams["max_episodes"], # old way.
        # New way TODO test compare diff agents make sure they work.
        max_epochs=hparams["max_episodes"]
        * hparams.get("replay_size", hparams["episode_length"])
        * environment_config["orders_per_day"],
        # early_stop_callback=False,
        val_check_interval=100,
        logger=wandb_logger,
        progress_bar_refresh_rate=0,  # Todo maybe parameterize
        # log_save_interval=1,
        # row_log_interval=1,  # the default of this may leave info behind.
        log_every_n_steps=1,
        gpus=count,  # use gpus if available
        callbacks=[
            MyPrintingCallback(),
            ShippingFacilityEnvironmentStorageCallback(
                experiment_name,
                base="data/results/",
                experiment_uploader=WandbDataUploader(),
            ),
            LearningRateMonitor(logging_interval="epoch", log_momentum=True),
        ],
    )
    return trainer, model


def _init_random_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    np.random.default_rng(0)
    random.seed(seed)  # not sure if actually used
    np.random.seed(seed)
    logging.info("Seed checker 1")
    logging.info(np.random.randint(0, 1000, size=5))
    logging.info("Seed checker 2")
    logging.info(np.random.randint(0, 1000, size=5))
