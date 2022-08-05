# Named tuple for storing experience steps gathered in training

import logging

from experiments_seminar_2 import ptl_wandb_run_builder

# Default big-ish

# Named tuple for storing experience steps gathered in training

if __name__ == "__main__":
    """
    Oct 17, small env to try first GNN implementation
    """
    import os

    # os.environ["WANDB_MODE"] = "offline"
    # os.environ["WANDB_MODE"] = "dryrun"
    # logging.root.level = logging.DEBUG
    logging.root.level = logging.INFO

    config_dict = {  # default if no hyperparams set for sweep.
        "env": {
            "num_dcs": 5,
            "num_customers": 3,
            "num_commodities": 1,
            "orders_per_day": 1,  # start with one, and then play with this.
            "dcs_per_customer": 2,
            "demand_mean": 600,
            "demand_var": 150,
            # "num_steps": 30,  # steps per episode
            "big_m_factor": 10000,  # how many times the customer cost is the big m.
            # New parameters 2021
            "version": "v2",
            "order_generator": "normal_multivariate",
            # "order_generator": "biased",
            "reward_function": "negative_log_cost_minus_log_big_m_units",
        },
        "hps": {
            "replay_size": 30,  # Size of the buffer.
            "warm_start_steps": 30,  # Number of steps to run in warmup to populate the buffer # OLD COMMENT: apparently has to be smaller than batch size
            # "max_episodes": 500,
            "max_episodes": 10,
            "episode_length": 30,
            # "batch_size": 64,
            "batch_size": 32,
            "gamma": 0.8,
            "hidden_size": 12,
            "gnn_hidden_units": 64,
            # "lr": 3e-3,
            "lr": 9.5e-3,
            # "lr": 1e-4,
            "eps_end": 0.01,
            "eps_start": 0.01,
            "sync_rate": 30,  # Rate to sync the target and learning network.
            "lambda_lr_sched_discount": 0.999992,
        },
        "seed": 655,
        # "agent": "best_fit",
        # "agent": "lookahead",
        # "agent": "nn_mask_plus_consumption",
        # "agent": "gnn_physnet_aggdemand",
        # "agent": "gnn_physnet_ntype_aggdemand",
        # "agent": "nn_debug_mlp_cheat",
        "agent": "lookahead",
        # "agent": "random_valid",
    }

    codename = ptl_wandb_run_builder.conf_to_env_codename(config_dict)
    trainer, model = ptl_wandb_run_builder.create_ptl_shipping_allocation_rl_runner(
        config_dict,
        # experiment_name=f"{codename}_{config_dict['agent'][:3]}",
        # experiment_name=f"{codename}_{config_dict['agent']}",
        experiment_name=f"{codename}_{config_dict['agent'][:8]}_debugsteps",
        run_mode="experiment",
        extra_tags=[
            f"env_{codename}",
            "debug",
            # f"gnn_v1_tuning",
        ],
    )
    trainer.fit(model)
