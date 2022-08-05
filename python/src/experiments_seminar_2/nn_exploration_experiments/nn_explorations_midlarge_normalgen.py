# Named tuple for storing experience steps gathered in training

import logging

from experiments_seminar_2 import ptl_wandb_run_builder

# Default big-ish

# Named tuple for storing experience steps gathered in training

if __name__ == "__main__":
    import os

    # os.environ["WANDB_MODE"] = "offline"
    # logging.root.level = logging.DEBUG
    logging.root.level = logging.INFO

    config_dict = {  # default if no hyperparams set for sweep.
        "env": {
            "num_dcs": 5,
            "num_customers": 50,
            "num_commodities": 25,
            "orders_per_day": 1,  # start with one, and then play with this.
            "dcs_per_customer": 2,
            "demand_mean": 600,
            "demand_var": 150,
            # "num_steps": 30,  # steps per episode
            "big_m_factor": 10000,  # how many times the customer cost is the big m.
            # New parameters 2021
            "version": "v2",
            "order_generator": "normal_multivariate",
            "reward_function": "negative_log_cost_minus_log_big_m_units",
            # "reward_function": "negative_if_bigm_else_denom_cost",
            # "reward_function": "negative_log_cost_minus_log_big_m_units_exp",
            # "reward_function": "negative_cost_times_bigm"
        },
        "hps": {
            "replay_size": 150,  # Size of the buffer.
            "warm_start_steps": 150,  # Number of steps to run in warmup to populate the buffer # OLD COMMENT: apparently has to be smaller than batch size
            "max_episodes": 200,
            "episode_length": 30,
            "batch_size": 30,
            "gamma": 0.8,
            "hidden_size": 12,
            "lr": 3e-3,
            # "lr": 8e-3,
            # "lr": 1e-4,
            "eps_end": 0.01,
            "eps_start": 0.01,
            "sync_rate": 30,  # Rate to sync the target and learning network.
            "lambda_lr_sched_discount": 0.999992,
        },
        "seed": 655,
        # "agent": "nn_warehouse_mask",
        # "agent": "nn_mask_plus_customer_onehot",
        # "agent": "nn_full_mlp",
        # "agent": "nn_full_mlp",  # Actually shooting myself in the foot
        # "agent": "random_valid",
        # "agent": "best_fit",
        # "agent": "lookahead",
        # "agent": "nn_debug_mlp_cheat",
        # "agent": "lookahead",
        "agent": "nn_mask_plus_consumption",
    }

    trainer, model = ptl_wandb_run_builder.create_ptl_shipping_allocation_rl_runner(
        config_dict,
        experiment_name=f"{config_dict['agent']}_ordergen_normal_50C5W25K",
        # experiment_name=f"nn_debug_mlp_cheat_nomask_small_debug_smallarch",
        run_mode="experiment",
        extra_tags=["ordergen_normal_experiment", "env_50C5W25K"],
    )
    trainer.fit(model)
