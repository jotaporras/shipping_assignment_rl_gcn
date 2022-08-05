import logging

logging.basicConfig(level=logging.INFO)
from experiments_seminar_2 import ptl_wandb_run_builder

if __name__ == "__main__":
    """
    First attempt at running the new lookahead agent.
    """
    config_dict = {
        "env": {
            "num_dcs": 5,
            "num_customers": 100,
            "num_commodities": 15,
            "orders_per_day": 1,  # start with one, and then play with this.
            "dcs_per_customer": 3,
            "demand_mean": 500,
            "demand_var": 150,
            "num_steps": 30,  # steps per episode
            "big_m_factor": 10000,  # how many times the customer cost is the big m.
            # New parameters 2021
            "version": "v2",
            "order_generator": "biased",
            "reward_function": "negative_cost",  # big_m_diff
        },
        "hps": {
            "env": "shipping-v0",  # openai env ID.
            "episode_length": 150,  # todo isn't this an env thing?
            "max_episodes": 20,  # to do is this num episodes, is it being used?
            "batch_size": 30,
        },
        "seed": 0,
        "agent": "best_fit",
    }

    trainer, model = ptl_wandb_run_builder.create_ptl_shipping_allocation_rl_runner(
        config_dict,
        # run_mode="local_debug",
        run_mode="experiment",
        experiment_name=f"{config_dict['agent']}_medium_size",
        project_name="rl_warehouse_assignment",
    )

    trainer.fit(model)
