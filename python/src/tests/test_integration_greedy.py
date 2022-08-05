from experiments_seminar_2 import ptl_wandb_run_builder

if __name__ == "__main__":
    """
    An end to end test of the greedy agent to ensure code integrity.
    """
    config_dict = {
        "env": {
            "num_dcs": 3,
            "num_customers": 4,
            "num_commodities": 3,
            "orders_per_day": 2,
            "dcs_per_customer": 2,
            "demand_mean": 500,
            "demand_var": 150,
            "num_steps": 5,  # steps per episode
            "big_m_factor": 10000,  # how many times the customer cost is the big m.
        },
        "hps": {
            "env": "shipping-v0",  # openai env ID.
            "episode_length": 5,  # todo isn't this an env thing?
            "max_episodes": 5,  # to do is this num episodes, is it being used?
            "batch_size": 5,
            "sync_rate": 1,  # Rate to sync the target and learning network.
        },
        "seed": 0,
        "agent": "best_fit"
        # "agent": "random_valid"
    }

    trainer, model = ptl_wandb_run_builder.create_ptl_shipping_allocation_rl_runner(
        config_dict,
        run_mode="debug",
        # run="experiment",
        project_name="rl_warehouse_assignment",
    )

    trainer.fit(model)
