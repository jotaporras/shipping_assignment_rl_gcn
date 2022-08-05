from experiments_seminar_2 import ptl_wandb_run_builder

if __name__ == "__main__":
    """
    Next gen (July 2021) greedy agent runner :)
    """
    config_dict = {
        "env": {
            "num_dcs": 3,
            "num_customers": 5,
            "num_commodities": 2,
            "orders_per_day": 2,
            "dcs_per_customer": 2,
            "demand_mean": 500,
            "demand_var": 150,
            "num_steps": 10,  # steps per episode
            "big_m_factor": 10000,  # how many times the customer cost is the big m.
            "version": "v2",
            "order_generator": "biased",
            "reward_function": "negative_log_cost_minus_log_big_m_units",
        },
        "hps": {
            "env": "shipping-v0",  # openai env ID.
            "episode_length": 30,  # todo isn't this an env thing?
            "max_episodes": 5,  # to do is this num episodes, is it being used?
            "batch_size": 10,
            "sync_rate": 2,  # Rate to sync the target and learning network.
        },
        "seed": 0,
        "agent": "best_fit"
        # "agent": "random_valid"
    }

    trainer, model = ptl_wandb_run_builder.create_ptl_shipping_allocation_rl_runner(
        config_dict,
        experiment_name=f"gr_best_fit_few_warehouses_debugreward",
        run_mode="debug",
        project_name="rl_warehouse_assignment",
    )

    trainer.fit(model)
