import ptl_wandb_run_builder

# TODO: there's a circular dependency if you import ptl_wandb_run_builder from the same level.
# Currently stuff only works if you run an experiment from a subpackage.
if __name__ == "__main__":
    """
    Next gen (July 2021) greedy agent runner :)
    """
    config_dict = {
        "env": {
            "num_dcs": 3,
            "num_customers": 50,
            "num_commodities": 35,
            "orders_per_day": int(100 * 0.05),
            "dcs_per_customer": 2,
            "demand_mean": 500,
            "demand_var": 150,
            "num_steps": 30,  # steps per episode
            "big_m_factor": 10000,  # how many times the customer cost is the big m.
            "version": "v2",
            "order_generator": "biased",
        },
        "hps": {
            "env": "shipping-v0",  # openai env ID.
            "episode_length": 30,  # todo isn't this an env thing?
            "max_episodes": 35,  # to do is this num episodes, is it being used?
            "batch_size": 30,
            "sync_rate": 2,  # Rate to sync the target and learning network.
        },
        "seed": 0,
        "agent": "best_fit"
        # "agent": "random_valid"
    }

    trainer, model = ptl_wandb_run_builder.create_ptl_shipping_allocation_rl_runner(
        config_dict,
        experiment_name=f"gr_best_fit_few_warehouses_debugreward",
        run_mode="local_debug",
        project_name="rl_warehouse_assignment",
    )

    trainer.fit(model)
