import logging

logging.basicConfig(level=logging.DEBUG)
from experiments_seminar_2 import ptl_wandb_run_builder

if __name__ == "__main__":
    """
    Next gen (July 2021) greedy agent runner :)
    """
    config_dict = {
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
            "reward_function": "negative_cost_diff",
        },
        "hps": {
            "env": "shipping-v0",  # openai env ID.
            "episode_length": 150,  # todo isn't this an env thing?
            "max_episodes": 20,  # to do is this num episodes, is it being used?
            # "batch_size": 30,
            # "sync_rate": 2,  # Rate to sync the target and learning network, not used with this agent
            "lr": 1e-3,
            "discount": 0.8,
            "epsilon": 0.01,
            "init_state_value": 0.001,
        },
        "seed": 0,
        "agent": "q_learning"
        # "agent": "random_valid"
    }

    trainer, model = ptl_wandb_run_builder.create_ptl_shipping_allocation_rl_runner(
        config_dict,
        # run_mode="local_debug",
        run_mode="debug",
        experiment_name=f"rl_{config_dict['agent']}_{config_dict['env']['reward_function']}_single_commodity",
        project_name="rl_warehouse_assignment",
    )

    trainer.fit(model)
