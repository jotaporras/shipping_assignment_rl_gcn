from experiment_utils import experiment_runner


def two_customers_dqn_debug_run():
    num_dcs = 10
    num_customers = 2
    num_commodities = 4
    orders_per_day = 2
    dcs_per_customer = 3
    demand_mean = 100
    demand_var = 20

    num_steps = 50
    num_episodes = 1000

    runner_dqn = experiment_runner.create_dqn_experiment_runner(
        num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        num_commodities,
        orders_per_day,
        num_steps,
    )
    runner_dqn.run_episodes(
        num_steps,
        num_episodes,
        orders_per_day,
        experiment_name="two_customers_dqn_debug",
    )


# one ep version for debugging the code.
def two_customers_dqn_debug_sample():
    num_dcs = 10
    num_customers = 2
    num_commodities = 4
    orders_per_day = 2
    dcs_per_customer = 3
    demand_mean = 100
    demand_var = 20

    num_steps = 50
    num_episodes = 1000

    runner_dqn = experiment_runner.create_dqn_experiment_runner(
        num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        num_commodities,
        orders_per_day,
        num_steps,
    )
    runner_dqn.run_episodes(
        num_steps,
        num_episodes,
        orders_per_day,
        experiment_name="two_customers_dqn_debug_sample",
    )


if __name__ == "__main__":
    # two_customers_dqn_debug_run()
    two_customers_dqn_debug_sample()
