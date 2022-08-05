from experiment_utils import experiment_runner
from experiments.experiment_params import *

if __name__ == "__main__":
    for i in reversed(list(range(num_experiments))):
        print("Running experiment id ", i)
        # parameters:
        num_dcs = dcs_list[i]
        num_customers = customers_list[i]
        num_commodities = num_commodities_list[i]
        orders_per_day = orders_per_day_list[i]
        # orders_per_day = 2#TODO FOR DEBUG
        dcs_per_customer = dcs_per_customer_list[i]
        demand_mean = 100
        demand_var = 20

        print("====parameters=====")
        print(
            (
                num_dcs,
                num_customers,
                dcs_per_customer,
                demand_mean,
                demand_var,
                num_commodities,
                orders_per_day,
                num_steps,
            )
        )

        # runner_random = experiment_runner.create_random_experiment_runner(
        #     num_dcs,
        #     num_customers,
        #     dcs_per_customer,
        #     demand_mean,
        #     demand_var,
        #     num_commodities,
        #     orders_per_day,
        #     num_steps,
        # )
        runner_bestfit = experiment_runner.create_bestfit_experiment_runner(
            num_dcs,
            num_customers,
            dcs_per_customer,
            demand_mean,
            demand_var,
            num_commodities,
            orders_per_day,
            num_steps,
        )
        # runner_random.run_episodes(
        #     num_steps, num_episodes, orders_per_day, experiment_name=f"dumb_agent_experiment_{i}"
        # )
        runner_bestfit.run_episodes(
            num_steps,
            num_episodes,
            orders_per_day,
            experiment_name=f"bestfit_agent_experiment_{i}",
        )
