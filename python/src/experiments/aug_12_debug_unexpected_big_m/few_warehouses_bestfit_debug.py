from experiment_utils import experiment_runner
from experiments import reproducibility
import numpy as np

# parameters:
# num_dcs = 3
# num_customers = 100  #scale up to 200
# num_commodities = 35  # scale up to 50
# orders_per_day = int(num_customers * 0.05) #was going to be 0.1 but ran too slow.
# # orders_per_day = 2
# dcs_per_customer = 2
# demand_mean = 500
# demand_var = 150
# num_steps = 30  # 1 month
# num_episodes = 1


# small version:
num_dcs = 3
num_customers = 2  # scale up to 200
num_commodities = 3  # scale up to 50
orders_per_day = 1  # was going to be 0.1 but ran too slow.
# orders_per_day = 2
dcs_per_customer = 2
demand_mean = 500
demand_var = 150
num_steps = 10  # 1 month
num_episodes = 1


def run_bestfit():
    print("===RUNNING BESTFIT===")
    reproducibility.set_seeds(0)
    print("Check this array to ensure reproducibility")
    print("Reproducibility BESTFIT", np.random.randint(0, 500, size=(5, 1)))
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
    runner_bestfit.run_episodes(
        num_steps,
        num_episodes,
        orders_per_day,
        experiment_name=f"z_debug_bestfit_few_warehouses",
    )
    print("===DONE BESTFIT===")


def run_agent_66_custom_agent():
    print("===RUNNING agent66===")
    reproducibility.set_seeds(0)
    print("Check this array to ensure reproducibility")
    print("Reproducibility agent66", np.random.randint(0, 500, size=(5, 1)))
    runner_bestfit = experiment_runner.create_agent_66_experiment_runner(
        num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        num_commodities,
        orders_per_day,
        num_steps,
    )
    runner_bestfit.run_episodes(
        num_steps,
        num_episodes,
        orders_per_day,
        experiment_name=f"z_debug_agent_highest",
    )
    print("===DONE agent66===")


if __name__ == "__main__":
    # run_bestfit()
    run_agent_66_custom_agent()
