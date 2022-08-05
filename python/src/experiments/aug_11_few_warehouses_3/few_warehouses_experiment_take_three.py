from multiprocessing.context import Process

from experiment_utils import experiment_runner
from experiments import reproducibility
import numpy as np
import random

"""
    Experiment with only a few warehouses but a lot of customers, which would be a more realistic scenario.
    
    Same as few warehouses v2 but with reproducibility setup, and donothing to track actions.
"""
# for i in reversed(list(range(num_experiments))):
# print("Running experiment id ", i)
# parameters:
num_dcs = 3
num_customers = 100  # scale up to 200
num_commodities = 35  # scale up to 50
orders_per_day = int(num_customers * 0.05)  # was going to be 0.1 but ran too slow.
# orders_per_day = 2
dcs_per_customer = 2
demand_mean = 500
demand_var = 150
num_steps = 30  # 1 month
num_episodes = 1000  # now with more eps ✨
# num_episodes = 1 #test reproducibility.


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
        experiment_name=f"bestfit_few_warehouses_v3",
    )
    print("===DONE BESTFIT===")


def run_random():
    print("***RUNNING RANDOM***")
    reproducibility.set_seeds(0)
    print("Check this array to ensure reproducibility")
    print("Reproducibility RANDOM", np.random.randint(0, 500, size=(5, 1)))
    runner_random = experiment_runner.create_random_experiment_runner(
        num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        num_commodities,
        orders_per_day,
        num_steps,
    )

    runner_random.run_episodes(
        num_steps,
        num_episodes,
        orders_per_day,
        experiment_name=f"dumb_few_warehouses_v3",
    )
    print("***DONE RANDOM***")


def run_dqn():
    print("!!!RUNNING DQN!!!")
    reproducibility.set_seeds(0)
    print("Check this array to ensure reproducibility")
    print("Reproducibility DQN", np.random.randint(0, 500, size=(5, 1)))
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
        experiment_name=f"dqn2_few_warehouses_v3",
    )

    runner_dqn.agent.save_weights("models/dqn2_few_warehouses_v3.h5")
    print("!!!DONE DQN!!!")


def run_donothing():
    print("!!!RUNNING donothing!!!")
    reproducibility.set_seeds(0)
    print("Check this array to ensure reproducibility")
    print("Reproducibility donothing", np.random.randint(0, 500, size=(5, 1)))
    runner_donothing = experiment_runner.create_donothing_experiment_runner(
        num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        num_commodities,
        orders_per_day,
        num_steps,
    )
    runner_donothing.run_episodes(
        num_steps,
        num_episodes,
        orders_per_day,
        experiment_name=f"donotrhing_few_warehouses_v3",
    )
    print("!!!DONE donothing!!!")


if __name__ == "__main__":
    # set seeds for reproducibility
    random.seed(0)  # not sure if actually used
    np.random.seed(0)

    # ====parameters===== (4, 100, 2, 500, 150, 50, 10, 90) Total elapsed 173.465825

    print("====parameters=====")
    print(
        (
            ("num_dcs", num_dcs),
            ("num_customers", num_customers),
            ("dcs_per_customer", dcs_per_customer),
            ("demand_mean", demand_mean),
            ("demand_var", demand_var),
            ("num_commodities", num_commodities),
            ("orders_per_day", orders_per_day),
            ("num_steps", num_steps),
        )
    )

    pbf = Process(target=run_bestfit, args=())
    pr = Process(target=run_random, args=())
    pdqn = Process(target=run_dqn, args=())
    pnothing = Process(target=run_donothing(), args=())

    pbf.start()
    pr.start()
    pdqn.start()
    pnothing.start()

    pbf.join()
    pr.join()
    pdqn.join()
    pnothing.join()
    print("All processes done")
