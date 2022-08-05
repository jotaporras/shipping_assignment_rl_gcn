from experiment_utils import experiment_runner
import pandas as pd
from experiments import reproducibility


def test_episodes_reproducible():
    num_dcs = 5
    num_customers = 2  # scale up to 200
    num_commodities = 10  # scale up to 50
    orders_per_day = 2  # was going to be 0.1 but ran too slow.
    # orders_per_day = 2
    dcs_per_customer = 2
    demand_mean = 500
    demand_var = 150
    num_steps = 10  # 1 month
    # num_episodes = 500
    num_episodes = 1

    reproducibility.set_seeds(0)

    runner_donothing_1 = experiment_runner.create_donothing_experiment_runner(
        num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        num_commodities,
        orders_per_day,
        num_steps,
    )
    runner_donothing_1.run_episodes(
        num_steps,
        num_episodes,
        orders_per_day,
        experiment_name=f"donothing_reproducible_test_1",
    )

    reproducibility.set_seeds(0)

    runner_donothing_2 = experiment_runner.create_donothing_experiment_runner(
        num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        num_commodities,
        orders_per_day,
        num_steps,
    )
    runner_donothing_2.run_episodes(
        num_steps,
        num_episodes,
        orders_per_day,
        experiment_name=f"donothing_reproducible_test_2",
    )

    a = pd.read_csv(
        "data/results/donothing_reproducible_test_1/ep_0/movement_detail_report.csv"
    )[["source_name", "destination_name"]]
    b = pd.read_csv(
        "data/results/donothing_reproducible_test_2/ep_0/movement_detail_report.csv"
    )[["source_name", "destination_name"]]
    compare = pd.concat([a, b], axis=1)
    compare.columns = ["s1", "d1", "s2", "d2"]

    print(compare)
    print("Equality: ", a.equals(b))

    assert a.equals(b)


if __name__ == "__main__":
    test_episodes_reproducible()
