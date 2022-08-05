import os

from visualization import shipping_allocation_result_viz


def create_summary_reports(experiment_directory):
    summary_movement_reports = (
        shipping_allocation_result_viz.read_all_summary_movement_reports(
            experiment_directory
        )
    )
    consolidated_summary = shipping_allocation_result_viz.consolidate_episodes(
        summary_movement_reports
    )
    consolidated_summary_path = os.path.join(
        experiment_directory, "consolidated_summary.csv"
    )
    print("Storing consolidated summary at", consolidated_summary_path)

    consolidated_summary.to_csv(consolidated_summary_path, index=False)


if __name__ == "__main__":
    experiment_names = [
        "bestfit_few_warehouses_v3",
        "dumb_few_warehouses_v3",
        "dqn2_few_warehouses_v3",
        "donotrhing_few_warehouses_v3",
    ]
    for experiment in experiment_names:
        create_summary_reports(f"data/results/{experiment}")
