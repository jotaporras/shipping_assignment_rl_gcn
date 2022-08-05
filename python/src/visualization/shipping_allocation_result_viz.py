from typing import List

import pandas as pd

# import seaborn as sns
import matplotlib.pyplot as plt
import os

# Reusable DF transformations and visualizations


def read_all_summary_movement_reports(experiment_base: str) -> List[pd.DataFrame]:
    eps = pd.Series(os.listdir(experiment_base))
    max_ep = eps.str.extract("ep_(\d+)", 0).fillna(0).astype(int).max().values[0]
    summary_movement_reports = [
        pd.read_csv(
            os.path.join(experiment_base, f"ep_{i}/summary_movement_report.csv")
        )
        for i in range(max_ep + 1)
    ]
    return summary_movement_reports


def consolidate_episodes(summary_movement_reports):
    ep_summaries = []
    for ep, episode_summary in enumerate(summary_movement_reports):
        episode_summary["step"] = range(episode_summary.shape[0])
        episode_summary["episode"] = ep
        ep_summaries.append(episode_summary)
    consolidated_summary = pd.concat(ep_summaries, axis=0)
    return consolidated_summary


def calculate_costs_per_episode(consolidated_summary, cost_col="total_cost"):
    costs_per_episode = (
        consolidated_summary.groupby("episode")[cost_col]
        .sum()
        .reset_index()
        .sort_values("episode")
    )
    return costs_per_episode


# Compare costs per episode for a number of runs
# def plot_compare_costs_per_episode(costs_per_episode_dfs: List,cost_col='total_cost'):
#     fig = plt.figure()
#     eps = costs_per_episode_dfs[0].episodes.tolist()
#     for costs_per_episode in costs_per_episode_dfs:
#         sns.lineplot(x=eps,y=costs_per_episode[cost_col])
#     return fig


def compare_two_runs(
    consolidated_summary_a, consolidated_summary_b, simplification_factor=1e9
):
    # copy
    consolidated_summary_a = consolidated_summary_a.copy()
    consolidated_summary_b = consolidated_summary_b.copy()
    # simplify costs:
    consolidated_summary_b.total_cost = (
        consolidated_summary_b.total_cost // simplification_factor
    )
    consolidated_summary_b.total_cost = (
        consolidated_summary_b.total_cost // simplification_factor
    )

    # Analyzing mean & STD of cost per episode
    print(consolidated_summary_a.total_cost.describe())
    print(consolidated_summary_b.total_cost.describe())

    # TODO plot


import os

if __name__ == "__main__":
    summary_movement_reports = read_all_summary_movement_reports(
        "data/results/bestfit_few_warehouses"
    )
    consolidated_summary = consolidate_episodes(summary_movement_reports)
    print(calculate_costs_per_episode(consolidated_summary) // 100000000)
