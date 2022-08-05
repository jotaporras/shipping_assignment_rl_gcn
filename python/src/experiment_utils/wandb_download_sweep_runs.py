"""
A utility to donwload runs from a given sweep.
"""
import argparse
import os
import sys

import wandb
import pandas as pd
import time
from datetime import timedelta

# These are the columns that are going to be fetched from W&B
# todo not being used actually lol sad.
FINAL_EXPERIMENT_KEYS = [
    "big_m_count",
    "trainer/global_step",
    #'incoming_interplants.dcs_0',
    "_step",
    #'deliveries_per_shipping_point_units.dcs_0',
    "mean_dcs_per_customer",
    "episode_process_time_s",
    #'deliveries_per_shipping_point_orders.dcs_1',
    "_runtime",
    "episode_process_time_ns",
    "average_cost_ep",
    "total_interplants",
    #'incoming_interplants.dcs_2',
    #'deliveries_per_shipping_point_orders.dcs_0',
    #'deliveries_per_shipping_point_units.dcs_1',
    #'deliveries_per_shipping_point_units.dcs_2',
    #'lr-Adam',
    #'deliveries_per_shipping_point_orders.dcs_2',
    #'incoming_interplants.dcs_1',
    #'lr-Adam-momentum',
    "_timestamp",
    "episode_reward",
    "loss",
    "episodes",
    "reward",
    "epoch",
    "",
]


def summary_to_dict(wandb_summary: "wandb.old.summary.HTTPSummary") -> dict:
    """Converts a wandb summary object (from my project) to a dict with columns flattened"""
    summary_dict = dict(wandb_summary)
    puredict = {}
    for k, v in summary_dict.items():
        if hasattr(v, "__dict__"):
            puredict[k] = dict(v)
        else:
            puredict[k] = v
    summary_df = pd.json_normalize(puredict, sep="_")
    summary_df.columns = [c.replace("/", "_") for c in summary_df.columns]
    [flattened_summary_dict] = summary_df.to_dict(orient="records")
    return flattened_summary_dict


if __name__ == "__main__":
    """
    Nov 9: Utility to download the runs from a Wandb sweep into Pandas DataFrames
    """
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_data_dir", type=str, required=True)
    parser.add_argument("--sweep_id", type=str, required=True)
    parser.add_argument("--project_name", type=str, default="thesis_final_experiments")# deleteme_final_experiment_test
    parser.add_argument("--entity", type=str, default="jotaporras")
    # fmt: on
    start = time.process_time()

    args = parser.parse_args()
    sweep_id = args.sweep_id
    project_name = args.project_name
    entity = args.entity
    base_data_dir = args.base_data_dir

    print("parsed arguments:")
    print(vars(args))

    api = wandb.Api()
    sweep_obj = api.sweep(f"{entity}/{project_name}/{sweep_id}")
    sweep_runs = list(sweep_obj.runs)
    sweep_name = sweep_obj.name

    # Storing datasets for each sweep run.
    if sweep_name == sweep_id:
        base_sweep_dir = os.path.join(base_data_dir, sweep_name)
    else:
        base_sweep_dir = os.path.join(base_data_dir, sweep_name + "_" + sweep_id)

    try:
        os.makedirs(base_sweep_dir)
    except (OSError, Exception) as error:
        print(error)
        print(
            "ERROR!!!! This sweep has already been downloaded. Delete the folder if you want to update."
        )
        sys.exit()

    run_info_dicts = []  # list of dicts for run info
    run_summary_dicts = []  # list of dicts for summary info
    for run in sweep_runs:
        run_id = run.id
        run_name = run.name
        print(f"Downloading data for {run_id} {run_name}")

        # History Metrics
        history_iterator = run.scan_history()
        history_df = pd.DataFrame(list(history_iterator))
        history_df.columns = [  # standardize columns
            c.replace("/", "_").replace("-", "_") for c in history_df.columns
        ]

        # todo uncomment.
        history_metrics_path = os.path.join(
            base_sweep_dir, f"{run_name}_{run_id}_history_metrics.csv"
        )
        history_df.to_csv(history_metrics_path, index=False)
        # Summary metrics
        summary_dict = summary_to_dict(run.summary)
        run_summary_dicts.append(summary_dict)

        # Run catalogue for the sweep
        run_info_dicts.append(
            {
                "run_id": run_id,
                "run_name": run_name,
                "run_dir": history_metrics_path,
                "run_state": run.state,
                "sweep_id": run.sweep.id,
                "sweep_name": run.sweep.name,
                "tags": str(run.tags),
            }
        )
    info_df = pd.DataFrame(run_info_dicts)
    summary_df = pd.DataFrame(run_summary_dicts)

    run_summary = pd.concat([info_df, summary_df], axis=1)

    run_summary.to_csv(os.path.join(base_sweep_dir, f"run_summary.csv"), index=False)
    end = time.process_time()
    elapsed = timedelta(seconds=end)
    print(f"Done, download script took {elapsed} to download {len(sweep_runs)} runs")
