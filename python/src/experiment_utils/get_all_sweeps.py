"""
A simple script to get all sweeps from a project in W&B
"""
import argparse
import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, required=True)
    args = parser.parse_args()

    api = wandb.Api()
    print(set([r.sweep.id for r in api.runs(f"jotaporras/{args.project}")]))
