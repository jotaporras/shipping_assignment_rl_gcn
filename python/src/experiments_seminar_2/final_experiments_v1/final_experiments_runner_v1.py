# Named tuple for storing experience steps gathered in training
import argparse
import json
import logging

from experiments_seminar_2 import ptl_wandb_run_builder

# Default big-ish

# Named tuple for storing experience steps gathered in training
from experiments_seminar_2.ptl_wandb_run_builder import codename_to_env_conf


def setup_argparser():
    parser = argparse.ArgumentParser()
    # fmt: off
    # Experiment specific
    parser.add_argument("--env_codename", type=str,required=True)
    parser.add_argument("--run_iteration_num", type=int, required=True)
    # env
    parser.add_argument("--demand_mean", type=int, default=600)
    parser.add_argument("--demand_var", type=int, default=150)
    parser.add_argument("--big_m_factor", type=int, default=10000)
    parser.add_argument("--version", type=str, default="v2")
    parser.add_argument("--order_generator", type=str, default="normal_multivariate")
    parser.add_argument("--reward_function", type=str, default="negative_log_cost_minus_log_big_m_units")
    parser.add_argument("--inventory_generator", type=str, default="dirichlet")
    # hps
    parser.add_argument("--replay_size", type=int, default=30)
    parser.add_argument("--warm_start_steps", type=int, default=30)
    parser.add_argument("--max_episodes", type=int, default=100)
    parser.add_argument("--episode_length", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--gnn_hidden_units", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.009)
    parser.add_argument("--eps_end", type=float, default=0.01)
    parser.add_argument("--eps_start", type=float, default=0.01)
    parser.add_argument("--sync_rate", type=int, default=30)
    parser.add_argument("--lambda_lr_sched_discount", type=float, default=0.999992)
    parser.add_argument("--time_limit_milliseconds", type=float, default=240*1000)
    # Others
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--agent", type=str, default="branch_and_bound_milp")
    parser.add_argument("--project_name", type=str, default="rl_warehouse_assignment")
    parser.add_argument("--experiment_name", type=str,nargs='?') # if experiment empty use the custom logic.
    parser.add_argument("--dry_run", action='store_true') # if experiment empty use the custom logic.
    parser.add_argument("--wandb_offline", action='store_true') # if experiment empty use the custom logic.
    parser.add_argument("--debug_mode", action='store_true') # if experiment empty use the custom logic.

    # fmt: on
    return parser


if __name__ == "__main__":
    """
    Oct 17, Running mid env on many agents to finally rule out existing architectures.
    """
    import os

    # logging.root.level = logging.DEBUG
    logging.root.level = logging.INFO
    parser = setup_argparser()
    args = parser.parse_args()
    args_dict = vars(args)
    if args.dry_run:
        logging.info("DRY RUNNING BABY")
        os.environ["WANDB_MODE"] = "dryrun"

    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    if args.debug_mode:
        logging.root.level = logging.DEBUG

    logging.info(f"Parsed args: {json.dumps(args_dict, sort_keys=False, indent=4)}")

    # parsing the environment size from a single parameter.
    env_subdict = codename_to_env_conf(args.env_codename)

    logging.info(f"Parsed env codename {args.env_codename} as ")
    logging.info(f"{json.dumps(env_subdict, sort_keys=False, indent=4)}")

    config_dict = {  # default if no hyperparams set for sweep.
        "env": {
            "num_dcs": env_subdict["num_dcs"],
            "num_customers": env_subdict["num_customers"],
            "num_commodities": env_subdict["num_commodities"],
            "orders_per_day": env_subdict["orders_per_day"],
            "dcs_per_customer": env_subdict["dcs_per_customer"],
            "demand_mean": args.demand_mean,
            "demand_var": args.demand_var,
            "big_m_factor": args.big_m_factor,
            "version": args.version,
            "order_generator": args.order_generator,
            "reward_function": args.reward_function,
            "inventory_generator": args.inventory_generator,
        },
        "hps": {
            "replay_size": args.replay_size,
            "warm_start_steps": args.warm_start_steps,
            "max_episodes": args.max_episodes,
            "episode_length": args.episode_length,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "gnn_hidden_units": args.gnn_hidden_units,
            "lr": args.lr,
            "eps_end": args.eps_end,
            "eps_start": args.eps_start,
            "sync_rate": args.sync_rate,
            "lambda_lr_sched_discount": args.lambda_lr_sched_discount,
            "time_limit_milliseconds": args.time_limit_milliseconds,
        },
        "seed": args.seed,
        "agent": args.agent,
        "run_iteration_num": args.run_iteration_num,
    }

    logging.info("Starting run with parameters")
    logging.info(json.dumps(config_dict, sort_keys=True, indent=4))

    codename = ptl_wandb_run_builder.conf_to_env_codename(config_dict)
    experiment_name = (
        args.experiment_name
        if args.experiment_name is not None
        else f"{codename}_{config_dict['agent']}"
    )
    logging.info(f"Experiment name: {experiment_name}")
    trainer, model = ptl_wandb_run_builder.create_ptl_shipping_allocation_rl_runner(
        config_dict,
        experiment_name=experiment_name,
        run_mode="experiment",
        extra_tags=[f"env_{codename}", "bb_scalability_pretest"],
        project_name=args.project_name,
    )
    trainer.fit(model)
