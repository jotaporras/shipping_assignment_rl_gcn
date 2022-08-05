# Named tuple for storing experience steps gathered in training
import os
import logging

from experiments_seminar_2 import ptl_wandb_run_builder

# Default big-ish

# Named tuple for storing experience steps gathered in training

if __name__ == "__main__":
    os.environ["WANDB_MODE"] = "offline"
    logging.root.level = logging.DEBUG
    # Small one copy pasted from onehot dqn v2
    config_dict = {  # default if no hyperparams set for sweep.
        "env": {
            "num_dcs": 3,
            "num_customers": 5,
            "num_commodities": 1,
            "orders_per_day": 1,  # start with one, and then play with this.
            "dcs_per_customer": 2,
            "demand_mean": 500,
            "demand_var": 150,
            # "num_steps": 30,  # steps per episode
            "big_m_factor": 10000,  # how many times the customer cost is the big m.
            # New parameters 2021
            "version": "v2",
            "order_generator": "biased",
            "reward_function": "negative_log_cost_minus_log_big_m_units",
        },
        "hps": {
            "replay_size": 150,  # Size of the buffer.
            "warm_start_steps": 150,  # Number of steps to run in warmup to populate the buffer # OLD COMMENT: apparently has to be smaller than batch size
            "max_episodes": 50,  # to do is this num episodes, is it being used?
            "episode_length": 30,
            "batch_size": 30,
            "gamma": 0.8,
            "hidden_size": 12,
            "lr": 1e-3,
            "eps_end": 0.01,
            "eps_start": 0.01,
            "sync_rate": 2,  # Rate to sync the target and learning network.
        },
        # TODO old parameters just in case.
        # "hps": {
        #     "env": "shipping-v0",  # openai env ID.
        #     "episode_length": 150,  # todo isn't this an env thing?
        #     # "max_episodes": 1,  # to do is this num episodes, is it being used?
        #     "max_episodes": 20,  # to do is this num episodes, is it being used?
        #     # "batch_size": 30,
        #     # "sync_rate": 2,  # Rate to sync the target and learning network, not used with this agent
        #     "lr": 1e-3,
        #     "discount": 0.8,
        #     "epsilon": 0.01,
        #     "init_state_value": 0.001,
        # },
        "seed": 655,
        "agent": "nn_customer_onehot"
        # "agent": "random_valid"
    }

    # the big one
    # config_dict = {  # default if no hyperparams set for sweep.
    #     "env": {
    #         "num_dcs": 5,
    #         "num_customers": 500,
    #         "num_commodities": 15,
    #         "orders_per_day": 10,
    #         "dcs_per_customer": 2,
    #         "demand_mean": 500,
    #         "demand_var": 150,
    #         "num_steps": 50,  # steps per episode
    #         "big_m_factor": 1000,  # how many times the customer cost is the big m.
    #         "version": "v2",
    #         "order_generator": "biased",
    #         "reward_function": "negative_log_cost_minus_log_big_m_units",
    #     },
    #     "hps": {
    #         "env": "shipping-v0",  # openai env ID.
    #         "replay_size": 50,
    #         "warm_start_steps": 50,  # apparently has to be smaller than batch size
    #         "max_episodes": 50,  # to do is this num episodes, is it being used?
    #         "episode_length": 50,  # todo isn't this an env thing?
    #         "batch_size": 32,
    #         "gamma": 0.99,
    #         "hidden_size": 12,  # todo unused probs
    #         "lr": 1e-5,
    #         "eps_end": 1.0,  # todo consider keeping constant to start.
    #         "eps_start": 0.99,  # todo consider keeping constant to start.
    #         "eps_last_frame": 1000,  # todo maybe drop
    #         "sync_rate": 2,  # Rate to sync the target and learning network.
    #     },
    #     "seed": 0,
    #     "agent": "dqn_agent_deep",
    # }
    # torch.manual_seed(config_dict["seed"])
    # np.random.seed(config_dict["seed"])
    # random.seed(config_dict["seed"])  # not sure if actually used
    # np.random.seed(config_dict["seed"])
    # run = wandb.init(
    #     config=config_dict,
    #     project="rl_warehouse_assignment",
    #     name="dqn_yolo",
    #     tags=["debug"],
    # )  # todo why not saving config???
    # config = wandb.config
    # environment_config = config.env
    # hparams = config.hps
    #
    # if "lr" in config:
    #     hparams["lr"] = config.lr
    #     hparams["gamma"] = config.gamma
    #
    # environment_instance = (
    #     network_flow_env_builder.build_next_gen_network_flow_environment(
    #         environment_config,
    #         hparams["episode_length"],
    #         order_gen=environment_config["order_generator"],
    #         reward_function_name=environment_config["reward_function"],
    #     )
    # )
    # net = CustomerOnehotDQN(
    #     num_customers=environment_config["num_customers"],
    #     num_dcs=environment_config["num_dcs"],
    # )
    #
    # agent = pytorch_agents.CustomerDQNAgent(
    #     environment_instance, net, hparams["eps_end"]
    # )
    #
    # model = ShippingAssignmentRunner(
    #     agent=agent, env=environment_instance, hparams=hparams
    # )
    # wandb_logger = WandbLogger(
    #     project="rl_warehouse_assignment",
    #     name="dqn_yolo",
    #     tags=["experiment"],
    #     log_model=False,
    # )
    # wandb_logger.log_hyperparams(dict(config))
    # wandb.watch(net, log_freq=5, log="all")
    # trainer = pl.Trainer(
    #     # early_stop_callback=False,
    #     track_grad_norm=5,
    #     val_check_interval=100,
    #     max_epochs=hparams["max_episodes"] * hparams["replay_size"],
    #     min_epochs=hparams["max_episodes"] * hparams["replay_size"],
    #     progress_bar_refresh_rate=0,
    #     logger=wandb_logger,
    #     # log_save_interval=1,
    #     # row_log_interval=1,  # the default` of this may leave info behind.
    #     callbacks=[
    #         MetricsAccumulatorCallback(),
    #         ShippingFacilityEnvironmentStorageCallback(
    #             "shipping_assignment_env",
    #             base="data/results/",
    #             experiment_uploader=WandbDataUploader(),
    #         ),
    #     ],
    # )
    trainer, model = ptl_wandb_run_builder.create_ptl_shipping_allocation_rl_runner(
        config_dict,
        experiment_name="customer_onehot_with_builder",
        run_mode="debug",
    )
    trainer.fit(model)
