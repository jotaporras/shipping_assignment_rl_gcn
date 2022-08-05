"""
This script is a scratchpad to explore the order generators: both the biased and the normal multivariate.
"""
import streamlit as st
import pandas as pd
import numpy as np
from envs import order_generators
from shipping_allocation import PhysicalNetwork
import seaborn as sns
import matplotlib.pyplot as plt

config_dict = {  # default if no hyperparams set for sweep.
    "env": {
        "num_dcs": 5,
        "num_customers": 20,
        "num_commodities": 3,
        "orders_per_day": 1,  # start with one, and then play with this.
        "dcs_per_customer": 2,
        "demand_mean": 600,
        "demand_var": 1,
        # "num_steps": 30,  # steps per episode
        "big_m_factor": 10000,  # how many times the customer cost is the big m.
        # New parameters 2021
        "version": "v2",
        "order_generator": "biased",
        "reward_function": "negative_log_cost_minus_log_big_m_units",
        # "reward_function": "negative_log_cost_minus_log_big_m_units_exp",
    },
    "hps": {
        "replay_size": 150,  # Size of the buffer.
        "warm_start_steps": 150,
        # Number of steps to run in warmup to populate the buffer # OLD COMMENT: apparently has to be smaller than batch size
        "max_episodes": 350,
        "episode_length": 30,
        "batch_size": 30,
        "gamma": 0.8,
        "hidden_size": 12,
        # "lr": 8e-3,
        "lr": 1e-4,
        "eps_end": 0.01,
        "eps_start": 0.01,
        "sync_rate": 30,  # Rate to sync the target and learning network.
        "lambda_lr_sched_discount": 0.999992,
    },
    "seed": 655,
    # "agent": "nn_warehouse_mask",
    # "agent": "nn_mask_plus_customer_onehot",
    # "agent": "nn_full_mlp",
    # "agent": "nn_full_mlp",  # Actually shooting myself in the foot
    # "agent": "random_valid",
    # "agent": "best_fit",
    # "agent": "lookahead",
    # "agent": "nn_debug_mlp_cheat",
    "agent": "lookahead",
    # "agent": "nn_mask_plus_consumption",
}


ORDERS_TO_GENERATE = st.sidebar.number_input(
    "orders_to_generate", min_value=1, max_value=10000, step=1
)
PZ_NUMERATOR = st.sidebar.number_input(
    "pz_numerator", min_value=1.0, max_value=10000.0, step=0.1
)
CUSTOMER_MEAN_PARAM = st.sidebar.number_input(
    "customer_mean_param",
    min_value=0.000001,
    max_value=1000000.0,
    step=1.0,
    value=600.0,
)
CUSTOMER_VAR_PARAM = st.sidebar.number_input(
    "customer_var_param", min_value=0.000001, max_value=1000000.0, step=1.0, value=1.0
)

pn = PhysicalNetwork(
    num_dcs=5,
    num_customers=20,
    dcs_per_customer=2,
    demand_mean=CUSTOMER_MEAN_PARAM,
    demand_var=CUSTOMER_VAR_PARAM,
    big_m_factor=10000,
    num_commodities=1,
    planning_horizon=5,
)
# order_gen = order_generators.BiasedOrderGenerator(
#     pn, orders_per_day=1, pz_numerator=PZ_NUMERATOR
# )

order_gen = order_generators.NormalOrderGenerator(
    pn,
    orders_per_day=1,
)
st.write(str(order_gen.generate_orders(0)))
orders = []
for i in range(ORDERS_TO_GENERATE):
    orders = orders + order_gen.generate_orders(i)
demands_np = np.stack([o.demand for o in orders])
demands_df = pd.DataFrame(demands_np, columns=["demand"])
st.write(str(orders[:5]))

"# Generator parameters"
"Customer means"
st.write(order_gen.commodity_means)
"Customer covariances"
st.write(np.stack(order_gen.customer_covariances))

"# Sample of demand"
st.write(demands_df.head(15))

"# Description of demand"

st.write(demands_df.demand.describe())
fig = plt.figure()
sns.distplot(demands_df.demand, kde=False)
st.pyplot(fig)
