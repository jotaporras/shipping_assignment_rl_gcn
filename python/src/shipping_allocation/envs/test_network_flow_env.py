from envs.network_flow_env import (
    EnvironmentParameters,
    ShippingFacilityEnvironment,
    RandomAgent,
)
from envs.order_generators import ActualOrderGenerator
from shipping_allocation import NaiveInventoryGenerator

from network import physical_network


def test_one_timestep():
    num_dcs = 2
    num_customers = 1
    num_commodities = 3
    orders_per_day = 1
    dcs_per_customer = 1
    demand_mean = 100
    demand_var = 20

    num_episodes = 1

    physical_network = physical_network(
        num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        num_commodities,
    )
    # order_generator = NaiveOrderGenerator(num_dcs, num_customers, orders_per_day)
    order_generator = ActualOrderGenerator(physical_network, orders_per_day)
    generator = NaiveInventoryGenerator()
    environment_parameters = EnvironmentParameters(
        physical_network, order_generator, generator, num_episodes
    )

    env = ShippingFacilityEnvironment(environment_parameters)
    agent = RandomAgent(env.action_space)

    obs = env.reset()
    reward = 0
    done = False
    print("=========== starting episode loop ===========")
    print("Initial environment: ")
    env.render()
    while not done:
        action = agent.get_action(obs, reward, done)
        # print(f"Agent is taking action: {action}")
        # the agent observes the first state and chooses an action
        # environment steps with the agent's action and returns new state and reward
        obs, reward, done, info = env.step(action)

        # print(f"Got reward {reward} done {done}")

        # Render the current state of the environment
        env.render()

        if done:
            print("===========Environment says we are DONE ===========")


def run_with_params(
    num_dcs,
    num_customers,
    dcs_per_customer,
    demand_mean,
    demand_var,
    num_commodities,
    orders_per_day,
    num_episodes,
):
    physical_network = physical_network(
        num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        num_commodities,
    )
    # order_generator = NaiveOrderGenerator(num_dcs, num_customers, orders_per_day)
    order_generator = ActualOrderGenerator(physical_network, orders_per_day)
    generator = NaiveInventoryGenerator()
    environment_parameters = EnvironmentParameters(
        physical_network, order_generator, generator, num_episodes
    )

    env = ShippingFacilityEnvironment(environment_parameters)
    agent = RandomAgent(env.action_space)

    obs = env.reset()
    reward = 0
    done = False
    print("=========== starting episode loop ===========")
    print("Initial environment: ")
    env.render()
    while not done:
        action = agent.get_action(obs, reward, done)
        # print(f"Agent is taking action: {action}")
        # the agent observes the first state and chooses an action
        # environment steps with the agent's action and returns new state and reward
        obs, reward, done, info = env.step(action)
        # print(f"Got reward {reward} done {done}")

        # Render the current state of the environment
        env.render()

        if done:
            print("===========Environment says we are DONE ===========")


def test_sliding_ten():
    num_dcs = 2
    num_customers = 2
    num_commodities = 2
    orders_per_day = 2
    dcs_per_customer = 1
    demand_mean = 100
    demand_var = 20

    num_episodes = 10

    run_with_params(
        num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        num_commodities,
        orders_per_day,
        num_episodes,
    )


def test_sliding_ten_2():
    num_dcs = 4
    num_customers = 3
    num_commodities = 2
    orders_per_day = 2
    dcs_per_customer = 4
    demand_mean = 100
    demand_var = 20

    num_episodes = 10

    run_with_params(
        num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        num_commodities,
        orders_per_day,
        num_episodes,
    )


def test_sliding_ten_3():
    num_dcs = 10
    num_customers = 20
    num_commodities = 2
    orders_per_day = 2
    dcs_per_customer = 4
    demand_mean = 100
    demand_var = 20

    num_episodes = 100

    run_with_params(
        num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        num_commodities,
        orders_per_day,
        num_episodes,
    )


import time

if __name__ == "__main__":
    total_start = time.process_time()
    for i in range(100):
        test_sliding_ten()
    for i in range(100):
        test_sliding_ten_2()
    for i in range(100):
        test_sliding_ten_3()
    total_end = time.process_time()
    print("Elapsed on all runs: ", total_end - total_start, "s")

    # test_one_timestep()
