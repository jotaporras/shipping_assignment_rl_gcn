"""
IIRC, this experiment_runner class is only used in the first 'experiments' package.
It was a utility that set up the environment, agent and manually did the runs.

This was previous to using PTL for running the experiments, so there's a chance
this will be useless. However, if I decide that PTL is not worth it and that
having the replay buffer etc is too much hassle, I might resort to this runner again,
because it's pure and written by me, so it's easier to debug.

SEP 18: This class probably has to go, it doesn't even compile.
"""

from shipping_allocation.envs.network_flow_env import (
    EnvironmentParameters,
    ShippingFacilityEnvironment,
)
from envs.order_generators import ActualOrderGenerator
from envs.inventory_generators import DirichletInventoryGenerator

from agents.concrete_agents import (
    RandomAgent,
    Agent,
    AlwaysZeroAgent,
    BestFitAgent,
    RandomValid,
    DoNothingAgent,
    AgentHighest,
)
from experiment_utils import report_generator
from network import physical_network
import numpy as np

DEBUG = True


class ExperimentRunner:
    environment_parameters: EnvironmentParameters
    agent: Agent

    def __init__(
        self,
        order_generator,
        inventory_generator,
        agent,
        env: ShippingFacilityEnvironment,
        experiment_name="",
    ):
        self.order_generator = order_generator
        self.inventory_generator = inventory_generator
        self.agent = agent
        self.environment_parameters = env.environment_parameters
        self.physical_network = self.environment_parameters.network
        self.env = env
        self.experiment_name = experiment_name

    def run_episode(self, ep):
        state = self.env.reset()
        reward = 0
        done = False
        print("=========== starting episode loop ===========")
        print("Initial environment: ")
        self.env.render()
        actions = []
        episode_rewards = []
        info = {}
        # demands_per_k = np.zeros((num_commodities,num_steps))
        # inventory_at_t = np.zeros((num_commodities,num_steps)) #todo llenar estos eventualmente
        while not done:
            # action = self.agent.train((obs,action,reward,obs, done))
            action = self.agent.get_action(state)

            # print(f"Agent is taking action: {action}")
            # the agent observes the first state and chooses an action
            # environment steps with the agent's action and returns new state and reward
            # obs, reward, done, info = self.env.step(action)#old
            next_state, reward, done, info = self.env.step(action)
            # print(f"Got reward {reward} done {done}")
            self.agent.train((state, action, next_state, reward, done))

            state = next_state

            # Render the current state of the environment
            self.env.render()
            actions.append(action)
            episode_rewards.append(reward)

            if done:
                print("===========Environment says we are DONE ===========")
                if self.experiment_name != "":
                    print("Writing costs to CSV")
                    report_generator.write_experiment_reports(
                        info, self.experiment_name + f"/ep_{ep}"
                    )  # todo consider writing only once instead of each ep.

        if DEBUG:
            print("Episode done, rewards per step: ", episode_rewards)
            print(
                "Episode done, average reward per step: ",
                sum(episode_rewards) / self.environment_parameters.num_steps,
            )
            print(
                "Episode done, average reward per order: ",
                sum(episode_rewards) / len(state["fixed"]),
            )

        return actions, episode_rewards, info

    def run_episodes(self, num_steps, num_episodes, orders_per_day, experiment_name):
        self.experiment_name = experiment_name  # hotfix
        total_rewards = []
        average_rewards = []

        total_actions = np.zeros(num_steps * orders_per_day)
        elapsed = []
        self.display_environment()
        for i in range(num_episodes):
            print("\n\nRunning episode: ", i)
            start_time = time.process_time()
            actions, episode_rewards, info = self.run_episode(i)
            end_time = time.process_time()

            total_rewards.append(sum(episode_rewards))
            average_rewards.append(np.mean(episode_rewards))
            elapsed.append(end_time - start_time)
            total_actions += np.array(actions)

        # Create datasets
        rewards_df = pd.DataFrame(
            data={
                "experiment_name": [experiment_name] * num_episodes,
                "episode": list(range(num_episodes)),
                "total_reward": total_rewards,
                "average_reward": average_rewards,
                "elapsed": elapsed,
            }
        )
        actions_df = pd.DataFrame(total_actions)

        base = f"data/results/{experiment_name}"
        if not os.path.exists("data"):
            os.mkdir("data")
        if not os.path.exists("data/results"):
            os.mkdir("data/results")
        if not os.path.exists(base):
            os.mkdir(base)
        rewards_df.to_csv(base + "/rewards.csv")
        actions_df.to_csv(base + "/actions.csv")
        print("done")
        if DEBUG:
            print("Experiment done, total rewards: ", total_rewards)
            print("Sum total rewards: ", sum(total_rewards))
            print(
                "Total fixed orders",
            )
            print("Elapsed", elapsed)
            print("Total elapsed", sum(elapsed))

    def display_environment(self):
        # Print things about env, parameters and network that might be useful for reference.
        physical_network = self.env.environment_parameters.network
        print("\n\n")
        print("================================================================")
        print("================================================================")
        print("================================================================")
        print("===== INITIALIZING RUN WITH CURRENT ENVIRONMENT PARAMS ======")
        print("===== DCS Per Customer Array ======")
        print(physical_network.dcs_per_customer_array)


def create_random_experiment_runner(
    num_dcs,
    num_customers,
    dcs_per_customer,
    demand_mean,
    demand_var,
    num_commodities,
    orders_per_day,
    num_steps,
):
    physical_network = physical_network(
        num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        num_commodities,
    )
    order_generator = ActualOrderGenerator(physical_network, orders_per_day)
    generator = DirichletInventoryGenerator(physical_network)

    environment_parameters = EnvironmentParameters(
        physical_network, order_generator, generator, num_steps
    )

    env = ShippingFacilityEnvironment(environment_parameters)
    agent = RandomAgent(env)

    return ExperimentRunner(order_generator, generator, agent, env)


def create_alwayszero_experiment_runner(
    num_dcs,
    num_customers,
    dcs_per_customer,
    demand_mean,
    demand_var,
    num_commodities,
    orders_per_day,
    num_steps,
):
    physical_network = physical_network(
        num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        num_commodities,
    )
    order_generator = ActualOrderGenerator(physical_network, orders_per_day)
    generator = DirichletInventoryGenerator(physical_network)

    environment_parameters = EnvironmentParameters(
        physical_network, order_generator, generator, num_steps
    )

    env = ShippingFacilityEnvironment(environment_parameters)
    agent = AlwaysZeroAgent(env)

    return ExperimentRunner(order_generator, generator, agent, env)


class AlwaysFirstAgent(object):
    """The world's DUMBEST agent!"""

    def act(self, observation, reward, done):
        return 0


# def create_always_first_dc_agent(num_dcs,
#         num_customers,
#         dcs_per_customer,
#         demand_mean,
#         demand_var,
#         num_commodities,
#         orders_per_day
#     ):
#     physical_network = PhysicalNetwork(
#         num_dcs,
#         num_customers,
#         dcs_per_customer,
#         demand_mean,
#         demand_var,
#         num_commodities,
#     )
#     order_generator = ActualOrderGenerator(physical_network, orders_per_day)
#     generator = NaiveInventoryGenerator()
#     agent = AlwaysFirstAgent()
#     return ExperimentRunner(order_generator,generator,agent,physical_network)


def create_dqn_experiment_runner(
    num_dcs,
    num_customers,
    dcs_per_customer,
    demand_mean,
    demand_var,
    num_commodities,
    orders_per_day,
    num_steps,
):
    physical_network = physical_network(
        num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        num_commodities,
    )
    order_generator = ActualOrderGenerator(physical_network, orders_per_day)
    generator = DirichletInventoryGenerator(physical_network)

    environment_parameters = EnvironmentParameters(
        physical_network, order_generator, generator, num_steps
    )

    env = ShippingFacilityEnvironment(environment_parameters)
    agent = QNAgent(env)

    return ExperimentRunner(order_generator, generator, agent, env)


def create_bestfit_experiment_runner(
    num_dcs,
    num_customers,
    dcs_per_customer,
    demand_mean,
    demand_var,
    num_commodities,
    orders_per_day,
    num_steps,
):
    physical_network = physical_network(
        num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        num_commodities,
    )
    order_generator = ActualOrderGenerator(physical_network, orders_per_day)
    generator = DirichletInventoryGenerator(physical_network)

    environment_parameters = EnvironmentParameters(
        physical_network, order_generator, generator, num_steps
    )

    env = ShippingFacilityEnvironment(environment_parameters)
    agent = BestFitAgent(env)

    return ExperimentRunner(
        order_generator, generator, agent, env, experiment_name="bestfit_validation"
    )


def create_randomvalid_experiment_runner(
    num_dcs,
    num_customers,
    dcs_per_customer,
    demand_mean,
    demand_var,
    num_commodities,
    orders_per_day,
    num_steps,
):
    physical_network = physical_network(
        num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        num_commodities,
    )
    order_generator = ActualOrderGenerator(physical_network, orders_per_day)
    generator = DirichletInventoryGenerator(physical_network)

    environment_parameters = EnvironmentParameters(
        physical_network, order_generator, generator, num_steps
    )

    env = ShippingFacilityEnvironment(environment_parameters)
    agent = RandomValid(env)

    return ExperimentRunner(
        order_generator, generator, agent, env, experiment_name="randomvalid_validation"
    )


def create_donothing_experiment_runner(
    num_dcs,
    num_customers,
    dcs_per_customer,
    demand_mean,
    demand_var,
    num_commodities,
    orders_per_day,
    num_steps,
):
    physical_network = physical_network(
        num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        num_commodities,
    )
    order_generator = ActualOrderGenerator(physical_network, orders_per_day)
    generator = DirichletInventoryGenerator(physical_network)

    environment_parameters = EnvironmentParameters(
        physical_network, order_generator, generator, num_steps
    )

    env = ShippingFacilityEnvironment(environment_parameters)
    agent = DoNothingAgent(env)

    return ExperimentRunner(
        order_generator, generator, agent, env, experiment_name="randomvalid_validation"
    )


def create_agent_66_experiment_runner(
    num_dcs,
    num_customers,
    dcs_per_customer,
    demand_mean,
    demand_var,
    num_commodities,
    orders_per_day,
    num_steps,
):
    physical_network = physical_network(
        num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        num_commodities,
    )
    order_generator = ActualOrderGenerator(physical_network, orders_per_day)
    generator = DirichletInventoryGenerator(physical_network)

    environment_parameters = EnvironmentParameters(
        physical_network, order_generator, generator, num_steps
    )

    env = ShippingFacilityEnvironment(environment_parameters)
    agent = AgentHighest(env)

    return ExperimentRunner(
        order_generator, generator, agent, env, experiment_name="randomvalid_validation"
    )


def run_with_params(
    num_dcs,
    num_customers,
    dcs_per_customer,
    demand_mean,
    demand_var,
    num_commodities,
    orders_per_day,
    num_steps,
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
    generator = DirichletInventoryGenerator(physical_network)
    environment_parameters = EnvironmentParameters(
        physical_network, order_generator, generator, num_steps
    )

    env = ShippingFacilityEnvironment(environment_parameters)
    agent = RandomAgent(env)

    obs = env.reset()
    reward = 0
    done = False
    print("=========== starting episode loop ===========")
    print("Initial environment: ")
    env.render()
    actions = []
    episode_rewards = []
    # demands_per_k = np.zeros((num_commodities,num_steps))
    # inventory_at_t = np.zeros((num_commodities,num_steps)) #todo llenar estos eventualmente
    while not done:
        action = agent.act(obs, reward, done)

        # print(f"Agent is taking action: {action}")
        # the agent observes the first state and chooses an action
        # environment steps with the agent's action and returns new state and reward
        obs, reward, done, info = env.step(action)
        # print(f"Got reward {reward} done {done}")

        # Render the current state of the environment
        env.render()
        actions.append(action)
        episode_rewards.append(reward)

        if done:
            print("===========Environment says we are DONE ===========")

    return actions, episode_rewards


import os
import pandas as pd
import time


def run_episodes(
    num_dcs,
    num_customers,
    dcs_per_customer,
    demand_mean,
    demand_var,
    num_commodities,
    orders_per_day,
    num_steps,
    num_episodes,
    experiment_name,
):
    total_rewards = []
    average_rewards = []
    total_actions = np.zeros(num_steps * orders_per_day)
    elapsed = []
    for i in range(num_episodes):
        start_time = time.process_time()
        actions, episode_rewards = run_with_params(
            num_dcs,
            num_customers,
            dcs_per_customer,
            demand_mean,
            demand_var,
            num_commodities,
            orders_per_day,
            num_steps,
        )
        end_time = time.process_time()

        total_rewards.append(sum(episode_rewards))
        average_rewards.append(np.mean(episode_rewards))
        elapsed.append(end_time - start_time)
        total_actions += np.array(actions)

    # Create datasets
    rewards_df = pd.DataFrame(
        data={
            "experiment_name": [experiment_name] * num_episodes,
            "episode": list(range(num_episodes)),
            "total_reward": total_rewards,
            "average_reward": average_rewards,
            "elapsed": elapsed,
        }
    )
    actions_df = pd.DataFrame(total_actions)

    base = f"data/results/{experiment_name}"
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/results"):
        os.mkdir("data/results")
    if not os.path.exists(base):
        os.mkdir(base)
    rewards_df.to_csv(base + "/rewards.csv")
    actions_df.to_csv(base + "/actions.csv")
    print("done")
