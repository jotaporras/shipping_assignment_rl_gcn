from envs import ShippingFacilityEnvironment, rewards
from envs.network_flow_env import (
    EnvironmentParameters,
)
from envs.order_generators import (
    ActualOrderGenerator,
    BiasedOrderGenerator,
    NormalOrderGenerator,
)
from envs.inventory_generators import DirichletInventoryGenerator, MinIDGenerator
from envs.shipping_assignment_env import ShippingAssignmentEnvironment

from network.physical_network import PhysicalNetwork


def build_next_gen_network_flow_environment(
    environment_config, episode_length, order_gen: str, reward_function_name: str
):
    """
    Second generation of shipping point assignment environment.
    Args:
        environment_config: A dictionary with the config to build the environment, see any PTL run for an example.
        episode_length:
        order_gen:
        reward_function:

    Returns:

    """
    physical_network = PhysicalNetwork(
        num_dcs=environment_config["num_dcs"],
        num_customers=environment_config["num_customers"],
        dcs_per_customer=environment_config["dcs_per_customer"],
        demand_mean=environment_config["demand_mean"],
        demand_var=environment_config["demand_var"],
        big_m_factor=environment_config["big_m_factor"],
        num_commodities=environment_config["num_commodities"],
    )

    order_generator = order_generator_chooser(
        physical_network, order_gen, environment_config["orders_per_day"]
    )
    # TODo temporarily replace dirichlet with a gen that allocates inventory differently.
    if environment_config["inventory_generator"] == "dirichlet":
        inventory_generator = DirichletInventoryGenerator(physical_network)
    elif environment_config["inventory_generator"] == "min_id":
        inventory_generator = MinIDGenerator(physical_network)
    else:
        raise RuntimeError("Illegal inveotory gen detected")

    reward_function = rewards.reward_chooser(reward_function_name)

    return ShippingAssignmentEnvironment(
        physical_network,
        order_generator,
        inventory_generator,
        reward_function,
        num_steps=episode_length,
    )


def build_network_flow_env_parameters(  # TODO receive individuals instead of all dict?
    environment_config, episode_length, order_gen: str
):
    """
    Deprecated
    Old way of creating environment (first step, parameters).
    """
    physical_network = PhysicalNetwork(
        num_dcs=environment_config["num_dcs"],
        num_customers=environment_config["num_customers"],
        dcs_per_customer=environment_config["dcs_per_customer"],
        demand_mean=environment_config["demand_mean"],
        demand_var=environment_config["demand_var"],
        big_m_factor=environment_config["big_m_factor"],
        num_commodities=environment_config["num_commodities"],
    )

    order_generator = order_generator_chooser(
        physical_network, order_gen, environment_config["orders_per_day"]
    )

    generator = DirichletInventoryGenerator(physical_network)

    environment_parameters = EnvironmentParameters(
        physical_network, order_generator, generator, episode_length
    )

    return environment_parameters


def order_generator_chooser(physical_network, order_gen, orders_per_day):
    """
    Picks an order generator based on configuraiton
    Args:
        orders_per_day:
        order_gen:
        physical_network:

    Returns:

    """
    if order_gen == "original":  #
        order_generator = ActualOrderGenerator(physical_network, orders_per_day)
    elif order_gen == "biased":
        order_generator = BiasedOrderGenerator(physical_network, orders_per_day)
    elif order_gen == "normal_multivariate":
        order_generator = NormalOrderGenerator(physical_network, orders_per_day)
    else:
        raise NotImplementedError("alternatives are original and biased")
    return order_generator
