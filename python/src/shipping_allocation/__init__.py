"""
This package contains the code for the Shipping Point Assignment (SPA) simulator/environment, using the Open AI interface.

To use this environment you need to install it as a package using pip install -e .
Otherwise some imports will not recognize "envs" as a package.
"""
from shipping_allocation.envs.network_flow_env import (
    EnvironmentParameters,
)
from envs.order_generators import NaiveOrderGenerator
from envs.inventory_generators import NaiveInventoryGenerator
from gym.envs.registration import register

from network.physical_network import PhysicalNetwork