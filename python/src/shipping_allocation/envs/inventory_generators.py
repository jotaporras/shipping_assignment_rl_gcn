from abc import ABC
from typing import List

import numpy as np
from network import physical_network

from experiment_utils.Order import Order


class InventoryGenerator(ABC):
    # Generates new inventory and distributes it somehow to keep the network balanced for the selected locations.
    # Returns a numpy array of shape (num_dcs,num_commodities) representing how much extra inventory is going to appear.
    def generate_new_inventory(
        self, network: physical_network, open_orders: List[Order]
    ):  # todo add type when it works.
        pass


class NaiveInventoryGenerator(InventoryGenerator):
    def generate_new_inventory(
        self, network: physical_network, open_orders: List[Order]
    ):
        # logging.info("==> inventory generator")
        total_inventory = sum(
            map(lambda o: o.demand, open_orders)
        )  # TODO rename and do for many commmodities.
        even = total_inventory // network.num_dcs
        dc_inv = np.array([even] * network.num_dcs).reshape(
            network.num_dcs, -1
        )  # To keep the (dc,product) shape. #todo validate with multiple commodities
        # logging.info("Demand", total_inventory)
        # logging.info("Pre level dc_inv")
        # logging.info(dc_inv)
        # logging.info("Total new inv",np.sum(dc_inv))
        imbalance = total_inventory - np.sum(dc_inv, axis=0)
        # if total_inventory // network.num_dcs != total_inventory / network.num_dcs:
        dc_inv[0, :] = dc_inv[0, :] + imbalance
        # logging.info("Rebalanced dc inv",dc_inv)
        # logging.info("Rebalanced sum",np.sum(dc_inv))
        if (np.sum(dc_inv, axis=0) != total_inventory).any():
            raise Exception("np.sum(dc_inv) != total_inventory")
        return dc_inv


class DirichletInventoryGenerator(InventoryGenerator):
    def __init__(self, network: physical_network):
        num_dcs = network.num_dcs
        num_commodities = network.num_commodities
        self.alpha = np.random.permutation(
            num_dcs / np.arange(1, num_dcs + 1)
        )  # trying to make it skewed.
        self.inventory_generation_distribution = np.random.dirichlet(
            self.alpha, num_commodities
        )  # (num_dc,num_k) of dc distribution of inventory.

    def generate_new_inventory(
        self, network: physical_network, open_orders: List[Order]
    ):
        # logging.info("==> inventory generator")
        total_inventory = sum(
            map(lambda o: o.demand, open_orders)
        )  # TODO rename and do for many commmodities.
        # even = total_inventory // network.num_dcs
        inventory_distribution = self.inventory_generation_distribution

        supply_per_dc = np.floor(
            total_inventory.reshape(-1, 1) * inventory_distribution
        )
        imbalance = total_inventory - np.sum(supply_per_dc, axis=1)
        supply_per_dc[:, 0] = supply_per_dc[:, 0] + imbalance

        # logging.info("Demand", total_inventory)
        # logging.info("Pre level dc_inv")
        # logging.info(dc_inv)
        # logging.info("Total new inv",np.sum(dc_inv))
        # if total_inventory // network.num_dcs != total_inventory / network.num_dcs:
        # logging.info("Rebalanced dc inv",dc_inv)
        # logging.info("Rebalanced sum",np.sum(dc_inv))
        if not np.isclose(np.sum(np.sum(supply_per_dc, axis=1) - total_inventory), 0.0):
            raise RuntimeError("Demand was not correctly balanced")
        return supply_per_dc.transpose()


class OracleGenerator(InventoryGenerator):
    """
    Cheats by seeing where the open orders are going to be.
    Used as a lower bound metric for when agents.
    #TODO: this might disrupt the RNG sequence when comparing runs, that's why it calls dirichlet dirichlet.
    """

    def __init__(self, network: physical_network):
        self.num_dcs = network.num_dcs
        self.num_commodities = network.num_commodities
        self.dirichlet_shim = DirichletInventoryGenerator(network)

    def generate_new_inventory(
        self, network: physical_network, open_orders: List[Order]
    ):
        # Call dirichlet to simulate that inventory was really generated
        shim_inv = self.dirichlet_shim.generate_new_inventory(network, open_orders)
        inventory = np.zeros((self.num_dcs, self.num_commodities))
        for order in open_orders:
            shipping_point_id = (
                order.shipping_point.node_id
            )  # Ship pt is equal to node id.
            # Add inventory exactly where it is required.
            inventory[shipping_point_id, :] = (
                inventory[shipping_point_id] + order.demand.transpose()
            )
        return inventory


class MinIDGenerator(InventoryGenerator):
    """
    Puts the inventory on the first warehouse of the valid ones.
    #TODO: this might disrupt the RNG sequence when comparing runs, that's why it calls dirichlet dirichlet.
    """

    def __init__(self, network: physical_network):
        self.num_dcs = network.num_dcs
        self.num_commodities = network.num_commodities
        self.dirichlet_shim = DirichletInventoryGenerator(network)

    def generate_new_inventory(
        self, network: "PhysicalNetwork", open_orders: List[Order]
    ):
        # Call dirichlet to simulate that inventory was really generated
        shim_inv = self.dirichlet_shim.generate_new_inventory(network, open_orders)

        inventory = np.zeros((self.num_dcs, self.num_commodities))
        for order in open_orders:
            customer_node_id = order.customer.node_id
            customer_id = network.get_customer_id(customer_node_id)
            first_warehouse_index = np.where(
                network.dcs_per_customer_array[customer_id, :] == 1
            )[0][0]
            # Add inventory exactly where it is required.
            inventory[first_warehouse_index, :] = (
                inventory[first_warehouse_index] + order.demand.transpose()
            )
        return inventory
