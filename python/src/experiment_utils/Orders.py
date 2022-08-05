from typing import List


from experiment_utils.Order import Order
import numpy as np

from network import physical_network


class Orders:
    """
    Splits the totalCapacity into a random number of locations

    :param int num_orders:
    :param Network network:
    """

    # def __init__(self, totalCapacity, num_orders, network):
    def __init__(self, num_orders, network: physical_network):
        self.totalTime = 0
        # self.totalCapacity = totalCapacity
        self.num_orders = num_orders
        self.network = network
        self.orders = []

        self.generate_orders()

    def generate_orders(self):
        # Choose orders to be generated this timestep.

        demand_var = self.network.demand_var
        # Generate demand
        demand = np.random.multivariate_normal(
            [200, 100, 50], np.eye(3) * demand_var, size=(10, 5)
        )  # (num_customers,commodities,orders)


# Vector with total demand per commodity for each DC in horizon for fixed orders
def summarize_order_demand(
    orders: List[Order], start: int, end: int, commodity_shape
) -> np.array:
    if len(orders) != 0:
        result = np.sum(
            [o.demand for o in orders if start <= o.due_timestep <= end], axis=0
        )
        return result.reshape(commodity_shape)
    else:
        return np.zeros(commodity_shape)


def summarize_order_demand_per_dc(orders, start, end, num_dcs, num_commodities):
    """
    Summarizes the demand in the list of orders to a (dc,commodity) matrix of
    how much each DC will have to satisfy of each commodity within the horizon
    (start<=due<=end)
    Args:
        orders:
        start:
        end:
        num_dcs:
        num_commodities:

    Returns: a (W,K) matrix of the demands fixed to each warehouse.

    """
    if len(orders) != 0:
        demand_per_warehouse = np.zeros((num_dcs, num_commodities))
        demands_in_horizon_per_w = [
            (o.demand, o.shipping_point.node_id)
            for o in orders
            if start <= o.due_timestep <= end
        ]
        for demand, warehouse in demands_in_horizon_per_w:
            demand_per_warehouse[warehouse, :] = (
                demand_per_warehouse[warehouse, :] + demand
            )
        return demand_per_warehouse
    else:
        return np.zeros((num_dcs, num_commodities))


def summarize_demand_per_customer_in_horizon(
    orders: List[Order],
    start,
    end,
    num_customers,
    num_commodities,
    physical_network: "PhysicalNetwork",
):
    """
    Summarizes the orders in horizon into the demand per customer-commodity. The indexing is customer_id (zero-based).
    This is because a customer theoretically could have more than one order per horizon.
    Args:
        orders:

    Returns: (C,K) vector of the summary of balances (negative) of demand

    """
    demands = np.zeros((num_customers, num_commodities))
    for o in orders:
        if start <= o.due_timestep <= end:
            customer_node_id = o.customer.node_id
            customer_id = physical_network.get_customer_id(customer_node_id)
            demands[customer_id, :] += o.demand
    # Customer balances are negative demand.
    balances = demands * -1

    return balances
