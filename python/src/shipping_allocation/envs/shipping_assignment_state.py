from collections import namedtuple

from torch_geometric.data import Data

from experiment_utils import Orders
import numpy as np
import torch

# Check the environment for state_vector impl. As of Sep 18, it's
# A concatenation of inventories, current order demand an some metadata neurons
# That indicate which customer is allocating the order.
ShippingAssignmentState = namedtuple(
    "ShippingAssignmentState",
    [
        "current_t",
        "physical_network",
        "fixed",
        "open",
        "inventory",
        "state_vector",
        "big_m_counter_per_commodity",
        "optimization_cost",
        "big_m_units_per_commodity",
    ],
)


def state_to_fixed_demand(state):
    """Converts a state to the currently relevant fixed orders in horizon (because raw fixed is historical)"""
    planning_horizon = state.physical_network.planning_horizon
    current_t = state.current_t
    end_t = current_t + planning_horizon - 1
    fixed_demand = Orders.summarize_order_demand(
        state.fixed, current_t, end_t, state.physical_network.num_commodities
    )
    return fixed_demand


def state_to_demand_per_warehouse_commodity(state):
    """TODO if works test if generalize to many commodites.
    Converts the demand of fixed orders in horizon into a vector of how much demand there is on each warehouse.
    """
    planning_horizon = state.physical_network.planning_horizon
    current_t = state.current_t
    end_t = current_t + planning_horizon - 1
    # (num_dcs,num_warehouses)
    demand_per_dc = Orders.summarize_order_demand_per_dc(
        state.fixed,
        current_t,
        end_t,
        state.physical_network.num_dcs,
        state.physical_network.num_commodities,
    )
    return demand_per_dc.reshape(
        1, -1
    ).flatten()  # shape (num_warehouses*num_commodities)


# Graph based networks TODO decide if this should go somewhere else
def state_to_agg_balance_in_horizon_gnn(
    state: ShippingAssignmentState,
) -> "torch_geometric.data.Data":
    """
    Converts an environment state to node features and adjacency list
    by using inventory and demand vectors as features. Adjacencies are valid DC->Customer.
    Features are the agg demand for that node in horizon or inventory if warehouse. No arc features yet.
    Returns:
        torch_geometric.data with the node features and edge_indices
    """
    inventory = state.inventory
    pn = state.physical_network
    latest_open_order = state.open[0]
    latest_order_demand = latest_open_order.demand

    planning_horizon = state.physical_network.planning_horizon
    current_t = state.current_t
    end_t = current_t + planning_horizon - 1

    all_orders = state.fixed + state.open

    # Create a node feature vector for the customers: 0 for all customers except for the latest open order.
    demand_summary_per_customer = Orders.summarize_demand_per_customer_in_horizon(
        all_orders,
        start=current_t,
        end=end_t,
        num_customers=state.physical_network.num_customers,
        num_commodities=state.physical_network.num_commodities,
        physical_network=state.physical_network,
    )
    node_features = np.vstack((inventory, demand_summary_per_customer))

    edge_index = pn.physical_adjacency_matrix  # TODo hope this is correct.

    graph_data = Data(
        x=torch.tensor(node_features), edge_index=torch.tensor(edge_index)
    )

    return graph_data


def state_to_node_type_graphmarkers(state: ShippingAssignmentState):
    """Given a state, get the node types (whether warehouse or customer, and whether next order).
    Shape should match the node vec tensor (nodes,features) so it should be (nodes,2)"""
    latest_order = state.open[0]
    num_dcs = state.physical_network.num_dcs
    num_nodes = state.physical_network.num_customers + num_dcs
    # Two features per node: (is customer and is_latest_order.
    node_type = np.zeros((num_nodes, 2))
    # Setting is_latest_order to True on the next order.
    node_type[latest_order.customer.node_id, 1] = 1.0
    # Setting DCS to one.
    node_type[num_dcs:-1, 0] = 1.0  # Set all from start of customers until last
    return node_type
