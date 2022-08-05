"""
2022-07-02 This module is very important, it consumes the information from the environment state and runs the network flows optimization for each of the K commodities
"""
import time
import numpy as np

from network.ExtendedNetwork import ExtendedNetwork
from ortools.graph import pywrapgraph

from network import physical_network
import logging

DEBUG = False
logger = logging.getLogger(__name__)


def optimize(physical_network, inventory, fixed, open, current_t):
    """
    Creates and optimizes an optimization problem based on the state variables.
    Args:
        See the network_flow_env module for details on these variables.

    Returns:
        total_cost: sum of the costs of the optimal solutions by all commodities
        transport_matrix: Total inventory movements for each warehouse and commodity, shaped like inventory.
        all_movements: List of arc flow objects. each element of the list is the arcs of commodity k
        big_m_counter: Number of Big M assignments done in this iteration.

    """
    planning_horizon_t = current_t + physical_network.planning_horizon - 1
    # Treat all orders as fixed.
    # extended_network = ExtendedNetwork(network, inventory, fixed_orders=open + fixed, open_orders=[])
    # only fixed TODO is this ok??? Could be. Consider the before and after stage impact.
    extended_network = ExtendedNetwork(
        physical_network, inventory, fixed_orders=fixed, open_orders=[]
    )
    extended_nodes, extended_arcs = extended_network.convert_to_extended(
        current_t, planning_horizon_t
    )

    inv_shape = inventory.shape
    transport_matrix = np.zeros(inv_shape)

    # Generate ortools.
    total_cost = 0.0
    total_big_m_count = 0
    all_movements = []
    big_m_counter = 0
    big_m_per_commodity = []
    big_m_units_per_commodity = []
    for k in range(physical_network.num_commodities):  # TODO one indexed commodities?
        (
            k_cost,
            tm,
            all_k_movements,
            big_m_counter_k,
            big_m_units_k,
        ) = _optimize_commodity(
            physical_network,
            k,
            extended_nodes,
            extended_arcs,
            current_t,
            inv_shape,
        )
        all_movements.append(all_k_movements)
        big_m_per_commodity.append(big_m_counter_k)
        big_m_units_per_commodity.append(big_m_units_k)
        total_cost += k_cost
        transport_matrix += tm

    if DEBUG:
        logger.info(f"Total optimization cost: {total_cost}")
        logger.info("Total transportation movements: ")
        logger.info(transport_matrix)

    # flatten all movements to one commodity
    all_movements = [item for sublist in all_movements for item in sublist]

    # Check if all movements contain dummy flows #TODO, shouldn't the dummy be a node instead of string??
    from_dummy = [m for m in all_movements if m[0].tail.location == "DUMMY"]
    to_dummy = [m for m in all_movements if m[0].head.location == "DUMMY"]
    if len(to_dummy) > 0:
        logger.debug("Found flows to dummy nodes, figure out why")
        logger.debug(to_dummy)
        logger.debug("All movements where dummy was found:")
        logger.debug(all_movements)
    if len(from_dummy) > 0:
        logger.debug("Found flows from dummy nodes, figure out why")
        logger.debug(from_dummy)
        logger.debug("All movements where dummy was found:")
        logger.debug(all_movements)

    return (
        total_cost,
        transport_matrix,
        all_movements,
        big_m_per_commodity,
        big_m_units_per_commodity,
    )


# TODO this is critical to test!!!
def _optimize_commodity(
    physical_network,
    k,
    extended_nodes,
    extended_arcs,
    current_t,
    inventory_shape,
    inf_capacity=9000000,
):
    """

    Returns:
        cost: Total cost of the optimization for this commodity
        transport_movements: Arc flows moving from DC to DC
        all_movements: All kinds of movements occuring at time t. (TODO, should I change it to be all flows?)
        big_m_count: Count of Big M movements in this commodity.
    """
    mcf = pywrapgraph.SimpleMinCostFlow()
    slack_node_id = len(extended_nodes)

    # logger.info("adding arcs and nodes")
    problem_balance = 0
    mcfarcs = {}
    for n in extended_nodes:
        if n.commodity == k:
            # logger.info(f"mcf.SetNodeSupply({n.node_id},int({n.balance})), node: {n.name},{n}")
            mcf.SetNodeSupply(n.node_id, int(n.balance))
            problem_balance += n.balance
    for a in extended_arcs:
        if a.commodity == k:
            # logger.info(f"mcf.AddArcWithCapacityAndUnitCost({a.tail.node_id}, {a.head.node_id}, {inf_capacity}, {a.cost}), arc: {a.name},{a}")
            mcfarcs[(a.tail.node_id, a.head.node_id)] = a
            mcf.AddArcWithCapacityAndUnitCost(
                a.tail.node_id, a.head.node_id, inf_capacity, a.cost
            )

    # This was the first attempt at handling imbalanced problems, but then I coded it directly into the orders.
    if problem_balance != 0:
        logger.warning(f"WARN!! MCF balance for {k} is {problem_balance}")

    start = time.process_time()
    status = mcf.Solve()
    end = time.process_time()
    elapsed_ms = (end - start) / 1000000

    transport_movements = np.zeros(inventory_shape)
    all_movements = []
    debug_flow_movements = []
    big_m_counter = 0
    big_m_units = 0
    if status == mcf.OPTIMAL:
        # logger.info("\nFlows: ")
        for ai in range(mcf.NumArcs()):
            tail = mcf.Tail(ai)
            head = mcf.Head(ai)
            a = mcfarcs[(tail, head)]

            # Accumulate all movements occurring at current_t
            if a.commodity == k and mcf.Flow(ai) > 0 and a.head.time == current_t:
                all_movements.append((a, mcf.Flow(ai)))

            if a.commodity == k and mcf.Flow(ai) > 0:
                debug_flow_movements.append((a, mcf.Flow(ai)))

            # logger.info(f"{a.name} = {mcf.Flow(ai)}",end="")
            if (
                a.commodity == k
                and a.transportation_arc()
                and mcf.Flow(ai) > 0
                and a.head.time == current_t
            ):
                transport_movements[a.tail.location.node_id, k] -= mcf.Flow(
                    ai
                )  # subtract from source
                transport_movements[a.head.location.node_id, k] += mcf.Flow(
                    ai
                )  # add to destination
            if a.cost >= physical_network.big_m_cost and mcf.Flow(ai) > 0:
                big_m_counter += 1
                big_m_units += mcf.Flow(ai)
                logger.debug(
                    f"This is a Big M cost found in the optimization {a} ==> {mcf.Flow(ai)}"
                )
                logger.debug(f"{a.tail.location}, {a.head.location}")
    else:
        logger.info(f"Status", status)
        raise Exception("Something happened")

    return (
        mcf.OptimalCost(),
        transport_movements,
        all_movements,
        big_m_counter,
        big_m_units,
    )


# MinCostFlowBase_BAD_COST_RANGE = 6
#
# MinCostFlowBase_BAD_RESULT = 5
#
# MinCostFlowBase_FEASIBLE = 2
# MinCostFlowBase_INFEASIBLE = 3
#
# MinCostFlowBase_NOT_SOLVED = 0
#
# MinCostFlowBase_OPTIMAL = 1
# MinCostFlowBase_UNBALANCED = 4
