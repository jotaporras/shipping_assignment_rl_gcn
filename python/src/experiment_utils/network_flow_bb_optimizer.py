"""
Nov 4, 2021. My first attempt at triyng to solve Multicommodity netflow with B&B.
A nasty copy paste of network_flow_k_optimizer up until today. WATCH OUT IF YOU NEED TO MODIFY THAT.
"""

# {
#             "physical_network": self.environment_parameters.network,
#             "inventory": self.inventory,
#             "open": self.open_orders,
#             "fixed": self.fixed_orders,
#             "current_t": self.current_t,
# }
from __future__ import annotations
import signal
import itertools
import logging
import operator
import time
from typing import List, Tuple

import numpy as np
from network.ExtendedNetwork import ExtendedNetwork, TENNode
from network.physical_network import Arc, PhysicalNetwork
from ortools.linear_solver import pywraplp

from experiment_utils.Order import Order

logger = logging.getLogger(__name__)


def optimize_branch_and_bound(
    physical_network,
    inventory,
    fixed,
    open,
    current_t,
    time_limit_milliseconds=240 * 1000,
):
    """
    Uses branch and bound to optimize a network flow.
    Args:
        state:

    Returns:
        (extended_nodes,extended_arcs_with_flow):
        extended_nodes:
            The TENNodes generated to optimize this BB problem.
        extended_arcs_with_flow:
            list of (Arc,int) tuples with the flow to each TEN Arc
            given by the B&B solver.


    """
    planning_horizon_t = current_t + physical_network.planning_horizon - 1
    # Treat all orders as fixed.
    # extended_network = ExtendedNetwork(network, inventory, fixed_orders=open + fixed, open_orders=[])
    # only fixed TODO is this ok??? Could be. Consider the before and after stage impact.
    extended_network = ExtendedNetwork(
        physical_network, inventory, fixed_orders=fixed, open_orders=open
    )
    extended_nodes, extended_arcs = extended_network.convert_to_extended(
        current_t, planning_horizon_t, generate_all_inbound_customer_arcs=True
    )

    (extended_arcs_with_flow, solver,) = optimize_expanded_with_branch_bound(
        physical_network, extended_nodes, extended_arcs, time_limit_milliseconds
    )
    logger.debug(
        f"Branch & Bound solver finished with cost {solver.Objective().Value()}"
    )
    # flatten all movements to one commodity

    # Check if all movements contain dummy flows #TODO, shouldn't the dummy be a node instead of string??
    from_dummy = [m for m in extended_arcs_with_flow if m[0].tail.location == "DUMMY"]
    to_dummy = [m for m in extended_arcs_with_flow if m[0].head.location == "DUMMY"]
    if len(to_dummy) > 0:
        logger.debug("Found flows to dummy nodes, figure out why")
        logger.debug(to_dummy)
        logger.debug("All movements where dummy was found:")
        logger.debug(extended_arcs_with_flow)
    if len(from_dummy) > 0:
        logger.debug("Found flows from dummy nodes, figure out why")
        logger.debug(from_dummy)
        logger.debug("All movements where dummy was found:")
        logger.debug(extended_arcs_with_flow)

    # TODO do more checks validations and logs here.

    return (extended_nodes, extended_arcs_with_flow)


def optimize_expanded_with_branch_bound(
    physical_network,
    extended_nodes: List[TENNode],
    extended_arcs: List[Arc],
    time_limit_milliseconds,
):
    """

    Returns:
        cost: Total cost of the optimization for this commodity
        transport_movements: Arc flows moving from DC to DC
        all_movements: All kinds of movements occuring at time t. (TODO, should I change it to be all flows?)
        big_m_count: Count of Big M movements in this commodity.
    """
    num_commodities = physical_network.num_commodities
    solver = pywraplp.Solver.CreateSolver("SCIP")
    infinity = solver.Infinity()
    slack_node_id = len(
        extended_nodes
    )  # TODO im assumming no slack for this. Damn son.

    balances_n_k = {}

    # todo fijo hay una forma más facil de modelar esto usando las estructuras de Orlin.
    # este monton de diccionarios no son muy elegantes.
    problem_balance = 0
    # Storage for my arc objects (tail_id,head_id) -> arc #probs gonna die.
    mcfarcs = {}
    # Storage for indicator vars l_ij^k of arcs (tail_id,head_id) -> [var]*k #TODO add time
    ind_arcs_vars = {}
    # (head) -> [var]*k*inbound
    ind_arcs_vars_by_c = {}
    # Storage for flow vars x_ij^k (tail_id,head_id) ->  [var]*k #TODO add time
    flow_arcs_vars = {}
    # Regular arc objects
    flow_arcs = {}
    # (j,k) -> [Arc(i,j) forall i]
    inflow_map_vars = {}
    # (i,k) -> [Arc(i,j) forall k]
    outflow_map_vars = {}

    for n in extended_nodes:
        # I think I dont need to worry about physical location on balances.
        balances_n_k.setdefault((n.ten_key(), n.commodity), n)

    # create flow, indicator and objective vars
    objective_vars = []
    for a in extended_arcs:
        i = a.tail.ten_key()
        j = a.head.ten_key()
        # t_i = a.tail.time
        # t_j = a.head.time
        k = a.commodity
        ij_key = (i, j)

        # Creating flow var

        flow_arc_k_vars = flow_arcs_vars.setdefault(ij_key, [None] * num_commodities)
        flow_solver_var = solver.IntVar(0.0, infinity, f"x^{k}_{i}->{j}")
        flow_arc_k_vars[k] = flow_solver_var

        # Creating inflow &outflow aggregator
        inflow_map_vars.setdefault((j, k), []).append(flow_solver_var)
        outflow_map_vars.setdefault((i, k), []).append(flow_solver_var)

        objective_vars.append(flow_solver_var * a.cost)

        # TODO potential optimization: technically I only need to do this for open orders
        if a.head.kind == "C":  # Destination customer, requires indicator var
            # This map is to set constraints by (i,j) ->[k]
            ind_arc_k_vars = ind_arcs_vars.setdefault(ij_key, [None] * num_commodities)
            indicator_var = solver.IntVar(0.0, 1.0, f"l^{k}_{i}->{j}")
            ind_arc_k_vars[k] = indicator_var
            # This map is to set constraints by (c)-> [all arcs]
            ind_arcs_vars_by_c.setdefault(j, []).append(indicator_var)

    # TODO mass balance constraints and upper bound on flow
    upper_bound = 0.0
    for n_k_key in balances_n_k.keys():  # for every TEN node
        ten_key, k = n_k_key
        node_k = balances_n_k[n_k_key]
        # expanded_node_id = node_k.node_id
        inflow_vars = inflow_map_vars.get(n_k_key, [])
        outflow_vars = outflow_map_vars.get(n_k_key, [])
        # todo make sure this is correct.
        # Adding mass balance constrant for the current n,k
        # Note: it's harmless if the var lists are empty.
        logger.debug(f"Constraint: mbc_{node_k.ten_key()}^{k}")
        logger.debug(
            f"Adding constraint -{solver.Sum(inflow_vars)} + {solver.Sum(outflow_vars)} = {node_k.balance}"
        )
        solver.Add(
            -solver.Sum(inflow_vars) + solver.Sum(outflow_vars) == node_k.balance,
            f"mbc_{node_k.ten_key()}^{k}",
        )
        upper_bound += abs(
            node_k.balance
        )  # a lazy upper bound, in reality it's way less.

    # linearized bundle constraints.
    for ij_key in ind_arcs_vars.keys():
        # x^k_ij <= xhat*l^k_ij
        used_commodities = 0
        for k in range(num_commodities):
            # Bind indicator variables to flow.
            if (
                flow_arcs_vars[ij_key][k] is not None
            ):  # Happens bc there's no node if no demand of commodity.
                solver.Add(
                    flow_arcs_vars[ij_key][k] <= upper_bound * ind_arcs_vars[ij_key][k],
                    f"x^{k}_{ij_key} <= xhat*l^{k}_{ij_key}",
                )
                solver.Add(
                    flow_arcs_vars[ij_key][k] >= ind_arcs_vars[ij_key][k],
                    f"x^{k}_{ij_key} >= l^{k}_{ij_key}",
                )
                used_commodities += 1
        # Bind indicator vars to each other: The sum of all indicators of an arc (one per commodity)
        # should sum to K if at least one of them is in use.
        # used commodities instead of num_commodities in case the order demand has zero somewhere.
        non_empty_indicators = [
            indicator for indicator in ind_arcs_vars[ij_key] if indicator is not None
        ]
        solver.Add(
            solver.Sum(non_empty_indicators)
            == used_commodities * non_empty_indicators[0],
            f"linearized_bundle_sum_k_{ij_key}",
        )
    # sum of all constraints
    for indicator_key, l_c_list in ind_arcs_vars_by_c.items():
        # l_c_list = [l_ic^k] for a given c.
        # sum of all indicators of a customer == K

        solver.Add(
            solver.Sum(l_c_list) == num_commodities,
            f"bundle_all_customers_sum_k_{indicator_key}",
        )

    # objective
    solver.Minimize(solver.Sum(objective_vars))

    start = time.process_time()
    logger.debug(f"Number of B&B Constrains: {solver.NumConstraints()}")
    logger.debug(f"Number of B&B Variables: {solver.NumVariables()}")
    logger.debug("Calling solve")
    solver.set_time_limit(time_limit_milliseconds)

    def _raise_timeout_exception():
        print("TIMED OUT ON BNB SOLVE.")
        raise Exception("TIMED OUT ON BNB SOLVE.")

    signal.signal(signal.SIGALRM, _raise_timeout_exception)
    # print(f"waiting for {time_limit_milliseconds}")
    signal.alarm(int(time_limit_milliseconds / 1000))
    status = solver.Solve()  # todo replace with correct solve.
    signal.alarm(0)
    logger.debug("Solve finished")
    end = time.process_time()
    elapsed_ms = end - start
    logger.debug(f"BB optimizer took {elapsed_ms}s")
    extended_arcs_with_flow = []
    if status == pywraplp.Solver.OPTIMAL:
        for a in extended_arcs:
            i = a.tail.ten_key()
            j = a.head.ten_key()
            k = a.commodity
            ij_key = (i, j)
            flow_ij = flow_arcs_vars[ij_key][k].solution_value()

            extended_arcs_with_flow.append((a, flow_ij))

            # Group orders by order id and commodity
            positive_flow_customer_arcs = []
            for arc, flow in extended_arcs_with_flow:
                if arc.head.kind == "C" and flow > 0:
                    key = arc.head.ten_key()
                    value = arc.tail.ten_key()
                    positive_flow_customer_arcs.append((key, value))
            dcs_used_per_order = [
                (k, set([dc for c, dc in v]))
                for k, v in itertools.groupby(
                    positive_flow_customer_arcs, operator.itemgetter(0)
                )
            ]
            orders_with_multiple_dcs = [
                (order, dcs) for order, dcs in dcs_used_per_order if len(dcs) > 1
            ]
            if len(orders_with_multiple_dcs) > 0:
                logger.error("FATAL. Found orders with more than one DC serving it. ")
                logger.error(orders_with_multiple_dcs)
                offending_flows = [
                    (arc, flow)
                    for arc, flow in extended_arcs_with_flow
                    if arc.head.ten_key() == orders_with_multiple_dcs[0]
                ]
                logger.error("flows for one of the offenders")
                logger.error(offending_flows)
                raise RuntimeError(
                    "Violation of constraints, some order is using more than one constraint."
                )
        logger.debug(f"Number of B&B nodes: {solver.nodes()}")
        logger.debug(f"Number of B&B Constrains: {solver.NumConstraints()}")
        logger.debug(f"Number of B&B Constrains: {solver.NumVariables()}")
        logger.debug(f"Number of iterations: {solver.iterations()}")
        logger.debug(f"Wall time B&B: {solver.wall_time()}")
        return extended_arcs_with_flow, solver
    else:
        logger.error(f"Optimizer status", status)
        logger.error(f"Problem that was used when error happened")
        logger.error(f"Number of B&B nodes: {solver.nodes()}")
        logger.error(f"Number of B&B Constrains: {solver.NumConstraints()}")
        logger.error(f"Number of B&B Constrains: {solver.NumVariables()}")
        logger.error(f"Number of iterations: {solver.iterations()}")
        logger.error(f"Wall time B&B: {solver.wall_time()}")
        for a in extended_arcs:
            i = a.tail.ten_key()
            j = a.head.ten_key()
            k = a.commodity
            ij_key = (i, j)
            flow_ij = flow_arcs_vars[ij_key][k].solution_value()
            if flow_ij > 0:
                logger.error(f"Arc {a} with flow flow_ij")
        raise Exception("Something happened in the BB optimizer")


def bb_solution_to_agent_action(
    open_orders: List[Order],
    extended_arcs_with_flow: List[Tuple[Arc, int]],
):
    """
    Converts a set of TEN arcs with flow (from an optimizer) to a set of actions on open orders.
    Args:
        extended_arcs_with_flow:

    Returns:
        A list with len(open_orders) specifying the actions to take for each.

    """

    def find_open_order(
        arc: Arc,
    ):  # todo this is a nasty n^2 but OO are small. If an issue, optimize by having a map of orders.
        for o in open_orders:
            if arc.head.location == o.customer:
                return o
        return None

    actions = {}  # (order_key,action)
    for arc, flow in extended_arcs_with_flow:
        if (
            arc.head.kind == "C" and arc.tail.location != "DUMMY"
        ):  # Todo not sure if dummy condition is even implemented. (nov 5)
            associated_order = find_open_order(arc)
            if associated_order is not None and flow > 0:
                actions[associated_order.order_key()] = arc.tail.location.node_id
    if len(actions.keys()) != len(open_orders):
        raise RuntimeError("Unexpected, not all open orders got assigned an action.")
    return actions


if __name__ == "__main__":
    # Given
    logging.root.setLevel(logging.DEBUG)
    physical_network = PhysicalNetwork(
        num_dcs=3,
        num_customers=1,
        dcs_per_customer=2,
        demand_mean=100,
        demand_var=25,
        big_m_factor=10000,
        num_commodities=2,
        planning_horizon=4,
    )
    # (num_dcs,num_commodities)
    # fmt:off
    inventory = np.array([
        [1.0,0.0],
        [0.0,0.0],
        [1.0,1.0],
    ])
    # fmt:on

    customer_node = physical_network.customers[0]
    dc_0_node = physical_network.dcs[0]
    open = [
        Order(
            demand=np.array([2.0, 1.0]),
            shipping_point=dc_0_node,
            customer=customer_node,
            delivery_time=3,
            name="o1",
        )
    ]
    fixed = []
    # fixed = [
    #     Order(
    #         demand=np.array([2.0]),
    #         shipping_point=dc_0_node,
    #         customer=customer_node,
    #         delivery_time=3,
    #         name="o1",
    #     )
    # ]
    current_t = 1
    planning_horizon_t = current_t + physical_network.planning_horizon - 1
    extended_network = ExtendedNetwork(
        physical_network, inventory, fixed_orders=fixed, open_orders=open
    )
    extended_nodes, extended_arcs_with_flow = optimize_branch_and_bound(
        physical_network, inventory, fixed, open, current_t
    )

    next_actions = bb_solution_to_agent_action(open, extended_arcs_with_flow)
    print("Soulitionr")
    print(
        next_actions
    )  
