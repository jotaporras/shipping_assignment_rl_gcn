from __future__ import print_function

import numpy as np


def generate_basic(num_dcs, num_customers, num_t, dcs_per_customer):  # todo test
    """

    Create a set of nodes and arcs for a flow over time network, with LT=1 for all dcs and customers, and
    delivery to customers on the last day.

    :param num_dcs:
    :param num_customers:
    :param num_t:
    :param dcs_per_customer:
    """
    # Choose dcs per customer
    base_dc_assignment = np.zeros(num_dcs)
    base_dc_assignment[0:dcs_per_customer] = 1

    dcs_per_customer_array = np.array(
        [np.random.permutation(base_dc_assignment) for c in range(num_customers)]
    )

    # generate dc nodes
    dc_nodes = [
        {
            "id": dc * num_t + t,
            "name": f"dc_{dc}:{t}",
            "t": t,
            "place": dc,
            "type": "dc",
        }
        for dc in range(num_dcs)
        for t in range(num_t)
    ]

    # generate customer nodes
    customer_nodes = [
        {
            "id": len(dc_nodes) + c,
            "name": f"c_{c}:{num_t - 1}",
            "t": num_t - 1,
            "place": len(dc_nodes) + c,
            "type": "c",
        }
        for c in range(num_customers)
    ]

    # print("dc_nodes",*dc_nodes, sep="\n")
    # print("customer_nodes",*customer_nodes, sep="\n")

    # generate dc timer arcs
    dc_time_arcs = [
        {
            "name": f"dc_{dc}:{t}->dc_{dc}:{t + 1}",
            "from": dc * num_t + t,
            "to": dc * num_t + t + 1,
        }
        for dc in range(num_dcs)
        for t in range(num_t - 1)
    ]

    # generate dc transport arcs, same lapse transportation, and loops allowed.
    dc_transport_arcs = [
        {
            "name": f"dc_{dc_a}:{t}->dc_{dc_b}:{t}",
            "from": dc_a * num_t + t,
            "to": dc_b * num_t + t,
        }
        for dc_a in range(num_dcs)
        for dc_b in range(num_dcs)
        for t in range(num_t)
        if dc_a != dc_b
    ]

    # generate dc to customer arcs
    dc_customer_arcs = [
        {
            "name": f"dc_{dc}:{num_t - 1}->c_{c}:{num_t - 1}",
            "from": dc * num_t + num_t - 1,
            "to": len(dc_nodes) + c,
        }
        for dc in range(num_dcs)
        for c in range(num_customers)
        if dcs_per_customer_array[c, dc] == 1
    ]

    # print("dc_time_arcs",*dc_time_arcs, sep="\n")
    # print("dc_transport_arcs",*dc_transport_arcs, sep="\n")
    # print("dc_customer_arcs",*dc_customer_arcs, sep="\n")

    nodes = dc_nodes + customer_nodes
    arcs = dc_time_arcs + dc_transport_arcs + dc_customer_arcs
    return nodes, arcs


def generate_basic_multicommodity(
    num_dcs, num_customers, num_t, dcs_per_customer, num_k, mean_k_demand
):
    # todo test. also, how to add reasonable capacities?  fixed capacities per dc-pair, constant over time. move up to 10% inventory.
    # todo also, how to have both the static and dynamic representation of the demand
    # todo add random dates for locations.
    nodes, arcs = generate_basic(num_dcs, num_customers, num_t, dcs_per_customer)
    demand = np.random.poisson(mean_k_demand, [num_k, num_customers])
    product_choices = np.random.binomial(1, 0.6, [num_k, num_customers])
    demand_per_order = demand * product_choices

    demand_per_product = np.sum(demand_per_order, axis=1)

    inventory_distribution = np.random.dirichlet(
        [5.0] * num_dcs, num_k
    )  # (num_k,num_dc) of dc distribution of inventory.

    supply_per_dc = np.floor(inventory_distribution * demand_per_product.reshape(4, 1))
    supply_per_dc[:, 0] = (
        supply_per_dc[:, 0] + demand_per_product - np.sum(supply_per_dc, axis=1)
    )
    if not np.isclose(np.sum(np.sum(supply_per_dc, axis=1) - demand_per_product), 0.0):
        raise RuntimeError("Demand was not correctly balanced")

    for n in nodes:
        if n["type"] == "dc" and n["t"] == 0:
            n["b"] = supply_per_dc[:, n["place"]]
        elif n["type"] == "dc" and n["t"] == num_t - 1:
            n["b"] = -demand_per_order[:, n["place"]]
        else:
            n["b"] = np.zeros(num_k)
