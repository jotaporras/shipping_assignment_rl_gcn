from ortools.graph import pywrapgraph

from experiment_utils.general_utils import round_to_1
from experiment_utils.mcf_solver import mcf_solve

from experiment_utils.problem_generator import generate_basic


def generate_hardwired_last_mile_mcf(
    nodes,
    arcs,
    num_customers,
    num_dcs,
    base_demand,
    default_capacity=10000,
    default_cost=1,
):
    base_supply = int(-base_demand * num_customers / num_dcs)
    mcf = pywrapgraph.SimpleMinCostFlow()
    for i in range(0, len(arcs)):
        mcf.AddArcWithCapacityAndUnitCost(
            arcs[i]["from"], arcs[i]["to"], default_capacity, default_cost
        )
        # Add node supplies.

    for i, n in enumerate(nodes):
        if n["type"] == "dc":
            if n["t"] == 0:
                mcf.SetNodeSupply(n["id"], base_supply)
            else:
                mcf.SetNodeSupply(n["id"], 0)
        else:
            mcf.SetNodeSupply(n["id"], base_demand)
    return mcf


def simple_unicommodity_min_cost_flow_benchmark(num_trials=3):
    print("running simple_unicommodity_min_cost_flow_benchmark")
    num_dcs = 50
    num_customers = 800
    dcs_per_customer = 5
    base_demand = -100

    results = []

    for num_t in range(0, 3000, 250):
        times = []
        costs = []
        num_arcs = 0
        num_nodes = 0
        for i in range(num_trials):
            nodes, arcs = generate_basic(
                num_dcs, num_customers, num_t, dcs_per_customer
            )
            mcf = generate_hardwired_last_mile_mcf(
                nodes, arcs, num_customers, num_dcs, base_demand
            )
            elapsed_ms = mcf_solve(mcf)
            times.append(elapsed_ms)
            costs.append(mcf.OptimalCost())
            num_arcs = len(arcs)
            num_nodes = len(nodes)
        results.append(
            {
                "num_customers": num_customers,
                "dcs_per_customer": dcs_per_customer,
                "base_demand": base_demand,
                "num_t": num_t,
                "num_arcs": num_arcs,
                "num_nodes": num_nodes,
                "costs": float(sum(costs)) / float(len(costs)),
                "elapsed_ms": float(sum(times)) / float(len(times)),
                "elapsed_s": str(
                    round_to_1(float(sum(times)) / float(len(times)) / 1000)
                )
                + "s",
            }
        )

    print(*results, sep="\n")
