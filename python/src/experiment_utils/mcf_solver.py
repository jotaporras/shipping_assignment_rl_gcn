"""
Using Ortools to solve Network Min Cost flow problems. it's only used in the benchmarks apparently.
"""
import time
from ortools.graph import pywrapgraph


def mcf_solve(mcf):
    balances = [mcf.Supply(n) for n in range(mcf.NumNodes())]
    # print(balances)
    # print("balance sum should be 0 = ",sum(balances))

    print("Running optimization")
    start = time.process_time_ns()
    mcf.Solve()
    end = time.process_time_ns()
    elapsed_ms = (end - start) / 1000000
    print(f"elapsed {elapsed_ms}ms")
    print(f"elapsed {round_to_1(elapsed_ms/1000)}s")

    if mcf.Solve() == mcf.OPTIMAL:
        print("Minimum cost:", mcf.OptimalCost())
        # print('')
        # print('  Arc    Flow / Capacity  FlowCost ArcCost')
        # for i in range(mcf.NumArcs()):
        #     cost = mcf.Flow(i) * mcf.UnitCost(i)
        # print('%s   %3s  / %3s       %3s\t\t\t%3s' % (
        #     arcs[i]['name'],
        #     # mcf.Tail(i),
        #     # mcf.Head(i),
        #     mcf.Flow(i),
        #     mcf.Capacity(i),
        #     # unscaled_double(cost)
        #     cost,
        #     mcf.UnitCost(i)
        # )
        #   )

    else:
        print("There was an issue with the min cost flow input.")
        # print(mcf)
        # print(mcf.NumArcs())
        # print('  Arc    Flow / Capacity  FlowCost ArcCost')
        # for i in range(mcf.NumArcs()):
        #     cost =  mcf.UnitCost(i)
        #     # print('%1s -> %1s   %3s  / %3s       %3s\t\t\t%3s' % (
        #         mcf.Tail(i),
        #         mcf.Head(i),
        #         0,
        #         mcf.Capacity(i),
        #         # unscaled_double(cost)
        #         cost,
        #         mcf.UnitCost(i)))
    return elapsed_ms
