"""
An incomplete, probably useless attempt at implementing Lagrangian Relaxation for a MCF problem.
"""

# from ortools.graph import pywrapgraph
# from typing import List
#
# class Arc:
#     def __init__(self, tail: int, head: int, capacity, costs: List[int]):
#         """
#         :param tail:
#         :param head:
#         :param costs: list of costs for each commodity
#         """
#         self.tail = tail
#         self.head = head
#         self.capacity = capacity
#         self.costs = costs
#
#     def num_commodities(self):
#         return len(self.costs)
#
# class Node:
#     def __init__(self, id: int, balance: int):
#         self.id = id
#         self.balance = balance
#
# #Following ortols.
# class LagrangianRelaxationSolver:
#     SCALING_FACTOR = 100
#
#     def __init__(self, nodes:List[Arc], arcs:List[Arc], num_commodities: int):
#         self.nodes = nodes
#         self.arcs = arcs
#         self.num_commodities = num_commodities
#
#     def solve(self):
#
#
#
#
# def solve_multicommodity(mcflows: List[pywrapgraph.SimpleMinCostFlow]):
#     # Product 1
#     SCALING_FACTOR = 100
#     ic = 1000  # infinite capacity.
#     arcs = [
#         (0, 1, 1, 5),
#         (0, 2, 5, ic),
#         (2, 3, 1, 10),
#         (3, 1, 5, ic),
#         (3, 5, 1, ic),
#         (4, 2, 1, ic),
#         (4, 5, 5, ic)
#     ]
#
#     arcs = [
#         {"tail": 0, "head": 1, "cost": 1, "capacity": 5},
#         {"tail": 0, "head": 2, "cost": 5, "capacity": ic},
#         {"tail": 2, "head": 3, "cost": 1, "capacity": 10},
#         {"tail": 3, "head": 1, "cost": 5, "capacity": ic},
#         {"tail": 3, "head": 5, "cost": 1, "capacity": ic},
#         {"tail": 4, "head": 2, "cost": 1, "capacity": ic},
#         {"tail": 4, "head": 5, "cost": 5, "capacity": ic}
#     ]
#
#
# def print_solution(mcf):
#     if mcf.Solve() == mcf.OPTIMAL:
#         print('Minimum cost:', mcf.OptimalCost())
#         print('')
#         print('  Arc    Flow / Capacity  FlowCost ArcCost')
#         for i in range(mcf.NumArcs()):
#             cost = mcf.Flow(i) * mcf.UnitCost(i)
#             print('%1s -> %1s   %3s  / %3s       %3s\t\t\t%3s' % (
#                 mcf.Tail(i),
#                 mcf.Head(i),
#                 mcf.Flow(i),
#                 mcf.Capacity(i),
#                 # unscaled_double(cost)
#                 cost,
#                 mcf.UnitCost(i)
#             )
#                   )
#     else:
#         print('There was an issue with the min cost flow input.')
#
#
# def calculate_original_cost(mcf, arcs):
#     acc = 0.0
#     for i in range(0, len(arcs)):
#         acc += mcf.Flow(i) * arcs[i]['cost']
#     return acc
#
#
# def calculate_unscaled_cost(mcf):
#     acc = 0.0
#     for i in range(0, len(arcs)):
#         acc += mcf.Flow(i) * unscaled(mcf.UnitCost(i))
#     return acc
#
#
# def scaled(cost):
#     return int(cost * SCALING_FACTOR)
#
#
# def unscaled(cost):
#     return int(cost / SCALING_FACTOR)
#
#
# def unscaled_double(cost):
#     return float(float(cost) / SCALING_FACTOR)
#
#     supplies_p1 = [10, -10, 0, 0, 0, 0]
#     supplies_p1 = [10, -10, 0, 0, 20, -20]
#
#     p1 = pywrapgraph.SimpleMinCostFlow()
#
#     for i in range(0, len(arcs)):
#         p1.AddArcWithCapacityAndUnitCost(arcs[i]['tail'], arcs[i]['head'],
#                                          arcs[i]['capacity'], arcs[i]['cost'])
#
#     # Add node supplies.
#
#     for i in range(0, len(supplies_p1)):
#         p1.SetNodeSupply(i, supplies_p1[i])
#
#     p1.Solve()
#
#     p2 = pywrapgraph.SimpleMinCostFlow()
#     supplies_p2 = [0, 0, 0, 0, 20, -20]
#     # supplies_p2 = [10, -10, 0, 0, 20, -20]
#
#     for i in range(0, len(arcs)):
#         p2.AddArcWithCapacityAndUnitCost(arcs[i]['tail'], arcs[i]['head'],
#                                          arcs[i]['capacity'], arcs[i]['cost'])
#
#     # Add node supplies.
#
#     for i in range(0, len(supplies_p2)):
#         p2.SetNodeSupply(i, supplies_p2[i])
#
#     p2.Solve()
#
#     lagrangian opt test.
#     steps = 15
#     steps = 25
#     original_theta = 1.0
#     theta = original_theta
#     #lagrangians = [0. for i in range(0, len(arcs))]
#     lagrangians = [5.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
#     mcf_p1 = None
#     mcf_p2 = None
#     print("initial lagrangian ", lagrangians)
#     for t in range(1, steps):
#         print("iteration ", t)
#         mcf_p1 = pywrapgraph.SimpleMinCostFlow()
#         for i in range(0, len(arcs)):
#             arc = arcs[i]
#             lagrangian = lagrangians[i]
#             scaled_cost = scaled(arc['cost'] + lagrangian)
#             # capacity = ic
#             capacity = arc['capacity']
#             mcf_p1.AddArcWithCapacityAndUnitCost(arc['tail'], arc['head'], capacity, scaled_cost)
#         for i in range(0, len(supplies_p1)):
#             mcf_p1.SetNodeSupply(i, supplies_p1[i])
#         mcf_p1.Solve()
#         # print("---P1 SOLUTION---")
#         # print_solution(mcf_p1)
#         mcf_p2 = pywrapgraph.SimpleMinCostFlow()
#         for i in range(0, len(arcs)):
#             arc = arcs[i]
#             lagrangian = lagrangians[i]
#             scaled_cost = scaled(arc['cost'] + lagrangian)
#             # capacity = ic
#             capacity = arc['capacity']
#             mcf_p2.AddArcWithCapacityAndUnitCost(arc['tail'], arc['head'], capacity, scaled_cost)
#         for i in range(0, len(supplies_p2)):
#             mcf_p2.SetNodeSupply(i, supplies_p2[i])
#         mcf_p2.Solve()
#         # print("---P2 SOLUTION---")
#         # print_solution(mcf_p2)
#         lagrangian_cost = calculate_unscaled_cost(mcf_p1) + calculate_unscaled_cost(mcf_p2)
#
#         # update the lagrangian multipler values
#         for i in range(0, len(arcs)):
#             arc = arcs[i]
#             p1_flow = mcf_p1.Flow(i)
#             p2_flow = mcf_p2.Flow(i)
#             # print("updating lagrangian for arc ", i)
#             # print('%1s -> %1s   %3s  / %3s' % (
#             #     mcf_p1.Tail(i),
#             #     mcf_p1.Head(i),
#             #     mcf_p1.Flow(i),
#             #     mcf_p1.Capacity(i)))
#             # print('%1s -> %1s   %3s  / %3s' % (
#             #     mcf_p2.Tail(i),
#             #     mcf_p2.Head(i),
#             #     mcf_p2.Flow(i),
#             #     mcf_p2.Capacity(i)))
#             overcapacity = (p1_flow+p2_flow - arc['capacity'])
#             # print("P1 Difference between flow in arc and lagrangian ", overcapacity)
#             lagrangians[i] = max(0, lagrangians[i] + theta * (overcapacity))
#
#         theta = original_theta/t
#
#         total_cost = calculate_original_cost(mcf_p1,arcs) + calculate_original_cost(mcf_p2,arcs)
