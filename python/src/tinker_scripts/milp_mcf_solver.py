"""
A -possibly- functioning attempt at solving a hard coded shipping point problem with CSP.
July 2021: I think I planned to use this at tema selecto but wasn't able to make it work with large problems?
"""
import functools

from ortools.sat.python import cp_model

from experiment_utils.mnetgen_format import *
from typing import List

from experiment_utils.problem_generator import (
    generate_basic_multicommodity,
    generate_basic,
)


class ShippingPointConsolidationCSPSolver:
    def __init__(
        self,
        nodes: List[Node],
        arcs: List[Arc],
        capacities: List[MutualCapacity],  # this empty for now.
        exclusive_arcs: List[List[Arc]],  # group of arcs that form an "exclusive set".
    ):
        self.nodes = nodes
        self.arcs = arcs
        self.capacities = capacities
        self.exclusive_arcs = exclusive_arcs
        self.model = None

        self.K = max(
            self.nodes, key=lambda n: n.commodity
        ).commodity  # WARNING: commodities must be continuous and monotonically increasing.

    def create_model(self):
        self.model = cp_model.CpModel()
        model = self.model
        infinity = sum([abs(n.supply) for n in self.nodes])
        print("infinity", infinity)

        # Generate arc variables.
        self.arcs = sorted(self.arcs, key=lambda a: a.name)

        arc_vars = dict(
            {(a.name, model.NewIntVar(0, infinity, str(a))) for a in self.arcs}
        )

        # Decision variables for arc subsets
        boolean_vars_by_commodity_destination = {}

        all_subset_boolean_vars = []
        for delivery_subset_arcs in self.exclusive_arcs:
            arc_subset_boolean_vars = []
            boolean_vars_by_commodity = (
                {}
            )  # todo this is not being used, i think it should
            boolean_vars_by_source = {}  # see if you need all.

            for arc in delivery_subset_arcs:
                # print("processing arc",arc)
                boolvar = model.NewBoolVar("l" + str(arc))
                arc_subset_boolean_vars.append(boolvar)

                # if the boolean is set to true, then there is flow in that arc.
                # print("Creating decision boolean var constraints") #todo this is probably wrong, not the expected arcs are being triggered.
                # print("CONSTRAINT: when its positive flow",arc_vars[arc.name] > 0,"then",boolvar)
                # print("CONSTRAINT:  when its zero flow",arc_vars[arc.name] == 0,"then",boolvar.Not())
                model.Add(arc_vars[arc.name] > 0).OnlyEnforceIf(boolvar)
                model.Add(arc_vars[arc.name] == 0).OnlyEnforceIf(boolvar.Not())

                # Add arc to commodity-destination dictionary.
                if (
                    arc.commodity,
                    arc.to_node,
                ) not in boolean_vars_by_commodity_destination:
                    boolean_vars_by_commodity_destination[
                        (arc.commodity, arc.to_node)
                    ] = []
                # print(f"Adding arc {arc} to boolean_vars_by_commodity_destination {(arc.commodity, arc.to_node)}")
                boolean_vars_by_commodity_destination[
                    (arc.commodity, arc.to_node)
                ].append(boolvar)

                if arc.from_node not in boolean_vars_by_source:
                    boolean_vars_by_source[arc.from_node] = []
                # print(f"Adding arc {arc} to boolean vars by source {arc.from_node}")
                boolean_vars_by_source[arc.from_node].append(boolvar)
                # print()
            all_subset_boolean_vars.append(arc_subset_boolean_vars)

            # print(f"whole boolean_vars_by_source: {boolean_vars_by_source}")
            # for bv_list in boolean_vars_by_source.values():
            #     print(bv_list)
            #     for bv1 in bv_list:
            #         list_except = list(bv_list)
            #         list_except.remove(bv1)
            #         for bv2 in list_except:
            #             if bv1 != bv2:
            #                 print("CONSTRAINT: Restricting by source constraint",bv1 == bv2)
            #                 model.Add(bv1 == bv2)

            # Only one serving arc per commodity
            for (
                commodity
            ) in (
                boolean_vars_by_commodity.keys()
            ):  # this is always empty. #todo see if delete.
                commodity_boolean_vars = boolean_vars_by_commodity[commodity]
                commodity_bundle = commodity_boolean_vars[0]
                for a in commodity_boolean_vars[1:]:
                    commodity_bundle = commodity_bundle + a
                # print("CONSTRAINT: By Commodity bundle constraint",commodity_bundle == 1)
                model.Add(commodity_bundle == 1)

            # Bundle all arcs to the destination.
            # bundle_restriction = arc_subset_set[0]
            # for arc in arc_subset_set[1:]:
            #     bundle_restriction = bundle_restriction + arc
            # model.Add(bundle_restriction <= self.K)
            # arc_subset_sets.append(arc_subset_set)

        # only one serving arc per (customer-commodity)
        for (
            commodity,
            to_node,
        ) in boolean_vars_by_commodity_destination.keys():  # verified correct.
            commodity_node_boolean_vars = boolean_vars_by_commodity_destination[
                (commodity, to_node)
            ]
            # print(f"Creating commodity-destination restrictions for {(commodity, to_node)}")
            commodity_destination_bundle = commodity_node_boolean_vars[0]
            for a in commodity_node_boolean_vars[1:]:
                commodity_destination_bundle = commodity_destination_bundle + a
            # print("CONSTRAINT: Commodity-destination constraint",commodity_destination_bundle==1)
            model.Add(commodity_destination_bundle == 1)

        # Mass balance constraints.
        mbc_outbound_map = {}
        mbc_inbound_map = {}
        for arc in self.arcs:
            if (arc.from_node, arc.commodity) not in mbc_outbound_map:
                mbc_outbound_map[(arc.from_node, arc.commodity)] = []
            mbc_outbound_map[(arc.from_node, arc.commodity)].append(arc)

            if (arc.to_node, arc.commodity) not in mbc_inbound_map:
                mbc_inbound_map[(arc.to_node, arc.commodity)] = []
            mbc_inbound_map[(arc.to_node, arc.commodity)].append(arc)

        # Create mass balance constraints
        for n in self.nodes:
            outbounds = mbc_outbound_map.get((n.node_id, n.commodity), [])
            inbounds = mbc_inbound_map.get((n.node_id, n.commodity), [])

            outbound_vars = list(map(lambda o: arc_vars[o.name], outbounds))
            inbound_vars = list(map(lambda i: arc_vars[i.name], inbounds))

            outbound_var_sum = functools.reduce(lambda a, b: a + b, outbound_vars, 0)
            inbound_var_sum = functools.reduce(lambda a, b: a + b, inbound_vars, 0)

            # Mass balance constraint
            # print(f"Node id {n.node_id}")
            # print(f"Inbound {inbound_var_sum}")
            # print(f"Inbound neg {-inbound_var_sum}")
            # print(f"Outbound {outbound_var_sum}")
            # print(f"CONSTRAINT: Adding Mass balance constraint node {n.node_id}",(-inbound_var_sum + outbound_var_sum) == n.supply)
            model.Add(
                (-inbound_var_sum + outbound_var_sum) == n.supply
            )  # this is the correct
            # model.Add((inbound_var_sum - outbound_var_sum) == n.supply)

        # Create objective function
        obj = arc_vars[self.arcs[0].name] * self.arcs[0].cost
        for a in self.arcs[1:]:
            obj = obj + arc_vars[a.name] * a.cost
        print("Model objective", obj)
        model.Minimize(obj)
        print("Validation: ", model.Validate())
        print("--")
        solver = cp_model.CpSolver()

        status = solver.Solve(model)

        DEBUG = False
        if status == cp_model.OPTIMAL:
            if DEBUG:
                print("Solution:")
                for a in self.arcs:
                    print(
                        f"arc^{a.commodity}({a.from_node},{a.to_node})={solver.Value(arc_vars[a.name])}"
                    )
                for i, eal in enumerate(self.exclusive_arcs):
                    print(f"eal {eal}")
                    bvcd = all_subset_boolean_vars[i]
                    print(f"bvcd {bvcd}")
                    for ai, eab in enumerate(bvcd):
                        ea = eal[ai]
                        print(f"exclusive arc {ea}")
                        print(
                            f"exclusive^{ea.commodity}({ea.from_node},{ea.to_node})={solver.Value(eab)}"
                        )

            # for k in list(x.keys()):
            #     # print(f"x[{k}] = {x[k].Value()}")
            #     print(f"x[{k}] = {solver.Value(x[k])}")
            # print(f"l^1_25 {solver.Value(l1_25)}")
            # print(f"l^2_25 {solver.Value(l2_25)}")
            # print(f"l^1_45 {solver.Value(l1_45)}")
            # print(f"l^2_45 {solver.Value(l2_45)}")
            print("Objective value =", solver.ObjectiveValue())
            print("UserTime =", solver.UserTime())
            print("WallTime() =", solver.WallTime())
        else:
            if status == cp_model.FEASIBLE:
                print("Feasible solution")
                print("Objective value =", solver.ObjectiveValue())
                print("UserTime =", solver.UserTime())
                print("WallTime() =", solver.WallTime())
            else:
                print(solver.StatusName(status))
                print("The problem does not have an optimal solution.")
                # print(model)


def example1():
    b = {
        "b^1_1": 10,
        "b^1_2": 0,
        "b^1_3": 0,
        "b^1_4": 0,
        "b^1_5": -10,
        "b^2_1": 0,
        "b^2_2": 0,
        "b^2_3": 10,
        "b^2_4": 0,
        "b^2_5": -10,
    }

    arc1_25 = Arc(3, 2, 5, 1, 1, 100, 0)
    arc1_45 = Arc(5, 4, 5, 1, 1, 100, 0)
    arc2_25 = Arc(8, 2, 5, 2, 1, 100, 0)
    arc2_45 = Arc(10, 4, 5, 2, 1, 100, 0)
    arcs = [
        Arc(1, 1, 2, 1, 1, 100, 0),
        Arc(2, 1, 3, 1, 1, 100, 0),
        arc1_25,
        Arc(4, 3, 4, 1, 1, 100, 0),
        arc1_45,
        Arc(6, 1, 2, 2, 1, 100, 0),
        Arc(7, 1, 3, 2, 1, 100, 0),
        arc2_25,
        Arc(9, 3, 4, 2, 1, 100, 0),
        arc2_45,
    ]

    nodes = [
        Node(1, 1, 10),
        Node(2, 1, 0),
        Node(3, 1, 0),
        Node(4, 1, 0),
        Node(5, 1, -10),
        Node(1, 2, 0),
        Node(2, 2, 0),
        Node(3, 2, 10),
        Node(4, 2, 0),
        Node(5, 2, -10),
    ]
    exclusive_arcs = [[arc1_25, arc1_45, arc2_25, arc2_45]]  # dc to c1 arcs
    # exclusive_arcs=[]
    x = ShippingPointConsolidationCSPSolver(nodes, arcs, [], exclusive_arcs)
    x.create_model()


def example2():
    commodities = range(1, 12)
    # commodities = [1]
    nodes = []
    for k in commodities:
        shift = 6 * (k - 1)
        nodes.append(Node(shift + 1, k, 5))
        nodes.append(Node(shift + 2, k, 5))
        nodes.append(Node(shift + 3, k, 5))
        nodes.append(Node(shift + 4, k, 5))

        nodes.append(Node(shift + 5, k, -10))
        nodes.append(Node(shift + 6, k, -10))
    # nodes = [
    #     Node(1, 1, 5),
    #     Node(2, 1, 5),
    #     Node(3, 1, 5),
    #     Node(4, 1, 5),
    #
    #     Node(5, 1, -10),
    #     Node(6, 1, -10),
    #
    #     # ###
    #     Node(7, 2, 5),
    #     Node(8, 2, 5),
    #     Node(9, 2, 5),
    #     Node(10, 2, 5),
    #
    #     Node(11, 2, -10),
    #     Node(12, 2, -10),
    #
    #     ##
    #     Node(13, 3, 5),
    #     Node(14, 3, 5),
    #     Node(15, 3, 5),
    #     Node(16, 3, 5),
    #
    #     Node(17, 3, -10),
    #     Node(18, 3, -10)
    # ]
    print(f"total sum (should be 0): {sum(map(lambda n:n.supply,nodes))}")

    arcs = []
    i = 1
    num_nodes = 6  # max(nodes,key=lambda n:n.node_id).node_id
    for k in commodities:
        shift = num_nodes * (k - 1)
        arcs.append(Arc(i, shift + 1, shift + 2, k, 1, 0, 0))
        i += 1
        arcs.append(Arc(i, shift + 1, shift + 5, k, 1, 0, 0))
        i += 1
        arcs.append(Arc(i, shift + 2, shift + 1, k, 1, 0, 0))
        i += 1
        arcs.append(Arc(i, shift + 2, shift + 3, k, 1, 0, 0))
        i += 1
        arcs.append(Arc(i, shift + 2, shift + 5, k, 1, 0, 0))
        i += 1
        arcs.append(Arc(i, shift + 2, shift + 6, k, 1, 0, 0))
        i += 1
        arcs.append(Arc(i, shift + 3, shift + 2, k, 1, 0, 0))
        i += 1
        arcs.append(Arc(i, shift + 3, shift + 4, k, 1, 0, 0))
        i += 1
        arcs.append(Arc(i, shift + 3, shift + 5, k, 1, 0, 0))
        i += 1
        arcs.append(Arc(i, shift + 3, shift + 6, k, 1, 0, 0))
        i += 1
        arcs.append(Arc(i, shift + 4, shift + 3, k, 1, 0, 0))
        i += 1
        arcs.append(Arc(i, shift + 4, shift + 6, k, 1, 0, 0))
        i += 1
    destinations = [5, 6]
    mutual_arcs = []

    for d in destinations:
        mutual_sublist = []
        for k in commodities:
            for a in arcs:
                base = num_nodes * (k - 1)
                real_id = a.to_node - base
                if real_id == d:
                    mutual_sublist.append(a)
        mutual_arcs.append(mutual_sublist)

    # for n in nodes:
    #     print(n)
    # for a in arcs:
    #     print(a)
    # print("mutual arcs")
    # for ma in mutual_arcs:
    #     print(ma)
    # x = ShippingPointConsolidationCSPSolver(nodes, arcs, [], [])
    print("==============CSP Solving==============")
    print("len(nodes)", len(nodes))
    print("len(arcs)", len(arcs))
    print("len(mutual_arcs)", len(mutual_arcs))
    print("K)", len(commodities))
    print([len(mas) for mas in mutual_arcs])

    x = ShippingPointConsolidationCSPSolver(nodes, arcs, [], mutual_arcs)
    # x = ShippingPointConsolidationCSPSolver(nodes, arcs, [], [])
    # x = ShippingPointConsolidationCSPSolver(nodes, arcs, [], [
    #     [Arc(2, 1, 5, 1, 1, 0, 0), Arc(5, 2, 5, 1, 1, 0, 0), Arc(9, 3, 5, 1, 1, 0, 0)],
    #     [Arc(6, 2, 6, 1, 1, 0, 0), Arc(10, 3, 6, 1, 1, 0, 0), Arc(12, 4, 6, 1, 1, 0, 0)]
    # ])
    x.create_model()


# super simplified two customer scenario to see why mutual restrction is failing/
def example3():
    nodes = [
        Node(1, 1, 5),
        Node(2, 1, 5),
        Node(3, 1, 5),
        Node(4, 1, -10),
        Node(5, 1, -5),
    ]
    arcs = [
        Arc(1, 1, 4, 1, 1, -1, -1),
        Arc(2, 2, 1, 1, 1, -1, -1),
        Arc(3, 2, 4, 1, 1, -1, -1),
        Arc(4, 3, 5, 1, 1, -1, -1),
    ]
    mutuals = [
        [
            Arc(1, 1, 4, 1, 1, -1, -1),
            Arc(3, 2, 4, 1, 1, -1, -1),
        ]
    ]
    x = ShippingPointConsolidationCSPSolver(nodes, arcs, [], mutuals)
    x.create_model()


# 3 failed so falling back to one customer with transshipment.
def example4():
    nodes = [
        Node(1, 1, 5),
        Node(2, 1, 5),
        Node(3, 1, -10),
    ]
    arcs = [
        Arc(1, 1, 3, 1, 1, -1, -1),
        Arc(2, 2, 1, 1, 1, -1, -1)
        # Arc(3,2,3,1,1,-1,-1),
    ]
    mutuals = [[Arc(1, 1, 3, 1, 1, -1, -1), Arc(3, 2, 3, 1, 1, -1, -1)]]
    # x = ShippingPointConsolidationCSPSolver(nodes,arcs,[],[])
    x = ShippingPointConsolidationCSPSolver(nodes, arcs, [], mutuals)
    x.create_model()


if __name__ == "__main__":
    # example1()
    example2()  
    # example3()
    # example4()
