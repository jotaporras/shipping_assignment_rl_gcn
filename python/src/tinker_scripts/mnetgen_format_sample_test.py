"""
A script to write a MCF problem into mnetgen to run with Fragioni's solver.
Eventually gave up on it because couldn't get the solver to run.

Test the format of mnetgen.C by Antonio Fragioni. Copy from there:

  The output file formats are (each record, fields tab-separated):

  Mutual capacity file (*.mut):

  < mutual capacity pointer > , < mutual capacity >

  Arc file (*.arc):

  < arc name > , < from node > , < to node > , < commodity > , < cost > ,
  < capacity > , < mutual capacity pointer >

  Arc name is an integer between 1 and the number of arcs (differently from
  the original mnetgen format), that is necessary to distinguish between
  multiple instances of an arc (i, j) for the same commodity, that are
  permitted

  Node supply file (*.nod if FOUR_F == 0, *.sup otherwise):

  < node > , < commodity > , < supply >

  Problem description file (*.nod, only if FOUR_F == 1)

  < commodities > , < nodes > , < arcs > , < capacitated arcs >
"""

# Recreate the sample from Network flows by Orlin
from typing import List


class Arc:
    def __init__(
        self,
        name: int,
        from_node: int,
        to_node: int,
        commodity: int,
        cost: float,
        capacity: int,
        mutual_capacity_id: int,
    ):
        self.name = name
        self.from_node = from_node
        self.to_node = to_node
        self.commodity = commodity
        self.cost = cost
        self.capacity = capacity
        self.mutual_capacity_id = mutual_capacity_id


class Node:
    def __init__(self, node_id: int, commodity: int, supply: int):
        self.node_id = node_id
        self.commodity = commodity
        self.supply = supply


class MutualCapacity:
    def __init__(self, mutual_capacity_id: int, capacity: int):
        self.mutual_capacity_id = mutual_capacity_id
        self.capacity = capacity


# class ProblemDescription:
#     def __init__(self, commodities: int, nodes: int, arcs: int, capacitated_arcs: int):
#         self.commodities = commodities
#         self.nodes = nodes
#         self.arcs = arcs
#         self.capacitated_arcs = capacitated_arcs


class MnetgenFormatWriter:
    __SEP = "\t"

    def __init__(
        self, nodes: List[Node], arcs: List[Arc], capacities: List[MutualCapacity]
    ):
        self.nodes = nodes
        self.arcs = arcs
        self.capacities = capacities

    def write(self, dir: str, filename: str):
        arc_lines = self.__arc_lines()
        node_lines = self.__node_lines()
        mutual_capacity_lines = self.__mutual_capacity_lines()
        summary_lines = self.__nod_lines()

        self.__write_lines(arc_lines, f"{dir}/{filename}.arc")
        self.__write_lines(node_lines, f"{dir}/{filename}.sup")
        self.__write_lines(mutual_capacity_lines, f"{dir}/{filename}.mut")
        self.__write_lines(summary_lines, f"{dir}/{filename}.nod")

    #   Arc file (*.arc):
    #
    #   < arc name > , < from node > , < to node > , < commodity > , < cost > ,
    #   < capacity > , < mutual capacity pointer >
    def __arc_lines(self):
        SEP = self.__SEP
        arc_lines = []
        for a in self.arcs:
            arc_lines.append(
                f"{a.name}{SEP}{a.from_node}{SEP}{a.to_node}{SEP}{a.commodity}{SEP}{a.cost}{SEP}{a.capacity}{SEP}{a.mutual_capacity_id}"
            )
        return arc_lines

    #   Node supply file (*.nod if FOUR_F == 0, *.sup otherwise):
    #
    #   < node > , < commodity > , < supply >
    def __node_lines(self):
        SEP = self.__SEP
        node_lines = []
        for n in self.nodes:
            node_lines.append(f"{n.node_id}{SEP}{n.commodity}{SEP}{n.supply}")
        return node_lines

    #   Mutual capacity file (*.mut):
    #
    #   < mutual capacity pointer > , < mutual capacity >
    def __mutual_capacity_lines(self):
        SEP = self.__SEP
        mc_lines = []
        for mc in self.capacities:
            mc_lines.append(f"{mc.mutual_capacity_id}{SEP}{mc.capacity}")
        return mc_lines

    def __write_lines(self, ds: List[str], _writedir: str):
        with open(_writedir, "w+") as f:
            for i, line in enumerate(ds):
                if i != len(ds) - 1:
                    f.write("%s\n" % line)
                else:
                    f.write("%s" % line)

    def __nod_lines(self):
        SEP = self.__SEP
        commodities = len(set([node.commodity for node in self.nodes]))
        nodes = len(self.nodes)
        arcs = len(self.arcs)
        capacitated = sum(
            [1 for arc in self.arcs if arc.mutual_capacity_id != 0]
        )  # first bundle is for uncapacitateds.
        nod_line = f"{commodities}{SEP}{nodes}{SEP}{arcs}{SEP}{capacitated}"
        print(f"nod_line {nod_line}")
        return [nod_line]


def test_write():
    supplies_p1 = [10, -10, 0, 0, 0, 0]
    supplies_p2 = [0, 0, 0, 0, 20, -20]
    ic = 1000  # infinite capacity
    arcs = [
        {"tail": 0, "head": 1, "cost": 1, "capacity": 5},
        {"tail": 0, "head": 2, "cost": 5, "capacity": ic},
        {"tail": 2, "head": 3, "cost": 1, "capacity": 10},
        {"tail": 3, "head": 1, "cost": 5, "capacity": ic},
        {"tail": 3, "head": 5, "cost": 1, "capacity": ic},
        {"tail": 4, "head": 2, "cost": 1, "capacity": ic},
        {"tail": 4, "head": 5, "cost": 5, "capacity": ic},
    ]

    bundle_capacities = {
        "-1": {"cap": ic * 100, "id": 0},
        "01": {"cap": 5, "id": 1},
        "23": {"cap": 10, "id": 2},
    }

    def bundle_id(arc):
        key = f"{arc['tail']}{arc['head']}"
        default_key = "-1"
        if key in bundle_capacities:
            return bundle_capacities[key]["id"]
        else:
            return bundle_capacities[default_key]["id"]

    num_arcs = len(arcs)
    # mnetgen_arcs = [Arc(i + (commodity-1)*num_arcs+1, a['tail']+1, a['head']+1, commodity, a['cost'], ic, bundle_id(a)) for commodity in [1, 2] for i, a in enumerate(arcs) ]

    # mmnetgen_nodes = [Node(i+1, 1, supplies_p1[i]) for i, a in enumerate(supplies_p1)] + [Node(i + len(supplies_p1)+1, 2, supplies_p2[i])
    #                                                                                       for i, a in
    #                                                                                       enumerate(supplies_p2)]

    # TODO single commodity debug deleteme
    mnetgen_arcs = [
        Arc(
            i + (commodity - 1) * num_arcs + 1,
            a["tail"] + 1,
            a["head"] + 1,
            commodity,
            a["cost"],
            ic,
            bundle_id(a),
        )
        for commodity in [1]
        for i, a in enumerate(arcs)
    ]

    mmnetgen_nodes = [Node(1, 1, 0)] + [
        Node(i + 2, 1, supplies_p1[i]) for i, a in enumerate(supplies_p1)
    ]
    # TODO single commodity debug deleteme

    mnetgen_mutual_capacities = [
        MutualCapacity(capacity["id"], capacity["cap"])
        for k, capacity in bundle_capacities.items()
    ]
    writer = MnetgenFormatWriter(
        mmnetgen_nodes, mnetgen_arcs, mnetgen_mutual_capacities
    )
    writer.write("/Users/aleph/Documents/jota/tesis/MMCFCoin/Main", "orlin_mcf0")


if __name__ == "__main__":
    import os
    import pathlib

    if not os.path.exists("../../data"):
        os.makedirs("../../data")
    print(pathlib.Path().absolute())
    test_write()
