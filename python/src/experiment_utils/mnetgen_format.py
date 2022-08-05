"""
This module contains the classes for Nodes, Arcs used in the optimizer with mnetgen format,
 for the solvers.

"""
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

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Arc(%s,%s,%s,%s,%s,%s,%s)" % (
            self.name,
            self.from_node,
            self.to_node,
            self.commodity,
            self.cost,
            self.capacity,
            self.mutual_capacity_id,
        )


class Node:
    def __init__(self, node_id: int, commodity: int, supply: int):
        self.node_id = node_id
        self.commodity = commodity
        self.supply = supply

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Node(%s,%s,%s)" % (self.node_id, self.commodity, self.supply)


class MutualCapacity:
    def __init__(self, mutual_capacity_id: int, capacity: int):
        self.mutual_capacity_id = mutual_capacity_id
        self.capacity = capacity


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
