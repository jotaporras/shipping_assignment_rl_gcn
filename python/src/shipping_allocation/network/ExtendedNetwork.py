import logging
from typing import List

from experiment_utils.Order import Order
from network.physical_network import PhysicalNetwork, Arc, Node
import numpy as np

DEBUG = False


class TENNode(Node):
    time: int

    def __init__(self, id, balance, flow, commodity, kind, time, location, name):
        super().__init__(id, balance, flow, commodity, kind, location, name)

    def __init__(self, id, balance, flow, commodity, kind, time, location, name):
        super().__init__(id, balance, flow, commodity, kind, location, name)
        self.time = time

    def ten_key(self):
        """
        If it's a TEN arc, return a tuple with location and time, used to group
        all repeated nodes of a commodity on a TEN problem (B&B solver)
        Returns:
            ten_key: tuple
                A tuple of (phys_node_id,time)
        """

        return f"{self.location.kind}{self.location.node_id}:{self.time}"


class ExtendedNetwork:
    """
    Definition of a Time Expanded Network

    :param Network network:
    :param Orders fixed_orders:

    """

    fixed_orders: List[Order]
    open_orders: List[Order]
    network: PhysicalNetwork
    inventory: np.array
    DUMMY_NAME: str = "DUMMY"

    def __init__(self, network, inventory, fixed_orders, open_orders=[]):
        self.network = network
        self.inventory = inventory
        self.fixed_orders = fixed_orders
        self.open_orders = open_orders
        self.location_time_nodemap = (
            {}
        )  # Map from (location,t) => Node if the node is relevant to this optimization.

    def convert_to_extended(
        self, current_t, planning_horizon_t, generate_all_inbound_customer_arcs=False
    ):
        return self._generate_nodes(current_t, planning_horizon_t), self._generate_arcs(
            current_t,
            planning_horizon_t,
            generate_all_inbound_customer_arcs=generate_all_inbound_customer_arcs,
        )

    def _generate_nodes(self, current_t, planning_horizon_t):
        # Creates all the dcs nodes
        nodes = []
        node_id_inc = 0
        total_time = self.network.planning_horizon

        self.location_time_nodemap = (
            {}
        )  # todo verify this on mulitple calls? Does this even get called multiple times? I think it's a disposable object.

        num_commodities = self.network.num_commodities
        for k in range(num_commodities):
            for dc in self.network.dcs:
                # for t in range(total_time):
                for t in range(current_t, planning_horizon_t + 1):
                    expanded_node_id = node_id_inc
                    dc_k_inventory = (
                        self.inventory[dc.node_id, k] if t == current_t else 0
                    )  # inventory is (dc,k)
                    node = TENNode(
                        expanded_node_id,
                        dc_k_inventory,
                        0,
                        k,
                        dc.kind,
                        t,
                        dc,
                        name=f"{expanded_node_id}__{dc.name}^{k}:{t}",
                    )
                    nodes.append(node)

                    self.location_time_nodemap.setdefault(
                        (dc.node_id, t), [None] * num_commodities
                    )[
                        k
                    ] = node  # Todo check this map after assignment.

                    node_id_inc += 1

        # Creates all the customer nodes.
        for order in self.fixed_orders + self.open_orders:
            if current_t <= order.due_timestep <= planning_horizon_t:
                for k in range(num_commodities):
                    k_demand = order.demand[k]  # type hints go crazy here
                    if (
                        k_demand > 0
                    ):  # only generate nodes for commodities that have it.
                        KIND = "C"
                        node = TENNode(
                            node_id_inc,
                            -k_demand,
                            0,
                            k,
                            KIND,
                            order.due_timestep,
                            order.customer,
                            name=f"{node_id_inc}__{order.name}^{k}:{order.due_timestep}",
                        )
                        nodes.append(node)
                        location_time_key = (order.customer.node_id, order.due_timestep)
                        self.location_time_nodemap.setdefault(
                            location_time_key, [None] * num_commodities
                        )[
                            k
                        ] = node  # Todo check this map after assignment.
                        node_id_inc += 1

        # TODO theres something wrong with this calculation because the balancer doesnt complain. probably has to do with time windows.
        # if DEBUG:
        #     balances = np.sum(self.inventory,axis=0)-sum(map(lambda o: o.demand, self.fixed_orders))
        #     #print("balances",balances)
        #     if np.sum(balances)!=0:
        #         print("SOME BALANCES NOT 0", np.argwhere(balances != 0))
        #         print(balances)

        # Sum inventory and demand vectors for imbalances
        balances = np.sum(self.inventory, axis=0) - sum(
            map(
                lambda o: o.demand,
                [
                    o
                    for o in self.fixed_orders + self.open_orders
                    if current_t <= o.due_timestep <= planning_horizon_t
                    and o.demand[k] > 0
                ],
            )
        )

        # Add the slack.
        for k in range(num_commodities):
            KIND = self.DUMMY_NAME
            dummy = Node(-1, 0, 0, 0, KIND)
            node = TENNode(
                node_id_inc,
                -balances[k],
                0,
                k,
                KIND,
                current_t,
                dummy,
                name=f"{node_id_inc}__{dummy}^{k}:{current_t}",
            )
            nodes.append(node)
            location_time_key = (
                self.DUMMY_NAME,
                current_t,
            )  # todo maybe replace dummy name with a node id for dummy. And add to physical network.
            self.location_time_nodemap.setdefault(
                location_time_key, [None] * num_commodities
            )[k] = node
            node_id_inc += 1

        return nodes

    def _generate_arcs(
        self,
        current_t,
        planning_horizon_t,
        dummy_cost=1,
        generate_all_inbound_customer_arcs=False,
    ):
        """Generates the Time Expanded Network Arcs. generate_all_inbound_customer_arcs flag is
        set to false when doing environment optimization (take only the order's shipping point) or
        True when doing B&B (consider all customer's valid shipping points)
        """
        arcs = []
        arcsToCostumers = []

        planning_horizon = self.network.planning_horizon
        num_commodities = self.network.num_commodities
        network = self.network
        # Generate the arcs to the same dcs across t
        arc_id_incr = 0
        for dc_node in self.network.dcs:
            for k in range(num_commodities):
                # for t in range(0, planning_horizon - 1):
                for t in range(current_t, planning_horizon_t):
                    tail = self.location_time_nodemap[(dc_node.node_id, t)][k]
                    head = self.location_time_nodemap[(dc_node.node_id, t + 1)][k]

                    cost = self.network.default_storage_cost
                    capacity = self.network.default_inf_capacity

                    ten_arc = Arc(
                        arc_id_incr,
                        tail,
                        head,
                        cost,
                        capacity,
                        k,
                        name=f"{tail.name}=>{head.name}",
                    )
                    arcs.append(ten_arc)
                    arc_id_incr += 1
        # Generates the arcs beetwen differentes DCs over time. DCs can be connected on the same t (could be hardwired to more duration.
        transport_duration = 0  # TODO hardwired and not implemented
        # TODO this is wrong because it's connecting based on physical network, will add all possible paths to orders when they are fixed.
        # It should only add all possible arcs between DC-Customer if they are open orders. Consider skipping orders altogether in this loop and do on next/
        # Otherwise,. remove next loop and add filters for only valid arcs for orders.
        # ==========CONNECTING PHYSICAL ARCS OF DCS==========
        for physical_arc in self.network.arcs:
            for k in range(num_commodities):
                for t in range(current_t, planning_horizon_t + 1):
                    physical_tail = physical_arc.tail
                    physical_head = physical_arc.head

                    # if tail or head are not in location time nodemap, or if head is a customer??? (
                    if (
                        (physical_head.node_id, t)
                        not in self.location_time_nodemap.keys()
                        or (physical_tail.node_id, t)
                        not in self.location_time_nodemap.keys()
                        or physical_head.kind == "C"
                    ):
                        continue

                    tail = self.location_time_nodemap[(physical_tail.node_id, t)][k]
                    head = self.location_time_nodemap[(physical_head.node_id, t)][k]

                    transport_cost = network.default_dc_transport_cost
                    capacity = (
                        self.network.default_inf_capacity
                    )  # todo i think the physical network was meant to have a capacity?
                    arc_name = f"{tail.name}=>{head.name}"

                    ten_arc = Arc(
                        arc_id_incr,
                        tail,
                        head,
                        transport_cost,
                        capacity,
                        k,
                        name=arc_name,
                    )

                    # nodeFrom = Node(physical_arc.tail.arc_id + "_" + str(t), physical_arc.tail.demand, physical_arc.tail.flow)
                    # nodeTo = Node(physical_arc.head.arc_id + "_" + str(t + physical_arc.cost), physical_arc.head.demand, physical_arc.head.flow)
                    # newArc = Arc(nodeFrom.id +"_to_" + nodeTo.id, nodeFrom, nodeTo, physical_arc.cost, physical_arc.demand)
                    if physical_head.kind == "C":
                        if (
                            physical_head.node_id,
                            t,
                        ) in self.location_time_nodemap.keys():  # if there is a
                            # print("Adding an order arc",ten_arc)
                            arcs.append(ten_arc)
                            # arcsToCostumers.append(ten_arc)
                    else:
                        arcs.append(
                            ten_arc
                        )  # TODO I THINK IT ALWAYS SHOULD BE ADDED TO ARCS BUT LET'S SEE WHY THIS LOGIC PREVAILED.

                        # TODO DELETE THIS WHEN PREVIOUS TODO OF ARCS AND ARCSTOCUSTOMERS IS RESOLVED.
                    # if physical_arc.cost + t < self.fixed_orders.totalTime and physical_arc.head.arc_id[0] != 'c': #TODO i think i dont need this pruning.
                    #     arcs.append(newArc)
                    # elif physical_arc.head.arc_id[0] == 'c':
                    #     arcsToCostumers.append(newArc)
                    arc_id_incr += 1

        for order in self.fixed_orders:
            for k in range(num_commodities):
                if (  # if is an order inside of horizon.
                    current_t <= order.due_timestep <= planning_horizon_t
                    and order.demand[k] > 0
                ):
                    arc = self._create_arc_for_order_at_k(
                        order, order.shipping_point, k, arc_id=arc_id_incr
                    )
                    arcs.append(arc)
                    arc_id_incr += 1

        for order in self.open_orders:
            dcs_to_connect = []
            if generate_all_inbound_customer_arcs:
                dc_ids = network.get_valid_dcs(order.customer.node_id)[0].tolist()
                dcs_to_connect = [network.dcs[dc_i] for dc_i in dc_ids]
            else:
                dcs_to_connect = [order.shipping_point]
            for k in range(num_commodities):
                if (  # if is an order inside of horizon.
                    current_t <= order.due_timestep <= planning_horizon_t
                    and order.demand[k] > 0
                ):
                    for dc in dcs_to_connect:
                        arc = self._create_arc_for_order_at_k(
                            order, dc, k, arc_id=arc_id_incr
                        )
                        arcs.append(arc)
                        arc_id_incr += 1

        # Add arcs to dummy nodes
        for dc_node in self.network.dcs:
            for k in range(num_commodities):
                # Get the DC at current_t
                tail = self.location_time_nodemap[(dc_node.node_id, current_t)][k]
                head = self.location_time_nodemap[(self.DUMMY_NAME, current_t)][k]
                cost = dummy_cost  # todo maybe extract to the network.

                arc = Arc(
                    arc_id_incr,
                    tail,
                    head,
                    cost,
                    network.default_inf_capacity,
                    k,
                    name=f"{tail.name}=>{head.name}",
                )
                arcs.append(arc)
                arc_id_incr += 1

        return arcs

    def _create_arc_for_order_at_k(self, order, dc, k, arc_id):
        """Create an arc from a customer node to a dc node"""
        # todo maybe modify to timestep-1
        tail = self.location_time_nodemap[(dc.node_id, order.due_timestep)][k]
        # TODO: transportation duration is hardcoded to zero here.
        head = self.location_time_nodemap[
            (
                order.customer.node_id,
                order.due_timestep,
            )
        ][k]
        valid_connection = self.network.is_valid_arc(dc.node_id, order.customer.node_id)
        cost = (
            self.network.default_customer_transport_cost
            if valid_connection
            else network.big_m_cost
        )
        return Arc(
            arc_id,
            tail,
            head,
            cost,
            self.network.default_inf_capacity,
            k,
            name=f"{tail.name}=>{head.name}",
        )
