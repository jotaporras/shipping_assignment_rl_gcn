from typing import List

import numpy as np
import logging

logger = logging.getLogger(__name__)


class Customer:
    def __init__(self, customer_id, allowed_dc_indices):
        self.customer_id = customer_id
        self.customer_name = f"C_{customer_id}"
        self.allowed_dc_indices = allowed_dc_indices
        self.kind = "Customer"

    def __repr__(self):
        return f"Customer(customer_id={self.customer_id}, customer_name={self.customer_name}, allowed_dc_indices={self.allowed_dc_indices})"


class DistributionCenter:
    def __init__(self, dc_id):
        self.dc_id = dc_id
        self.dc_name = f"W_{dc_id}"
        self.kind = "DC"

    def __repr__(self):
        return f"DistributionCenter(dc_id={self.dc_id}, dc_name={self.dc_name})"


class Node:
    """
    Definition of a node

    :param string name: Node identifier, must be unique
    :param int balance: store capacity
    ::param int load: the current load carried
    """

    def __init__(self, id, balance, flow, commodity, kind, location=None, name=""):
        self.node_id = id
        self.balance = balance
        self.flow = flow
        self.commodity = commodity
        self.name = name
        self.kind = kind  # DC or C
        self.location = location  # None if this is a physical node.

    def __repr__(self):
        return self.name + "  b=" + str(int(self.balance))

    # todo define if I need to implement this method (might be good for type inference)
    # def ten_key(self):


class Arc:
    """
    Definition of an arc

    :param string id: arc identifier, must be unique
    :param node tail: Node from where the arc starts
    :param node head: Node from where the arc ends
    :param int cost: cost of transit this arc
    :param int capacity: the load the arc can carry
    """

    arc_id: int
    tail: Node
    head: Node
    cost: int
    capacity: int

    def __init__(self, arc_id: int, tail, head, cost, capacity, commodity, name=""):
        self.arc_id = arc_id
        self.tail = tail
        self.head = head
        self.cost = cost
        self.capacity = capacity
        self.commodity = commodity
        self.name = name

    def transportation_arc(self) -> bool:
        return (
            self.tail.kind == "DC"
            and self.head.kind == "DC"
            and self.tail.location.node_id != self.head.location.node_id
        )

    def inventory_arc(self) -> bool:
        return (
            self.tail.kind == "DC"
            and self.head.kind == "DC"
            and self.tail.location.node_id == self.head.location.node_id
        )

    def to_customer_arc(self) -> bool:
        return self.tail.kind == "DC" and self.head.kind == "C"

    def __repr__(self):
        big_m_marker = (
            "*" if self.cost > 10000 else ""
        )  # todo hacky and ad hoc, won't work if normal costs are set high (but why would u)
        # return f"Arc(arc_id={self.arc_id}, tail={self.tail}, head={self.head}, cost={self.cost}, capacity={self.capacity}, commodity={self.commodity}, name={self.name})"
        return self.name + big_m_marker  # short repr.


class PhysicalNetwork:
    """
    Definition of a physical network, including the graph of DCS and customers

    :param int num_dcs: num of nodes that are not costumers
    :param int num_customers: num of nodes that are costumers
    :param int dcs_per_customer: dcs per costumer
    """

    dcs: List[Node]
    customers: List[Node]
    dcs_debug: List[DistributionCenter]
    customers_debug: List[Customer]
    inventory_dirichlet_parameters: np.array
    planning_horizon: int

    def __init__(
        self,
        num_dcs,
        num_customers,
        dcs_per_customer,
        demand_mean,
        demand_var,
        big_m_factor=10000,  # factor of how customre cost to apply to big m arcs.
        num_commodities=1,
        planning_horizon=5,
    ):
        logger.info("Calling physical network gen")
        # ======= HARDWIRED CONSTANTS RELATED TO TIME AND COSTS =====
        self.default_storage_cost = 1  # TODO HARDWIRED CONSTANTS
        self.default_delivery_time = 3  # TODO HARDWIRED CONSTANTS
        # self.default_dc_transport_cost = 10 #TODO HARDWIRED CONSTANTS
        self.default_dc_transport_cost = 150  # TODO HARDWIRED CONSTANTS
        self.default_customer_transport_cost = 10  # TODO HARDWIRED CONSTANTS
        self.default_inf_capacity = 999999
        # self.big_m_cost = self.default_customer_transport_cost*100000
        self.big_m_cost = self.default_customer_transport_cost * big_m_factor
        self.demand_var = demand_var
        self.demand_mean = demand_mean
        self.planning_horizon = planning_horizon
        assert num_dcs >= dcs_per_customer

        self.num_dcs = num_dcs
        self.num_customers = num_customers
        self.dcs_per_customer = dcs_per_customer
        self.num_commodities = num_commodities
        self.dcs = []
        self.dcs_debug = []
        self.customers = []
        self.customers_debug = []
        self.arcs = []
        self.dcs_per_customer_array: np.array = None
        self.inventory_dirichlet_parameters = None

        self._generate()

    def _generate(self):
        """
        Generates the dcs and customer nodes and the arcs
        """

        # Generate allowed DCs per customer
        base_dc_assignment = np.zeros(self.num_dcs)
        base_dc_assignment[0 : self.dcs_per_customer] = 1

        self.dcs_per_customer_array = np.array(
            [
                np.random.permutation(base_dc_assignment)
                for c in range(self.num_customers)
            ]
        )  # Shape (num_customers,num_dcs)

        # Generates the dcs nodes
        location_id = 0
        for i in range(self.num_dcs):
            self.dcs.append(
                Node(location_id, 0, 0, -1, kind="DC", name="dcs_" + str(location_id))
            )
            self.dcs_debug.append(DistributionCenter(i))
            location_id += 1

        # Generates the customer nodes
        for i in range(self.num_customers):
            self.customers.append(
                Node(location_id, 0, 0, 1, kind="C", name="c_" + str(location_id))
            )
            self.customers_debug.append(
                Customer(i, self.dcs_per_customer_array[i, :])
            )  # this is only for breakpoint debugging, could be used for something else in the future.
            location_id += 1

        # Generates the arcs between dcs and dcs
        arc_id = 0
        for node1 in self.dcs:
            for node2 in self.dcs:
                if node1.node_id != node2.node_id:
                    self.arcs.append(
                        Arc(
                            arc_id,
                            node1,
                            node2,
                            1,
                            0,
                            1,
                            name=str(node1.node_id) + "_to_" + str(node2.node_id),
                        )
                    )
                    arc_id += 1

        # heavymetal distribution.
        total_demand_mean = self.demand_mean * self.num_customers * self.num_commodities
        # Dont remember what this was but is dirichlet with different parameter, maybe this was more skewed.
        # self.demand_mean_matrix = np.floor(
        #     np.random.dirichlet(self.num_customers / np.arange(1, self.num_customers + 1),
        #                         size=1) * total_demand_mean).reshape(-1) #(cust)
        self.demand_mean_matrix = np.floor(
            np.random.dirichlet([5.0] * self.num_customers, size=1) * total_demand_mean
        ).reshape(
            -1
        )  # (cust)

        # Generate distribution parameters for the customers.
        # self.customer_means = np.random.poisson(self.demand_mean, size=self.num_customers)
        self.customer_means = (
            np.floor(
                np.random.dirichlet(
                    self.num_customers / np.arange(1, self.num_customers + 1), size=1
                )
                * total_demand_mean
            ).reshape(-1)
            + self.demand_mean
        )  # (cust) #sum mean at the end to avoid negs.

        logger.info(f"Current customer means")
        logger.info(self.customer_means)

        # Parameters for inventory distribution hardwired for now.
        self.inventory_dirichlet_parameters = [
            5.0
        ] * self.num_dcs  # todo deprecated not used

        # Generate a random arc between dcs and costumers
        for cid in range(self.num_customers):
            # counter = 0
            # nodeDc = np.random.choice(self.dcs, size=self.dcs_per_customer, replace=False)
            customer_node = self.customers[cid]
            for dc in np.argwhere(self.dcs_per_customer_array[cid, :] > 0).reshape(-1):
                dcO_node = self.dcs[dc]
                cost = self.default_customer_transport_cost
                capacity = self.default_inf_capacity
                self.arcs.append(
                    Arc(
                        arc_id,
                        dcO_node,
                        customer_node,
                        cost,
                        capacity,
                        -1,
                        name=f"{dcO_node.name}_to_{customer_node.name}",
                    )
                )
                arc_id += 1
                # counter += 1
        # After generation, create utility physical adjacency matrix for GNN methods/
        self.physical_adjacency_matrix = self.generate_physical_adjacency_matrix()

    def is_valid_arc(self, dc, customer):
        base_customer_id = customer - self.num_dcs
        return self.dcs_per_customer_array[base_customer_id, dc] == 1

    def get_valid_dcs(self, customer_node_id):
        """Returns a tuple where the first element are the ids of the dcs with a 1 on dcs_per_customer_array"""
        base_customer_id = customer_node_id - self.num_dcs
        return np.where(
            self.dcs_per_customer_array[base_customer_id, :] == 1
        )  # TODO: would be more user friendly if it returned nodes instead of ids. Also, np where is returing a tuple, kind of unexpected.

    def generate_physical_adjacency_matrix(self):
        """
        Generates an adjacency matrix of DC->Customer Arcs in the
        expected format by PyG.
        Returns: A (2, num_arcs) matrix specifying valid movements
        #TODO test.
        """
        arcs = []
        for dc_id in range(self.num_dcs):
            for j in range(self.num_customers):
                customer_id = self.num_dcs + j  # first customer id is after first dc.
                if self.dcs_per_customer_array[j, dc_id]:  # if valid arc
                    arcs.append(
                        [dc_id, customer_id]
                    )  # verified manually that it works with 3 dcs and 5 custs.
        return np.array(arcs).T

    def get_customer_id(self, customer_node_id):
        """Customer ID is a zero based indicator of the customer. It's node_id=-num_wdcs"""
        # TODO refactor all uses of customer - self.num_dcs and test
        return customer_node_id - self.num_dcs

    def _log_summary_of_network(self):
        logger.debug("Generated physical network. Summary")
        logger.debug("Valid DCs")
        logger.debug(self.dcs_per_customer_array)
        # TODO these no longer apply, they're generated independently within inventorygen
        # TODO cleanup
        # logger.debug("Customer means")
        # logger.debug(self.customer_means)
        # logger.debug("Customer variances")
        # logger.debug(self.demand_var)
        # logger.debug("Inventory dirichlet params")
        # logger.debug(self.inventory_dirichlet_parameters)

    def __repr__(self):
        return str(self.__dict__)
