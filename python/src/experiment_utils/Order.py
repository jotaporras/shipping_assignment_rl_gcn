import numpy as np
from network.physical_network import Node


class Order:
    """
    Definition of an order

    :param np.array demand: demand vector of size K.
    :param node shipping_point: the initial node from where it will start
    :param node customer: the final node where it should be delivered
    :param int delivery_time: the time it takes from shipping point to customer.
    """

    demand: np.array  # (k,1)
    shipping_point: Node
    customer: Node
    due_timestep: int

    def __init__(self, demand: np.array, shipping_point, customer, delivery_time, name):
        self.demand = demand
        self.shipping_point = shipping_point
        self.customer = customer
        self.due_timestep = delivery_time
        self.name = name

    def __repr__(self):
        return f"Order(demand={self.demand}, shipping_point={self.shipping_point}, customer={self.customer}, deliveryTime={self.due_timestep})"

    def order_key(self):
        # TODO NOV 5 THIS ASSUMES ONE ORDER PER DAY (which makes sense to me but maybe not Luis)
        return (self.customer.node_id, self.due_timestep)
