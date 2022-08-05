import logging
import random
from abc import ABC
from typing import List

import numpy as np
import scipy.stats
from network import physical_network

from experiment_utils.Order import Order


class OrderGenerator(ABC):
    # Generates a set of locations for the next timestep.
    def generate_orders(self, current_t: int) -> List[Order]:
        pass


class NaiveOrderGenerator(OrderGenerator):
    default_delivery_time = 1

    def __init__(self, num_dcs, num_customers, orders_per_day):
        self.num_dcs = num_dcs
        self.num_customers = num_customers
        self.orders_per_day = orders_per_day

    def generate_orders(self):  # TODO: needs a list of commodities, also needs the
        customer = "c_" + str(np.random.choice(np.arange(self.num_customers)))
        dc = "dc_" + str(np.random.choice(np.arange(self.num_dcs)))
        demand = random.randint(0, 50)
        return [
            Order(demand, dc, customer, self.default_delivery_time)
            for it in range(self.orders_per_day)
        ]


class ActualOrderGenerator(OrderGenerator):
    """
    The original is independent means for each product customer.
    """

    network: physical_network
    orders_per_day: int
    logger = logging.getLogger(__name__)

    def __init__(self, network: physical_network, orders_per_day):
        self.network = network
        self.orders_per_day = orders_per_day

    def generate_orders(self, current_t) -> List[Order]:
        return self._generate_orders(self.orders_per_day, current_t)

    def _generate_orders(
        self, orders_per_day: int, current_t
    ):  # TODO test and validate.
        # Choose customers to generate orders with OUT replacement, orders per day must be <= customers
        chosen_customers = np.random.choice(
            np.arange(self.network.num_customers), size=orders_per_day, replace=False
        )
        order_means = self.network.customer_means[chosen_customers]

        demand = np.floor(
            np.random.multivariate_normal(
                order_means,
                np.eye(orders_per_day) * self.network.demand_var,
                size=self.network.num_commodities,
            )
        )  # shape (num_commodities,num_orders)
        if (demand < 0).any():
            self.logger.info("Customer means that caused negatives")
            self.logger.info(order_means)
            # raise Exception("Generated a negative order")
            demand = np.abs(demand)
        # Create order objects
        orders = []
        for ci in range(len(chosen_customers)):
            order_demand_vector = demand[:, ci]
            _chosen_customer = chosen_customers[ci]
            customer_node = self.network.customers[_chosen_customer]
            chosen_initial_point = np.random.choice(
                np.argwhere(self.network.dcs_per_customer_array[ci, :]).reshape(-1)
            )
            initial_point_physical_node = self.network.dcs[chosen_initial_point]
            time = (
                current_t + self.network.planning_horizon - 1
            )  # Orders appear on the edge of PH.
            orders.append(
                Order(
                    order_demand_vector,
                    initial_point_physical_node,
                    customer_node,
                    time,
                    name=f"oc_{customer_node.node_id}:{time}",
                )
            )
        return orders


class BiasedOrderGenerator(OrderGenerator):
    """
    # biased is more skewed and there's correlations in products.
    """

    network: physical_network
    orders_per_day: int
    commodity_means: np.array
    pz_numerator: float  # this is a test

    def __init__(self, network: physical_network, orders_per_day, pz_numerator=1.0):
        self.network = network
        self.orders_per_day = orders_per_day
        self.customer_covariances = (
            self._generate_customer_covariances()
        )  # shape:(C,K,K)
        self.commodity_means = self._generate_commodity_means()
        self.pz_numerator = pz_numerator

    def _generate_customer_covariances(self):
        """

        Returns: A covariance matrix with shape (num_customers,K,K)

        """

        K = self.network.num_commodities
        num_customers = self.network.num_customers
        return (
            scipy.stats.invwishart(K, np.ones(K))
            .rvs(size=num_customers)
            .reshape(num_customers, K, K)
        )

    def _generate_commodity_means(self):
        # total_demand_mean = self.network.demand_mean * self.network.num_customers * self.network.num_commodities
        return np.random.poisson(
            self.network.demand_mean / self.network.num_commodities,
            size=self.network.num_commodities,
        )
        # return np.floor(
        #     np.random.dirichlet(self.network.num_commodities / np.arange(1, self.network.num_commodities + 1),
        #                         size=1) * total_demand_mean).reshape(-1) + self.network.demand_mean # shape (commodities)

    def generate_orders(self, current_t) -> List[Order]:
        # todo params
        chosen_customers = np.random.choice(
            np.arange(self.network.num_customers),
            size=self.orders_per_day,
            replace=False,
        )
        order_means = self.network.customer_means[
            chosen_customers
        ]  # getting the means from the network but the covariances from here for legacy reasons.
        K = self.network.num_commodities

        ####

        # Generating covariance matrix with inverse Wishart distribution. What does that parameter do?
        # Like Chi^2 but high dimensional, for generating covariance matrices.
        covar = scipy.stats.invwishart(K, np.ones(K)).rvs(size=1)

        orders = []
        for ci in range(len(chosen_customers)):
            means = self.commodity_means
            covar = self.customer_covariances[ci, :, :]

            # Sampling X from a multivariate normal with the covariance from Wishart.
            multivariate_normal_x = np.random.multivariate_normal(
                np.zeros(means.shape), covar, size=1
            )

            # Extract the probability density of the sampled values. Is the sqrt(diag(covar)) arbitrary?
            px = scipy.stats.norm(0, np.sqrt(np.diagonal(covar))).cdf(
                multivariate_normal_x
            )

            # Take those quantiles and plug them into a geometric. This is going to skew the data and project it into the range that we want starting at 0.
            # qgeom(x,prob). X is a vector of quantiles of the probability of failures in a Bernoulli (shape K). Second param is probabilities.  Why pz(1-pz)?? Something related to MLE?
            # pz = 1 / means
            # TODO just to check if means are impacting in any way
            pz = self.pz_numerator / means
            order_demand = scipy.stats.geom(p=pz * (1 - pz)).ppf(px).flatten()

            _chosen_customer = chosen_customers[ci]
            customer_node = self.network.customers[_chosen_customer]
            chosen_initial_point = np.random.choice(
                np.argwhere(
                    self.network.dcs_per_customer_array[_chosen_customer, :]
                ).reshape(-1)
            )
            initial_point_physical_node = self.network.dcs[chosen_initial_point]
            time = (
                current_t + self.network.planning_horizon - 1
            )  # Orders appear on the edge of PH.

            orders.append(
                Order(
                    order_demand,
                    initial_point_physical_node,
                    customer_node,
                    time,
                    name=f"oc_{customer_node.node_id}:{time}",
                )
            )
        return orders


class NormalOrderGenerator(BiasedOrderGenerator):
    """
    A makeshift, normal multivariate attempt to reduce the variance by Javier.
    """

    def __init__(self, network: physical_network, orders_per_day):
        super(NormalOrderGenerator, self).__init__(network, orders_per_day, 1.0)

    def generate_orders(self, current_t):
        # todo params
        chosen_customers = np.random.choice(
            np.arange(self.network.num_customers),
            size=self.orders_per_day,
            replace=False,
        )
        order_means = self.network.customer_means[
            chosen_customers
        ]  # getting the means from the network but the covariances from here for legacy reasons.
        K = self.network.num_commodities

        ####

        # Generating covariance matrix with inverse Wishart distribution. What does that parameter do?
        # Like Chi^2 but high dimensional, for generating covariance matrices.
        covar = scipy.stats.invwishart(K, np.ones(K)).rvs(size=1)

        orders = []
        for ci in range(len(chosen_customers)):
            means = self.commodity_means
            covar = self.customer_covariances[ci, :, :] * self.network.demand_var

            # Round down the ints and add 1 to avoid zero demands.
            order_demand = (
                np.random.multivariate_normal(means, covar).astype(int) + 1
            ).astype(float)
            order_demand = np.where(order_demand < 1.0, 1.0, order_demand)

            _chosen_customer = chosen_customers[ci]
            customer_node = self.network.customers[_chosen_customer]
            customer_node_id = customer_node.node_id
            chosen_initial_point = np.random.choice(
                np.argwhere(
                    self.network.dcs_per_customer_array[_chosen_customer, :]
                ).reshape(-1)
            )
            initial_point_physical_node = self.network.dcs[chosen_initial_point]
            time = (
                current_t + self.network.planning_horizon - 1
            )  # Orders appear on the edge of PH.

            orders.append(
                Order(
                    order_demand,
                    initial_point_physical_node,
                    customer_node,
                    time,
                    name=f"oc_{customer_node.node_id}:{time}",
                )
            )
        return orders
