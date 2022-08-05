import torch
from envs import network_flow_env_builder
import numpy as np
from torch import Tensor
from torch_geometric.nn import GCNConv


class DemoEnvGCN(torch.nn.Module):
    def __init__(self, in_channels=2, hidden_channels=16, out_channels=3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = x.float()
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


def run():
    environment_config = {
        "num_dcs": 3,
        "num_customers": 5,
        "num_commodities": 2,
        "orders_per_day": 2,
        "dcs_per_customer": 2,
        "demand_mean": 500,
        "demand_var": 150,
        "num_steps": 10,  # steps per episode
        "big_m_factor": 10000,  # how many times the customer cost is the big m.
        "version": "v2",
        "order_generator": "biased",
        "reward_function": "negative_log_cost_minus_log_big_m_units",
    }

    env_instance = network_flow_env_builder.build_next_gen_network_flow_environment(
        environment_config,
        5,
        environment_config["order_generator"],
        environment_config["reward_function"],
    )
    env_instance.reset()
    state, reward, done, info = env_instance.step(0)
    pn = state.physical_network
    # Generate a matrix of the current env.
    inventory = env_instance.inventory  # (dc,commodity)
    demand = state.open[0].demand  # (K)
    print(inventory.shape)
    print(demand.shape)
    demand_node_feat = demand.reshape(1, -1) * -1  # Demand as negative
    customer_nodes = np.zeros((pn.num_customers, pn.num_commodities))
    customer_id = pn.get_customer_id(state.open[0].customer.node_id)
    customer_nodes[customer_id, :] = demand_node_feat

    # Node features: inventory availabilities, then customer demands.
    node_features = np.vstack((inventory, customer_nodes))

    # physical network arcs. This should be done only once.
    edge_index = pn.physical_adjacency_matrix
    import torch_geometric.data

    # TODo implement a dataset that loads stuff like this.
    # torch_geometric.data.Data(x=node_features,edge_index=edge_index,)
    model = DemoEnvGCN()
    output = model(torch.tensor(node_features), torch.tensor(edge_index))
    print(output)


if __name__ == "__main__":
    run()
