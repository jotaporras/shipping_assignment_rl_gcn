import torch_geometric.loader
from envs import shipping_assignment_state
from shipping_allocation import PhysicalNetwork

from agents.pytorch_agents import PhysnetAggDemandGCNAgent, PhysnetAggDemandGCN
from dqn import dqn_common
from dqn.dqn_common import RLDataset, ReplayBuffer, Experience, GeometricReplayBuffer
import numpy as np
import torch
import torch.utils.data
from torch_geometric.data import Data


def _random_state(num_customers, num_commodities, num_dcs, num_edges):
    # State for GNN: Tuple of two matrices: (C,K) features, and (E,2) adjacency Returns torch_geometric.Data
    num_nodes = num_customers + num_dcs
    node_features = np.random.randn(num_nodes, num_commodities)
    adjacencies = np.random.randint(low=0, high=num_nodes, size=(2, num_edges))
    state = Data(x=torch.tensor(node_features), edge_index=torch.tensor(adjacencies))
    return state


def test_gnn_dataloading():
    # integration test to make sure I have the pyg shapes in the correct fashion.
    # Given
    buffer = GeometricReplayBuffer(10)
    num_customers = 5
    num_dcs = 3
    num_commodities = 1
    num_edges = 6

    # generating random experience
    for i in range(10):
        state = _random_state(num_customers, num_commodities, num_dcs, num_edges)
        new_state = _random_state(num_customers, num_commodities, num_dcs, num_edges)
        action = np.random.randint(low=0, high=num_dcs)
        done = False
        reward = np.random.randn()
        buffer.append(Experience(state, action, reward, done, new_state))

    ds = RLDataset(buffer, sample_size=5)
    dataloader = torch.utils.data.DataLoader(
        dataset=ds, batch_size=2, collate_fn=dqn_common.gnn_rl_collate_fn
    )

    # When
    elem = next(iter(dataloader))
    print(elem)

    # Then


def test_gnn_dataloading_pygdataset():
    # integration test to make sure I have the pyg shapes in the correct fashion.
    # Given
    buffer = GeometricReplayBuffer(10)
    num_customers = 5
    num_dcs = 3
    num_commodities = 1
    num_edges = 6

    # generating random experience
    state_datas = []
    for i in range(100):
        state = _random_state(num_customers, num_commodities, num_dcs, num_edges)
        state_data = Data(x=torch.tensor(state[0]), edge_index=torch.tensor(state[1]))
        state_datas.append(state_data)

    dataloader = torch_geometric.loader.DataLoader(
        dataset=state_datas,
        batch_size=5,
    )

    # When
    elem = next(iter(dataloader))
    
    print(elem)
    network = PhysnetAggDemandGCN(num_commodities, num_dcs, num_customers)
    network(elem)

    # Then
