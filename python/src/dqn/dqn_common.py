"""
    Common components to DQN agent training
"""

from collections import deque, namedtuple
from typing import Tuple, Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import IterableDataset
from torch_geometric.data import Batch

Experience = namedtuple(
    "Experience", field_names=["state", "action", "reward", "done", "new_state"]
)


class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """
        Add experience to the buffer

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        # Unpacking the Experiences named tuple
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices]
        )

        if isinstance(states[0], np.ndarray):  
            states = np.array(states)
            next_states = np.array(next_states)
        else:  # its a tuple of arrays from PyG, leave it be.
            states = states
            next_states = next_states

        return (
            states,
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool),
            next_states,
        )


class GeometricReplayBuffer(ReplayBuffer):
    """TODO Oct 19 disposable attempt at adapting the code to PyG DataLoading strategy. Not sure
    if it's going to work or if I'm thinking of stacking in an inappropriate way."""

    def sample(self, batch_size: int):
        regular_sample = super().sample(batch_size)
        states, actions, rewards, dones, next_states = regular_sample
        # tensor_states = [(torch.tensor(n), torch.tensor(e)) for n, e in states]
        # tensor_next_states = [
        #   (torch.tensor(n), torch.tensor(e)) for n, e in next_states
        # ]
        return (
            states,
            torch.tensor(actions),
            torch.tensor(rewards),
            torch.tensor(dones),
            next_states,
        )


class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the ExperienceBuffer
    which will be updated with new experiences during training

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator:
        states, actions, rewards, dones, new_states = self.buffer.sample(
            self.sample_size
        )
        for i in range(len(dones)):
            # yield iter([(states[i], actions[i], rewards[i], dones[i], new_states[i])])
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]


class ShippingFacilityEpisodesDataset(IterableDataset):
    """
    Simple dataset to guide the PTL training based on the number of steps we're going to run.
    It's just a fancy iterator. Deprecated since we unified the dataloaders and stuff.
    """

    def __init__(self, num_steps, orders_per_day) -> None:
        self.num_steps = num_steps
        self.orders_per_day = orders_per_day

    def __iter__(self) -> Iterator:
        for step in range(self.num_steps):
            for num_order in range(self.orders_per_day):
                ep_start = step == 0 and num_order == 0
                yield step, num_order, ep_start


def gnn_rl_collate_fn(batch):
    """
    Converts a batch of experiences into a set of tensors and batches of gnn data.
    Args:
        batch: list[Experience] but the states are torch_geometric.Data

    Returns:

    """
    unzipped_batch = list(zip(*batch))
    states, actions, rewards, done, next_states = unzipped_batch
    batched_states = Batch.from_data_list(states)
    batched_next_states = Batch.from_data_list(next_states)
    return (
        batched_states,
        torch.tensor(actions),
        torch.tensor(rewards),
        torch.tensor(done),
        batched_next_states,
    )
