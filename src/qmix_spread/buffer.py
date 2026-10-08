"""Replay buffer that stores whole episodes.

The agent network is recurrent, so it has to be trained on whole episodes in order
(the memory at step t depends on steps 0..t-1). Every Simple Spread episode has exactly
max_cycles = 25 steps, so episodes can be stored in fixed-size arrays without padding.
"""

import numpy as np
import torch

from .rollout import Episode


class EpisodeBuffer:
    def __init__(self, capacity: int, episode_limit: int, n_agents: int, obs_dim: int, state_dim: int):
        T = episode_limit
        self.capacity = capacity
        self.episode_limit = episode_limit
        # Observations and states have T + 1 entries: the one after the last step is
        # needed for the TD target of the last step.
        self.obs = np.zeros((capacity, T + 1, n_agents, obs_dim), dtype=np.float32)
        self.state = np.zeros((capacity, T + 1, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, T, n_agents), dtype=np.int64)
        self.reward = np.zeros((capacity, T), dtype=np.float32)
        self.terminated = np.zeros((capacity, T), dtype=np.float32)
        self._next = 0  # where the next episode goes (oldest is overwritten when full)
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, episode: Episode) -> None:
        if len(episode.rewards) != self.episode_limit:
            raise ValueError(
                f"expected an episode of exactly {self.episode_limit} steps, got {len(episode.rewards)}; "
                "this buffer has no padding, so episodes must not end early"
            )
        i = self._next
        self.obs[i] = episode.obs
        self.state[i] = episode.states
        self.actions[i] = episode.actions
        self.reward[i] = episode.rewards
        self.terminated[i] = episode.terminated
        self._next = (self._next + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        """Random episodes (without repeats) as torch tensors with batch as the first axis."""
        idx = rng.choice(self._size, size=batch_size, replace=False)
        return {
            "obs": torch.from_numpy(self.obs[idx]),  # [B, T+1, n, O]
            "state": torch.from_numpy(self.state[idx]),  # [B, T+1, S]
            "actions": torch.from_numpy(self.actions[idx]),  # [B, T, n]
            "reward": torch.from_numpy(self.reward[idx]),  # [B, T]
            "terminated": torch.from_numpy(self.terminated[idx]),  # [B, T]
        }
