"""Simple Spread with a small numpy-friendly interface.

PettingZoo returns dictionaries keyed by agent name ({"agent_0": ..., "agent_1": ...}).
QMIX works with arrays where agent i is row i, so this wrapper converts once, here,
and the rest of the code never sees dictionaries.

What one agent observes (12 numbers for N=2):
    [own velocity (2), own position (2),
     landmark positions relative to the agent (2 landmarks x 2),
     other agent position relative to the agent (1 x 2),
     communication from the other agent (2, always 0: agents are silent here)]
Full state used by the mixing network = observations of all agents glued together (24 numbers).
"""

import numpy as np
from mpe2 import simple_spread_v3

from .config import QMIXConfig

ACTION_NAMES = ["no_action", "move_left", "move_right", "move_down", "move_up"]


class SpreadEnv:
    def __init__(self, config: QMIXConfig, render_mode: str | None = None):
        self._env = simple_spread_v3.parallel_env(
            N=config.n_agents,
            local_ratio=config.local_ratio,
            max_cycles=config.max_cycles,
            continuous_actions=False,  # QMIX picks one of 5 discrete actions
            render_mode=render_mode,
        )
        self.agent_names = list(self._env.possible_agents)
        self.n_agents = len(self.agent_names)
        # int(...): gymnasium gives numpy integers, which JSON and torch.load(weights_only=True) reject
        self.n_actions = int(self._env.action_space(self.agent_names[0]).n)
        self.obs_dim = int(self._env.observation_space(self.agent_names[0]).shape[0])
        self.state_dim = int(self._env.state_space.shape[0])
        self.episode_limit = config.max_cycles

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Start a new episode.

        With a seed the start positions are fixed (used for evaluation).
        Without a seed the env continues its own random stream (used for training).
        Returns obs [n_agents, obs_dim] and state [state_dim].
        """
        obs_dict, _ = self._env.reset(seed=seed)
        return self._stack(obs_dict), self._env.state()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool, bool]:
        """Apply one action per agent.

        Returns next obs [n_agents, obs_dim], next state [state_dim], team reward,
        terminated (the task itself ended) and truncated (time limit of max_cycles reached).
        """
        action_dict = {name: int(a) for name, a in zip(self.agent_names, actions)}
        obs_dict, rewards, terminations, truncations, _ = self._env.step(action_dict)

        # With local_ratio=0 every agent gets the same global reward. QMIX needs exactly
        # one team reward per step, so we check the assumption instead of trusting it.
        reward_values = [rewards[name] for name in self.agent_names]
        assert all(r == reward_values[0] for r in reward_values), (
            f"agents received different rewards {reward_values}; is local_ratio=0?"
        )

        terminated = any(terminations.values())
        truncated = any(truncations.values())
        return self._stack(obs_dict), self._env.state(), reward_values[0], terminated, truncated

    def render(self) -> np.ndarray:
        """RGB frame [700, 700, 3]; needs render_mode="rgb_array"."""
        return self._env.render()

    def positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Absolute positions: agents [n_agents, 2] and landmarks [n_agents, 2] (for plots)."""
        world = self._env.unwrapped.world
        agents = np.array([agent.state.p_pos for agent in world.agents])
        landmarks = np.array([landmark.state.p_pos for landmark in world.landmarks])
        return agents, landmarks

    def close(self) -> None:
        self._env.close()

    def _stack(self, obs_dict: dict) -> np.ndarray:
        return np.stack([obs_dict[name] for name in self.agent_names]).astype(np.float32)
