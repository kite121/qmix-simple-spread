"""Playing episodes: the QMIX policy, a random policy for comparison, and the episode loop.

Both policies have the same two methods, so the same code can run either of them:
    policy.reset()                          at the start of every episode
    policy.act(obs, epsilon, rng) -> [n]    one action index per agent
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .env import SpreadEnv
from .networks import AgentNetwork


@dataclass
class Episode:
    obs: np.ndarray  # [T+1, n, O]  observation before every step and after the last one
    states: np.ndarray  # [T+1, S]    full state at the same moments
    actions: np.ndarray  # [T, n]
    rewards: np.ndarray  # [T]         shared team reward of every step
    terminated: np.ndarray  # [T]      1.0 if the task itself ended at this step (never in Simple Spread)
    agent_positions: np.ndarray | None = None  # [T+1, n, 2] absolute positions, if recorded
    landmark_positions: np.ndarray | None = None  # [n, 2]
    frames: list | None = None  # rendered RGB images, if recorded

    @property
    def total_reward(self) -> float:
        """Episode return: the sum of the team rewards of all steps."""
        return float(self.rewards.sum())

    @property
    def final_mean_distance(self) -> float:
        """Average distance from a landmark to its nearest agent after the last step.

        With local_ratio=0 the step reward is minus the sum of these distances,
        so this is just -last_reward / number_of_landmarks.
        """
        n_landmarks = self.actions.shape[1]
        return float(-self.rewards[-1] / n_landmarks)


class QMIXPolicy:
    """Decentralised execution of a trained QMIX model.

    Every agent picks its action from its own Q_i, computed from its own observation
    and memory only. The mixing network and the full state are not needed here.
    """

    def __init__(self, agent_net: AgentNetwork):
        self.agent_net = agent_net
        self.n_agents = agent_net.n_agents
        self.n_actions = agent_net.n_actions
        self.reset()

    def reset(self) -> None:
        self.hidden = self.agent_net.init_hidden(1)  # [1, n, H]
        self.last_actions_onehot = torch.zeros(1, self.n_agents, self.n_actions)  # no previous action yet
        self.last_q = None  # Q_i of the latest step [n, A], for inspection (the tests compare it with training)

    @torch.no_grad()
    def act(self, obs: np.ndarray, epsilon: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
        """obs [n, O] -> one action per agent [n].

        epsilon = 0 means fully greedy (no random actions), as in evaluation.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)  # [1, n, O]
        q, self.hidden = self.agent_net(obs_t, self.last_actions_onehot, self.hidden)  # q [1, n, A]
        self.last_q = q[0].clone()  # [n, A]

        actions = q[0].argmax(dim=-1).numpy()  # [n] best action of every agent
        if epsilon > 0.0:
            # Epsilon-greedy, independently for every agent.
            explore = rng.random(self.n_agents) < epsilon
            random_actions = rng.integers(0, self.n_actions, size=self.n_agents)
            actions = np.where(explore, random_actions, actions)

        # F.one_hot([2, 0], 5) = [[0, 0, 1, 0, 0], [1, 0, 0, 0, 0]] -> next step's input
        self.last_actions_onehot = F.one_hot(torch.as_tensor(actions), self.n_actions).float().unsqueeze(0)
        return actions


class RandomPolicy:
    """Every agent picks one of the actions uniformly at random. Baseline for comparison."""

    def __init__(self, n_agents: int, n_actions: int):
        self.n_agents = n_agents
        self.n_actions = n_actions

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, epsilon: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
        return rng.integers(0, self.n_actions, size=self.n_agents)


def run_episode(
    env: SpreadEnv,
    policy,
    epsilon: float = 0.0,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    record_positions: bool = False,
    record_frames: bool = False,
) -> Episode:
    """Play one episode. With a seed the start positions are fixed, so runs are comparable."""
    obs, state = env.reset(seed=seed)
    policy.reset()

    obs_list, state_list = [obs], [state]
    actions_list, reward_list, terminated_list = [], [], []
    positions = [env.positions()[0]] if record_positions else None
    frames = [env.render()] if record_frames else None

    for _ in range(env.episode_limit):
        actions = policy.act(obs, epsilon, rng)
        obs, state, reward, terminated, truncated = env.step(actions)

        obs_list.append(obs)
        state_list.append(state)
        actions_list.append(actions)
        reward_list.append(reward)
        terminated_list.append(float(terminated))
        if record_positions:
            positions.append(env.positions()[0])
        if record_frames:
            frames.append(env.render())
        if terminated or truncated:
            break

    return Episode(
        obs=np.stack(obs_list),
        states=np.stack(state_list).astype(np.float32),
        actions=np.stack(actions_list).astype(np.int64),
        rewards=np.array(reward_list, dtype=np.float64),
        terminated=np.array(terminated_list, dtype=np.float32),
        agent_positions=np.stack(positions) if record_positions else None,
        landmark_positions=env.positions()[1] if record_positions else None,
        frames=frames,
    )


def evaluate(
    env: SpreadEnv, policy, seeds: list[int], epsilon: float = 0.0, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Episode returns of the policy, one per seed (one fixed start position per seed)."""
    return np.array([run_episode(env, policy, epsilon, seed=s, rng=rng).total_reward for s in seeds])
