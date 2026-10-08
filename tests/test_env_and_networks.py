import itertools

import numpy as np
import torch

from qmix_spread.config import QMIXConfig
from qmix_spread.env import ACTION_NAMES, SpreadEnv
from qmix_spread.networks import AgentNetwork, QMixer
from qmix_spread.rollout import QMIXPolicy, RandomPolicy, run_episode


def test_environment_matches_the_assignment():
    config = QMIXConfig()
    env = SpreadEnv(config)
    assert (env.n_agents, env.n_actions, env.obs_dim, env.state_dim) == (2, 5, 12, 24)
    assert len(ACTION_NAMES) == env.n_actions

    obs, state = env.reset(seed=3)
    assert obs.shape == (2, 12) and state.shape == (24,)
    # The full state is the observations of both agents glued together.
    assert np.allclose(state, obs.reshape(-1))

    # SpreadEnv.step asserts that both agents receive the same reward on every step.
    episode = run_episode(env, RandomPolicy(2, 5), rng=np.random.default_rng(0), seed=3)
    assert len(episode.rewards) == config.max_cycles == 25
    assert episode.terminated.sum() == 0  # Simple Spread only ends by the time limit
    assert np.all(episode.rewards < 0)  # reward = minus the sum of distances


def test_seeded_reset_gives_the_same_start():
    env = SpreadEnv(QMIXConfig())
    obs_a, _ = env.reset(seed=123)
    env.reset()
    obs_b, _ = env.reset(seed=123)
    assert np.array_equal(obs_a, obs_b)


def test_agent_network_shapes_and_memory():
    net = AgentNetwork(obs_dim=12, n_agents=2, n_actions=5, hidden_dim=64)
    obs = torch.randn(3, 2, 12)
    last_actions = torch.zeros(3, 2, 5)
    q, hidden = net(obs, last_actions, net.init_hidden(3))
    assert q.shape == (3, 2, 5) and hidden.shape == (3, 2, 64)
    # The memory changes the output: same input, different hidden state -> different Q.
    q_again, _ = net(obs, last_actions, hidden)
    assert not torch.allclose(q, q_again)


def test_mixer_accepts_any_leading_shape():
    mixer = QMixer(n_agents=2, state_dim=24, embed_dim=32, state_limit=3.0)
    q_tot = mixer(torch.randn(4, 25, 2), torch.randn(4, 25, 24))
    assert q_tot.shape == (4, 25)


def test_mixer_is_monotonic_in_every_agent_q():
    torch.manual_seed(0)
    mixer = QMixer(n_agents=2, state_dim=24, embed_dim=32, state_limit=3.0)
    agent_qs = (torch.randn(1000, 2) * 20).requires_grad_(True)
    states = torch.randn(1000, 24) * 2
    mixer(agent_qs, states).sum().backward()
    # dQ_tot/dQ_i >= 0 for every sample and every agent.
    assert (agent_qs.grad >= 0).all()


def test_individual_best_actions_give_the_best_joint_action():
    """With a monotonic mixer, greedy per agent = greedy over all 5 x 5 joint actions."""
    torch.manual_seed(1)
    mixer = QMixer(n_agents=2, state_dim=24, embed_dim=32, state_limit=3.0)
    for _ in range(200):
        q_table = torch.randn(2, 5) * 10  # Q_i for every agent and action
        state = torch.randn(24)
        joint = list(itertools.product(range(5), range(5)))
        q_tot = torch.stack([mixer(torch.stack([q_table[0, a1], q_table[1, a2]]), state) for a1, a2 in joint])
        best_joint = joint[int(q_tot.argmax())]
        best_individual = tuple(q_table.argmax(dim=1).tolist())
        assert best_joint == best_individual


def test_policy_uses_only_local_observations():
    torch.manual_seed(0)
    net = AgentNetwork(obs_dim=12, n_agents=2, n_actions=5, hidden_dim=64)
    policy = QMIXPolicy(net)
    actions = policy.act(np.zeros((2, 12), dtype=np.float32))  # no state, no mixer
    assert actions.shape == (2,) and actions.dtype.kind == "i"
    assert policy.last_q.shape == (2, 5)
