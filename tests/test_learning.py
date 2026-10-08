import numpy as np
import pytest
import torch

from qmix_spread.buffer import EpisodeBuffer
from qmix_spread.checkpoint import build_networks
from qmix_spread.config import QMIXConfig
from qmix_spread.env import SpreadEnv
from qmix_spread.learner import QMIXLearner, unroll_agent_network
from qmix_spread.rollout import RandomPolicy, run_episode
from qmix_spread.td_example import check_with_autograd, hand_td_example, real_transition_example
from qmix_spread.train import train


def make_setup(seed=0, n_episodes=8):
    torch.manual_seed(seed)
    config = QMIXConfig(batch_size=4)
    env = SpreadEnv(config)
    agent_net, mixer = build_networks(config, env)
    buffer = EpisodeBuffer(16, env.episode_limit, env.n_agents, env.obs_dim, env.state_dim)
    rng = np.random.default_rng(seed)
    episodes = [run_episode(env, RandomPolicy(2, 5), rng=rng, seed=100 + i) for i in range(n_episodes)]
    for episode in episodes:
        buffer.add(episode)
    return config, env, agent_net, mixer, buffer, rng, episodes


def test_buffer_shapes_and_overwrite():
    config, env, *_, buffer, rng, episodes = make_setup()
    batch = buffer.sample(4, rng)
    assert batch["obs"].shape == (4, 26, 2, 12)
    assert batch["state"].shape == (4, 26, 24)
    assert batch["actions"].shape == (4, 25, 2)
    assert batch["reward"].shape == (4, 25)
    for episode in episodes * 2:  # 24 more episodes into a buffer of 16
        buffer.add(episode)
    assert len(buffer) == 16


def test_buffer_rejects_short_episodes():
    *_, buffer, _, episodes = make_setup(n_episodes=1)
    short = episodes[0]
    short.rewards = short.rewards[:10]
    with pytest.raises(ValueError):
        buffer.add(short)


def test_td_loss_matches_a_manual_computation():
    config, env, agent_net, mixer, buffer, rng, _ = make_setup()
    learner = QMIXLearner(agent_net, mixer, config)
    batch = buffer.sample(4, rng)
    loss, d = learner.compute_td_loss(batch)

    # Chosen Q: pick by plain indexing instead of gather.
    q_all = d["q_all"]
    b, t, i = 2, 7, 1
    assert d["chosen_q"][b, t, i] == q_all[b, t, i, batch["actions"][b, t, i]]

    # Target recomputed for one step: the online network picks the next actions (double Q),
    # the target networks rate them.
    assert config.double_q
    with torch.no_grad():
        target_q = unroll_agent_network(learner.target_agent_net, batch["obs"], batch["actions"])
        next_actions = q_all[b, t + 1].argmax(dim=-1)  # [n]
        next_q = target_q[b, t + 1, torch.arange(2), next_actions]  # [n]
        next_q_tot = learner.target_mixer(next_q, batch["state"][b, t + 1])
        y = batch["reward"][b, t] + config.gamma * next_q_tot  # never terminated in Simple Spread
    assert torch.allclose(d["td_target"][b, t], y, atol=1e-5)
    assert torch.allclose(loss, ((d["td_target"] - d["q_tot"]) ** 2).mean())


def test_training_steps_reduce_the_loss_on_a_fixed_batch():
    config, env, agent_net, mixer, buffer, rng, _ = make_setup()
    learner = QMIXLearner(agent_net, mixer, config)
    batch = buffer.sample(4, rng)
    first = learner.compute_td_loss(batch)[0].item()
    for _ in range(100):
        learner.train_step(batch)
    assert learner.compute_td_loss(batch)[0].item() < first


def test_hand_example_numbers_and_autograd():
    ex = hand_td_example()
    assert ex["q_tot"] == pytest.approx(-10.0)
    assert ex["td_target"] == pytest.approx(-8.6)
    assert ex["td_error"] == pytest.approx(1.4)
    assert ex["new_q_tot"] == pytest.approx(-9.258)
    assert abs(ex["new_td_error"]) < abs(ex["td_error"])
    assert check_with_autograd(ex)["max_abs_difference"] < 1e-9


def test_real_example_moves_q_tot_towards_the_target():
    config, env, agent_net, mixer, _, _, episodes = make_setup()
    before = [p.clone() for p in agent_net.parameters()]
    r = real_transition_example(agent_net, mixer, config, episodes[0], step=5, lr=1e-4)
    # Q_tot moves towards the target without jumping over it.
    assert abs(r["new_td_error"]) < abs(r["td_error"])
    assert np.sign(r["new_td_error"]) == np.sign(r["td_error"])
    assert all(torch.equal(a, b) for a, b in zip(before, agent_net.parameters()))  # original untouched


def test_tiny_training_run_is_deterministic():
    config = QMIXConfig(total_env_steps=1_000, batch_size=4, buffer_size=50, target_update_interval=10,
                        eval_interval_steps=500, n_eval_episodes=2)
    a = train(config, progress=lambda _: None)
    b = train(config, progress=lambda _: None)
    assert a.env_steps == 1_000 and a.episodes == 40
    assert [row["env_steps"] for row in a.eval_log] == [0, 500, 1_000]
    assert a.eval_log == b.eval_log
    assert [row["loss"] for row in a.train_log[3:]] == [row["loss"] for row in b.train_log[3:]]
