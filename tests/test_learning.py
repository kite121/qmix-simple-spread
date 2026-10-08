import numpy as np
import pytest
import torch
import torch.nn.functional as F

from qmix_spread.buffer import EpisodeBuffer
from qmix_spread.checkpoint import build_networks
from qmix_spread.config import QMIXConfig
from qmix_spread.env import SpreadEnv
from qmix_spread.learner import QMIXLearner, unroll_agent_network
from qmix_spread.rollout import QMIXPolicy, RandomPolicy, evaluate, run_episode
from qmix_spread.td_example import check_with_autograd, episodes_to_batch, hand_td_example, real_transition_example
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


def test_buffer_shapes():
    config, env, *_, buffer, rng, episodes = make_setup()
    batch = buffer.sample(4, rng)
    assert batch["obs"].shape == (4, 26, 2, 12)
    assert batch["state"].shape == (4, 26, 24)
    assert batch["actions"].shape == (4, 25, 2)
    assert batch["reward"].shape == (4, 25)


def test_full_buffer_replaces_the_oldest_episodes():
    config, env, *_, episodes = make_setup(n_episodes=6)
    buffer = EpisodeBuffer(4, env.episode_limit, env.n_agents, env.obs_dim, env.state_dim)
    for episode in episodes:  # 6 episodes into 4 slots: episodes 4 and 5 replace 0 and 1
        buffer.add(episode)
    assert len(buffer) == 4
    for slot, episode_index in enumerate([4, 5, 2, 3]):
        assert np.array_equal(buffer.obs[slot], episodes[episode_index].obs)
        assert np.array_equal(buffer.actions[slot], episodes[episode_index].actions)


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


def test_td_target_matches_an_independent_step_by_step_computation():
    """Q_tot and the TD target recomputed one episode and one step at a time, with plain indexing,
    and with target networks that differ from the online ones (as after many updates)."""
    config, env, agent_net, mixer, buffer, rng, _ = make_setup()
    learner = QMIXLearner(agent_net, mixer, config)
    torch.manual_seed(123)
    other_agent_net, other_mixer = build_networks(config, env)
    learner.target_agent_net.load_state_dict(other_agent_net.state_dict())
    learner.target_mixer.load_state_dict(other_mixer.state_dict())
    batch = buffer.sample(4, rng)
    _, d = learner.compute_td_loss(batch)

    def q_values_of_episode(net, b):
        hidden, last_action, out = net.init_hidden(1), torch.zeros(1, 2, 5), []
        for t in range(26):
            q, hidden = net(batch["obs"][b, t].unsqueeze(0), last_action, hidden)
            out.append(q[0])  # [n, A]
            if t < 25:
                last_action = F.one_hot(batch["actions"][b, t], 5).float().unsqueeze(0)
        return out

    with torch.no_grad():
        for b in range(4):
            online = q_values_of_episode(learner.agent_net, b)
            target = q_values_of_episode(learner.target_agent_net, b)
            for t in range(25):
                taken = batch["actions"][b, t]
                chosen = torch.stack([online[t][i, taken[i]] for i in range(2)])
                q_tot = learner.mixer(chosen, batch["state"][b, t])
                best_next = online[t + 1].argmax(dim=1)  # double Q: the online network picks ...
                next_q = torch.stack([target[t + 1][i, best_next[i]] for i in range(2)])  # ... the target rates
                y = batch["reward"][b, t] + config.gamma * learner.target_mixer(next_q, batch["state"][b, t + 1])
                assert torch.allclose(d["q_tot"][b, t], q_tot, atol=1e-5)
                assert torch.allclose(d["td_target"][b, t], y, atol=1e-5)


def test_q_values_while_playing_equal_q_values_in_training():
    """The memory, the previous-action input and the batching must be the same in both places."""
    torch.manual_seed(0)
    config = QMIXConfig()
    env = SpreadEnv(config)
    agent_net, _ = build_networks(config, env)
    policy = QMIXPolicy(agent_net)
    rng = np.random.default_rng(0)

    episodes, played_q = [], []
    for seed in [1, 2, 3]:
        q_per_step = []

        class Recorder:  # plays like the policy and remembers its Q values
            def reset(self):
                policy.reset()

            def act(self, obs, epsilon, rng):
                actions = policy.act(obs, epsilon, rng)
                q_per_step.append(policy.last_q)
                return actions

        episodes.append(run_episode(env, Recorder(), epsilon=0.5, seed=seed, rng=rng))  # many non-greedy actions
        played_q.append(torch.stack(q_per_step))  # [T, n, A]

    batch = episodes_to_batch(episodes)
    with torch.no_grad():
        trained_q = unroll_agent_network(agent_net, batch["obs"], batch["actions"])[:, :-1]  # [3, T, n, A]
    assert torch.allclose(trained_q, torch.stack(played_q), atol=1e-6)


def test_agent_q_values_do_not_depend_on_the_other_agent():
    config, env, agent_net, mixer, buffer, rng, _ = make_setup()
    batch = buffer.sample(2, rng)
    changed_obs, changed_actions = batch["obs"].clone(), batch["actions"].clone()
    changed_obs[:, :, 1] += 3.0  # agent 2's observations ...
    changed_actions[:, :, 1] = (changed_actions[:, :, 1] + 1) % 5  # ... and actions
    with torch.no_grad():
        q = unroll_agent_network(agent_net, batch["obs"], batch["actions"])
        q_changed = unroll_agent_network(agent_net, changed_obs, changed_actions)
    assert torch.equal(q[:, :, 0], q_changed[:, :, 0])
    assert not torch.equal(q[:, :, 1], q_changed[:, :, 1])


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


TINY = QMIXConfig(total_env_steps=1_000, batch_size=4, buffer_size=50, target_update_interval=10,
                  eval_interval_steps=500, n_eval_episodes=2)


def test_tiny_training_run_is_deterministic_and_evaluates_greedily():
    a = train(TINY, progress=lambda _: None)
    b = train(TINY, progress=lambda _: None)
    assert a.env_steps == 1_000 and a.episodes == 40
    assert [row["env_steps"] for row in a.eval_log] == [0, 500, 1_000]
    assert a.eval_log == b.eval_log
    assert [row["loss"] for row in a.train_log[3:]] == [row["loss"] for row in b.train_log[3:]]
    # The logged evaluation is the greedy policy (no random actions) on the fixed evaluation seeds.
    greedy = evaluate(SpreadEnv(TINY), QMIXPolicy(a.agent_net), TINY.eval_seeds())
    assert a.eval_log[-1]["mean_return"] == pytest.approx(float(greedy.mean()), abs=1e-12)


def test_target_networks_are_refreshed_every_interval(monkeypatch):
    refreshed_correctly = []
    original = QMIXLearner.update_targets

    def checking_update(self):
        original(self)
        same = [torch.equal(p, q) for p, q in zip(self.agent_net.state_dict().values(),
                                                   self.target_agent_net.state_dict().values())]
        same += [torch.equal(p, q) for p, q in zip(self.mixer.state_dict().values(),
                                                    self.target_mixer.state_dict().values())]
        refreshed_correctly.append(all(same))

    monkeypatch.setattr(QMIXLearner, "update_targets", checking_update)
    train(TINY, progress=lambda _: None)  # 40 episodes, refresh every 10
    assert refreshed_correctly == [True, True, True, True]
