import matplotlib

matplotlib.use("Agg")  # draw without a screen

import numpy as np

from qmix_spread.config import QMIXConfig
from qmix_spread.env import SpreadEnv
from qmix_spread.plotting import plot_episode_trajectories, plot_td_loss, plot_training_curve
from qmix_spread.rollout import RandomPolicy, run_episode
from qmix_spread.train import read_csv, train, write_csv


def test_all_figures_are_drawn_and_saved(tmp_path):
    config = QMIXConfig(total_env_steps=500, batch_size=4, buffer_size=50, eval_interval_steps=250, n_eval_episodes=2)
    result = train(config, progress=lambda _: None)
    # The curve must also work from logs read back from CSV (all values become floats there).
    write_csv(tmp_path / "train.csv", result.train_log)
    write_csv(tmp_path / "eval.csv", result.eval_log)
    plot_training_curve(read_csv(tmp_path / "train.csv"), read_csv(tmp_path / "eval.csv"), -42.0, "t", tmp_path / "a.png")
    plot_td_loss(result.train_log, "t", tmp_path / "b.png")

    env = SpreadEnv(config)
    rng = np.random.default_rng(0)
    episodes = [run_episode(env, RandomPolicy(2, 5), seed=s, rng=rng, record_positions=True) for s in (1, 2)]
    plot_episode_trajectories([(episodes[0], "a"), (episodes[1], "b")], tmp_path / "c.png")
    for name in ["a.png", "b.png", "c.png"]:
        assert (tmp_path / name).stat().st_size > 10_000
