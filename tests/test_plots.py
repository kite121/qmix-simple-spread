import matplotlib

matplotlib.use("Agg")  # draw without a screen

import numpy as np

from qmix_spread.config import QMIXConfig
from qmix_spread.env import SpreadEnv
from qmix_spread.plotting import (
    plot_episode_trajectories,
    plot_joint_action_q,
    plot_paired_returns,
    plot_td_loss,
    plot_training_curve,
    save_episode_gif,
)
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
    plot_joint_action_q(np.arange(25.0).reshape(5, 5), ["a", "b", "c", "d", "e"], "t", tmp_path / "d.png")
    plot_paired_returns(np.array([-10.0, -20.0]), np.array([-30.0, -40.0]), tmp_path / "e.png")
    for name in ["a.png", "b.png", "c.png", "d.png", "e.png"]:
        assert (tmp_path / name).stat().st_size > 10_000

    video_env = SpreadEnv(config, render_mode="rgb_array")
    with_frames = [run_episode(video_env, RandomPolicy(2, 5), seed=s, rng=rng, record_frames=True) for s in (1, 2)]
    gif = save_episode_gif([(with_frames[0], "a"), (with_frames[1], "b")], tmp_path / "f.gif")
    from PIL import Image

    with Image.open(gif) as image:
        assert image.n_frames == config.max_cycles + 1  # start frame + one frame per step
