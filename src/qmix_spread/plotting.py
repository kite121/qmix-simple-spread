"""Figures for the notebook and the slides (static PNG, light background)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .rollout import Episode

# Colors: categorical slots 1 and 2 of a colorblind-checked palette, plus neutral text/grid tones.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#ffffff"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.title.set_color(TEXT)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Mean of the last `window` values at every point (shorter window at the start)."""
    values = np.asarray(values, dtype=float)
    cumsum = np.concatenate([[0.0], np.cumsum(values)])  # cumsum[k] = sum of the first k values
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)
    return (cumsum[end] - cumsum[start]) / (end - start)


def plot_training_curve(
    train_log: list[dict],
    eval_log: list[dict],
    random_mean: float,
    title: str,
    path: str | Path | None = None,
    smoothing_episodes: int = 200,
):
    """Team reward per episode during training.

    Blue: greedy policy (no random actions) on fixed start positions, mean +- std.
    Orange: training episodes (with epsilon exploration), moving average.
    Gray dashed: random policy on the same fixed start positions.
    """
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    train_steps = np.array([row["env_steps"] for row in train_log]) / 1000
    train_returns = np.array([row["episode_return"] for row in train_log])
    ax.plot(
        train_steps,
        moving_average(train_returns, smoothing_episodes),
        color=ORANGE,
        linewidth=2,
        label=f"обучающие эпизоды (со случайными действиями), среднее по {smoothing_episodes}",
    )

    eval_steps = np.array([row["env_steps"] for row in eval_log]) / 1000
    eval_mean = np.array([row["mean_return"] for row in eval_log])
    eval_std = np.array([row["std_return"] for row in eval_log])
    n_eval = eval_log[0]["n_episodes"]
    ax.fill_between(eval_steps, eval_mean - eval_std, eval_mean + eval_std, color=BLUE, alpha=0.1, linewidth=0)
    ax.plot(
        eval_steps,
        eval_mean,
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        label=f"QMIX без случайных действий, {n_eval} фиксированные расстановки (среднее ± ст. откл.)",
    )

    ax.axhline(random_mean, color=TEXT_SECONDARY, linewidth=1.2, linestyle=(0, (5, 4)))
    ax.annotate(
        f"случайная политика: {random_mean:.1f}",
        xy=(eval_steps[-1], random_mean),
        xytext=(0, 6),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=TEXT_SECONDARY,
        fontsize=10,
    )
    ax.annotate(
        f"{eval_mean[-1]:.1f}",
        xy=(eval_steps[-1], eval_mean[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        color=TEXT,
        fontsize=10,
        fontweight="bold",
    )

    ax.set_xlabel("шаги среды, тыс.")
    ax.set_ylabel("награда команды за эпизод (сумма за 25 шагов)")
    ax.set_title(title, fontsize=13, loc="left")
    ax.set_xlim(0, eval_steps[-1] * 1.06)
    legend = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT)
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor=SURFACE)
    return fig


def plot_td_loss(train_log: list[dict], title: str, path: str | Path | None = None, smoothing_episodes: int = 200):
    """TD loss (mean squared TD error) during training, log scale."""
    rows = [row for row in train_log if np.isfinite(row["loss"])]
    steps = np.array([row["env_steps"] for row in rows]) / 1000
    loss = np.array([row["loss"] for row in rows])

    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.plot(steps, moving_average(loss, smoothing_episodes), color=BLUE, linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("шаги среды, тыс.")
    ax.set_ylabel(f"TD-ошибка в квадрате\n(среднее по {smoothing_episodes} шагам обучения)")
    ax.set_title(title, fontsize=13, loc="left")
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor=SURFACE)
    return fig


def plot_episode_trajectories(episodes: list[tuple[Episode, str]], path: str | Path | None = None):
    """Paths of the agents, one panel per (episode, title); hollow circle = start, filled = end, cross = landmark."""
    fig, axes = plt.subplots(1, len(episodes), figsize=(5.5 * len(episodes), 5.8), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    colors = [BLUE, ORANGE]
    for ax, (episode, title) in zip(axes[0], episodes):
        if episode.agent_positions is None:
            raise ValueError("run the episode with record_positions=True")
        _style(ax)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for i in range(episode.agent_positions.shape[1]):
            path_i = episode.agent_positions[:, i]  # [T+1, 2]
            color = colors[i % len(colors)]
            ax.plot(path_i[:, 0], path_i[:, 1], color=color, linewidth=2, label=f"агент {i + 1}")
            ax.plot(*path_i[0], marker="o", markersize=9, markerfacecolor=SURFACE, markeredgecolor=color,
                    markeredgewidth=2)
            ax.plot(*path_i[-1], marker="o", markersize=9, color=color, markeredgecolor=SURFACE, markeredgewidth=1.5)
        landmarks = episode.landmark_positions
        ax.scatter(landmarks[:, 0], landmarks[:, 1], marker="X", s=140, color=TEXT, edgecolor=SURFACE,
                   linewidth=1.5, label="ориентиры (цели)", zorder=5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{title}\nнаграда за эпизод {episode.total_reward:.1f}", fontsize=11, loc="left")
        legend = ax.legend(loc="best", frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(TEXT)
    fig.text(0.01, 0.01, "○ старт   ● конец эпизода (25 шагов)", color=TEXT_SECONDARY, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor=SURFACE)
    return fig
