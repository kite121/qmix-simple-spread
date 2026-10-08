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
        label=f"training episodes (with random exploration), moving average of {smoothing_episodes}",
    )

    eval_steps = np.array([row["env_steps"] for row in eval_log]) / 1000
    eval_mean = np.array([row["mean_return"] for row in eval_log])
    eval_std = np.array([row["std_return"] for row in eval_log])
    n_eval = int(eval_log[0]["n_episodes"])  # int(): logs read back from CSV hold floats
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
        label=f"QMIX greedy policy on {n_eval} fixed start positions (mean ± std)",
    )

    ax.axhline(random_mean, color=TEXT_SECONDARY, linewidth=1.2, linestyle=(0, (5, 4)))
    ax.annotate(
        f"random agents: {random_mean:.1f}",
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

    ax.set_xlabel("environment steps, thousands")
    ax.set_ylabel("team return per episode (sum of 25 rewards)")
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
    """Loss (mean squared TD error of a batch) during training, log scale."""
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
    ax.set_xlabel("environment steps, thousands")
    ax.set_ylabel(f"loss = mean squared TD error\n(moving average of {smoothing_episodes} updates)")
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

    # The same square area in every panel, so the paths can be compared by eye.
    points = np.concatenate([np.concatenate([e.agent_positions.reshape(-1, 2), e.landmark_positions])
                             for e, _ in episodes])
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    half_size = (points.max(axis=0) - points.min(axis=0)).max() / 2 * 1.1 + 0.05
    for ax, (episode, title) in zip(axes[0], episodes):
        if episode.agent_positions is None:
            raise ValueError("run the episode with record_positions=True")
        _style(ax)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for i in range(episode.agent_positions.shape[1]):
            path_i = episode.agent_positions[:, i]  # [T+1, 2]
            color = colors[i % len(colors)]
            ax.plot(path_i[:, 0], path_i[:, 1], color=color, linewidth=2, label=f"agent {i + 1}")
            ax.plot(*path_i[0], marker="o", markersize=9, markerfacecolor=SURFACE, markeredgecolor=color,
                    markeredgewidth=2)
            ax.plot(*path_i[-1], marker="o", markersize=9, color=color, markeredgecolor=SURFACE, markeredgewidth=1.5)
        landmarks = episode.landmark_positions
        ax.scatter(landmarks[:, 0], landmarks[:, 1], marker="X", s=140, color=TEXT, edgecolor=SURFACE,
                   linewidth=1.5, label="landmarks (targets)", zorder=5)
        ax.set_xlim(center[0] - half_size, center[0] + half_size)
        ax.set_ylim(center[1] - half_size, center[1] + half_size)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{title}\nepisode return {episode.total_reward:.1f}", fontsize=11, loc="left")
        legend = ax.legend(loc="best", frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(TEXT)
    fig.text(0.01, 0.01, "○ start   ● end of episode (25 steps)", color=TEXT_SECONDARY, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor=SURFACE)
    return fig


def plot_joint_action_q(q_tot_table: np.ndarray, action_names: list[str], title: str, path: str | Path | None = None):
    """Q_tot for every joint action of 2 agents as a heatmap; the best cell is framed.

    q_tot_table [A, A]: row = action of agent 1, column = action of agent 2.
    """
    table = np.asarray(q_tot_table, dtype=float)
    best = np.unravel_index(table.argmax(), table.shape)
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    image = ax.imshow(table, cmap="Blues")
    # Enough decimals to tell the cells apart (an untrained mixer gives nearly equal values).
    spread = table.max() - table.min()
    decimals = 2 if spread > 1 else 3 if spread > 0.1 else 4
    threshold = table.min() + 0.6 * spread
    for (row, col), value in np.ndenumerate(table):
        ax.text(col, row, f"{value:.{decimals}f}", ha="center", va="center", fontsize=8,
                color=SURFACE if value > threshold else TEXT)
    ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False, edgecolor=ORANGE, linewidth=3))
    ax.set_xticks(range(len(action_names)), action_names, rotation=30, ha="right")
    ax.set_yticks(range(len(action_names)), action_names)
    ax.set_xlabel("action of agent 2")
    ax.set_ylabel("action of agent 1")
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.set_title(title, fontsize=11, loc="left", color=TEXT)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).set_label("$Q_{tot}$", color=TEXT_SECONDARY)
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor=SURFACE)
    return fig


def plot_paired_returns(qmix_returns: np.ndarray, random_returns: np.ndarray, path: str | Path | None = None):
    """One dot per start position: random agents (x) vs QMIX (y). Above the diagonal = QMIX is better."""
    qmix_returns, random_returns = np.asarray(qmix_returns), np.asarray(random_returns)
    fig, ax = plt.subplots(figsize=(5.6, 5.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    low = min(qmix_returns.min(), random_returns.min()) - 3
    high = max(qmix_returns.max(), random_returns.max()) + 3
    ax.plot([low, high], [low, high], color=TEXT_SECONDARY, linewidth=1.2, linestyle=(0, (5, 4)))
    ax.scatter(random_returns, qmix_returns, s=22, color=BLUE, alpha=0.8, edgecolor=SURFACE, linewidth=0.5)
    better = int((qmix_returns > random_returns).sum())
    # The area below the diagonal (QMIX worse) is where the labels do not cover any dots.
    ax.text(0.97, 0.06, f"QMIX better on {better} of {len(qmix_returns)}\nstart positions (dots above the line)",
            transform=ax.transAxes, ha="right", va="bottom", color=TEXT, fontsize=10)
    label_at = low + 0.12 * (high - low)
    ax.text(label_at, label_at, "same return  ", rotation=45, rotation_mode="anchor", ha="left", va="bottom",
            color=TEXT_SECONDARY, fontsize=9)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("random agents: episode return")
    ax.set_ylabel("QMIX: episode return")
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor=SURFACE)
    return fig


def save_episode_gif(episodes: list[tuple[Episode, str]], path: str | Path, scale: float = 0.5,
                     frame_ms: int = 150) -> Path:
    """Rendered episodes side by side as one animated GIF (episodes must have frames recorded)."""
    from PIL import Image, ImageDraw, ImageFont  # Pillow comes with matplotlib

    try:
        font = ImageFont.load_default(size=20)
    except TypeError:  # Pillow < 10.1 has one fixed-size default font
        font = ImageFont.load_default()
    n_frames = max(len(episode.frames) for episode, _ in episodes)
    header, gap = 34, 6  # label strip above the frames, gray gap between the panels
    gif_frames = []
    for t in range(n_frames):
        panels = []
        for episode, _ in episodes:
            frame = episode.frames[min(t, len(episode.frames) - 1)]  # [700, 700, 3]
            panels.append(Image.fromarray(frame).resize((int(frame.shape[1] * scale), int(frame.shape[0] * scale))))
        width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
        canvas = Image.new("RGB", (width, panels[0].height + header), GRID)
        draw = ImageDraw.Draw(canvas)
        x = 0
        for panel, (episode, title) in zip(panels, episodes):
            step = min(t, len(episode.rewards))
            canvas.paste(Image.new("RGB", (panel.width, header), SURFACE), (x, 0))
            canvas.paste(panel, (x, header))
            draw.text((x + 10, 7), f"{title}   step {step:2d}   return {episode.rewards[:step].sum():6.1f}",
                      fill=TEXT, font=font)
            x += panel.width + gap
        gif_frames.append(canvas)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Hold the last frame longer, so the final positions are easy to see.
    durations = [frame_ms] * (n_frames - 1) + [frame_ms * 8]
    gif_frames[0].save(path, save_all=True, append_images=gif_frames[1:], duration=durations, loop=0)
    return path
