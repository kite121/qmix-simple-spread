"""The QMIX training loop, saving of all outputs, and a command-line entry point.

From the repository root:
    PYTHONPATH=src python -m qmix_spread.train --seed 0
"""

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .buffer import EpisodeBuffer
from .checkpoint import build_networks, library_versions, save_checkpoint, save_json
from .config import QMIXConfig
from .env import FIXED_ENV_ARGS, SpreadEnv
from .learner import QMIXLearner
from .networks import AgentNetwork, QMixer
from .plotting import plot_td_loss, plot_training_curve
from .rollout import QMIXPolicy, RandomPolicy, evaluate, run_episode

RANDOM_BASELINE_RNG_SEED = 12345  # random-policy actions; separate from training randomness


@dataclass
class TrainResult:
    agent_net: AgentNetwork
    mixer: QMixer
    train_log: list[dict]  # one row per training episode
    eval_log: list[dict]  # one row per evaluation of the greedy policy
    env_steps: int
    episodes: int
    wall_time_sec: float


def epsilon_at(env_steps: int, config: QMIXConfig) -> float:
    """Exploration rate: goes linearly from epsilon_start to epsilon_finish, then stays."""
    progress = min(env_steps / config.epsilon_anneal_steps, 1.0)
    return config.epsilon_start + progress * (config.epsilon_finish - config.epsilon_start)


def train(config: QMIXConfig, progress=print) -> TrainResult:
    """Train QMIX from random weights. Same config and library versions -> same result on CPU."""
    torch.manual_seed(config.seed)  # network initialisation
    rng = np.random.default_rng(config.seed)  # exploration and buffer sampling

    env = SpreadEnv(config)
    eval_env = SpreadEnv(config)  # separate copy, so evaluation does not disturb the training env
    agent_net, mixer = build_networks(config, env)
    learner = QMIXLearner(agent_net, mixer, config)
    buffer = EpisodeBuffer(config.buffer_size, env.episode_limit, env.n_agents, env.obs_dim, env.state_dim)
    policy = QMIXPolicy(agent_net)  # acts with the current (online) agent network
    eval_seeds = config.eval_seeds()

    train_log: list[dict] = []
    eval_log: list[dict] = []
    env_steps = episodes = last_target_update = 0
    next_eval = 0
    start_time = time.perf_counter()

    def run_evaluation() -> None:
        returns = evaluate(eval_env, policy, eval_seeds, epsilon=0.0)
        eval_log.append({
            "env_steps": env_steps,
            "episodes": episodes,
            "mean_return": float(returns.mean()),
            "std_return": float(returns.std()),
            "n_episodes": len(returns),
        })
        recent = [row["episode_return"] for row in train_log[-100:]]
        recent_losses = [row["loss"] for row in train_log[-100:] if np.isfinite(row["loss"])]
        recent_q = [row["q_tot_mean"] for row in train_log[-100:] if np.isfinite(row["q_tot_mean"])]
        progress(
            f"steps {env_steps:>7} | episodes {episodes:>6} | epsilon {epsilon_at(env_steps, config):.2f} | "
            f"train return (last 100) {np.mean(recent) if recent else float('nan'):7.2f} | "
            f"greedy eval {returns.mean():7.2f} +- {returns.std():5.2f} | "
            f"loss {np.mean(recent_losses) if recent_losses else float('nan'):8.4f} | "
            f"Q_tot {np.mean(recent_q) if recent_q else float('nan'):8.2f} | "
            f"{time.perf_counter() - start_time:6.0f} s"
        )

    while env_steps < config.total_env_steps:
        if env_steps >= next_eval:
            run_evaluation()
            next_eval += config.eval_interval_steps

        # 1) Play one episode with epsilon-greedy exploration and store it.
        epsilon = epsilon_at(env_steps, config)
        # Only the first reset is seeded; later resets continue the env's own random stream.
        episode = run_episode(env, policy, epsilon, seed=config.seed if episodes == 0 else None, rng=rng)
        buffer.add(episode)
        env_steps += len(episode.rewards)
        episodes += 1

        # 2) One gradient step on a batch of stored episodes.
        stats = {"loss": float("nan"), "q_tot_mean": float("nan")}
        if len(buffer) >= config.batch_size:
            stats = learner.train_step(buffer.sample(config.batch_size, rng))

        # 3) Refresh the target networks every target_update_interval episodes.
        if episodes - last_target_update >= config.target_update_interval:
            learner.update_targets()
            last_target_update = episodes

        train_log.append({
            "episode": episodes,
            "env_steps": env_steps,
            "epsilon": epsilon,
            "episode_return": episode.total_reward,
            "loss": stats["loss"],
            "q_tot_mean": stats["q_tot_mean"],
        })

    run_evaluation()  # always evaluate the final model
    return TrainResult(agent_net, mixer, train_log, eval_log, env_steps, episodes, time.perf_counter() - start_time)


def random_baseline(config: QMIXConfig, seeds: list[int]) -> np.ndarray:
    """Returns of the random policy on the given start positions."""
    env = SpreadEnv(config)
    policy = RandomPolicy(env.n_agents, env.n_actions)
    return evaluate(env, policy, seeds, rng=np.random.default_rng(RANDOM_BASELINE_RNG_SEED))


def write_csv(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(f)]


def run_name(config: QMIXConfig) -> str:
    return f"qmix_spread_n{config.n_agents}_seed{config.seed}"


def save_run(result: TrainResult, config: QMIXConfig, random_returns: np.ndarray, root: str | Path) -> dict[str, Path]:
    """Save checkpoint, logs, run config and plots under the repository layout rooted at `root`."""
    root = Path(root)
    name = run_name(config)
    paths = {
        "checkpoint": root / "models" / f"{name}.pt",
        "train_log": root / "results" / "tables" / f"{name}_train_log.csv",
        "eval_log": root / "results" / "tables" / f"{name}_eval_log.csv",
        "run_config": root / "configs" / f"{name}.json",
        "curve": root / "results" / "plots" / f"{name}_training_curve.png",
        "loss": root / "results" / "plots" / f"{name}_td_loss.png",
    }
    final_eval = result.eval_log[-1]
    sha256 = save_checkpoint(
        paths["checkpoint"],
        result.agent_net,
        result.mixer,
        config,
        metadata={
            "env_steps": result.env_steps,
            "episodes": result.episodes,
            "wall_time_sec": round(result.wall_time_sec, 1),
            "torch_threads": torch.get_num_threads(),
            "final_eval_mean_return": final_eval["mean_return"],
            "final_eval_std_return": final_eval["std_return"],
            "eval_seeds": config.eval_seeds(),
            "random_policy_mean_return_on_eval_seeds": float(random_returns.mean()),
        },
    )
    write_csv(paths["train_log"], result.train_log)
    write_csv(paths["eval_log"], result.eval_log)
    save_json(paths["run_config"], {
        "config": config.to_dict(),
        "environment": {
            "package": "mpe2",
            "env": "simple_spread_v3.parallel_env",
            "N": config.n_agents,
            "local_ratio": config.local_ratio,
            "max_cycles": config.max_cycles,
            "continuous_actions": False,
            **FIXED_ENV_ARGS,
        },
        "library_versions": library_versions(),
        "checkpoint": {"file": paths["checkpoint"].name, "sha256": sha256},
    })
    title = f"Обучение QMIX на Simple Spread (N={config.n_agents}), seed {config.seed}"
    plt.close(plot_training_curve(result.train_log, result.eval_log, float(random_returns.mean()), title, paths["curve"]))
    plt.close(plot_td_loss(result.train_log, f"Квадрат TD-ошибки (loss) во время обучения, seed {config.seed}",
                           paths["loss"]))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QMIX on Simple Spread and save all outputs.")
    parser.add_argument("--seed", type=int, default=QMIXConfig.seed)
    parser.add_argument("--total-env-steps", type=int, default=QMIXConfig.total_env_steps)
    parser.add_argument("--root", default=".", help="repository root; outputs go to models/, results/, configs/")
    parser.add_argument("--threads", type=int, default=1, help="torch CPU threads (1 is fastest for these tiny nets)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override any other QMIXConfig field, e.g. --set gamma=0.95 lr=0.001")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    overrides = {}
    for item in args.set:
        key, value = item.split("=", 1)
        default = getattr(QMIXConfig, key)  # fails loudly on a typo in the field name
        if value.lower() == "none":
            overrides[key] = None
        elif isinstance(default, bool):
            overrides[key] = value.lower() in ("1", "true", "yes")  # bool("0") would be True
        else:
            overrides[key] = type(default)(value)
    config = QMIXConfig(seed=args.seed, total_env_steps=args.total_env_steps, **overrides)
    result = train(config)
    random_returns = random_baseline(config, config.eval_seeds())
    paths = save_run(result, config, random_returns, args.root)
    for key, path in paths.items():
        print(f"{key:>10}: {path}")


if __name__ == "__main__":
    main()
