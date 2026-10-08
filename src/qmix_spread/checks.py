"""Checks of trained models after training (numbers for the report).

From the repository root:
    PYTHONPATH=src python -m qmix_spread.checks --seeds 0 1 2

Writes two tables to results/tables/:
    qmix_training_check_heldout.csv  - QMIX vs random agents on 100 start positions not used in training
    qmix_long_episode_check.csv      - the same start positions with 150-step episodes instead of 25
These are sanity checks of the training; the official evaluation is done separately (Aslan).
"""

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .env import SpreadEnv
from .learner import unroll_agent_network
from .rollout import QMIXPolicy, RandomPolicy, run_episode
from .td_example import episodes_to_batch
from .train import read_csv, run_name, write_csv

HELDOUT_SEEDS = list(range(30_000, 30_100))  # start positions used neither in training nor in training-time evaluation
RANDOM_POLICY_RNG_SEED = 0
LONG_EPISODE_STEPS = 150  # gamma^150 = 0.95^150 < 0.001, so the discounted return beyond is negligible


def heldout_check(root: Path, seed: int) -> dict:
    """QMIX (no random actions) vs random agents on the same 100 start positions, plus criterion C2."""
    model = load_checkpoint(root / "models" / f"qmix_spread_n2_seed{seed}.pt")
    env = SpreadEnv(model.config)
    qmix_episodes = [run_episode(env, QMIXPolicy(model.agent_net), seed=s) for s in HELDOUT_SEEDS]
    rng = np.random.default_rng(RANDOM_POLICY_RNG_SEED)
    random_episodes = [run_episode(env, RandomPolicy(env.n_agents, env.n_actions), seed=s, rng=rng) for s in HELDOUT_SEEDS]
    qmix = np.array([e.total_reward for e in qmix_episodes])
    random = np.array([e.total_reward for e in random_episodes])
    difference = qmix - random  # paired: the same start position
    standard_error = difference.std(ddof=1) / np.sqrt(len(difference))

    # C2: share of the best improvement over the random policy that is kept at the end of training.
    eval_means = np.array([row["mean_return"] for row in read_csv(root / "results" / "tables" / f"{run_name(model.config)}_eval_log.csv")])
    random_on_eval_seeds = model.metadata["random_policy_mean_return_on_eval_seeds"]
    kept = (eval_means[-3:].mean() - random_on_eval_seeds) / (eval_means.max() - random_on_eval_seeds)
    return {
        "seed": seed,
        "eval32_final": round(float(eval_means[-1]), 2),
        "eval32_best": round(float(eval_means.max()), 2),
        "eval32_random": round(float(random_on_eval_seeds), 2),
        "c2_kept_share": round(float(kept), 3),
        "heldout_qmix_mean": round(float(qmix.mean()), 2),
        "heldout_qmix_std": round(float(qmix.std()), 2),
        "heldout_random_mean": round(float(random.mean()), 2),
        "heldout_random_std": round(float(random.std()), 2),
        "heldout_difference_mean": round(float(difference.mean()), 2),
        "heldout_difference_in_standard_errors": round(float(difference.mean() / standard_error), 1),
        "heldout_qmix_better_share": round(float(np.mean(difference > 0)), 2),
        "final_distance_qmix": round(float(np.mean([e.final_mean_distance for e in qmix_episodes])), 3),
        "final_distance_random": round(float(np.mean([e.final_mean_distance for e in random_episodes])), 3),
    }


def long_episode_check(root: Path, seed: int) -> dict:
    """Do the agents stay at the landmarks after step 25, which training never saw?

    Also compares the network's own value estimate Q_tot at the first step with the real
    discounted return of the long episode (gamma from the config).
    """
    model = load_checkpoint(root / "models" / f"qmix_spread_n2_seed{seed}.pt")
    env = SpreadEnv(dataclasses.replace(model.config, max_cycles=LONG_EPISODE_STEPS))
    episodes = [run_episode(env, QMIXPolicy(model.agent_net), seed=s) for s in HELDOUT_SEEDS]
    distance = np.array([-e.rewards / env.n_agents for e in episodes])  # [episodes, steps] landmark-to-nearest-agent

    gamma = model.config.gamma
    discounts = gamma ** np.arange(LONG_EPISODE_STEPS)
    real_return = np.array([(discounts * e.rewards).sum() for e in episodes])
    batch = episodes_to_batch(episodes)
    with torch.no_grad():
        q = unroll_agent_network(model.agent_net, batch["obs"][:, :2], batch["actions"][:, :1])  # steps 0 and 1
        chosen = torch.gather(q[:, :1], dim=3, index=batch["actions"][:, :1].unsqueeze(-1)).squeeze(-1)  # [E, 1, n]
        predicted = model.mixer(chosen, batch["state"][:, :1])[:, 0].numpy()  # Q_tot at step 0

    row = {"seed": seed}
    for step in [25, 50, 100, 150]:
        row[f"mean_distance_step{step}"] = round(float(distance[:, step - 1].mean()), 3)
    row["share_within_0.5_at_step150"] = round(float(np.mean(distance[:, -1] < 0.5)), 2)
    row["max_distance_at_step150"] = round(float(distance[:, -1].max()), 2)
    row["worst_start_position"] = HELDOUT_SEEDS[int(distance[:, -1].argmax())]
    row["q_tot_step0_predicted"] = round(float(predicted.mean()), 2)
    row["discounted_return_step0_real"] = round(float(real_return.mean()), 2)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity checks of trained QMIX checkpoints.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--root", default=".", help="repository root")
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(args.root)

    heldout = [heldout_check(root, seed) for seed in args.seeds]
    long = [long_episode_check(root, seed) for seed in args.seeds]
    write_csv(root / "results" / "tables" / "qmix_training_check_heldout.csv", heldout)
    write_csv(root / "results" / "tables" / "qmix_long_episode_check.csv", long)
    for row in heldout + long:
        print(row)


if __name__ == "__main__":
    main()
