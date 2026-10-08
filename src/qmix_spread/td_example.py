"""A fully numeric TD update, step by step.

Part 1 (hand_td_example): round numbers, computed with plain arithmetic, for the slides.
    To keep it countable by hand, the mixer is one layer: Q_tot = w1*Q1 + w2*Q2 + b with w1, w2 >= 0.
    The real QMixer has a hidden layer and gets w and b from the state through hypernetworks,
    but the TD step is exactly the same.
Part 2 (check_with_autograd): the same numbers recomputed by PyTorch autograd, to prove part 1 is right.
Part 3 (real_transition_example): the same quantities taken from the trained networks for one real step.
"""

import copy

import numpy as np
import torch

from .config import QMIXConfig
from .learner import QMIXLearner
from .networks import AgentNetwork, QMixer
from .rollout import Episode


def hand_td_example() -> dict:
    """All numbers of one TD update. Every value is computed from the ones above it."""
    ex = {}
    # --- Given ---
    ex["gamma"] = 0.95  # the same discount factor as in training
    ex["reward"] = -1.0  # team reward after the step (minus the distances to the landmarks)
    # Current step: Q of the action each agent took, and mixer weights for the current state.
    ex["q1"], ex["q2"] = -4.0, -6.0
    ex["w1"], ex["w2"], ex["b"] = 0.5, 1.0, -2.0
    # Next step: Q values of every action (no_action, left, right, down, up) and mixer weights
    # for the next state (they differ from the current ones: weights depend on the state).
    # In our code the online network picks the best action and the target network rates it
    # (double Q); here both agree, so "best" is simply the maximum.
    ex["next_q1_all"] = [-6.0, -5.5, -5.0, -7.0, -6.5]
    ex["next_q2_all"] = [-3.0, -2.5, -4.0, -3.5, -2.8]
    ex["next_w1"], ex["next_w2"], ex["next_b"] = 0.8, 0.4, -3.0
    ex["lr"] = 0.005  # step size, large on purpose so that the change is visible

    # --- 1. Current estimate Q_tot ---
    ex["q_tot"] = ex["w1"] * ex["q1"] + ex["w2"] * ex["q2"] + ex["b"]  # 0.5*(-4) + 1*(-6) - 2 = -10

    # --- 2. Next estimate: every agent takes its own best action ---
    ex["next_q1_max"] = max(ex["next_q1_all"])  # -5.0 (move_right)
    ex["next_q2_max"] = max(ex["next_q2_all"])  # -2.5 (move_left)
    ex["next_q_tot"] = ex["next_w1"] * ex["next_q1_max"] + ex["next_w2"] * ex["next_q2_max"] + ex["next_b"]  # -8

    # --- 3. TD target, TD error, loss ---
    ex["td_target"] = ex["reward"] + ex["gamma"] * ex["next_q_tot"]  # -1 + 0.95*(-8) = -8.6
    ex["td_error"] = ex["td_target"] - ex["q_tot"]  # -8.6 - (-10) = 1.4
    ex["loss"] = ex["td_error"] ** 2  # 1.96

    # --- 4. Gradients: loss = (y - Q_tot)^2, so dloss/dQ_tot = -2 * (y - Q_tot) ---
    ex["dloss_dq_tot"] = -2.0 * ex["td_error"]  # -2.8
    # How the blame is shared between agents: dQ_tot/dQ_i = w_i
    ex["dloss_dq1"] = ex["dloss_dq_tot"] * ex["w1"]  # -1.4
    ex["dloss_dq2"] = ex["dloss_dq_tot"] * ex["w2"]  # -2.8
    # Gradients for the mixer parameters: dQ_tot/dw_i = Q_i, dQ_tot/db = 1
    ex["dloss_dw1"] = ex["dloss_dq_tot"] * ex["q1"]  # 11.2
    ex["dloss_dw2"] = ex["dloss_dq_tot"] * ex["q2"]  # 16.8
    ex["dloss_db"] = ex["dloss_dq_tot"]  # -2.8

    # --- 5. One gradient descent step: parameter <- parameter - lr * gradient ---
    ex["new_w1"] = ex["w1"] - ex["lr"] * ex["dloss_dw1"]  # 0.444
    ex["new_w2"] = ex["w2"] - ex["lr"] * ex["dloss_dw2"]  # 0.916
    ex["new_b"] = ex["b"] - ex["lr"] * ex["dloss_db"]  # -1.986
    ex["new_q_tot"] = ex["new_w1"] * ex["q1"] + ex["new_w2"] * ex["q2"] + ex["new_b"]  # -9.258
    ex["new_td_error"] = ex["td_target"] - ex["new_q_tot"]  # 0.658: the error got about 2x smaller
    return ex


def check_with_autograd(ex: dict) -> dict:
    """Recompute the hand example with PyTorch and return the largest difference to the hand numbers."""
    t = {k: torch.tensor(ex[k], dtype=torch.float64, requires_grad=True) for k in ["q1", "q2", "w1", "w2", "b"]}
    q_tot = t["w1"] * t["q1"] + t["w2"] * t["q2"] + t["b"]
    next_q1 = torch.tensor(ex["next_q1_all"], dtype=torch.float64).max()
    next_q2 = torch.tensor(ex["next_q2_all"], dtype=torch.float64).max()
    next_q_tot = ex["next_w1"] * next_q1 + ex["next_w2"] * next_q2 + ex["next_b"]
    td_target = (ex["reward"] + ex["gamma"] * next_q_tot).detach()  # target is a constant
    loss = (td_target - q_tot) ** 2
    loss.backward()

    with torch.no_grad():
        new = {k: t[k] - ex["lr"] * t[k].grad for k in ["w1", "w2", "b"]}
        new_q_tot = new["w1"] * t["q1"] + new["w2"] * t["q2"] + new["b"]

    autograd = {
        "q_tot": q_tot.item(),
        "td_target": td_target.item(),
        "loss": loss.item(),
        "dloss_dq1": t["q1"].grad.item(),
        "dloss_dq2": t["q2"].grad.item(),
        "dloss_dw1": t["w1"].grad.item(),
        "dloss_dw2": t["w2"].grad.item(),
        "dloss_db": t["b"].grad.item(),
        "new_q_tot": new_q_tot.item(),
    }
    autograd["max_abs_difference"] = max(abs(autograd[k] - ex[k]) for k in autograd)
    return autograd


def episodes_to_batch(episodes: list[Episode]) -> dict[str, torch.Tensor]:
    """Episodes as one batch in the format of EpisodeBuffer.sample (batch axis first)."""
    return {
        "obs": torch.as_tensor(np.stack([e.obs for e in episodes]), dtype=torch.float32),
        "state": torch.as_tensor(np.stack([e.states for e in episodes]), dtype=torch.float32),
        "actions": torch.as_tensor(np.stack([e.actions for e in episodes]), dtype=torch.int64),
        "reward": torch.as_tensor(np.stack([e.rewards for e in episodes]), dtype=torch.float32),
        "terminated": torch.as_tensor(np.stack([e.terminated for e in episodes]), dtype=torch.float32),
    }


def real_transition_example(
    agent_net: AgentNetwork, mixer: QMixer, config: QMIXConfig, episode: Episode, step: int, lr: float
) -> dict:
    """The TD update for one real step of one real episode, through the real QMIXLearner code.

    Works on copies, so the trained model is not changed. The target networks are copies of
    the trained networks (as right after a target update). One plain gradient descent step
    with the loss of this single step shows how Q_tot moves towards the TD target.
    """
    learner = QMIXLearner(copy.deepcopy(agent_net), copy.deepcopy(mixer), config)
    batch = episodes_to_batch([episode])

    _, d = learner.compute_td_loss(batch)
    result = {
        "step": step,
        "q1_all": d["q_all"][0, step, 0].tolist(),  # Q_1 for each of the 5 actions
        "q2_all": d["q_all"][0, step, 1].tolist(),
        "actions": episode.actions[step].tolist(),
        "chosen_q": d["chosen_q"][0, step].tolist(),  # [Q_1(a_1), Q_2(a_2)]
        "q_tot": d["q_tot"][0, step].item(),
        "reward": float(episode.rewards[step]),
        "gamma": config.gamma,
        "next_max_q": d["target_max_q"][0, step].tolist(),  # Q at step+1 of each agent's best action
        "next_q_tot": d["target_q_tot"][0, step].item(),
        "td_target": d["td_target"][0, step].item(),
        "td_error": d["td_error"][0, step].item(),
        "lr": lr,
    }
    result["loss"] = result["td_error"] ** 2

    # One plain gradient descent step on the loss of this single step: theta <- theta - lr * grad.
    # (Training uses RMSprop on batches of 32 episodes; plain descent is easier to follow here.)
    step_loss = d["td_error"][0, step] ** 2
    learner.optimizer.zero_grad()
    step_loss.backward()
    with torch.no_grad():
        for p in learner.params:
            p -= lr * p.grad
    _, d_after = learner.compute_td_loss(batch)
    result["new_q_tot"] = d_after["q_tot"][0, step].item()
    result["new_td_error"] = result["td_target"] - result["new_q_tot"]
    return result


def format_hand_example(ex: dict) -> str:
    """The hand example as a Markdown table (Russian labels, for the notebook and slides)."""
    rows = [
        ("Дано: награда за шаг r", "", ex["reward"]),
        ("Дано: коэффициент дисконтирования γ", "", ex["gamma"]),
        ("Дано: оценки агентов для выбранных действий Q₁, Q₂", "", f"{ex['q1']}, {ex['q2']}"),
        ("Дано: веса смешивания для текущего состояния w₁, w₂, b", "", f"{ex['w1']}, {ex['w2']}, {ex['b']}"),
        ("1. Текущая общая оценка Q_tot", "w₁·Q₁ + w₂·Q₂ + b", ex["q_tot"]),
        ("2. Лучшая оценка агента 1 в следующем шаге", f"max{ex['next_q1_all']}", ex["next_q1_max"]),
        ("2. Лучшая оценка агента 2 в следующем шаге", f"max{ex['next_q2_all']}", ex["next_q2_max"]),
        ("2. Следующая общая оценка Q_tot′ (веса следующего состояния "
         f"{ex['next_w1']}, {ex['next_w2']}, {ex['next_b']})", "w₁′·maxQ₁′ + w₂′·maxQ₂′ + b′", ex["next_q_tot"]),
        ("3. TD-цель y", "r + γ·Q_tot′", ex["td_target"]),
        ("3. TD-ошибка δ", "y − Q_tot", ex["td_error"]),
        ("3. Ошибка, которую минимизируем", "δ²", ex["loss"]),
        ("4. Градиент по Q_tot", "−2·δ", ex["dloss_dq_tot"]),
        ("4. Сигнал агенту 1", "−2·δ·w₁", ex["dloss_dq1"]),
        ("4. Сигнал агенту 2", "−2·δ·w₂", ex["dloss_dq2"]),
        ("4. Градиенты по w₁, w₂, b", "−2·δ·Q₁, −2·δ·Q₂, −2·δ",
         f"{ex['dloss_dw1']:.4g}, {ex['dloss_dw2']:.4g}, {ex['dloss_db']:.4g}"),
        (f"5. Шаг с размером {ex['lr']}: новые w₁, w₂, b", "параметр − шаг·градиент",
         f"{ex['new_w1']:.4f}, {ex['new_w2']:.4f}, {ex['new_b']:.4f}"),
        ("5. Новая общая оценка Q_tot", "новые w₁·Q₁ + w₂·Q₂ + b", ex["new_q_tot"]),
        ("5. Новая TD-ошибка", "y − новая Q_tot", ex["new_td_error"]),
    ]
    lines = ["| Шаг | Формула | Значение |", "| --- | --- | --- |"]
    for name, formula, value in rows:
        shown = f"{value:.4f}".rstrip("0").rstrip(".") if isinstance(value, float) else value
        lines.append(f"| {name} | {formula} | {shown} |")
    return "\n".join(lines)


def format_real_example(r: dict, action_names: list[str]) -> str:
    """The real-network example as a Markdown table."""
    a1, a2 = r["actions"]
    q1 = np.round(r["q1_all"], 3).tolist()
    q2 = np.round(r["q2_all"], 3).tolist()
    rows = [
        ("Q₁ для 5 действий (сеть агента 1)", str(q1)),
        ("Q₂ для 5 действий (сеть агента 2)", str(q2)),
        ("Выбранные действия", f"агент 1: {action_names[a1]}, агент 2: {action_names[a2]}"),
        ("Q₁, Q₂ выбранных действий", f"{r['chosen_q'][0]:.3f}, {r['chosen_q'][1]:.3f}"),
        ("Q_tot (смешивающая сеть)", f"{r['q_tot']:.4f}"),
        ("Награда r", f"{r['reward']:.4f}"),
        ("Q лучших действий агентов в следующем шаге (выбирает обучаемая сеть, оценивает целевая)",
         f"{r['next_max_q'][0]:.3f}, {r['next_max_q'][1]:.3f}"),
        ("Q_tot′ следующего шага (целевая смешивающая сеть)", f"{r['next_q_tot']:.4f}"),
        ("TD-цель y = r + γ·Q_tot′", f"{r['reward']:.4f} + {r['gamma']}·({r['next_q_tot']:.4f}) = {r['td_target']:.4f}"),
        ("TD-ошибка δ = y − Q_tot", f"{r['td_error']:.4f}"),
        ("Ошибка δ²", f"{r['loss']:.6f}"),
        (f"Q_tot после одного шага градиентного спуска (шаг {r['lr']})", f"{r['new_q_tot']:.4f}"),
        ("TD-ошибка после шага", f"{r['new_td_error']:.4f}"),
    ]
    lines = [f"| Шаг эпизода {r['step']} | Значение |", "| --- | --- |"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    return "\n".join(lines)
