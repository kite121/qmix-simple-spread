"""A fully numeric TD update, step by step.

Part 1 (hand_td_example): round numbers, computed with plain arithmetic, for the slides.
    To keep it countable by hand, the mixer is one layer: Q_tot = w1*Q1 + w2*Q2 + b with w1, w2 >= 0.
    The real QMixer has a hidden layer and gets w and b from the state through hypernetworks,
    but the TD step is exactly the same.
Part 2 (check_with_autograd): the same numbers recomputed by PyTorch autograd. The formulas are the same,
    so this checks the arithmetic of part 1 (gradients by hand vs. by autograd), not the formulas themselves.
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
    # Only the mixer parameters change here, to keep it countable. In the real network the
    # signals of step 4 (dloss_dq1, dloss_dq2) also flow on into the agent networks.
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


def _num(value: float, digits: int = 4) -> str:
    """Short number for formulas: 0.5, -8.6, 0.444 (no trailing zeros)."""
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _par(value: float, digits: int = 4) -> str:
    """Number for a product: negative values in parentheses, e.g. 0.5 \\cdot (-4)."""
    text = _num(value, digits)
    return f"({text})" if text.startswith("-") else text


def _vector(values: list[float], bold_index: int | None = None, digits: int = 3) -> str:
    parts = [_num(v, digits) for v in values]
    if bold_index is not None:
        parts[bold_index] = rf"\mathbf{{{parts[bold_index]}}}"
    return r"\big(" + r",\ ".join(parts) + r"\big)"


def format_hand_example(ex: dict) -> str:
    """The hand example as a chain of formulas with every number filled in (Markdown + LaTeX)."""
    best1 = int(np.argmax(ex["next_q1_all"]))
    best2 = int(np.argmax(ex["next_q2_all"]))
    lines = [
        r"\text{given:}\quad & r = " + _num(ex["reward"]) + r",\ \gamma = " + _num(ex["gamma"])
        + r",\ \eta = " + _num(ex["lr"]),
        r"& (Q_1, Q_2) = (" + _num(ex["q1"]) + ", " + _num(ex["q2"]) + r"),\ (w_1, w_2, b) = ("
        + ", ".join(_num(ex[k]) for k in ("w1", "w2", "b")) + ")",
        r"Q_{tot} &= w_1 Q_1 + w_2 Q_2 + b = " + _num(ex["w1"]) + r" \cdot " + _par(ex["q1"]) + " + "
        + _num(ex["w2"]) + r" \cdot " + _par(ex["q2"]) + " + " + _par(ex["b"]) + " = " + _num(ex["q_tot"]),
        r"Q_1(\tau_1', \cdot) &= " + _vector(ex["next_q1_all"], best1) + r"\quad \text{(max in bold)}",
        r"Q_2(\tau_2', \cdot) &= " + _vector(ex["next_q2_all"], best2),
        r"Q_{tot}' &= w_1' \max Q_1' + w_2' \max Q_2' + b' = " + _num(ex["next_w1"]) + r" \cdot "
        + _par(ex["next_q1_max"]) + " + " + _num(ex["next_w2"]) + r" \cdot " + _par(ex["next_q2_max"]) + " + "
        + _par(ex["next_b"]) + " = " + _num(ex["next_q_tot"]),
        r"y &= r + \gamma\, Q_{tot}' = " + _num(ex["reward"]) + " + " + _num(ex["gamma"]) + r" \cdot "
        + _par(ex["next_q_tot"]) + " = " + _num(ex["td_target"]),
        r"\delta &= y - Q_{tot} = " + _num(ex["td_target"]) + " - " + _par(ex["q_tot"]) + " = "
        + _num(ex["td_error"]) + r", \qquad \mathcal{L} = \delta^2 = " + _num(ex["loss"]),
        r"\partial \mathcal{L} / \partial Q_{tot} &= -2\delta = " + _num(ex["dloss_dq_tot"]),
        r"\partial \mathcal{L} / \partial (Q_1, Q_2) &= -2\delta\, (w_1, w_2) = (" + _num(ex["dloss_dq1"]) + ", "
        + _num(ex["dloss_dq2"]) + r") \quad \text{(credit to each agent)}",
        r"\partial \mathcal{L} / \partial (w_1, w_2, b) &= -2\delta\, (Q_1, Q_2, 1) = ("
        + ", ".join(_num(ex[k]) for k in ("dloss_dw1", "dloss_dw2", "dloss_db")) + ")",
        r"(w_1, w_2, b) &\leftarrow (w_1, w_2, b) - \eta \nabla \mathcal{L} = ("
        + ", ".join(_num(ex[k]) for k in ("new_w1", "new_w2", "new_b")) + ")",
        r"Q_{tot}^{\text{new}} &= " + _num(ex["new_q_tot"]) + r", \qquad \delta^{\text{new}} = y - Q_{tot}^{\text{new}} = "
        + _num(ex["new_td_error"]) + r" \quad (\text{was } " + _num(ex["td_error"]) + ")",
    ]
    return "$$\n\\begin{aligned}\n" + " \\\\\n".join(lines) + "\n\\end{aligned}\n$$"


def format_real_example(r: dict, action_names: list[str]) -> str:
    """The real-network example as a chain of formulas (Markdown + LaTeX)."""
    a1, a2 = r["actions"]
    lines = [
        r"Q_1(\tau_1, \cdot) &= " + _vector(r["q1_all"], a1) + r"\quad \text{(taken action in bold)}",
        r"Q_2(\tau_2, \cdot) &= " + _vector(r["q2_all"], a2),
        r"Q_{tot}(\boldsymbol{\tau}, \mathbf{a}, s) &= g\big(" + _num(r["chosen_q"][0], 3) + ",\\ "
        + _num(r["chosen_q"][1], 3) + r";\ s\big) = " + _num(r["q_tot"]),
        r"Q^{-}_{tot}(\boldsymbol{\tau}', \hat{\mathbf{a}}', s') &= g^{-}\big(" + _num(r["next_max_q"][0], 3) + ",\\ "
        + _num(r["next_max_q"][1], 3) + r";\ s'\big) = " + _num(r["next_q_tot"]),
        r"y &= r + \gamma\, Q^{-}_{tot} = " + _num(r["reward"]) + " + " + _num(r["gamma"]) + r" \cdot "
        + _par(r["next_q_tot"]) + " = " + _num(r["td_target"]),
        r"\delta &= y - Q_{tot} = " + _num(r["td_error"]) + r", \qquad \mathcal{L} = \delta^2 = " + _num(r["loss"], 6),
        r"\text{one gradient step } (\eta = " + _num(r["lr"], 6) + r"):\quad Q_{tot} &\to " + _num(r["new_q_tot"])
        + r", \qquad \delta \to " + _num(r["new_td_error"]),
    ]
    header = (f"Step {r['step'] + 1} of the episode (index {r['step']} in code): agent 1 `{action_names[a1]}`, "
              f"agent 2 `{action_names[a2]}`")
    return header + "\n\n$$\n\\begin{aligned}\n" + " \\\\\n".join(lines) + "\n\\end{aligned}\n$$"
