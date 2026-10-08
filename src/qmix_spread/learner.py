"""One QMIX training step: individual Q_i -> Q_tot -> TD target -> TD error -> loss -> gradient step."""

import copy

import torch
import torch.nn.functional as F

from .config import QMIXConfig
from .networks import AgentNetwork, QMixer


def unroll_agent_network(agent_net: AgentNetwork, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Run the recurrent agent network over whole episodes, step by step.

    obs [B, T+1, n, O], actions [B, T, n]  ->  Q values for every step and action [B, T+1, n, A]
    """
    batch_size, n_steps = obs.shape[0], obs.shape[1]  # n_steps = T + 1

    # The input at step t contains the agent's action from step t-1; at step 0 there is none.
    # Example for one agent: actions over time [2, 0, 4] -> previous action input [none, 2, 0, 4],
    # where "none" is the all-zero vector and 2 is the one-hot [0, 0, 1, 0, 0].
    actions_onehot = F.one_hot(actions, agent_net.n_actions).float()  # [B, T, n, A]
    no_action = torch.zeros_like(actions_onehot[:, :1])  # [B, 1, n, A]
    last_actions_onehot = torch.cat([no_action, actions_onehot], dim=1)  # [B, T+1, n, A]

    hidden = agent_net.init_hidden(batch_size)  # [B, n, H] empty memory at the start
    q_per_step = []
    for t in range(n_steps):
        q, hidden = agent_net(obs[:, t], last_actions_onehot[:, t], hidden)  # q [B, n, A]
        q_per_step.append(q)
    return torch.stack(q_per_step, dim=1)  # T+1 tensors [B, n, A] -> [B, T+1, n, A]


class QMIXLearner:
    def __init__(self, agent_net: AgentNetwork, mixer: QMixer, config: QMIXConfig):
        self.agent_net = agent_net
        self.mixer = mixer
        # Target networks: frozen copies used only to compute TD targets.
        # They are refreshed every config.target_update_interval episodes (see update_targets),
        # which keeps the target from moving after every single gradient step.
        self.target_agent_net = copy.deepcopy(agent_net)
        self.target_mixer = copy.deepcopy(mixer)

        self.gamma = config.gamma
        self.double_q = config.double_q
        self.grad_clip_norm = config.grad_clip_norm
        # Agent network and mixer are trained together by one loss (end-to-end).
        self.params = list(agent_net.parameters()) + list(mixer.parameters())
        self.optimizer = torch.optim.RMSprop(
            self.params, lr=config.lr, alpha=config.rmsprop_alpha, eps=config.rmsprop_eps
        )

    def compute_td_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Squared TD error of Q_tot, averaged over all episodes and steps of the batch.

        Batch (from EpisodeBuffer.sample): obs [B, T+1, n, O], state [B, T+1, S],
        actions [B, T, n], reward [B, T], terminated [B, T].
        Returns the loss and the intermediate tensors (used by the numeric example).
        """
        obs, state, actions = batch["obs"], batch["state"], batch["actions"]
        reward, terminated = batch["reward"], batch["terminated"]

        # 1) Individual Q_i of every agent, for every action, at every step.
        q_all = unroll_agent_network(self.agent_net, obs, actions)  # [B, T+1, n, A]

        # Keep only the Q value of the action each agent actually took at steps 0..T-1.
        # gather picks, along the action axis (dim 3), the entry whose index is in `actions`:
        #   Q of one agent at one step = [0.1, 0.7, -0.2, 0.0, 0.3], action taken = 1  ->  0.7
        # actions.unsqueeze(-1) adds the axis gather needs: [B, T, n] -> [B, T, n, 1].
        chosen_q = torch.gather(q_all[:, :-1], dim=3, index=actions.unsqueeze(-1)).squeeze(-1)  # [B, T, n]

        # 2) Mix the agents' chosen Q_i into the team value Q_tot, using the full state.
        q_tot = self.mixer(chosen_q, state[:, :-1])  # [B, T]

        # 3) TD target y = r + gamma * Q_tot_target(next step, best joint action).
        with torch.no_grad():  # the target is a fixed number for this step, no gradient through it
            target_q_all = unroll_agent_network(self.target_agent_net, obs, actions)  # [B, T+1, n, A]
            # Value of every agent's best action at the next step (steps 1..T).
            # Because the mixer is monotonic, "every agent takes its own best action" gives the
            # best joint action for Q_tot as well, so we never need to try all 5 x 5 combinations.
            if self.double_q:
                # Double Q-learning: the online network picks the action, the target network rates it.
                # A single network that both picks and rates the max tends to overestimate.
                next_actions = q_all[:, 1:].argmax(dim=3, keepdim=True)  # [B, T, n, 1]
                target_max_q = torch.gather(target_q_all[:, 1:], dim=3, index=next_actions).squeeze(3)  # [B, T, n]
            else:
                target_max_q = target_q_all[:, 1:].max(dim=3).values  # [B, T, n]
            target_q_tot = self.target_mixer(target_max_q, state[:, 1:])  # [B, T]
            # (1 - terminated) drops the future only if the task itself ended. The 25-step time
            # limit is not such an end: the task does not depend on time and the state has no
            # step counter, so we still bootstrap from the next state, as pymarl does
            # (Pardo et al., 2018, "Time Limits in Reinforcement Learning").
            td_target = reward + self.gamma * (1.0 - terminated) * target_q_tot  # [B, T]

        # 4) TD error and the loss that is minimised: mean of squared TD errors.
        td_error = td_target - q_tot  # [B, T]  > 0 means "better than the network expected"
        loss = (td_error**2).mean()

        details = {
            "q_all": q_all,
            "chosen_q": chosen_q,
            "q_tot": q_tot,
            "target_max_q": target_max_q,
            "target_q_tot": target_q_tot,
            "td_target": td_target,
            "td_error": td_error,
        }
        return loss, details

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        loss, details = self.compute_td_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip_norm)
        self.optimizer.step()
        return {
            "loss": loss.item(),
            "grad_norm": grad_norm.item(),
            "q_tot_mean": details["q_tot"].mean().item(),
        }

    def update_targets(self) -> None:
        """Copy the current weights into the target networks."""
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
