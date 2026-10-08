"""The two networks of QMIX.

Letters used in the shape comments:
    B - batch size (number of episodes, or 1 when acting)
    n - number of agents (2)
    O - observation size of one agent (12)
    A - number of actions (5)
    H - hidden size of the agent network (64)
    S - size of the full state (24)
    E - hidden size of the mixing network (32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentNetwork(nn.Module):
    """Computes Q_i: one value per action for every agent.

    This is the DRQN agent from the QMIX paper:
        input  = [own observation, own previous action (one-hot), agent id (one-hot)]
        layers = Linear -> ReLU -> GRU cell -> Linear
        output = Q_i(history_i, a) for each of the A actions

    One set of weights is shared by all agents (faster learning). The agent id in the
    input still lets the network behave differently for different agents if that helps.
    The GRU keeps a memory (hidden state) of what this agent has seen so far in the episode.
    """

    def __init__(self, obs_dim: int, n_agents: int, n_actions: int, hidden_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        input_dim = obs_dim + n_actions + n_agents  # 12 + 5 + 2 = 19 for Simple Spread N=2

        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, n_actions)

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Empty memory at the start of an episode: zeros [B, n, H]."""
        return torch.zeros(batch_size, self.n_agents, self.hidden_dim)

    def forward(
        self, obs: torch.Tensor, last_actions_onehot: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One time step for all agents at once.

        obs                 [B, n, O]
        last_actions_onehot [B, n, A]  (all zeros at the first step of an episode)
        hidden              [B, n, H]
        returns q [B, n, A] and the new hidden [B, n, H]
        """
        batch_size = obs.shape[0]

        # One-hot agent ids, the same for every episode in the batch.
        # torch.eye(2) = [[1, 0],      agent 0
        #                 [0, 1]]      agent 1
        # .expand(B, -1, -1) repeats this [n, n] matrix B times without copying memory -> [B, n, n]
        agent_ids = torch.eye(self.n_agents).expand(batch_size, -1, -1)

        # Glue the three parts along the last axis: [B, n, O + A + n]
        inputs = torch.cat([obs, last_actions_onehot, agent_ids], dim=-1)

        # Weights are shared, so every (episode, agent) pair is just one more row.
        # reshape [B, n, 19] -> [B*n, 19]; e.g. B=3, n=2 gives 6 rows:
        #   row 0 = episode 0 agent 0, row 1 = episode 0 agent 1, row 2 = episode 1 agent 0, ...
        x = inputs.reshape(batch_size * self.n_agents, -1)
        h = hidden.reshape(batch_size * self.n_agents, self.hidden_dim)

        x = F.relu(self.fc_in(x))  # [B*n, H]
        h = self.gru(x, h)  # [B*n, H]  memory updated with the current input
        q = self.fc_out(h)  # [B*n, A]  <- the individual Q_i values

        # Back to one row per episode and one slot per agent.
        return q.view(batch_size, self.n_agents, self.n_actions), h.view(
            batch_size, self.n_agents, self.hidden_dim
        )


class QMixer(nn.Module):
    """Mixes the chosen Q_i of all agents into one team value Q_tot, using the full state.

        hidden = ELU(Q_agents . W1 + b1)      W1 [n, E] >= 0, b1 [E]
        Q_tot  = hidden . W2 + b2             W2 [E, 1] >= 0, b2 scalar

    W1, b1, W2, b2 are not fixed parameters: small "hypernetworks" produce them from the
    full state s, so the way agents are combined can change from situation to situation.
    W1 and W2 pass through abs(), so they are never negative. ELU is an increasing function.
    Together this gives monotonicity: if any Q_i goes up, Q_tot cannot go down.
    That is why each agent can simply take its own best action at execution time:
    the joint action made of individual best actions is also the best for Q_tot.

    The state is used only here, i.e. only during training.
    """

    def __init__(self, n_agents: int, state_dim: int, embed_dim: int, state_limit: float | None):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        # Stored as a buffer, so it is saved in the state_dict together with the weights.
        # None switches the limit off (the mixer is then exactly the one from the paper).
        self.register_buffer("state_limit", None if state_limit is None else torch.tensor(float(state_limit)))

        # Hypernetworks for the weights: one linear layer each, as in the paper.
        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        # Hypernetworks for the biases. Biases may be negative, so no abs() here.
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        # The final bias V(s) is a small two-layer network, as in the paper.
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1)
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """agent_qs [..., n] (one chosen Q_i per agent), states [..., S]  ->  Q_tot [...].

        Any leading shape works, e.g. [B, T] for a batch of whole episodes.
        """
        leading_shape = agent_qs.shape[:-1]
        qs = agent_qs.reshape(-1, 1, self.n_agents)  # [N, 1, n]  N = all leading dims flattened
        s = states.reshape(-1, self.state_dim)  # [N, S]

        # Not in the paper: soft limit of the state values, limit * tanh(s / limit). With limit = 3:
        #   0.5 -> 0.50,  1.0 -> 0.96,  2.0 -> 1.75,  4.5 -> 2.71,  50.0 -> 3.00
        # Most state values lie within +-1.5 and are almost unchanged. Values up to about 4.5
        # (agents far from the start area at the end of an episode) are rare in the data; the
        # hypernetworks are linear in s and extrapolated badly there, so Q_tot for such states
        # grew far too large in our pilot runs (see the report). The limit compresses these tails.
        if self.state_limit is not None:
            s = self.state_limit * torch.tanh(s / self.state_limit)

        # First layer weights, generated from the state and made non-negative.
        w1 = torch.abs(self.hyper_w1(s)).view(-1, self.n_agents, self.embed_dim)  # [N, n, E]
        b1 = self.hyper_b1(s).view(-1, 1, self.embed_dim)  # [N, 1, E]

        # torch.bmm = one matrix product per row of the batch: [N, 1, n] x [N, n, E] -> [N, 1, E]
        # Example for one row with n=2, E=3:
        #   qs = [[2.0, 1.0]]   w1 = [[0.5, 0.1, 0.0],
        #                             [0.2, 0.3, 1.0]]
        #   qs @ w1 = [[2*0.5 + 1*0.2, 2*0.1 + 1*0.3, 2*0.0 + 1*1.0]] = [[1.2, 0.5, 1.0]]
        # Each hidden unit is a weighted sum of the agents' Q values with non-negative weights.
        hidden = F.elu(torch.bmm(qs, w1) + b1)  # [N, 1, E]

        # Second layer: E hidden units -> one number.
        w2 = torch.abs(self.hyper_w2(s)).view(-1, self.embed_dim, 1)  # [N, E, 1]
        b2 = self.hyper_b2(s).view(-1, 1, 1)  # [N, 1, 1]
        q_tot = torch.bmm(hidden, w2) + b2  # [N, 1, E] x [N, E, 1] -> [N, 1, 1]

        return q_tot.view(leading_shape)
