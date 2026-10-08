"""All settings of one QMIX run, kept in one place so they can be saved with the model."""

from dataclasses import asdict, dataclass


@dataclass
class QMIXConfig:
    # --- Environment (Simple Spread from the mpe2 package) ---
    n_agents: int = 2  # N: number of agents, which is also the number of landmarks
    local_ratio: float = 0.0  # 0.0 -> every agent receives only the shared team reward
    max_cycles: int = 25  # episode length in environment steps

    # --- Networks (sizes from the QMIX paper, Rashid et al. 2018) ---
    agent_hidden_dim: int = 64  # width of the GRU inside the agent network
    mixing_embed_dim: int = 32  # width of the hidden layer of the mixing network
    # Not in the paper: state -> limit * tanh(state / limit) before the hypernetworks.
    # Without it Q_tot ran away to large positive values in our pilot runs (see the report).
    mixer_state_limit: float = 3.0

    # --- Learning ---
    # The paper uses 0.99. Episodes here are only 25 steps long, and with 0.99 Q_tot slowly
    # ran away to positive values in pilot runs even with the state limit; 0.95 was stable.
    gamma: float = 0.95
    # Double Q-learning, as in the authors' reference code (pymarl): the next action is chosen
    # by the online network and rated by the target network, which reduces overestimation.
    double_q: bool = True
    lr: float = 5e-4  # RMSprop learning rate
    rmsprop_alpha: float = 0.99
    rmsprop_eps: float = 1e-5
    grad_clip_norm: float = 10.0  # gradients with a larger total norm are scaled down
    batch_size: int = 32  # episodes per gradient step
    buffer_size: int = 5000  # episodes kept in the replay buffer
    target_update_interval: int = 200  # episodes between copies into the target networks

    # --- Exploration (epsilon-greedy, epsilon goes down linearly) ---
    epsilon_start: float = 1.0
    epsilon_finish: float = 0.05
    epsilon_anneal_steps: int = 50_000  # environment steps

    # --- Run ---
    total_env_steps: int = 200_000
    seed: int = 0  # seeds network init, exploration, buffer sampling and the training env
    eval_interval_steps: int = 10_000  # how often the greedy policy is evaluated
    eval_seeds_start: int = 10_000  # evaluation uses env seeds eval_seeds_start, +1, ...
    n_eval_episodes: int = 32

    def eval_seeds(self) -> list[int]:
        return list(range(self.eval_seeds_start, self.eval_seeds_start + self.n_eval_episodes))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "QMIXConfig":
        return cls(**values)
