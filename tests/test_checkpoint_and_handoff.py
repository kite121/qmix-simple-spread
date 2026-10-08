import json
import re
from pathlib import Path

import numpy as np
import torch

from qmix_spread.checkpoint import build_networks, file_sha256, load_checkpoint, save_checkpoint
from qmix_spread.config import QMIXConfig
from qmix_spread.env import SpreadEnv
from qmix_spread.learner import unroll_agent_network
from qmix_spread.rollout import QMIXPolicy, run_episode
from qmix_spread.td_example import episodes_to_batch

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_checkpoint(path: Path):
    torch.manual_seed(0)
    config = QMIXConfig()
    env = SpreadEnv(config)
    agent_net, mixer = build_networks(config, env)
    sha256 = save_checkpoint(path, agent_net, mixer, config, metadata={"env_steps": 0})
    return config, env, agent_net, mixer, sha256


def test_loaded_model_gives_exactly_the_same_outputs(tmp_path):
    path = tmp_path / "model.pt"
    config, env, agent_net, mixer, sha256 = make_checkpoint(path)
    loaded = load_checkpoint(path)

    episode = run_episode(env, QMIXPolicy(agent_net), seed=30_000)
    batch = episodes_to_batch([episode])
    with torch.no_grad():
        q_saved = unroll_agent_network(agent_net, batch["obs"], batch["actions"])
        q_loaded = unroll_agent_network(loaded.agent_net, batch["obs"], batch["actions"])
        chosen = q_saved[:, :-1].max(dim=3).values
        assert torch.equal(q_saved, q_loaded)
        assert torch.equal(mixer(chosen, batch["state"][:, :-1]), loaded.mixer(chosen, batch["state"][:, :-1]))

    # The same seed gives the same greedy episode with the loaded model.
    replay = run_episode(env, QMIXPolicy(loaded.agent_net), seed=30_000)
    assert np.array_equal(episode.actions, replay.actions)

    description = json.loads(path.with_suffix(".json").read_text())
    assert description["sha256"] == sha256 == file_sha256(path)
    assert loaded.config == config
    assert set(loaded.metadata["library_versions"]) >= {"torch", "mpe2", "pettingzoo"}


def test_the_snippet_for_aslan_runs(tmp_path, capsys):
    """Runs the code block from models/README.md against a freshly saved checkpoint."""
    path = tmp_path / "model.pt"
    make_checkpoint(path)
    readme = (REPO_ROOT / "models" / "README.md").read_text(encoding="utf-8")
    code = re.search(r"```python\n(.*?)```", readme, re.DOTALL).group(1)
    assert 'CHECKPOINT = "models/qmix_spread_n2_seed0.pt"' in code
    code = code.replace('"models/qmix_spread_n2_seed0.pt"', repr(str(path)))
    exec(compile(code, "models/README.md", "exec"), {})
    out = capsys.readouterr().out
    assert "QMIX:" in out and "random:" in out and "26 (700, 700, 3)" in out
