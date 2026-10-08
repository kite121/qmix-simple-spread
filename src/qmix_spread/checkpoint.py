"""Saving and loading a trained model together with everything needed to rebuild it."""

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch

from .config import QMIXConfig
from .env import SpreadEnv
from .networks import AgentNetwork, QMixer

CHECKPOINT_FORMAT = 1


def library_versions() -> dict[str, str]:
    """Versions that matter for reproducing a run."""
    versions = {"python": platform.python_version()}
    for package in ["torch", "numpy", "mpe2", "pettingzoo", "gymnasium"]:
        versions[package] = version(package)
    return versions


def build_networks(config: QMIXConfig, env: SpreadEnv) -> tuple[AgentNetwork, QMixer]:
    """Fresh (untrained) networks with sizes taken from the config and the environment."""
    agent_net = AgentNetwork(env.obs_dim, env.n_agents, env.n_actions, config.agent_hidden_dim)
    mixer = QMixer(env.n_agents, env.state_dim, config.mixing_embed_dim, config.mixer_state_limit)
    return agent_net, mixer


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_checkpoint(
    path: str | Path, agent_net: AgentNetwork, mixer: QMixer, config: QMIXConfig, metadata: dict
) -> str:
    """Write the model to `path` (.pt) and a readable description next to it (.json).

    metadata: plain numbers and strings only (steps, wall time, evaluation result, ...).
    Returns the SHA-256 of the .pt file, so others can say exactly which file they used.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dims = {
        "obs_dim": agent_net.obs_dim,
        "state_dim": mixer.state_dim,
        "n_agents": agent_net.n_agents,
        "n_actions": agent_net.n_actions,
    }
    full_metadata = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "library_versions": library_versions(),
        **metadata,
    }
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "config": config.to_dict(),
            "dims": dims,
            "agent_net": agent_net.state_dict(),
            "mixer": mixer.state_dict(),
            "metadata": full_metadata,
        },
        path,
    )
    sha256 = file_sha256(path)
    description = {
        "checkpoint_file": path.name,
        "sha256": sha256,
        "size_bytes": path.stat().st_size,
        "n_parameters": {
            "agent_net": sum(p.numel() for p in agent_net.parameters()),
            "mixer": sum(p.numel() for p in mixer.parameters()),
        },
        "dims": dims,
        "config": config.to_dict(),
        "metadata": full_metadata,
    }
    save_json(path.with_suffix(".json"), description)
    return sha256


@dataclass
class LoadedModel:
    agent_net: AgentNetwork
    mixer: QMixer
    config: QMIXConfig
    metadata: dict


def load_checkpoint(path: str | Path) -> LoadedModel:
    """Rebuild both networks from a checkpoint. No training happens here."""
    # weights_only=True: only tensors and plain Python values are read, no arbitrary code.
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["format"] != CHECKPOINT_FORMAT:
        raise ValueError(f"unknown checkpoint format {data['format']}")
    config = QMIXConfig.from_dict(data["config"])
    dims = data["dims"]
    agent_net = AgentNetwork(dims["obs_dim"], dims["n_agents"], dims["n_actions"], config.agent_hidden_dim)
    mixer = QMixer(dims["n_agents"], dims["state_dim"], config.mixing_embed_dim, config.mixer_state_limit)
    agent_net.load_state_dict(data["agent_net"])
    mixer.load_state_dict(data["mixer"])
    agent_net.eval()
    mixer.eval()
    return LoadedModel(agent_net, mixer, config, data["metadata"])


def save_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
