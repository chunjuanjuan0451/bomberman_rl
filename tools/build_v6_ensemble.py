"""Build an auditable inference bundle from stable-v6 residual checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_v6_ensemble.config import architecture_name, load_config, sha256_file
from agent_code.model_a_v6_stable.config import architecture_name as stable_architecture_name
from agent_code.model_a_v6_stable.network import torch


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def build(config_path: Path, output_path: Path) -> dict:
    config, _, config_sha256 = load_config(config_path)
    parent_state = _load(_resolve(config["parent_checkpoint"]))["online_net"]
    residual_states = []
    for member in config["members"]:
        checkpoint = _load(_resolve(member["checkpoint"]))
        if checkpoint.get("architecture") != stable_architecture_name():
            raise RuntimeError(f"member architecture mismatch: {member['training_seed']}")
        if checkpoint.get("config_sha256") != member["config_sha256"]:
            raise RuntimeError(f"member config metadata mismatch: {member['training_seed']}")
        if checkpoint.get("parent_sha256") != config["parent_sha256"]:
            raise RuntimeError(f"member parent metadata mismatch: {member['training_seed']}")
        state = checkpoint["online_net"]
        if any(not torch.equal(value, state[f"base.{name}"]) for name, value in parent_state.items()):
            raise RuntimeError(f"member frozen base mismatch: {member['training_seed']}")
        residual_states.append({
            name.removeprefix("residual."): value
            for name, value in state.items() if name.startswith("residual.")
        })
    return {
        "architecture": architecture_name(),
        "aggregation": "mean",
        "config_sha256": config_sha256,
        "parent_sha256": config["parent_sha256"],
        "member_sha256": [member["sha256"] for member in config["members"]],
        "residual_states": residual_states,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite ensemble bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build(args.config.resolve(), output), output)
    print(f"{output}\nsha256={sha256_file(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
