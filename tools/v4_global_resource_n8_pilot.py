"""Run the preregistered local7/full33 n=8 resource-visibility rescue pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_v4_global_resource_n8.callbacks import _torch_load  # noqa: E402
from agent_code.model_a_v4_global_resource_n8.config import (  # noqa: E402
    ARMS, ENDPOINT_ROUND, MILESTONES, REPLICAS, architecture_name,
    checkpoint_path, load_protocol, milestone_path, sha256_file,
)
from agent_code.model_a_v4_global_resource_n8.network import torch  # noqa: E402
from tools import v4_global_resource_stage1 as base  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-global-resource-n8-pilot-s141000.json"
TRAINING_KIND = "model-a-v4-global-resource-n8-pilot-training"
EVALUATION_KIND = "model-a-v4-global-resource-n8-pilot-evaluation"
REPORT_KIND = "model-a-v4-global-resource-n8-pilot-report"
TARGET_AGENT = "model_a_v4_global_resource_n8"


def _configure_base() -> None:
    base.ARMS = ARMS
    base.REPLICAS = REPLICAS
    base.ENDPOINT_ROUND = ENDPOINT_ROUND
    base.DEFAULT_PROTOCOL = DEFAULT_PROTOCOL
    base.TRAINING_KIND = TRAINING_KIND
    base.EVALUATION_KIND = EVALUATION_KIND
    base.REPORT_KIND = REPORT_KIND
    base.TARGET_AGENT = TARGET_AGENT
    base.load_protocol = load_protocol
    base.architecture_name = architecture_name
    base.checkpoint_path = checkpoint_path
    base.milestone_path = milestone_path
    base.sha256_file = sha256_file
    base._torch_load = _torch_load
    base.validate_diagnostic = validate_diagnostic


def validate_checkpoint(
    protocol: dict,
    protocol_hash: str,
    arm: str,
    replica: str,
    round_number: int = ENDPOINT_ROUND,
) -> dict:
    path = milestone_path(protocol, arm, replica, round_number)
    if not path.is_file():
        raise RuntimeError(f"global-resource n8 checkpoint missing: {path}")
    payload = _torch_load(path)
    expected = {
        "architecture": architecture_name(),
        "protocol_sha256": protocol_hash,
        "parent_sha256": protocol["frozen_v4"]["sha256"],
        "arm": arm,
        "replica": replica,
        "completed_rounds": round_number,
        "single_training_variable": protocol["single_training_variable"],
        "epsilon": 0.10,
        "n_step": 8,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"global-resource n8 checkpoint {key} mismatch: {path}")
    parent = _torch_load(ROOT / protocol["frozen_v4"]["path"])["online_net"]
    online = payload.get("online_net", {})
    if set(f"base.{key}" for key in parent) - set(online):
        raise RuntimeError(f"global-resource n8 base schema mismatch: {path}")
    if any(not torch.equal(online[f"base.{key}"], value) for key, value in parent.items()):
        raise RuntimeError(f"frozen-v4 tensors changed: {path}")
    diagnostics = payload.get("training_diagnostics", {})
    histogram = diagnostics.get("return_steps_histogram", {})
    if (
        diagnostics.get("raw_transitions") != payload.get("training_steps")
        or diagnostics.get("matured_targets") != diagnostics.get("raw_transitions")
        or sum(histogram.values()) != diagnostics.get("matured_targets")
        or len(diagnostics.get("per_round", ())) != round_number
        or sum(item["transitions"] for item in diagnostics.get("per_round", ())) != diagnostics.get("raw_transitions")
        or sum(item["optimizer_updates"] for item in diagnostics.get("per_round", ())) != diagnostics.get("gradient_updates")
    ):
        raise RuntimeError(f"global-resource n8 accounting mismatch: {path}")
    return payload


def validate_diagnostic(path: Path, protocol_hash: str, arm: str, replica: str, rounds: int) -> dict:
    if not path.is_file():
        raise RuntimeError(f"global-resource n8 evaluation diagnostic missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "kind": "model-a-v4-global-resource-n8-pilot-evaluation-diagnostic",
        "protocol_sha256": protocol_hash,
        "arm": arm,
        "replica": replica,
        "policy_updates": 0,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"global-resource n8 evaluation diagnostic mismatch: {path}")
    entries = payload.get("rounds", [])
    if len(entries) != rounds or any(sum(item["actions"].values()) != item["transitions"] for item in entries):
        raise RuntimeError(f"global-resource n8 evaluation accounting mismatch: {path}")
    return payload


def train_one(protocol_path: Path, protocol: dict, protocol_hash: str, arm: str, replica: str) -> dict:
    manifest = base.training_manifest_path(protocol, arm, replica)
    existing = base.load_completed(manifest, TRAINING_KIND)
    if existing is not None:
        validate_checkpoint(protocol, protocol_hash, arm, replica)
        if existing["checkpoint"]["sha256"] != sha256_file(checkpoint_path(protocol, arm, replica)):
            raise RuntimeError(f"completed global-resource n8 checkpoint drift: {arm}/{replica}")
        return existing
    endpoint = checkpoint_path(protocol, arm, replica)
    stats = manifest.with_suffix(".stats.json")
    if endpoint.parent.exists() or manifest.exists() or stats.exists():
        raise RuntimeError(f"refusing orphaned global-resource n8 artifacts: {arm}/{replica}")
    case = protocol["training"]["seeds_by_replica"][replica]
    command = [
        sys.executable, "main.py", "play", "--agents", TARGET_AGENT,
        "--train", "1", "--continue-without-training", "--no-gui",
        "--scenario", "classic", "--n-rounds", str(ENDPOINT_ROUND),
        "--seed", str(case["world_seed"]), "--save-stats", str(stats),
    ]
    overrides = {
        "MODEL_A_GLOBAL_PROTOCOL_PATH": str(protocol_path),
        "MODEL_A_GLOBAL_RUN_MODE": "train",
        "MODEL_A_GLOBAL_ARM": arm,
        "MODEL_A_GLOBAL_REPLICA": replica,
        "MODEL_A_GLOBAL_SEED": str(case["agent_seed"]),
        "MODEL_A_GLOBAL_CHECKPOINT_PATH": str(endpoint),
    }
    record = {
        "schema_version": 1,
        "kind": TRAINING_KIND,
        "status": "running",
        "started_at_utc": base.utc_now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_path": base.relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "arm": arm,
        "replica": replica,
        "training": {**case, "scenario": "classic", "opponents": [], "rounds": ENDPOINT_ROUND},
        "command": command,
        "environment_overrides": overrides,
        "checkpoint": {"path": base.relative(endpoint), "sha256": None},
        "raw_stats": base.relative(stats),
    }
    base.atomic_json(manifest, record)
    child = os.environ.copy()
    child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = base.utc_now()
    record["exit_code"] = completed.returncode
    try:
        if completed.returncode != 0 or not stats.is_file():
            raise RuntimeError("global-resource n8 training process or stats failed")
        payload = validate_checkpoint(protocol, protocol_hash, arm, replica)
        for round_number in MILESTONES:
            validate_checkpoint(protocol, protocol_hash, arm, replica, round_number)
        record["checkpoint"]["sha256"] = sha256_file(endpoint)
        record["training_diagnostics"] = payload["training_diagnostics"]
        record["metrics_by_agent"] = base.metrics_by_agent(json.loads(stats.read_text(encoding="utf-8")))
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    base.atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"global-resource n8 training failed: {arm}/{replica}: {record.get('error')}")
    return record


def execute(protocol_path: Path, protocol: dict, protocol_hash: str, preflight_result: dict) -> dict:
    _configure_base()
    if not preflight_result["passed"]:
        raise RuntimeError("global-resource n8 preflight failed")
    training = []
    for replica in REPLICAS:
        for arm in ARMS:
            training.append(train_one(protocol_path, protocol, protocol_hash, arm, replica))
    effect = base.parameter_effect(protocol)
    evaluations = []
    details = None
    if effect["evaluation_authorized"]:
        for stratum, spec in protocol["evaluation"]["strata"].items():
            for label in protocol["evaluation"]["labels"]:
                for case in spec["cases"]:
                    evaluations.append(base.evaluate_one(protocol_path, protocol, protocol_hash, stratum, label, case))
        old_decision, details = base.decide(protocol, evaluations, effect)
        decision = (
            "n8_rescues_fullboard_signal_stop_before_confirmation"
            if old_decision == "fullboard_resource_visibility_supported_stop_before_confirmation"
            else "n8_does_not_rescue_fullboard_signal_stop"
        )
    else:
        decision = "global_resource_n8_no_parameter_effect_stop"
    report = {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "status": "completed",
        "completed_at_utc": base.utc_now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_path": base.relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "preflight": preflight_result,
        "parameter_effect": effect,
        "training_manifests": [
            {"path": base.relative(base.training_manifest_path(protocol, run["arm"], run["replica"])),
             "sha256": sha256_file(base.training_manifest_path(protocol, run["arm"], run["replica"]))}
            for run in training
        ],
        "evaluation_manifests": [
            {"path": base.relative(base.evaluation_manifest_path(
                protocol, run["stratum"], run["label"], run["evaluation"]["world_seed"])),
             "sha256": sha256_file(base.evaluation_manifest_path(
                protocol, run["stratum"], run["label"], run["evaluation"]["world_seed"]))}
            for run in evaluations
        ],
        "decision": decision,
        "decision_details": details,
        "training_started": True,
        "evaluation_started": bool(evaluations),
        "intermediate_checkpoint_selection_used": False,
        "checkpoint_selected": False,
        "checkpoint_copied": False,
        "automatic_followup_started": False,
        "awaiting_user_instruction": True,
    }
    base.atomic_json(base.report_path(protocol), report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    _configure_base()
    protocol, protocol_path, protocol_hash = load_protocol(args.protocol)
    base.assert_registered_seeds_untouched(protocol_path, protocol)
    preflight_result = base.preflight(protocol)
    dry = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "single_training_variable": protocol["single_training_variable"],
        "arms": ["frozen-v4", *ARMS],
        "replicas": list(REPLICAS),
        "n_step": 8,
        "training_rounds": 2 * len(REPLICAS) * ENDPOINT_ROUND,
        "evaluation_rounds_if_authorized": sum(
            len(protocol["evaluation"]["labels"]) * len(spec["cases"]) * spec["rounds_per_case"]
            for spec in protocol["evaluation"]["strata"].values()
        ),
        "candidate_endpoint": ENDPOINT_ROUND,
        "intermediate_selection_allowed": False,
        "new_attack_reward_added": False,
        "kills_used_for_gate": False,
        "automatic_followup_started": False,
        "formal_execution_started": False,
        "preflight": preflight_result,
    }
    if not args.execute:
        print(json.dumps(dry, indent=2, sort_keys=True))
        return 0
    terminal = base.report_path(protocol)
    if terminal.exists():
        raise SystemExit(f"terminal report already exists: {terminal}")
    report = execute(protocol_path, protocol, protocol_hash, preflight_result)
    print(json.dumps({
        "status": report["status"],
        "decision": report["decision"],
        "report": base.relative(terminal),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
