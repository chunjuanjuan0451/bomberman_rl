"""Fresh-seed shared-roster confirmation of the post-hoc local7-n8 r3 candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_dqn.callbacks import MODEL_ARCHITECTURE  # noqa: E402
from agent_code.model_a_dqn.network import torch  # noqa: E402
from agent_code.model_a_v4_global_resource_n8.config import architecture_name  # noqa: E402
from tools.v4_opponent_shift import pinned_external_checkout  # noqa: E402
from tools.v4_task4_duel_training import (  # noqa: E402
    atomic_json, load_completed, metrics_by_agent, relative, seed_values,
)


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-local7-r3-final-roster-confirmation-s143000.json"
LABELS = ("v4", "source-r2", "local7-r3", "external-strong")
INTERNAL_LABELS = LABELS[:3]
AGENT_CODES = {
    "v4": "model_a_oppshift_v4",
    "source-r2": "model_a_oppshift_source",
    "local7-r3": "model_a_v4_global_resource_n8",
    "external-strong": "feature_is_everything_eval",
}
ROSTERS = (
    ["v4", "source-r2", "local7-r3", "external-strong"],
    ["source-r2", "local7-r3", "external-strong", "v4"],
    ["local7-r3", "external-strong", "v4", "source-r2"],
    ["external-strong", "v4", "source-r2", "local7-r3"],
) * 2
EXPECTED_CASES = tuple(
    {
        "case_id": f"c{index + 1}",
        "world_seed": 143001 + index,
        "agent_seeds": {
            "v4": 243001 + index,
            "source-r2": 343001 + index,
            "local7-r3": 443001 + index,
            "external-strong": 543001 + index,
        },
        "roster": roster,
    }
    for index, roster in enumerate(ROSTERS)
)
SOURCE_PATHS = {
    "agent_code/model_a_dqn/callbacks.py",
    "agent_code/model_a_dqn/features.py",
    "agent_code/model_a_dqn/network.py",
    "agent_code/model_a_opponent_shift/common.py",
    "agent_code/model_a_oppshift_v4/callbacks.py",
    "agent_code/model_a_oppshift_source/callbacks.py",
    "agent_code/model_a_v4_global_resource_n8/config.py",
    "agent_code/model_a_v4_global_resource_n8/features.py",
    "agent_code/model_a_v4_global_resource_n8/network.py",
    "agent_code/model_a_v4_global_resource_n8/callbacks.py",
    "agent_code/model_a_v4_global_resource/features.py",
    "agent_code/model_a_v4_global_resource/network.py",
    "agent_code/feature_is_everything_eval/callbacks.py",
    "tools/v4_opponent_shift.py",
    "tools/v4_task4_duel_training.py",
    "tools/v4_local7_r3_final_roster_confirmation.py",
}
EVALUATION_KIND = "model-a-v4-local7-r3-final-roster-confirmation-evaluation"
REPORT_KIND = "model-a-v4-local7-r3-final-roster-confirmation-report"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "online_net" not in payload:
        raise ValueError(f"invalid s143000 checkpoint: {path}")
    return payload


def registered_seeds(protocol: dict) -> set[int]:
    values = set()
    for case in protocol["evaluation"]["cases"]:
        values.add(int(case["world_seed"]))
        values.update(int(seed) for seed in case["agent_seeds"].values())
    return values


def assert_registered_seeds_untouched(protocol_path: Path, protocol: dict) -> None:
    registered = registered_seeds(protocol)
    output_root = (ROOT / protocol["evaluation_manifest_directory"]).resolve()
    excluded = {protocol_path.resolve(), (ROOT / protocol["report_path"]).resolve()}
    conflicts = []
    for path in (ROOT / "experiments").rglob("*.json"):
        resolved = path.resolve()
        if resolved in excluded or output_root == resolved or output_root in resolved.parents:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        overlap = sorted(registered & seed_values(payload))
        if overlap:
            conflicts.append({"path": relative(path), "seeds": overlap})
    if conflicts:
        raise ValueError(f"registered s143000 seeds were already used: {conflicts}")


def _validate_checkpoint(protocol: dict, label: str) -> None:
    item = protocol["checkpoint_inventory"][label]
    path = ROOT / item["path"]
    if not path.is_file() or sha256_file(path) != item["sha256"]:
        raise ValueError(f"s143000 checkpoint mismatch: {label}")
    payload = _torch_load(path)
    expected_architecture = architecture_name() if label == "local7-r3" else MODEL_ARCHITECTURE
    if payload.get("architecture") not in (None, expected_architecture):
        raise ValueError(f"s143000 architecture mismatch: {label}")
    for key, value in item.get("required_metadata", {}).items():
        if payload.get(key) != value:
            raise ValueError(f"s143000 lineage mismatch: {label}/{key}")
    if label == "local7-r3":
        training_protocol = ROOT / item["training_protocol_path"]
        if sha256_file(training_protocol) != item["required_metadata"]["protocol_sha256"]:
            raise ValueError("s143000 local7 training protocol drift")
    elif label == "source-r2":
        source_protocol = ROOT / item["protocol_path"]
        if sha256_file(source_protocol) != item["required_metadata"]["protocol_sha256"]:
            raise ValueError("s143000 source-r2 protocol drift")


def _expected_external() -> dict:
    return {
        "repository": "https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman.git",
        "commit": "a7fe5041b02548ce4438e502ca3adb11576bae75",
        "agent_subdirectory": "agent_code/feature_is_everything",
        "known_previous_opponent": True,
        "training_or_teacher_use_allowed": False,
        "submission_candidate": False,
        "license_file_present": False,
        "file_sha256": {
            "callbacks.py": "cadc00f69e4b0e04a1063e6760deea1ed48e23855992201c3edcebe6284071fe",
            "features.py": "8baf96ae76f685a5dbfb74bae644e2cbb816952103859081801888ebe6b9f620",
            "model.py": "27736c5afa4ccf50582141b6aea57740699365e35879ad6d5cb4842b3fc7829c",
            "my-saved-model.pt": "0fb1df0b3c3f255db121e8ba7c6061d1293f55d3e7ae12772259643ee2060577",
            "symmetry.py": "cac8ad3d8b197d1cca6fa67ca35d3db00d45c4c81afa549d797bb8b45657f82e",
        },
    }


def load_protocol(path: Path) -> tuple[dict, Path, str]:
    resolved = path if path.is_absolute() else ROOT / path
    resolved = resolved.resolve()
    raw = resolved.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("schema_version") != 1 or protocol.get("kind") != "model-a-v4-opponent-shift":
        raise ValueError("wrong s143000 protocol kind")
    if protocol.get("protocol_id") != "model-a-v4-local7-r3-final-roster-confirmation-s143000":
        raise ValueError("wrong s143000 protocol id")
    if tuple(protocol.get("labels", ())) != LABELS or protocol.get("post_hoc_candidate_fixed_before_s143000") != "local7-r3":
        raise ValueError("s143000 identity contract changed")
    if any(protocol.get(key) is not False for key in (
        "training_allowed", "selection_allowed", "checkpoint_copy_allowed", "automatic_followup",
    )):
        raise ValueError("s143000 must remain evaluation-only and terminal")
    evaluation = protocol.get("evaluation", {})
    if (
        evaluation.get("scenario") != "classic"
        or int(evaluation.get("rounds_per_case", 0)) != 50
        or tuple(evaluation.get("cases", ())) != EXPECTED_CASES
        or evaluation.get("shared_games_total") != 400
    ):
        raise ValueError("s143000 evaluation contract changed")
    if len(registered_seeds(protocol)) != 40:
        raise ValueError("s143000 requires forty globally unique seed values")
    if set(protocol.get("checkpoint_inventory", {})) != set(INTERNAL_LABELS):
        raise ValueError("s143000 checkpoint inventory changed")
    for label in INTERNAL_LABELS:
        _validate_checkpoint(protocol, label)
    reference = protocol["candidate_discovery_reference"]
    reference_path = ROOT / reference["path"]
    if not reference_path.is_file() or sha256_file(reference_path) != reference["sha256"]:
        raise ValueError("s142000 discovery reference drift")
    reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
    if (
        reference_payload.get("status") != "completed"
        or reference_payload.get("result", {}).get("decision") != reference["formal_decision"]
        or reference_payload.get("result", {}).get("individual_replica_selected") is not False
    ):
        raise ValueError("s142000 terminal state changed")
    if protocol.get("external_stress_opponent") != _expected_external():
        raise ValueError("s143000 external stress binding changed")
    if set(protocol.get("source_bindings", {})) != SOURCE_PATHS:
        raise ValueError("s143000 source binding set changed")
    for source, expected in protocol["source_bindings"].items():
        source_path = ROOT / source
        if not source_path.is_file() or sha256_file(source_path) != expected:
            raise ValueError(f"s143000 source binding mismatch: {source}")
    assert_registered_seeds_untouched(resolved, protocol)
    return protocol, resolved, hashlib.sha256(raw).hexdigest()


def manifest_path(protocol: dict, case_id: str) -> Path:
    return ROOT / protocol["evaluation_manifest_directory"] / f"{case_id}.json"


def report_path(protocol: dict) -> Path:
    return ROOT / protocol["report_path"]


def evaluate_one(
    protocol_path: Path,
    protocol: dict,
    protocol_hash: str,
    case: dict,
    external_agent_dir: Path,
) -> dict:
    output = manifest_path(protocol, case["case_id"])
    expected = {
        "case_id": case["case_id"], "world_seed": case["world_seed"],
        "agent_seeds": case["agent_seeds"], "roster": case["roster"],
        "scenario": "classic", "rounds": 50,
    }
    existing = load_completed(output, EVALUATION_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash or existing.get("evaluation") != expected:
            raise RuntimeError(f"completed s143000 evaluation drift: {output}")
        return existing
    stats = output.with_suffix(".stats.json")
    if output.exists() or stats.exists():
        raise RuntimeError(f"refusing orphaned s143000 output: {output}")
    codes = [AGENT_CODES[label] for label in case["roster"]]
    command = [
        sys.executable, "main.py", "play", "--agents", *codes,
        "--train", "0", "--continue-without-training", "--no-gui",
        "--scenario", "classic", "--n-rounds", "50", "--seed", str(case["world_seed"]),
        "--save-stats", str(stats),
    ]
    candidate = protocol["checkpoint_inventory"]["local7-r3"]
    overrides = {
        "MODEL_A_OPPONENT_SHIFT_PROTOCOL_PATH": str(protocol_path),
        "MODEL_A_OPPONENT_SHIFT_CASE_ID": case["case_id"],
        "OPPONENT_SHIFT_EXTERNAL_AGENT_DIR": str(external_agent_dir),
        "MODEL_A_GLOBAL_PROTOCOL_PATH": str((ROOT / candidate["training_protocol_path"]).resolve()),
        "MODEL_A_GLOBAL_RUN_MODE": "evaluate",
        "MODEL_A_GLOBAL_ARM": "local7",
        "MODEL_A_GLOBAL_REPLICA": "r3",
        "MODEL_A_GLOBAL_SEED": str(case["agent_seeds"]["local7-r3"]),
        "MODEL_A_GLOBAL_CHECKPOINT_PATH": str((ROOT / candidate["path"]).resolve()),
    }
    hashes_before = {
        label: sha256_file(ROOT / protocol["checkpoint_inventory"][label]["path"])
        for label in INTERNAL_LABELS
    }
    record = {
        "schema_version": 1, "kind": EVALUATION_KIND, "status": "running",
        "started_at_utc": utc_now(), "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path), "protocol_sha256": protocol_hash,
        "evaluation": expected, "command": command,
        "environment_overrides": {**overrides, "OPPONENT_SHIFT_EXTERNAL_AGENT_DIR": "<temporary-pinned-checkout>"},
        "checkpoint_sha256_before": hashes_before,
        "external_commit": protocol["external_stress_opponent"]["commit"],
        "raw_stats": relative(stats), "policy_updates": 0,
    }
    atomic_json(output, record)
    child = os.environ.copy()
    child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = utc_now()
    record["exit_code"] = completed.returncode
    try:
        hashes_after = {
            label: sha256_file(ROOT / protocol["checkpoint_inventory"][label]["path"])
            for label in INTERNAL_LABELS
        }
        if completed.returncode != 0 or not stats.is_file() or hashes_after != hashes_before:
            raise RuntimeError("s143000 process, stats, or checkpoint immutability failed")
        raw = metrics_by_agent(json.loads(stats.read_text(encoding="utf-8")))
        reverse = {code: label for label, code in AGENT_CODES.items()}
        if set(raw) != set(reverse):
            raise RuntimeError(f"unexpected s143000 agents: {sorted(raw)}")
        record["metrics_by_label"] = {reverse[code]: metrics for code, metrics in raw.items()}
        record["checkpoint_sha256_after"] = hashes_after
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(output, record)
    if record["status"] != "completed":
        raise RuntimeError(f"s143000 evaluation failed: {case['case_id']}: {record.get('error')}")
    return record


def aggregate_label(runs: list[dict], label: str) -> dict:
    additive = (
        "rounds", "score", "coins", "kills", "crates", "bombs", "moves",
        "waits", "suicides", "invalid_actions", "steps",
    )
    totals = {key: sum(int(run["metrics_by_label"][label][key]) for run in runs) for key in additive}
    rounds, steps = totals["rounds"], totals["steps"]
    return {
        **totals,
        "score_per_round": totals["score"] / rounds,
        "coins_per_round": totals["coins"] / rounds,
        "kills_per_round": totals["kills"] / rounds,
        "crates_per_round": totals["crates"] / rounds,
        "bombs_per_round": totals["bombs"] / rounds,
        "suicides_per_round": totals["suicides"] / rounds,
        "invalid_actions_per_round": totals["invalid_actions"] / rounds,
        "move_fraction": totals["moves"] / steps if steps else 0.0,
        "wait_fraction": totals["waits"] / steps if steps else 0.0,
        "episode_length": steps / rounds,
    }


def _ranking(metrics: dict[str, dict]) -> list[str]:
    return sorted(
        LABELS,
        key=lambda label: (
            metrics[label]["score_per_round"], metrics[label]["kills_per_round"],
            -metrics[label]["suicides_per_round"], -LABELS.index(label),
        ),
        reverse=True,
    )


def summarize(runs: list[dict]) -> tuple[dict, list[dict]]:
    pooled = {label: aggregate_label(runs, label) for label in LABELS}
    blocks = []
    for run in runs:
        metrics = run["metrics_by_label"]
        candidate_gain = metrics["local7-r3"]["score_per_round"] - metrics["v4"]["score_per_round"]
        blocks.append({
            "case_id": run["evaluation"]["case_id"],
            "world_seed": run["evaluation"]["world_seed"],
            "roster": run["evaluation"]["roster"],
            "metrics": metrics,
            "ranking": _ranking(metrics),
            "candidate_minus_v4_score_per_round": candidate_gain,
            "candidate_leads_v4_by_score": candidate_gain > 1e-12,
        })
    return pooled, blocks


def decide(protocol: dict, pooled: dict, blocks: list[dict]) -> dict:
    gate = protocol["confirmation_gate"]
    candidate = pooled["local7-r3"]
    v4 = pooled["v4"]
    score_gain = candidate["score_per_round"] - v4["score_per_round"]
    lead_blocks = sum(bool(block["candidate_leads_v4_by_score"]) for block in blocks)
    checks = {
        "pooled_score_gain": score_gain >= float(gate["minimum_candidate_score_gain_over_v4"]) - 1e-12,
        "block_support": lead_blocks >= int(gate["minimum_candidate_lead_blocks_over_v4"]),
        "internal_score_leader": candidate["score_per_round"] >= max(
            pooled["v4"]["score_per_round"], pooled["source-r2"]["score_per_round"],
        ) - 1e-12,
        "suicide_guard": candidate["suicides_per_round"] <= v4["suicides_per_round"] + float(
            gate["maximum_candidate_suicide_increase_over_v4"],
        ) + 1e-12,
        "invalid_guard": candidate["invalid_actions_per_round"] <= v4["invalid_actions_per_round"] + float(
            gate["maximum_candidate_invalid_increase_over_v4"],
        ) + 1e-12,
    }
    passed = all(checks.values())
    return {
        "decision": protocol["decision_branches"]["pass" if passed else "fail"],
        "passed": passed,
        "checks": checks,
        "candidate_minus_v4_score_per_round": score_gain,
        "candidate_minus_v4_coins_per_round": candidate["coins_per_round"] - v4["coins_per_round"],
        "candidate_minus_v4_kills_per_round": candidate["kills_per_round"] - v4["kills_per_round"],
        "candidate_minus_v4_suicides_per_round": candidate["suicides_per_round"] - v4["suicides_per_round"],
        "candidate_minus_v4_invalid_actions_per_round": candidate["invalid_actions_per_round"] - v4["invalid_actions_per_round"],
        "candidate_lead_blocks": lead_blocks,
        "minimum_candidate_lead_blocks": gate["minimum_candidate_lead_blocks_over_v4"],
        "external_minus_candidate_score_per_round": pooled["external-strong"]["score_per_round"] - candidate["score_per_round"],
        "pooled_ranking": _ranking(pooled),
        "training_started": False,
        "selection_made_within_s143000": False,
        "checkpoint_copied": False,
        "incumbent_replaced": False,
        "external_used_for_training_or_teacher_labels": False,
        "automatic_followup_started": False,
        "awaiting_user_instruction": True,
    }


def dry_run(protocol: dict) -> dict:
    cases = len(protocol["evaluation"]["cases"])
    rounds = int(protocol["evaluation"]["rounds_per_case"])
    return {
        "mode": "dry-run", "protocol_id": protocol["protocol_id"],
        "labels": list(LABELS), "fixed_candidate": "local7-r3",
        "cases": cases, "rounds_per_case": rounds,
        "shared_games_total": cases * rounds,
        "rounds_observed_per_label": cases * rounds,
        "roster_orders": [case["roster"] for case in protocol["evaluation"]["cases"]],
        "external_commit": protocol["external_stress_opponent"]["commit"],
        "external_checkout_started": False, "formal_evaluation_started": False,
        "training_started": False, "selection_allowed": False,
        "checkpoint_copy_allowed": False, "automatic_followup_started": False,
        "writes_terminal_report_and_stops": True,
    }


def execute(protocol_path: Path, protocol: dict, protocol_hash: str) -> dict:
    with pinned_external_checkout(protocol) as external_agent_dir:
        runs = [
            evaluate_one(protocol_path, protocol, protocol_hash, case, external_agent_dir)
            for case in protocol["evaluation"]["cases"]
        ]
    pooled, blocks = summarize(runs)
    report = {
        "schema_version": 1, "kind": REPORT_KIND, "status": "completed",
        "completed_at_utc": utc_now(), "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path), "protocol_sha256": protocol_hash,
        "evaluation_manifests": [
            {"path": relative(manifest_path(protocol, case["case_id"])),
             "sha256": sha256_file(manifest_path(protocol, case["case_id"]))}
            for case in protocol["evaluation"]["cases"]
        ],
        "pooled": pooled, "case_blocks": blocks,
        "result": decide(protocol, pooled, blocks),
        "source_artifacts_modified": False, "default_checkpoint_modified": False,
        "external_source_persisted_in_repository": False,
        "training_started": False, "checkpoint_copied": False,
        "automatic_followup_started": False, "awaiting_user_instruction": True,
    }
    atomic_json(report_path(protocol), report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol, protocol_path, protocol_hash = load_protocol(args.protocol)
    if not args.execute:
        print(json.dumps({**dry_run(protocol), "protocol_sha256": protocol_hash}, indent=2, sort_keys=True))
        return 0
    terminal = report_path(protocol)
    if terminal.exists():
        raise SystemExit(f"terminal report already exists: {terminal}")
    report = execute(protocol_path, protocol, protocol_hash)
    print(json.dumps({
        "status": report["status"], "report": relative(terminal),
        "report_sha256": sha256_file(terminal), **report["result"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
