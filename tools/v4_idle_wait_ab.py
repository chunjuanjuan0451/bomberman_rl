"""Run the preregistered exact-v4 conditional idle-WAIT short-training A/B."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_dqn.network import torch  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-idle-wait-ab-s146000.json"
ARMS = ("control", "idle-wait-001")
REPLICAS = ("r1", "r2", "r3")
TRAINING_KIND = "model-a-v4-idle-wait-ab-training"
EVALUATION_KIND = "model-a-v4-idle-wait-ab-evaluation"
REPORT_KIND = "model-a-v4-idle-wait-ab-report"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def seed_values(value) -> set[int]:
    found: set[int] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"seed", "world_seed", "agent_seed"} and isinstance(item, int):
                found.add(item)
            found.update(seed_values(item))
    elif isinstance(value, list):
        for item in value:
            found.update(seed_values(item))
    return found


def registered_seeds(protocol: dict) -> set[int]:
    return seed_values({"training": protocol["training"], "evaluation": protocol["evaluation"]})


def checkpoint_path(protocol: dict, arm: str, replica: str) -> Path:
    return ROOT / protocol["checkpoint_directory"] / arm / replica / "round-0100.pt"


def training_manifest_path(protocol: dict, arm: str, replica: str) -> Path:
    return ROOT / protocol["training_manifest_directory"] / arm / f"{replica}.json"


def evaluation_manifest_path(protocol: dict, stratum: str, label: str, seed: int) -> Path:
    return ROOT / protocol["evaluation_manifest_directory"] / stratum / f"{label}-s{seed}.json"


def report_path(protocol: dict) -> Path:
    return ROOT / protocol["report_path"]


def _validate_source_bindings(protocol: dict) -> None:
    bindings = protocol.get("source_bindings", {})
    if not bindings:
        raise ValueError("source bindings are not frozen")
    for name, spec in bindings.items():
        path = ROOT / spec["path"]
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise ValueError(f"source binding drift: {name}")


def load_protocol(path: Path) -> tuple[dict, Path, str]:
    resolved = path if path.is_absolute() else ROOT / path
    resolved = resolved.resolve()
    raw = resolved.read_bytes()
    protocol = json.loads(raw)
    if (
        protocol.get("kind") != "model-a-v4-idle-wait-ab"
        or protocol.get("protocol_id") != "model-a-v4-idle-wait-ab-s146000"
        or protocol.get("status") != "preregistered_stop_before_execution"
    ):
        raise ValueError("wrong idle-WAIT A/B protocol")
    single = protocol["single_changed_variable"]
    if single.get("control_penalty") != 0.0 or single.get("candidate_penalty") != -0.01:
        raise ValueError("idle-WAIT treatment changed")
    training = protocol["training"]
    if (
        tuple(training.get("arms", ())) != ARMS
        or tuple(training.get("replicas", ())) != REPLICAS
        or training.get("scenario") != "classic"
        or training.get("opponents") != []
        or training.get("rounds_per_arm") != 100
        or training.get("total_rounds") != 600
    ):
        raise ValueError("idle-WAIT training contract changed")
    evaluation = protocol["evaluation"]
    if evaluation.get("rounds_per_label") != 125 or evaluation.get("total_rounds") != 875:
        raise ValueError("idle-WAIT evaluation budget changed")
    if any(protocol.get(key) is not value for key, value in (
        ("training_allowed", True), ("checkpoint_selection_allowed", False),
        ("checkpoint_copy_allowed", False), ("automatic_followup", False),
    )):
        raise ValueError("idle-WAIT stopping contract changed")
    parent = ROOT / protocol["parent_checkpoint"]["path"]
    if not parent.is_file() or sha256_file(parent) != protocol["parent_checkpoint"]["sha256"]:
        raise ValueError("frozen-v4 parent mismatch")
    payload = torch_load(parent)
    for key in ("completed_rounds", "training_steps", "gradient_steps"):
        if int(payload.get(key, -1)) != int(protocol["parent_checkpoint"][key]):
            raise ValueError(f"frozen-v4 parent {key} mismatch")
    if abs(float(payload.get("epsilon", -1)) - float(protocol["parent_checkpoint"]["epsilon"])) > 1e-12:
        raise ValueError("frozen-v4 parent epsilon mismatch")
    for name, item in protocol["audit_lineage"].items():
        audit = ROOT / item["path"]
        if not audit.is_file() or sha256_file(audit) != item["sha256"]:
            raise ValueError(f"audit lineage drift: {name}")
    task2 = json.loads((ROOT / protocol["audit_lineage"]["task2_report"]["path"]).read_text())
    if task2.get("summary", {}).get("recommendation") != protocol["audit_lineage"]["task2_report"]["recommendation"]:
        raise ValueError("Task2 audit no longer authorizes the pilot")
    _validate_source_bindings(protocol)
    return protocol, resolved, hashlib.sha256(raw).hexdigest()


def assert_registered_seeds_untouched(protocol_path: Path, protocol: dict) -> None:
    registered = registered_seeds(protocol)
    excluded_roots = {
        (ROOT / protocol[key]).resolve() for key in (
            "checkpoint_directory", "training_manifest_directory", "evaluation_manifest_directory",
        )
    }
    excluded_files = {protocol_path.resolve(), report_path(protocol).resolve()}
    conflicts = []
    for path in (ROOT / "experiments").rglob("*.json"):
        resolved = path.resolve()
        if resolved in excluded_files or any(root == resolved or root in resolved.parents for root in excluded_roots):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        overlap = sorted(registered & seed_values(payload))
        if overlap:
            conflicts.append({"path": relative(path), "seeds": overlap})
    if conflicts:
        raise ValueError(f"registered s146000 seeds were already used: {conflicts}")


def load_completed(path: Path, kind: str) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != kind or payload.get("status") != "completed":
        raise RuntimeError(f"refusing incomplete/incompatible artifact: {path}")
    return payload


def metrics_by_agent(stats: dict) -> dict[str, dict]:
    result = {}
    for name, raw in stats.get("by_agent", {}).items():
        rounds = int(raw.get("rounds", 0)); steps = int(raw.get("steps", 0))
        moves = int(raw.get("moves", 0)); bombs = int(raw.get("bombs", 0))
        invalid = int(raw.get("invalid", 0)); waits = max(0, steps - moves - bombs - invalid)
        values = {
            "rounds": rounds, "score": int(raw.get("score", 0)), "coins": int(raw.get("coins", 0)),
            "kills": int(raw.get("kills", 0)), "crates": int(raw.get("crates", 0)), "bombs": bombs,
            "moves": moves, "waits": waits, "suicides": int(raw.get("suicides", 0)),
            "invalid_actions": invalid, "steps": steps,
        }
        values.update({
            "score_per_round": values["score"] / rounds if rounds else 0.0,
            "coins_per_round": values["coins"] / rounds if rounds else 0.0,
            "crates_per_round": values["crates"] / rounds if rounds else 0.0,
            "bombs_per_round": bombs / rounds if rounds else 0.0,
            "suicides_per_round": values["suicides"] / rounds if rounds else 0.0,
            "invalid_actions_per_round": invalid / rounds if rounds else 0.0,
            "move_fraction": moves / steps if steps else 0.0,
            "wait_fraction": waits / steps if steps else 0.0,
        })
        result[name] = values
    return result


def aggregate(runs: list[dict]) -> dict:
    keys = ("rounds", "score", "coins", "kills", "crates", "bombs", "moves", "waits", "suicides", "invalid_actions", "steps")
    totals = {key: sum(int(run["target_metrics"][key]) for run in runs) for key in keys}
    return metrics_by_agent({"by_agent": {"pooled": {
        "rounds": totals["rounds"], "score": totals["score"], "coins": totals["coins"],
        "kills": totals["kills"], "crates": totals["crates"], "bombs": totals["bombs"],
        "moves": totals["moves"], "suicides": totals["suicides"],
        "invalid": totals["invalid_actions"], "steps": totals["steps"],
    }}})["pooled"]


def validate_checkpoint(protocol: dict, protocol_hash: str, arm: str, replica: str) -> dict:
    path = checkpoint_path(protocol, arm, replica)
    if not path.is_file():
        raise RuntimeError(f"candidate checkpoint missing: {path}")
    payload = torch_load(path)
    expected = {
        "architecture": "model-a-mlp-v4-global7", "protocol_sha256": protocol_hash,
        "parent_sha256": protocol["parent_checkpoint"]["sha256"], "arm": arm,
        "replica": replica, "experiment_completed_rounds": 100,
        "completed_rounds": int(protocol["parent_checkpoint"]["completed_rounds"]) + 100,
        "idle_wait_penalty": 0.0 if arm == "control" else -0.01,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"checkpoint {key} mismatch: {arm}/{replica}")
    if (
        int(payload.get("training_steps", 0)) <= int(protocol["parent_checkpoint"]["training_steps"])
        or int(payload.get("gradient_steps", 0)) <= int(protocol["parent_checkpoint"]["gradient_steps"])
        or abs(float(payload.get("epsilon", -1)) - 0.05) > 1e-12
    ):
        raise RuntimeError(f"checkpoint training state mismatch: {arm}/{replica}")
    diagnostics = payload.get("experiment_training_diagnostics", {})
    if (
        len(diagnostics.get("per_round", ())) != 100
        or sum(item["transitions"] for item in diagnostics.get("per_round", ())) != diagnostics.get("raw_transitions")
        or sum(diagnostics.get("action_counts", {}).values()) != diagnostics.get("raw_transitions")
        or diagnostics.get("eligible_idle_waits", 0) < diagnostics.get("penalized_idle_waits", 0)
    ):
        raise RuntimeError(f"checkpoint diagnostic accounting mismatch: {arm}/{replica}")
    expected_penalized = diagnostics.get("eligible_idle_waits", 0) if arm != "control" else 0
    if diagnostics.get("penalized_idle_waits") != expected_penalized:
        raise RuntimeError(f"checkpoint treatment accounting mismatch: {arm}/{replica}")
    return payload


def train_one(protocol_path: Path, protocol: dict, protocol_hash: str, arm: str, replica: str) -> dict:
    manifest = training_manifest_path(protocol, arm, replica)
    existing = load_completed(manifest, TRAINING_KIND)
    endpoint = checkpoint_path(protocol, arm, replica)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash or existing["checkpoint"]["sha256"] != sha256_file(endpoint):
            raise RuntimeError(f"completed training drift: {arm}/{replica}")
        validate_checkpoint(protocol, protocol_hash, arm, replica)
        return existing
    stats = manifest.with_suffix(".stats.json")
    if endpoint.parent.exists() or manifest.exists() or stats.exists():
        raise RuntimeError(f"refusing orphaned training artifacts: {arm}/{replica}")
    case = protocol["training"]["seeds"][replica]
    endpoint.parent.mkdir(parents=True)
    shutil.copy2(ROOT / protocol["parent_checkpoint"]["path"], endpoint)
    command = [
        sys.executable, "main.py", "play", "--agents", "model_a_v4_idle_wait", "--train", "1",
        "--continue-without-training", "--no-gui", "--scenario", "classic", "--n-rounds", "100",
        "--seed", str(case["world_seed"]), "--save-stats", str(stats),
    ]
    overrides = {
        "MODEL_A_CHECKPOINT_PATH": str(endpoint), "MODEL_A_RESUME": "1",
        "MODEL_A_SEED": str(case["agent_seed"]), "MODEL_A_IDLE_PROTOCOL_PATH": str(protocol_path),
        "MODEL_A_IDLE_ARM": arm, "MODEL_A_IDLE_REPLICA": replica,
    }
    record = {
        "schema_version": 1, "kind": TRAINING_KIND, "status": "running", "started_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_hash, "arm": arm, "replica": replica,
        "training": {**case, "scenario": "classic", "opponents": [], "rounds": 100},
        "command": command, "environment_overrides": overrides,
        "checkpoint": {"path": relative(endpoint), "sha256": None}, "raw_stats": relative(stats),
    }
    atomic_json(manifest, record)
    child = os.environ.copy(); child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record.update({"ended_at_utc": utc_now(), "exit_code": completed.returncode})
    try:
        if completed.returncode or not stats.is_file():
            raise RuntimeError("training process or stats failed")
        payload = validate_checkpoint(protocol, protocol_hash, arm, replica)
        record["checkpoint"]["sha256"] = sha256_file(endpoint)
        record["training_diagnostics"] = payload["experiment_training_diagnostics"]
        record["metrics_by_agent"] = metrics_by_agent(json.loads(stats.read_text(encoding="utf-8")))
        record["status"] = "completed"
    except Exception as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"training failed: {arm}/{replica}: {record.get('error')}")
    return record


def evaluation_checkpoint(protocol: dict, label: str) -> Path:
    if label == "frozen-v4":
        return ROOT / protocol["parent_checkpoint"]["path"]
    arm, replica = label.rsplit("-", 1)
    if arm not in ARMS or replica not in REPLICAS:
        raise ValueError(f"invalid label: {label}")
    return checkpoint_path(protocol, arm, replica)


def evaluate_one(protocol: dict, protocol_hash: str, stratum: str, label: str, case: dict) -> dict:
    checkpoint = evaluation_checkpoint(protocol, label)
    expected_hash = sha256_file(checkpoint)
    manifest = evaluation_manifest_path(protocol, stratum, label, int(case["world_seed"]))
    existing = load_completed(manifest, EVALUATION_KIND)
    spec = protocol["evaluation"]["strata"][stratum]
    expected = {**case, "scenario": spec["scenario"], "opponents": [], "rounds": spec["rounds_per_case"]}
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash or existing.get("evaluation") != expected or existing["checkpoint"]["sha256"] != expected_hash:
            raise RuntimeError(f"completed evaluation drift: {stratum}/{label}")
        return existing
    stats = manifest.with_suffix(".stats.json")
    if manifest.exists() or stats.exists():
        raise RuntimeError(f"refusing orphaned evaluation artifacts: {stratum}/{label}")
    command = [
        sys.executable, "main.py", "play", "--agents", "model_a_dqn", "--train", "0",
        "--continue-without-training", "--no-gui", "--scenario", spec["scenario"],
        "--n-rounds", str(spec["rounds_per_case"]), "--seed", str(case["world_seed"]),
        "--save-stats", str(stats),
    ]
    overrides = {"MODEL_A_CHECKPOINT_PATH": str(checkpoint), "MODEL_A_SEED": str(case["agent_seed"])}
    record = {
        "schema_version": 1, "kind": EVALUATION_KIND, "status": "running", "started_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_hash,
        "stratum": stratum, "label": label, "evaluation": expected, "command": command,
        "environment_overrides": overrides, "checkpoint": {"path": relative(checkpoint), "sha256": expected_hash},
        "raw_stats": relative(stats),
    }
    atomic_json(manifest, record)
    child = os.environ.copy(); child.update(overrides)
    before = sha256_file(checkpoint)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record.update({"ended_at_utc": utc_now(), "exit_code": completed.returncode})
    if completed.returncode == 0 and stats.is_file() and sha256_file(checkpoint) == before:
        record["metrics_by_agent"] = metrics_by_agent(json.loads(stats.read_text(encoding="utf-8")))
        record["target_metrics"] = record["metrics_by_agent"].get("model_a_dqn")
        record["status"] = "completed" if record["target_metrics"] else "failed"
    else:
        record["status"] = "failed"
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"evaluation failed: {stratum}/{label}/{case['world_seed']}")
    return record


def family_metrics(rows: dict, stratum: str, family: str) -> dict:
    return aggregate([
        {"target_metrics": rows[stratum][f"{family}-{replica}"]} for replica in REPLICAS
    ])


def decide(protocol: dict, rows: dict) -> dict:
    gate = protocol["gate"]
    control2 = family_metrics(rows, "task2", "control")
    candidate2 = family_metrics(rows, "task2", "idle-wait-001")
    control1 = family_metrics(rows, "task1", "control")
    candidate1 = family_metrics(rows, "task1", "idle-wait-001")
    pairs = {}
    supportive = 0
    for replica in REPLICAS:
        delta = rows["task2"][f"idle-wait-001-{replica}"]["score_per_round"] - rows["task2"][f"control-{replica}"]["score_per_round"]
        support = delta >= gate["minimum_replica_task2_score_gain"]
        supportive += int(support)
        pairs[replica] = {"task2_score_gain": delta, "supportive": support}
    def retention(candidate: float, control: float, threshold: float) -> bool:
        return candidate >= threshold * control if control > 0 else candidate >= control
    checks = {
        "task2_score_gain": candidate2["score_per_round"] - control2["score_per_round"] >= gate["minimum_candidate_task2_score_gain_over_control"],
        "replica_support": supportive >= gate["minimum_supportive_replica_pairs"],
        "task2_wait_reduction": control2["wait_fraction"] - candidate2["wait_fraction"] >= gate["minimum_task2_wait_fraction_reduction"],
        "task2_crate_retention": retention(candidate2["crates_per_round"], control2["crates_per_round"], gate["minimum_task2_crate_retention_ratio"]),
        "task2_bomb_retention": retention(candidate2["bombs_per_round"], control2["bombs_per_round"], gate["minimum_task2_bomb_retention_ratio"]),
        "task1_score_retention": retention(candidate1["score_per_round"], control1["score_per_round"], gate["minimum_task1_score_retention_ratio"]),
        "task2_suicide_guard": candidate2["suicides_per_round"] <= control2["suicides_per_round"] + gate["maximum_task2_suicide_increase"] + 1e-12,
        "task2_invalid_guard": candidate2["invalid_actions_per_round"] <= control2["invalid_actions_per_round"] + gate["maximum_task2_invalid_increase"] + 1e-12,
    }
    passed = all(checks.values())
    return {
        "passed": passed, "decision": protocol["decision_branches"]["pass" if passed else "fail"],
        "checks": checks, "supportive_replica_pairs": supportive, "pair_results": pairs,
        "pooled": {"task1": {"control": control1, "idle-wait-001": candidate1},
                   "task2": {"control": control2, "idle-wait-001": candidate2}},
    }


def parameter_effect(protocol: dict, protocol_hash: str) -> dict:
    rows = {}
    tensor_differences = {}
    for arm in ARMS:
        rows[arm] = {}
        for replica in REPLICAS:
            payload = validate_checkpoint(protocol, protocol_hash, arm, replica)
            rows[arm][replica] = payload["experiment_training_diagnostics"]
    for replica in REPLICAS:
        control = torch_load(checkpoint_path(protocol, "control", replica))["online_net"]
        candidate = torch_load(checkpoint_path(protocol, "idle-wait-001", replica))["online_net"]
        tensor_differences[replica] = sum(
            int(not torch.equal(control[name], candidate[name])) for name in control
        )
    controls_clean = all(item["penalized_idle_waits"] == 0 for item in rows["control"].values())
    candidates_active = all(item["penalized_idle_waits"] > 0 for item in rows["idle-wait-001"].values())
    all_pairs_diverged = all(value > 0 for value in tensor_differences.values())
    result = {
        "control_penalty_events_zero": controls_clean,
        "candidate_penalty_events_positive": candidates_active,
        "matched_online_tensor_differences": tensor_differences,
        "all_matched_pairs_diverged": all_pairs_diverged,
        "diagnostics": rows,
    }
    result["evaluation_authorized"] = controls_clean and candidates_active and all_pairs_diverged
    return result


def execute(protocol_path: Path, protocol: dict, protocol_hash: str) -> dict:
    parent = ROOT / protocol["parent_checkpoint"]["path"]
    parent_before = sha256_file(parent)
    training = [train_one(protocol_path, protocol, protocol_hash, arm, replica) for replica in REPLICAS for arm in ARMS]
    if sha256_file(parent) != parent_before:
        raise RuntimeError("frozen-v4 parent changed during training")
    effect = parameter_effect(protocol, protocol_hash)
    evaluations = []
    rows: dict[str, dict[str, dict]] = {name: {} for name in protocol["evaluation"]["strata"]}
    outcome = None
    if effect["evaluation_authorized"]:
        for stratum, spec in protocol["evaluation"]["strata"].items():
            for label in protocol["evaluation"]["labels"]:
                runs = [evaluate_one(protocol, protocol_hash, stratum, label, case) for case in spec["cases"]]
                evaluations.extend(runs)
                rows[stratum][label] = aggregate(runs)
        outcome = decide(protocol, rows)
        decision = outcome["decision"]
    else:
        decision = "idle_wait_treatment_not_applied_stop"
    report = {
        "schema_version": 1, "kind": REPORT_KIND, "status": "completed",
        "completed_at_utc": utc_now(), "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path), "protocol_sha256": protocol_hash,
        "parent_checkpoint_unchanged": sha256_file(parent) == parent_before,
        "parameter_effect": effect, "evaluation_rows": rows, "outcome": outcome, "decision": decision,
        "training_manifests": [{"path": relative(training_manifest_path(protocol, run["arm"], run["replica"])),
                                "sha256": sha256_file(training_manifest_path(protocol, run["arm"], run["replica"]))} for run in training],
        "evaluation_manifest_count": len(evaluations), "training_started": True,
        "checkpoint_selected": False, "checkpoint_copied": False,
        "automatic_followup_started": False, "awaiting_user_instruction": True,
    }
    atomic_json(report_path(protocol), report)
    return report


def dry_run(protocol: dict, protocol_hash: str) -> dict:
    return {
        "mode": "dry-run", "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_hash,
        "single_changed_variable": protocol["single_changed_variable"], "arms": list(ARMS),
        "replicas": list(REPLICAS), "training_rounds": 600, "evaluation_rounds": 875,
        "total_rounds": 1475, "checkpoint_selection_allowed": False,
        "automatic_followup_started": False, "formal_execution_started": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol, protocol_path, protocol_hash = load_protocol(args.protocol)
    assert_registered_seeds_untouched(protocol_path, protocol)
    if not args.execute:
        print(json.dumps(dry_run(protocol, protocol_hash), indent=2, sort_keys=True))
        return 0
    if report_path(protocol).exists():
        raise SystemExit(f"terminal report already exists: {report_path(protocol)}")
    report = execute(protocol_path, protocol, protocol_hash)
    print(json.dumps({"report": relative(report_path(protocol)), "decision": report["decision"],
                      "checkpoint_selected": False, "automatic_followup_started": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
