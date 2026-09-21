"""Run selection-free outer confirmation of the fixed b1 peaceful candidate.

Dry-run is the default. ``--execute`` evaluates exactly one preregistered
candidate and two baselines on the untouched s119000 outer seeds, writes one
terminal report, and never trains, copies a checkpoint, or starts Task4.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_v4_task3_retention.config import (  # noqa: E402
    load_protocol as load_source_protocol,
)
from tools.v4_task3_retention import (  # noqa: E402
    aggregate, metrics_by_agent, sha256_file, validate_new_checkpoint,
)


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-task3-b1-outer-confirmation-s120000.json"
LABELS = ("v4", "source-r2", "candidate-b1-peaceful-r0100")
STRATA = ("task1", "task2", "task3_peaceful", "task3_coin")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_completed(path: Path, kind: str) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed" or payload.get("kind") != kind:
        raise RuntimeError(f"refusing incomplete/incompatible artifact: {path}")
    return payload


def load_protocol(path: Path) -> tuple[dict, str, dict, str]:
    path = path.resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or protocol.get("kind") != "model-a-v4-task3-b1-outer-confirmation":
        raise ValueError("wrong b1 outer-confirmation protocol")
    if tuple(protocol.get("labels", ())) != LABELS:
        raise ValueError("outer confirmation labels changed")

    source_info = protocol.get("source_experiment", {})
    source_path = (ROOT / source_info["protocol_path"]).resolve()
    source_protocol, source_hash = load_source_protocol(source_path)
    if source_hash != source_info["protocol_sha256"]:
        raise ValueError("source Task3 protocol hash mismatch")
    expected_outer = source_protocol["outer_evaluation"]["strata"]
    if protocol.get("evaluation", {}).get("strata") != expected_outer:
        raise ValueError("confirmation must use the untouched source outer suite exactly")
    if tuple(expected_outer) != STRATA:
        raise ValueError("source outer strata changed")
    if any(
        int(spec.get("rounds_per_case", 0)) != 25 or len(spec.get("cases", ())) != 2
        for spec in expected_outer.values()
    ):
        raise ValueError("outer confirmation requires two 25-round cases per stratum")
    seeds = []
    for spec in expected_outer.values():
        for case in spec["cases"]:
            if set(case) != {"world_seed", "agent_seed", "opponent_seed"}:
                raise ValueError("incomplete outer seed tuple")
            seeds.extend(int(value) for value in case.values())
    if len(seeds) != len(set(seeds)):
        raise ValueError("outer confirmation seeds must be globally unique")

    bindings = protocol.get("source_bindings", {})
    if set(bindings) != {"tools/v4_task3_b1_outer_confirmation.py"}:
        raise ValueError("outer-confirmation source binding set mismatch")
    for name, expected_hash in bindings.items():
        source = ROOT / name
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise ValueError(f"outer-confirmation source binding mismatch: {name}")

    inventory = protocol.get("checkpoint_inventory", {})
    expected_v4 = {
        "path": source_protocol["frozen_v4_baseline"]["checkpoint_path"],
        "sha256": source_protocol["frozen_v4_baseline"]["checkpoint_sha256"],
    }
    if inventory.get("v4") != expected_v4:
        raise ValueError("frozen-v4 inventory mismatch")
    if inventory.get("source_r2") != source_protocol["source_parent"]["checkpoint"]:
        raise ValueError("source-r2 inventory mismatch")
    for name in ("v4", "source_r2", "candidate_b1"):
        item = inventory[name]
        checkpoint = ROOT / item["path"]
        if not checkpoint.is_file() or sha256_file(checkpoint) != item["sha256"]:
            raise ValueError(f"outer checkpoint binding mismatch: {name}")
    candidate = (ROOT / inventory["candidate_b1"]["path"]).resolve()
    validate_new_checkpoint(
        source_protocol, source_hash, candidate, "b1", "task3_peaceful",
        stage_round=100, selected=True,
    )
    source_snapshot = (ROOT / inventory["candidate_b1"]["source_snapshot_path"]).resolve()
    if (
        not source_snapshot.is_file()
        or sha256_file(source_snapshot) != inventory["candidate_b1"]["source_snapshot_sha256"]
    ):
        raise ValueError("b1 peaceful source-snapshot binding mismatch")
    validate_new_checkpoint(
        source_protocol, source_hash, source_snapshot, "b1", "task3_peaceful",
        stage_round=100, selected=False,
    )

    selection_item = source_info["b1_peaceful_selection_manifest"]
    selection_path = ROOT / selection_item["path"]
    if not selection_path.is_file() or sha256_file(selection_path) != selection_item["sha256"]:
        raise ValueError("b1 peaceful selection-manifest binding mismatch")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        selection.get("status") != "completed"
        or selection.get("protocol_sha256") != source_hash
        or selection.get("branch") != "b1"
        or selection.get("stage") != "task3_peaceful"
        or int(selection.get("selected_stage_round", -1)) != 100
        or selection.get("selected_checkpoint", {}).get("sha256") != inventory["candidate_b1"]["sha256"]
        or selection.get("selected_checkpoint", {}).get("source_snapshot_sha256") != inventory["candidate_b1"]["source_snapshot_sha256"]
    ):
        raise ValueError("b1 peaceful selection identity mismatch")

    salvage_item = source_info["salvage_report"]
    salvage_path = ROOT / salvage_item["path"]
    if not salvage_path.is_file() or sha256_file(salvage_path) != salvage_item["sha256"]:
        raise ValueError("salvage report binding mismatch")
    salvage = json.loads(salvage_path.read_text(encoding="utf-8"))
    if (
        salvage.get("status") != "completed"
        or salvage.get("result", {}).get("decision") != "no_viable_coin_snapshot"
        or salvage.get("result", {}).get("provisional_selected_label") is not None
    ):
        raise ValueError("salvage report decision mismatch")

    rule = protocol.get("decision_rule", {})
    expected_rule = {
        "minimum_task1_retained_fraction_vs_v4": 0.9,
        "maximum_task2_score_gap_to_v4": 0.25,
        "minimum_peaceful_score_gain_over_v4": 0.25,
        "minimum_peaceful_kills_per_round": 0.1,
        "minimum_peaceful_kills_delta_to_v4": 0.0,
        "maximum_coin_score_gap_to_v4": 0.25,
        "minimum_coin_kills_per_round": 0.05,
        "minimum_coin_kills_delta_to_v4": 0.0,
        "maximum_task3_suicide_increase_over_v4": 0.05,
        "maximum_task3_invalid_increase_over_v4": 0.05,
    }
    for key, value in expected_rule.items():
        if rule.get(key) != value:
            raise ValueError(f"outer confirmation rule changed: {key}")
    if (
        rule.get("pass_decision") != "b1_peaceful_confirmed_for_task3"
        or rule.get("fail_decision") != "b1_peaceful_outer_rejected"
    ):
        raise ValueError("outer confirmation decision labels changed")
    if (
        protocol.get("selection_free") is not True
        or protocol.get("candidate_count") != 1
        or protocol.get("automatic_checkpoint_copy") is not False
        or protocol.get("automatic_task4") is not False
    ):
        raise ValueError("outer confirmation must remain selection-free and terminal")

    original_report = ROOT / source_protocol["report_path"]
    if original_report.exists():
        raise ValueError("source Task3 terminal report unexpectedly exists")
    original_outer = ROOT / source_protocol["evaluation_manifest_directory"] / "outer"
    if original_outer.exists() and any(original_outer.rglob("*.json")):
        raise ValueError("source outer manifests unexpectedly exist")
    return protocol, sha256_file(path), source_protocol, source_hash


def checkpoint_identity(protocol: dict, label: str) -> tuple[str, Path, dict[str, str]]:
    inventory = protocol["checkpoint_inventory"]
    if label == "v4":
        checkpoint = (ROOT / inventory["v4"]["path"]).resolve()
        return "model_a_dqn", checkpoint, {}
    if label == "source-r2":
        checkpoint = (ROOT / inventory["source_r2"]["path"]).resolve()
        return "model_a_v4_curriculum", checkpoint, {
            "MODEL_A_V4C_PROTOCOL_PATH": str(ROOT / protocol["source_experiment"]["clean_protocol_path"]),
            "MODEL_A_V4C_ARM": "curriculum",
            "MODEL_A_V4C_REPLICA": "r2",
            "MODEL_A_V4C_STAGE": "task2",
        }
    if label == "candidate-b1-peaceful-r0100":
        checkpoint = (ROOT / inventory["candidate_b1"]["path"]).resolve()
        return "model_a_v4_task3_retention", checkpoint, {
            "MODEL_A_V4T3R_PROTOCOL_PATH": str(ROOT / protocol["source_experiment"]["protocol_path"]),
            "MODEL_A_V4T3R_BRANCH": "b1",
            "MODEL_A_V4T3R_STAGE": "task3_peaceful",
        }
    raise ValueError(f"invalid outer-confirmation label: {label}")


def manifest_path(protocol: dict, stratum: str, label: str, world_seed: int) -> Path:
    return ROOT / protocol["evaluation_manifest_directory"] / stratum / f"{label}-s{world_seed}.json"


def evaluate_one(
    protocol_path: Path,
    protocol: dict,
    protocol_hash: str,
    stratum: str,
    spec: dict,
    label: str,
    case: dict,
) -> dict:
    target, checkpoint, overrides = checkpoint_identity(protocol, label)
    checkpoint_hash = sha256_file(checkpoint)
    inventory_key = {"v4": "v4", "source-r2": "source_r2"}.get(label, "candidate_b1")
    if checkpoint_hash != protocol["checkpoint_inventory"][inventory_key]["sha256"]:
        raise RuntimeError(f"checkpoint drift before outer confirmation: {label}")
    output = manifest_path(protocol, stratum, label, int(case["world_seed"]))
    expected = {
        **case,
        "scenario": spec["scenario"],
        "opponents": spec["opponents"],
        "rounds": int(spec["rounds_per_case"]),
        "target_agent": target,
    }
    existing = load_completed(output, "model-a-v4-task3-b1-outer-confirmation-evaluation")
    if existing is not None:
        if (
            existing.get("protocol_sha256") != protocol_hash
            or existing.get("evaluation") != expected
            or existing.get("checkpoint", {}).get("sha256") != checkpoint_hash
        ):
            raise RuntimeError(f"completed outer-confirmation evaluation drift: {output}")
        return existing
    stats_path = output.with_suffix(".stats.json")
    if stats_path.exists():
        raise RuntimeError(f"refusing orphaned outer-confirmation stats: {stats_path}")
    agents = [target, *spec["opponents"]]
    command = [
        sys.executable, "main.py", "play", "--agents", *agents,
        "--train", "0", "--continue-without-training", "--no-gui",
        "--scenario", spec["scenario"], "--n-rounds", str(spec["rounds_per_case"]),
        "--seed", str(case["world_seed"]), "--save-stats", str(stats_path),
    ]
    overrides.update({
        {
            "v4": "MODEL_A_CHECKPOINT_PATH",
            "source-r2": "MODEL_A_V4C_CHECKPOINT_PATH",
            "candidate-b1-peaceful-r0100": "MODEL_A_V4T3R_CHECKPOINT_PATH",
        }[label]: str(checkpoint),
        {
            "v4": "MODEL_A_SEED",
            "source-r2": "MODEL_A_V4C_SEED",
            "candidate-b1-peaceful-r0100": "MODEL_A_V4T3R_SEED",
        }[label]: str(case["agent_seed"]),
    })
    if spec["opponents"]:
        overrides["TASK3_OPPONENT_SEED"] = str(case["opponent_seed"])
    record = {
        "schema_version": 1,
        "kind": "model-a-v4-task3-b1-outer-confirmation-evaluation",
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "stratum": stratum,
        "label": label,
        "status": "running",
        "started_at_utc": utc_now(),
        "evaluation": expected,
        "checkpoint": {"path": relative(checkpoint), "sha256": checkpoint_hash},
        "command": command,
        "environment_overrides": overrides,
        "artifacts": {"raw_stats": relative(stats_path)},
    }
    atomic_json(output, record)
    child = os.environ.copy()
    child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = utc_now()
    record["exit_code"] = completed.returncode
    if completed.returncode == 0 and stats_path.is_file() and sha256_file(checkpoint) == checkpoint_hash:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        record["metrics_by_agent"] = metrics_by_agent(stats)
        record["target_metrics"] = record["metrics_by_agent"].get(target)
        record["status"] = "completed" if record["target_metrics"] else "failed"
    else:
        record["status"] = "failed"
    atomic_json(output, record)
    if record["status"] != "completed":
        raise RuntimeError(f"outer confirmation failed: {stratum}/{label}/{case['world_seed']}")
    return record


def run_evaluations(
    protocol_path: Path, protocol: dict, protocol_hash: str,
) -> tuple[dict, dict]:
    rows = {}
    manifests = {}
    for stratum, spec in protocol["evaluation"]["strata"].items():
        rows[stratum] = {}
        manifests[stratum] = {}
        for label in LABELS:
            runs = [
                evaluate_one(protocol_path, protocol, protocol_hash, stratum, spec, label, case)
                for case in spec["cases"]
            ]
            rows[stratum][label] = aggregate(runs)
            manifests[stratum][label] = [
                relative(manifest_path(protocol, stratum, label, int(case["world_seed"])))
                for case in spec["cases"]
            ]
    return rows, manifests


def decide(protocol: dict, rows: dict) -> dict:
    candidate = "candidate-b1-peaceful-r0100"
    rule = protocol["decision_rule"]
    v4 = {stratum: rows[stratum]["v4"] for stratum in STRATA}
    item = {stratum: rows[stratum][candidate] for stratum in STRATA}
    task1_fraction = item["task1"]["score_per_round"] / v4["task1"]["score_per_round"] if v4["task1"]["score_per_round"] > 0 else 1.0
    task3_safety = {}
    for stratum in ("task3_peaceful", "task3_coin"):
        task3_safety[stratum] = {
            "suicide": item[stratum]["suicides_per_round"] <= v4[stratum]["suicides_per_round"] + float(
                rule["maximum_task3_suicide_increase_over_v4"]
            ) + 1e-12,
            "invalid": item[stratum]["invalid_actions_per_round"] <= v4[stratum]["invalid_actions_per_round"] + float(
                rule["maximum_task3_invalid_increase_over_v4"]
            ) + 1e-12,
        }
    gates = {
        "task1_retention": task1_fraction + 1e-12 >= float(rule["minimum_task1_retained_fraction_vs_v4"]),
        "task2_score": item["task2"]["score_per_round"] + 1e-12 >= v4["task2"]["score_per_round"] - float(
            rule["maximum_task2_score_gap_to_v4"]
        ),
        "peaceful_score": item["task3_peaceful"]["score_per_round"] + 1e-12 >= v4["task3_peaceful"]["score_per_round"] + float(
            rule["minimum_peaceful_score_gain_over_v4"]
        ),
        "peaceful_kills_absolute": item["task3_peaceful"]["kills_per_round"] + 1e-12 >= float(
            rule["minimum_peaceful_kills_per_round"]
        ),
        "peaceful_kills_vs_v4": item["task3_peaceful"]["kills_per_round"] + 1e-12 >= v4["task3_peaceful"]["kills_per_round"] + float(
            rule["minimum_peaceful_kills_delta_to_v4"]
        ),
        "coin_score": item["task3_coin"]["score_per_round"] + 1e-12 >= v4["task3_coin"]["score_per_round"] - float(
            rule["maximum_coin_score_gap_to_v4"]
        ),
        "coin_kills_absolute": item["task3_coin"]["kills_per_round"] + 1e-12 >= float(
            rule["minimum_coin_kills_per_round"]
        ),
        "coin_kills_vs_v4": item["task3_coin"]["kills_per_round"] + 1e-12 >= v4["task3_coin"]["kills_per_round"] + float(
            rule["minimum_coin_kills_delta_to_v4"]
        ),
        "task3_safety": all(value for layer in task3_safety.values() for value in layer.values()),
    }
    passed = all(gates.values())
    return {
        "decision": "b1_peaceful_confirmed_for_task3" if passed else "b1_peaceful_outer_rejected",
        "passed": passed,
        "gates": gates,
        "task1_retained_fraction_vs_v4": task1_fraction,
        "task3_safety_gates": task3_safety,
        "candidate_minus_v4": {
            stratum: {
                metric: item[stratum][metric] - v4[stratum][metric]
                for metric in (
                    "score_per_round", "coins_per_round", "kills_per_round",
                    "suicides_per_round", "invalid_actions_per_round",
                )
            }
            for stratum in STRATA
        },
        "fixed_candidate": candidate,
        "candidate_count": 1,
        "selection_free": True,
        "automatic_checkpoint_copied": False,
        "training_started": False,
        "task4_started": False,
    }


def report_path(protocol: dict) -> Path:
    return ROOT / protocol["report_path"]


def write_report(
    protocol_path: Path, protocol: dict, protocol_hash: str, rows: dict, manifests: dict,
) -> tuple[dict, str]:
    path = report_path(protocol)
    existing = load_completed(path, "model-a-v4-task3-b1-outer-confirmation-report")
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("completed b1 outer report protocol drift")
        return existing, sha256_file(path)
    report = {
        "schema_version": 1,
        "kind": "model-a-v4-task3-b1-outer-confirmation-report",
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "rows": rows,
        "evaluation_manifests": manifests,
        "result": decide(protocol, rows),
        "awaiting_user_instruction": True,
        "source_artifacts_modified": False,
        "default_checkpoint_modified": False,
    }
    atomic_json(path, report)
    return report, sha256_file(path)


def dry_run(protocol: dict) -> dict:
    rounds = sum(
        len(spec["cases"]) * int(spec["rounds_per_case"])
        for spec in protocol["evaluation"]["strata"].values()
    ) * len(LABELS)
    return {
        "mode": "dry-run",
        "protocol_id": protocol["protocol_id"],
        "labels": list(LABELS),
        "candidate_count": 1,
        "strata": list(STRATA),
        "total_evaluation_rounds": rounds,
        "selection_free": True,
        "uses_untouched_source_outer_seeds": True,
        "formal_evaluation_started": False,
        "training_started": False,
        "checkpoint_copy_started": False,
        "task4_started": False,
        "writes_terminal_report_and_stops": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    try:
        protocol, protocol_hash, _, _ = load_protocol(protocol_path)
        if not args.execute:
            print(json.dumps(dry_run(protocol), indent=2, sort_keys=True))
            return 0
        existing = load_completed(report_path(protocol), "model-a-v4-task3-b1-outer-confirmation-report")
        if existing is not None:
            if existing.get("protocol_sha256") != protocol_hash:
                raise RuntimeError("completed b1 outer report protocol drift")
            print(json.dumps({
                "status": "completed",
                "report": relative(report_path(protocol)),
                "report_sha256": sha256_file(report_path(protocol)),
                "decision": existing["result"]["decision"],
                "awaiting_user_instruction": True,
                "training_started": False,
                "task4_started": False,
            }, indent=2, sort_keys=True))
            return 0
        rows, manifests = run_evaluations(protocol_path.resolve(), protocol, protocol_hash)
        report, report_hash = write_report(protocol_path.resolve(), protocol, protocol_hash, rows, manifests)
        print(json.dumps({
            "status": "completed",
            "report": relative(report_path(protocol)),
            "report_sha256": report_hash,
            "decision": report["result"]["decision"],
            "passed": report["result"]["passed"],
            "gates": report["result"]["gates"],
            "awaiting_user_instruction": True,
            "automatic_checkpoint_copied": False,
            "training_started": False,
            "task4_started": False,
        }, indent=2, sort_keys=True))
        return 0
    except (OSError, KeyError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
