"""Selection-free confirmation of the post-hoc Task4A r3 round-100 signal.

The sole candidate is compared with source-r2 and frozen v4 in classic games
against three independently seeded full-strength rule agents.  This runner is
evaluation-only: it never trains, copies a checkpoint, or starts Task4C.
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

from agent_code.model_a_v4_curriculum.config import sha256_file  # noqa: E402
from agent_code.model_a_v4_task4_split.config import (  # noqa: E402
    load_protocol as load_task4a_protocol,
)
from tools.v4_task3_source_r2_final_confirmation import (  # noqa: E402
    aggregate,
    atomic_json,
    load_completed,
    metrics_by_agent,
    relative,
)
from tools.v4_task4_rule_baseline_gate import opponent_summary  # noqa: E402
from tools.v4_task4_split_a import validate_candidate_checkpoint  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-task4-r3-round100-confirmation-s126000.json"
LABELS = ("v4", "source-r2", "candidate-r3-round-0100")
STRATUM = "task4c_three_rule"
EVALUATION_KIND = "model-a-v4-task4-r3-round100-confirmation-evaluation"
REPORT_KIND = "model-a-v4-task4-r3-round100-confirmation-report"
EXPECTED_CASES = tuple(
    {
        "world_seed": 126100 + index,
        "agent_seed": 226100 + index,
        "opponent_seed": 326100 + index,
    }
    for index in range(1, 5)
)
EXPECTED_RULE = {
    "minimum_score_delta_to_each_baseline": 0.0,
    "minimum_kills_delta_to_each_baseline": 0.0,
    "minimum_suicide_reduction_from_each_baseline": 0.1,
    "minimum_score_margin_over_rule_mean": 0.0,
    "minimum_case_blocks_not_below_both_baselines": 2,
    "pass_decision": "r3_round100_signal_replicated_for_task4c_consideration_stop",
    "fail_decision": "r3_round100_rejected_retain_source_r2_stop",
}
SOURCE_PATHS = {
    "tools/v4_task4_r3_round100_confirmation.py",
    "tools/v4_task4_rule_baseline_gate.py",
    "tools/v4_task3_source_r2_final_confirmation.py",
    "tools/v4_task4_split_a.py",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def seed_values(payload: object) -> set[int]:
    found: set[int] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "seed" or key.endswith("_seed") or key.endswith("_seeds"):
                values = value if isinstance(value, list) else [value]
                found.update(item for item in values if isinstance(item, int))
            found.update(seed_values(value))
    elif isinstance(payload, list):
        for value in payload:
            found.update(seed_values(value))
    return found


def assert_registered_seeds_untouched(protocol_path: Path, protocol: dict) -> None:
    registered = {int(value) for case in EXPECTED_CASES for value in case.values()}
    excluded_root = (ROOT / protocol["evaluation_manifest_directory"]).resolve()
    excluded_files = {protocol_path.resolve(), (ROOT / protocol["report_path"]).resolve()}
    conflicts = []
    for path in (ROOT / "experiments").rglob("*.json"):
        resolved = path.resolve()
        if resolved in excluded_files or excluded_root in resolved.parents:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        overlap = sorted(registered & seed_values(payload))
        if overlap:
            conflicts.append({"path": relative(path), "seeds": overlap})
    if conflicts:
        raise ValueError(f"registered s126000 seeds were already used: {conflicts}")


def _bound_json(protocol: dict, key: str) -> dict:
    item = protocol["candidate_lineage"][key]
    path = ROOT / item["path"]
    if not path.is_file() or sha256_file(path) != item["sha256"]:
        raise ValueError(f"candidate lineage artifact mismatch: {key}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(path: Path) -> tuple[dict, str, dict, str]:
    path = path.resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or protocol.get("kind") != "model-a-v4-task4-r3-round100-confirmation":
        raise ValueError("wrong r3 round-100 confirmation protocol")
    if tuple(protocol.get("labels", ())) != LABELS:
        raise ValueError("confirmation labels changed")
    if protocol.get("fixed_candidate") != "candidate-r3-round-0100":
        raise ValueError("r3 round-100 must remain the sole candidate")
    if protocol.get("selection_free") is not True:
        raise ValueError("confirmation must remain selection-free")
    if protocol.get("training_allowed") is not False or protocol.get("automatic_followup") is not False:
        raise ValueError("confirmation must remain evaluation-only and terminal")

    evaluation = protocol.get("evaluation", {})
    if (
        evaluation.get("stratum") != STRATUM
        or evaluation.get("scenario") != "classic"
        or evaluation.get("opponents") != ["seeded_rule_based_agent"] * 3
        or int(evaluation.get("rounds_per_case", 0)) != 25
        or tuple(evaluation.get("cases", ())) != EXPECTED_CASES
    ):
        raise ValueError("confirmation distribution, budget, or seeds changed")
    seeds = [int(value) for case in evaluation["cases"] for value in case.values()]
    if len(seeds) != 12 or len(set(seeds)) != 12:
        raise ValueError("confirmation requires 12 globally unique seeds")
    assert_registered_seeds_untouched(path, protocol)

    decision_rule = protocol.get("decision_rule", {})
    for key, value in EXPECTED_RULE.items():
        if decision_rule.get(key) != value:
            raise ValueError(f"confirmation decision rule changed: {key}")
    if decision_rule.get("any_decision_action") != (
        "write a terminal report and stop; never train, copy a checkpoint, replace the incumbent, or start Task4C"
    ):
        raise ValueError("confirmation terminal action changed")

    if set(protocol.get("source_bindings", {})) != SOURCE_PATHS:
        raise ValueError("confirmation source binding set mismatch")
    for name, expected_hash in protocol["source_bindings"].items():
        source = ROOT / name
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise ValueError(f"confirmation source binding mismatch: {name}")

    inventory = protocol.get("checkpoint_inventory", {})
    if set(inventory) != {"v4", "source_r2", "candidate_r3_round_0100"}:
        raise ValueError("confirmation checkpoint inventory changed")
    for key, item in inventory.items():
        checkpoint = ROOT / item["path"]
        if not checkpoint.is_file() or sha256_file(checkpoint) != item["sha256"]:
            raise ValueError(f"confirmation checkpoint mismatch: {key}")

    task4a_item = protocol["candidate_lineage"]["protocol"]
    task4a_protocol, task4a_hash = load_task4a_protocol(ROOT / task4a_item["path"])
    if task4a_hash != task4a_item["sha256"]:
        raise ValueError("Task4A protocol lineage mismatch")
    candidate = ROOT / inventory["candidate_r3_round_0100"]["path"]
    validate_candidate_checkpoint(task4a_protocol, task4a_hash, candidate, "r3", 100)

    report = _bound_json(protocol, "terminal_report")
    if (
        report.get("status") != "completed"
        or report.get("result", {}).get("decision") != "task4a_training_not_reproducible_stop"
        or report.get("result", {}).get("supportive_replica_count") != 0
    ):
        raise ValueError("candidate is not bound to the terminal Task4A failure")
    selection = _bound_json(protocol, "selection_manifest")
    if selection.get("status") != "completed" or selection.get("result", {}).get("selected_candidate") is not None:
        raise ValueError("Task4A selection status changed")
    observed = selection.get("candidates", {}).get("r3-round-0100", {})
    if (
        observed.get("path") != inventory["candidate_r3_round_0100"]["path"]
        or observed.get("sha256") != inventory["candidate_r3_round_0100"]["sha256"]
        or observed.get("replica") != "r3"
        or observed.get("stage_round") != 100
    ):
        raise ValueError("post-hoc candidate identity mismatch")
    target = observed.get("rows", {}).get(STRATUM, {}).get("target", {})
    expected_observation = protocol["candidate_lineage"]["revealed_inner_observation"]
    for key in ("score_per_round", "kills_per_round", "suicides_per_round"):
        if target.get(key) != expected_observation[key]:
            raise ValueError(f"revealed candidate observation mismatch: {key}")
    training = _bound_json(protocol, "r3_training_manifest")
    if training.get("status") != "completed" or training.get("replica") != "r3":
        raise ValueError("candidate training-manifest lineage mismatch")
    return protocol, sha256_file(path), task4a_protocol, task4a_hash


def checkpoint_identity(protocol: dict, label: str) -> tuple[str, Path, dict[str, str]]:
    inventory = protocol["checkpoint_inventory"]
    clean = protocol["source_lineage"]["clean_protocol"]
    if label == "v4":
        return "model_a_dqn", ROOT / inventory["v4"]["path"], {}
    if label == "source-r2":
        return "model_a_v4_curriculum", ROOT / inventory["source_r2"]["path"], {
            "MODEL_A_V4C_PROTOCOL_PATH": str(ROOT / clean["path"]),
            "MODEL_A_V4C_ARM": "curriculum",
            "MODEL_A_V4C_REPLICA": "r2",
            "MODEL_A_V4C_STAGE": "task2",
        }
    if label == "candidate-r3-round-0100":
        task4a = protocol["candidate_lineage"]["protocol"]
        return "model_a_v4_task4_split", ROOT / inventory["candidate_r3_round_0100"]["path"], {
            "MODEL_A_V4T4S_PROTOCOL_PATH": str(ROOT / task4a["path"]),
            "MODEL_A_V4T4S_REPLICA": "r3",
        }
    raise ValueError(f"unknown confirmation label: {label}")


def inventory_key(label: str) -> str:
    return {
        "v4": "v4",
        "source-r2": "source_r2",
        "candidate-r3-round-0100": "candidate_r3_round_0100",
    }[label]


def manifest_path(protocol: dict, label: str, world_seed: int) -> Path:
    return ROOT / protocol["evaluation_manifest_directory"] / f"{label}-s{world_seed}.json"


def evaluate_one(protocol_path: Path, protocol: dict, protocol_hash: str, label: str, case: dict) -> dict:
    target, checkpoint, overrides = checkpoint_identity(protocol, label)
    checkpoint = checkpoint.resolve()
    expected_hash = protocol["checkpoint_inventory"][inventory_key(label)]["sha256"]
    if sha256_file(checkpoint) != expected_hash:
        raise RuntimeError(f"confirmation checkpoint drift: {label}")
    output = manifest_path(protocol, label, int(case["world_seed"]))
    expected = {
        **case,
        "scenario": "classic",
        "opponents": ["seeded_rule_based_agent"] * 3,
        "rounds": int(protocol["evaluation"]["rounds_per_case"]),
        "target_agent": target,
    }
    existing = load_completed(output, EVALUATION_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash or existing.get("evaluation") != expected:
            raise RuntimeError(f"completed confirmation evaluation drift: {output}")
        return existing
    stats_path = output.with_suffix(".stats.json")
    if stats_path.exists():
        raise RuntimeError(f"refusing orphaned confirmation stats: {stats_path}")
    rounds = int(protocol["evaluation"]["rounds_per_case"])
    command = [
        sys.executable, "main.py", "play", "--agents", target,
        "seeded_rule_based_agent", "seeded_rule_based_agent", "seeded_rule_based_agent",
        "--train", "0", "--continue-without-training", "--no-gui",
        "--scenario", "classic", "--n-rounds", str(rounds),
        "--seed", str(case["world_seed"]), "--save-stats", str(stats_path),
    ]
    if label == "v4":
        overrides.update({
            "MODEL_A_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_SEED": str(case["agent_seed"]),
        })
    elif label == "source-r2":
        overrides.update({
            "MODEL_A_V4C_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_V4C_SEED": str(case["agent_seed"]),
        })
    else:
        overrides.update({
            "MODEL_A_V4T4S_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_V4T4S_SEED": str(case["agent_seed"]),
        })
    overrides["TASK4_RULE_SEED"] = str(case["opponent_seed"])
    record = {
        "schema_version": 1,
        "kind": EVALUATION_KIND,
        "status": "running",
        "started_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "stratum": STRATUM,
        "label": label,
        "evaluation": expected,
        "checkpoint": {"path": relative(checkpoint), "sha256": expected_hash},
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
    if completed.returncode == 0 and stats_path.is_file() and sha256_file(checkpoint) == expected_hash:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        record["metrics_by_agent"] = metrics_by_agent(stats)
        record["target_metrics"] = record["metrics_by_agent"].get(target)
        record["status"] = "completed" if record["target_metrics"] else "failed"
    else:
        record["status"] = "failed"
    atomic_json(output, record)
    if record["status"] != "completed":
        raise RuntimeError(f"confirmation evaluation failed: {label}/{case['world_seed']}")
    return record


def run_evaluations(protocol_path: Path, protocol: dict, protocol_hash: str) -> tuple[dict, dict, dict]:
    rows: dict[str, dict] = {}
    manifests: dict[str, list[str]] = {}
    runs_by_label: dict[str, list[dict]] = {}
    for label in LABELS:
        target, _, _ = checkpoint_identity(protocol, label)
        runs = [evaluate_one(protocol_path, protocol, protocol_hash, label, case) for case in EXPECTED_CASES]
        runs_by_label[label] = runs
        rows[label] = {
            "target": aggregate(runs),
            "opponents": opponent_summary(runs, target),
        }
        manifests[label] = [
            relative(manifest_path(protocol, label, int(case["world_seed"])))
            for case in EXPECTED_CASES
        ]
    cases = {}
    for index, case in enumerate(EXPECTED_CASES):
        cases[str(case["world_seed"])] = {
            label: {
                "target": runs_by_label[label][index]["target_metrics"],
                "opponents": opponent_summary(
                    [runs_by_label[label][index]],
                    checkpoint_identity(protocol, label)[0],
                ),
            }
            for label in LABELS
        }
    return rows, cases, manifests


def decide(protocol: dict, rows: dict, cases: dict) -> dict:
    rule = protocol["decision_rule"]
    candidate = rows["candidate-r3-round-0100"]
    source = rows["source-r2"]
    v4 = rows["v4"]
    c = candidate["target"]
    s = source["target"]
    v = v4["target"]
    score_delta = float(rule["minimum_score_delta_to_each_baseline"])
    kills_delta = float(rule["minimum_kills_delta_to_each_baseline"])
    suicide_reduction = float(rule["minimum_suicide_reduction_from_each_baseline"])
    case_support = {
        seed: data["candidate-r3-round-0100"]["target"]["score_per_round"] + 1e-12
        >= max(
            data["source-r2"]["target"]["score_per_round"],
            data["v4"]["target"]["score_per_round"],
        ) + score_delta
        for seed, data in cases.items()
    }
    gates = {
        "score_not_below_source_r2": c["score_per_round"] + 1e-12 >= s["score_per_round"] + score_delta,
        "score_not_below_v4": c["score_per_round"] + 1e-12 >= v["score_per_round"] + score_delta,
        "kills_not_below_source_r2": c["kills_per_round"] + 1e-12 >= s["kills_per_round"] + kills_delta,
        "kills_not_below_v4": c["kills_per_round"] + 1e-12 >= v["kills_per_round"] + kills_delta,
        "suicide_reduction_vs_source_r2": c["suicides_per_round"] <= s["suicides_per_round"] - suicide_reduction + 1e-12,
        "suicide_reduction_vs_v4": c["suicides_per_round"] <= v["suicides_per_round"] - suicide_reduction + 1e-12,
        "beats_rule_mean": c["score_per_round"] + 1e-12
        >= candidate["opponents"]["mean_score_per_agent_round"] + float(rule["minimum_score_margin_over_rule_mean"]),
        "case_block_reproducibility": sum(case_support.values())
        >= int(rule["minimum_case_blocks_not_below_both_baselines"]),
    }
    passed = all(gates.values())
    deltas = {
        baseline: {
            metric: c[metric] - rows[baseline]["target"][metric]
            for metric in (
                "score_per_round", "kills_per_round", "suicides_per_round",
                "invalid_actions_per_round", "coins_per_round",
            )
        }
        for baseline in ("source-r2", "v4")
    }
    return {
        "decision": rule["pass_decision"] if passed else rule["fail_decision"],
        "passed": passed,
        "gates": gates,
        "candidate_minus_baselines": deltas,
        "candidate_rule_score_margin": c["score_per_round"] - candidate["opponents"]["mean_score_per_agent_round"],
        "case_blocks_not_below_both_baselines": sum(case_support.values()),
        "case_block_support": case_support,
        "fixed_candidate": "candidate-r3-round-0100",
        "selection_free": True,
        "training_started": False,
        "checkpoint_copied": False,
        "incumbent_replaced": False,
        "automatic_followup_started": False,
        "task4c_started": False,
        "task4_complete": False,
        "next_action": (
            "candidate may be considered as a separately authorized Task4C parent; stop for user review"
            if passed else
            "reject r3 round-100 and retain source-r2 as the incumbent; stop for user review"
        ),
    }


def report_path(protocol: dict) -> Path:
    return ROOT / protocol["report_path"]


def write_report(protocol_path: Path, protocol: dict, protocol_hash: str, rows: dict, cases: dict, manifests: dict) -> tuple[dict, str]:
    path = report_path(protocol)
    existing = load_completed(path, REPORT_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("completed confirmation report protocol drift")
        return existing, sha256_file(path)
    for key, item in protocol["checkpoint_inventory"].items():
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"checkpoint changed during confirmation: {key}")
    report = {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "rows": rows,
        "case_rows": cases,
        "evaluation_manifests": manifests,
        "result": decide(protocol, rows, cases),
        "awaiting_user_instruction": True,
        "source_artifacts_modified": False,
        "default_checkpoint_modified": False,
    }
    atomic_json(path, report)
    return report, sha256_file(path)


def dry_run(protocol: dict) -> dict:
    rounds = len(LABELS) * len(EXPECTED_CASES) * int(protocol["evaluation"]["rounds_per_case"])
    return {
        "mode": "dry-run",
        "protocol_id": protocol["protocol_id"],
        "fixed_candidate": protocol["fixed_candidate"],
        "labels": list(LABELS),
        "stratum": STRATUM,
        "case_blocks": len(EXPECTED_CASES),
        "rounds_per_case": int(protocol["evaluation"]["rounds_per_case"]),
        "total_evaluation_rounds": rounds,
        "selection_free": True,
        "formal_evaluation_started": False,
        "training_started": False,
        "checkpoint_copy_allowed": False,
        "incumbent_replacement_allowed": False,
        "automatic_followup_started": False,
        "task4c_started": False,
        "writes_terminal_report_and_stops": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    protocol, protocol_hash, _, _ = load_protocol(protocol_path)
    if not args.execute:
        print(json.dumps(dry_run(protocol), indent=2, sort_keys=True))
        return 0
    existing = load_completed(report_path(protocol), REPORT_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("completed confirmation report protocol drift")
        report, report_hash = existing, sha256_file(report_path(protocol))
    else:
        rows, cases, manifests = run_evaluations(protocol_path.resolve(), protocol, protocol_hash)
        report, report_hash = write_report(
            protocol_path.resolve(), protocol, protocol_hash, rows, cases, manifests,
        )
    print(json.dumps({
        "status": report["status"],
        "report": relative(report_path(protocol)),
        "report_sha256": report_hash,
        **report["result"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
