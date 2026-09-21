"""Run the terminal Task4C matched A/B for exact-v4 BOMB temporal credit."""

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

from agent_code.model_a_v4_bomb_credit.config import (  # noqa: E402
    ARMS, EVALUATION_STRATA, REPLICAS, SNAPSHOT_ROUNDS, final_checkpoint_path,
    load_protocol, sha256_file, snapshot_path,
)
from tools.v4_task4_duel_training import (  # noqa: E402
    aggregate, atomic_json, load_completed, metrics_by_agent, opponent_summary,
    relative, seed_values,
)


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-task4-bomb-credit-s130000.json"
TRAINING_KIND = "model-a-v4-task4-bomb-credit-training"
EVALUATION_KIND = "model-a-v4-task4-bomb-credit-evaluation"
REPORT_KIND = "model-a-v4-task4-bomb-credit-report"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def training_manifest_path(protocol: dict, arm: str, replica: str) -> Path:
    return ROOT / protocol["training_manifest_directory"] / arm / f"{replica}.json"


def evaluation_manifest_path(protocol: dict, stratum: str, label: str, world_seed: int) -> Path:
    return ROOT / protocol["evaluation_manifest_directory"] / stratum / f"{label}-s{world_seed}.json"


def report_path(protocol: dict) -> Path:
    return ROOT / protocol["report_path"]


def registered_seeds(protocol: dict) -> set[int]:
    values = set()
    for case in protocol["training"]["seeds_by_replica"].values():
        values.update(int(value) for value in case.values())
    for spec in protocol["evaluation"]["strata"].values():
        for case in spec["cases"]:
            values.update(int(value) for value in case.values())
    return values


def assert_registered_seeds_untouched(protocol_path: Path, protocol: dict) -> None:
    registered = registered_seeds(protocol)
    excluded_roots = {
        (ROOT / protocol["checkpoint_directory"]).resolve(),
        (ROOT / protocol["training_manifest_directory"]).resolve(),
        (ROOT / protocol["evaluation_manifest_directory"]).resolve(),
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
        raise ValueError(f"registered s130000 seeds were already used: {conflicts}")


def _torch_load(path: Path) -> dict:
    from agent_code.model_a_v4_bomb_credit.callbacks import _torch_load as load
    return load(path)


def validate_checkpoint(protocol: dict, protocol_hash: str, arm: str, replica: str, path: Path, stage_round: int) -> dict:
    if not path.is_file():
        raise RuntimeError(f"BOMB-credit checkpoint missing: {path}")
    payload = _torch_load(path)
    expected = {
        "architecture": "model-a-mlp-v4-global7",
        "protocol_sha256": protocol_hash,
        "arm": arm,
        "replica": replica,
        "stage_id": "task4c_bomb_credit",
        "stage_completed_rounds": stage_round,
        "parent_sha256": protocol["source_parent"]["sha256"],
        "training_variable": "bomb_transition_return_horizon_1_vs_5",
        "bomb_return_steps": 1 if arm == "control" else 5,
        "non_bomb_return_steps": 1,
        "bomb_maturation_steps": 5,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"BOMB-credit checkpoint {key} mismatch: {path}")
    if abs(float(payload.get("epsilon", -1.0)) - 0.05) > 1e-12:
        raise RuntimeError(f"BOMB-credit epsilon changed: {path}")
    diagnostics = payload.get("target_diagnostics", {})
    required = {
        "raw_transitions", "non_bomb_one_step_targets", "bomb_targets_matured",
        "bomb_targets_full_five_step_observed", "bomb_targets_terminal_truncated",
        "bomb_target_return_steps_histogram", "bomb_target_kill_events_included",
        "bomb_target_self_events_included", "observed_kill_events", "observed_self_events",
        "uniquely_attributed_kill_events", "uniquely_attributed_self_events", "attribution_errors",
    }
    if set(diagnostics) != required:
        raise RuntimeError(f"BOMB-credit target diagnostics changed: {path}")
    if diagnostics["raw_transitions"] != diagnostics["non_bomb_one_step_targets"] + diagnostics["bomb_targets_matured"]:
        raise RuntimeError(f"BOMB-credit transition accounting mismatch: {path}")
    if diagnostics["attribution_errors"] != 0:
        raise RuntimeError(f"BOMB-credit outcome attribution failed: {path}")
    for observed, attributed in (
        ("observed_kill_events", "uniquely_attributed_kill_events"),
        ("observed_self_events", "uniquely_attributed_self_events"),
    ):
        if diagnostics[observed] != diagnostics[attributed]:
            raise RuntimeError(f"BOMB-credit outcome coverage mismatch: {path}")
    histogram = diagnostics["bomb_target_return_steps_histogram"]
    if set(histogram) != {"1", "2", "3", "4", "5"} or sum(histogram.values()) != diagnostics["bomb_targets_matured"]:
        raise RuntimeError(f"BOMB-credit return histogram mismatch: {path}")
    if arm == "control":
        if histogram["1"] != diagnostics["bomb_targets_matured"] or any(histogram[str(i)] for i in range(2, 6)):
            raise RuntimeError(f"control received a multi-step BOMB target: {path}")
        if diagnostics["bomb_target_kill_events_included"] or diagnostics["bomb_target_self_events_included"]:
            raise RuntimeError(f"control directly included delayed BOMB outcomes: {path}")
    else:
        if histogram["5"] != diagnostics["bomb_targets_full_five_step_observed"]:
            raise RuntimeError(f"credit full-horizon target mismatch: {path}")
        if diagnostics["bomb_target_kill_events_included"] != diagnostics["observed_kill_events"]:
            raise RuntimeError(f"credit did not include every observed kill in BOMB target: {path}")
        if diagnostics["bomb_target_self_events_included"] != diagnostics["observed_self_events"]:
            raise RuntimeError(f"credit did not include every observed self-kill in BOMB target: {path}")
    return payload


def validate_training_artifacts(protocol: dict, protocol_hash: str, arm: str, replica: str) -> list[dict]:
    result = []
    for stage_round in SNAPSHOT_ROUNDS:
        path = snapshot_path(protocol, arm, replica, stage_round)
        payload = validate_checkpoint(protocol, protocol_hash, arm, replica, path, stage_round)
        result.append({
            "stage_round": stage_round,
            "path": relative(path),
            "sha256": sha256_file(path),
            "training_steps": int(payload["training_steps"]),
            "gradient_steps": int(payload["gradient_steps"]),
            "epsilon": float(payload["epsilon"]),
            "target_diagnostics": payload["target_diagnostics"],
        })
    final = final_checkpoint_path(protocol, arm, replica)
    validate_checkpoint(protocol, protocol_hash, arm, replica, final, SNAPSHOT_ROUNDS[-1])
    if sha256_file(final) != result[-1]["sha256"]:
        raise RuntimeError(f"BOMB-credit final is not immutable round-100: {arm}/{replica}")
    return result


def train_one(protocol_path: Path, protocol: dict, protocol_hash: str, arm: str, replica: str) -> dict:
    manifest = training_manifest_path(protocol, arm, replica)
    existing = load_completed(manifest, TRAINING_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError(f"completed BOMB-credit training protocol drift: {arm}/{replica}")
        validate_training_artifacts(protocol, protocol_hash, arm, replica)
        return existing
    output = final_checkpoint_path(protocol, arm, replica)
    stats_path = manifest.with_suffix(".stats.json")
    if output.parent.exists() or stats_path.exists() or manifest.exists():
        raise RuntimeError(f"refusing orphaned BOMB-credit training artifacts: {arm}/{replica}")
    parent = (ROOT / protocol["source_parent"]["path"]).resolve()
    spec = protocol["training"]
    seeds = spec["seeds_by_replica"][replica]
    command = [
        sys.executable, "main.py", "play", "--agents", "model_a_v4_bomb_credit", *spec["opponents"],
        "--train", "1", "--continue-without-training", "--no-gui",
        "--scenario", spec["scenario"], "--n-rounds", str(spec["rounds_per_arm"]),
        "--seed", str(seeds["world_seed"]), "--save-stats", str(stats_path),
    ]
    overrides = {
        "MODEL_A_BOMB_CREDIT_PROTOCOL_PATH": str(protocol_path),
        "MODEL_A_BOMB_CREDIT_CHECKPOINT_PATH": str(output),
        "MODEL_A_BOMB_CREDIT_PARENT_PATH": str(parent),
        "MODEL_A_BOMB_CREDIT_ARM": arm,
        "MODEL_A_BOMB_CREDIT_REPLICA": replica,
        "MODEL_A_BOMB_CREDIT_SEED": str(seeds["agent_seed"]),
        "TASK4_RULE_SEED": str(seeds["opponent_seed"]),
    }
    record = {
        "schema_version": 1,
        "kind": TRAINING_KIND,
        "status": "running",
        "started_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "arm": arm,
        "replica": replica,
        "training": {**seeds, "scenario": spec["scenario"], "opponents": spec["opponents"], "rounds": spec["rounds_per_arm"]},
        "target_contract": {
            "bomb_maturation_steps": 5,
            "bomb_return_steps": 1 if arm == "control" else 5,
            "non_bomb_return_steps": 1,
        },
        "parent_checkpoint": {"path": relative(parent), "sha256": sha256_file(parent)},
        "checkpoint": {"path": relative(output), "sha256": None},
        "command": command,
        "environment_overrides": overrides,
        "artifacts": {"raw_stats": relative(stats_path)},
    }
    atomic_json(manifest, record)
    child = os.environ.copy()
    child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = utc_now()
    record["exit_code"] = completed.returncode
    try:
        if completed.returncode != 0 or not stats_path.is_file():
            raise RuntimeError("BOMB-credit training process or stats failed")
        record["snapshots"] = validate_training_artifacts(protocol, protocol_hash, arm, replica)
        record["metrics_by_agent"] = metrics_by_agent(json.loads(stats_path.read_text(encoding="utf-8")))
        record["checkpoint"]["sha256"] = sha256_file(output)
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"BOMB-credit training failed: {arm}/{replica}: {record.get('error')}")
    return record


def checkpoint_environment(protocol_path: Path, protocol: dict, label: str, case: dict) -> tuple[str, Path, dict[str, str]]:
    if label == "v4":
        checkpoint = (ROOT / protocol["frozen_v4_baseline"]["path"]).resolve()
        return "model_a_dqn", checkpoint, {
            "MODEL_A_CHECKPOINT_PATH": str(checkpoint), "MODEL_A_SEED": str(case["agent_seed"]),
        }
    if label == "source-r2":
        checkpoint = (ROOT / protocol["source_parent"]["path"]).resolve()
        lineage = protocol["source_parent"]["lineage"]
        return "model_a_v4_curriculum", checkpoint, {
            "MODEL_A_V4C_PROTOCOL_PATH": str(ROOT / lineage["protocol_path"]),
            "MODEL_A_V4C_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_V4C_ARM": "curriculum", "MODEL_A_V4C_REPLICA": "r2",
            "MODEL_A_V4C_STAGE": "task2", "MODEL_A_V4C_SEED": str(case["agent_seed"]),
        }
    arm, replica = label.split("-", 1)
    checkpoint = final_checkpoint_path(protocol, arm, replica)
    return "model_a_v4_bomb_credit", checkpoint, {
        "MODEL_A_BOMB_CREDIT_PROTOCOL_PATH": str(protocol_path),
        "MODEL_A_BOMB_CREDIT_CHECKPOINT_PATH": str(checkpoint),
        "MODEL_A_BOMB_CREDIT_ARM": arm,
        "MODEL_A_BOMB_CREDIT_REPLICA": replica,
        "MODEL_A_BOMB_CREDIT_SEED": str(case["agent_seed"]),
    }


def evaluate_one(protocol_path: Path, protocol: dict, protocol_hash: str, stratum: str, label: str, case: dict) -> dict:
    spec = protocol["evaluation"]["strata"][stratum]
    target, checkpoint, overrides = checkpoint_environment(protocol_path, protocol, label, case)
    checkpoint_hash = sha256_file(checkpoint)
    output = evaluation_manifest_path(protocol, stratum, label, int(case["world_seed"]))
    expected = {**case, "scenario": spec["scenario"], "opponents": spec["opponents"],
                "rounds_per_case": spec["rounds_per_case"], "target_agent": target}
    existing = load_completed(output, EVALUATION_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash or existing.get("evaluation") != expected:
            raise RuntimeError(f"completed BOMB-credit evaluation drift: {output}")
        if existing.get("checkpoint", {}).get("sha256") != checkpoint_hash:
            raise RuntimeError(f"completed BOMB-credit evaluation checkpoint drift: {output}")
        return existing
    stats_path = output.with_suffix(".stats.json")
    if output.exists() or stats_path.exists():
        raise RuntimeError(f"refusing orphaned BOMB-credit evaluation: {output}")
    command = [
        sys.executable, "main.py", "play", "--agents", target, *spec["opponents"],
        "--train", "0", "--continue-without-training", "--no-gui",
        "--scenario", spec["scenario"], "--n-rounds", str(spec["rounds_per_case"]),
        "--seed", str(case["world_seed"]), "--save-stats", str(stats_path),
    ]
    if any(name == "seeded_rule_based_agent" for name in spec["opponents"]):
        overrides["TASK4_RULE_SEED"] = str(case["opponent_seed"])
    elif spec["opponents"]:
        overrides["TASK3_OPPONENT_SEED"] = str(case["opponent_seed"])
    record = {
        "schema_version": 1, "kind": EVALUATION_KIND, "status": "running",
        "started_at_utc": utc_now(), "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path), "protocol_sha256": protocol_hash,
        "stratum": stratum, "label": label, "evaluation": expected,
        "checkpoint": {"path": relative(checkpoint), "sha256": checkpoint_hash},
        "command": command, "environment_overrides": overrides,
        "artifacts": {"raw_stats": relative(stats_path)},
    }
    atomic_json(output, record)
    child = os.environ.copy()
    child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = utc_now()
    record["exit_code"] = completed.returncode
    if completed.returncode == 0 and stats_path.is_file() and sha256_file(checkpoint) == checkpoint_hash:
        record["metrics_by_agent"] = metrics_by_agent(json.loads(stats_path.read_text(encoding="utf-8")))
        record["target_metrics"] = record["metrics_by_agent"].get(target)
        record["status"] = "completed" if record["target_metrics"] else "failed"
    else:
        record["status"] = "failed"
    atomic_json(output, record)
    if record["status"] != "completed":
        raise RuntimeError(f"BOMB-credit evaluation failed: {stratum}/{label}")
    return record


def run_evaluations(protocol_path: Path, protocol: dict, protocol_hash: str) -> tuple[dict, dict, dict]:
    labels = ["v4", "source-r2", *[f"{arm}-{replica}" for replica in REPLICAS for arm in ARMS]]
    rows, manifests, runs = {}, {}, {}
    for label in labels:
        rows[label], manifests[label], runs[label] = {}, {}, {}
        for stratum in EVALUATION_STRATA:
            spec = protocol["evaluation"]["strata"][stratum]
            case_runs = [evaluate_one(protocol_path, protocol, protocol_hash, stratum, label, case) for case in spec["cases"]]
            target, _, _ = checkpoint_environment(protocol_path, protocol, label, spec["cases"][0])
            rows[label][stratum] = {"target": aggregate(case_runs), "opponents": opponent_summary(case_runs, target)}
            runs[label][stratum] = case_runs
            manifests[label][stratum] = [
                relative(evaluation_manifest_path(protocol, stratum, label, int(case["world_seed"])))
                for case in spec["cases"]
            ]
    return rows, manifests, runs


def _mean_metrics(rows: dict, labels: list[str], stratum: str) -> dict:
    fields = ("score_per_round", "kills_per_round", "suicides_per_round", "invalid_actions_per_round")
    return {field: sum(rows[label][stratum]["target"][field] for label in labels) / len(labels) for field in fields}


def decide(protocol: dict, rows: dict, training: dict) -> dict:
    rule = protocol["evaluation"]["decision_rule"]
    source = {name: rows["source-r2"][name]["target"] for name in EVALUATION_STRATA}
    pair_results = {}
    for replica in REPLICAS:
        control = rows[f"control-{replica}"]["task4c_three_rule"]["target"]
        credit = rows[f"credit-{replica}"]["task4c_three_rule"]["target"]
        retention = {
            "task1": rows[f"credit-{replica}"]["task1"]["target"]["score_per_round"]
                     >= rule["minimum_task1_fraction_of_source"] * source["task1"]["score_per_round"] - 1e-12,
            "task2": rows[f"credit-{replica}"]["task2"]["target"]["score_per_round"]
                     >= source["task2"]["score_per_round"] - rule["maximum_task2_score_drop_from_source"] - 1e-12,
        }
        for stratum in ("task3_peaceful", "task3_coin"):
            candidate = rows[f"credit-{replica}"][stratum]["target"]
            retention[f"{stratum}_score"] = candidate["score_per_round"] >= source[stratum]["score_per_round"] - rule["maximum_task3_score_drop_from_source"] - 1e-12
            retention[f"{stratum}_kills"] = candidate["kills_per_round"] >= source[stratum]["kills_per_round"] - rule["maximum_task3_kills_drop_from_source"] - 1e-12
            retention[f"{stratum}_suicide"] = candidate["suicides_per_round"] <= source[stratum]["suicides_per_round"] + rule["maximum_task3_suicide_increase_over_source"] + 1e-12
        checks = {
            "kills_gain": credit["kills_per_round"] >= control["kills_per_round"] + rule["pair_minimum_task4c_kills_gain"] - 1e-12,
            "score_safe": credit["score_per_round"] >= control["score_per_round"] + rule["pair_minimum_task4c_score_delta"] - 1e-12,
            "suicide_safe": credit["suicides_per_round"] <= control["suicides_per_round"] + rule["pair_maximum_task4c_suicide_delta"] + 1e-12,
            "retention": all(retention.values()),
        }
        pair_results[replica] = {
            "checks": checks, "retention_checks": retention, "supportive": all(checks.values()),
            "task4c_deltas": {field: credit[field] - control[field] for field in ("score_per_round", "kills_per_round", "suicides_per_round")},
        }
    controls = [f"control-{replica}" for replica in REPLICAS]
    credits = [f"credit-{replica}" for replica in REPLICAS]
    pooled_control = _mean_metrics(rows, controls, "task4c_three_rule")
    pooled_credit = _mean_metrics(rows, credits, "task4c_three_rule")
    pooled_checks = {
        "kills_gain_over_control": pooled_credit["kills_per_round"] >= pooled_control["kills_per_round"] + rule["pooled_minimum_task4c_kills_gain_over_control"] - 1e-12,
        "score_safe_vs_control": pooled_credit["score_per_round"] >= pooled_control["score_per_round"] + rule["pooled_minimum_task4c_score_delta_to_control"] - 1e-12,
        "suicide_safe_vs_control": pooled_credit["suicides_per_round"] <= pooled_control["suicides_per_round"] + rule["pooled_maximum_task4c_suicide_delta_to_control"] + 1e-12,
        "kills_gain_over_source": pooled_credit["kills_per_round"] >= source["task4c_three_rule"]["kills_per_round"] + rule["pooled_minimum_task4c_kills_gain_over_source"] - 1e-12,
        "score_safe_vs_source": pooled_credit["score_per_round"] >= source["task4c_three_rule"]["score_per_round"] + rule["pooled_minimum_task4c_score_delta_to_source"] - 1e-12,
        "suicide_safe_vs_source": pooled_credit["suicides_per_round"] <= source["task4c_three_rule"]["suicides_per_round"] + rule["pooled_maximum_task4c_suicide_delta_to_source"] + 1e-12,
    }
    target_checks = {}
    for arm in ARMS:
        for replica in REPLICAS:
            diagnostics = training[arm][replica]["snapshots"][-1]["target_diagnostics"]
            key = f"{arm}-{replica}"
            target_checks[key] = (
                diagnostics["attribution_errors"] == 0
                and diagnostics["observed_kill_events"] == diagnostics["uniquely_attributed_kill_events"]
                and diagnostics["observed_self_events"] == diagnostics["uniquely_attributed_self_events"]
                and (
                    arm == "control"
                    or diagnostics["observed_kill_events"]
                    >= rule["minimum_observed_kill_events_per_credit_replica"]
                )
                and (
                    diagnostics["bomb_target_kill_events_included"] == 0
                    and diagnostics["bomb_target_self_events_included"] == 0
                    if arm == "control" else
                    diagnostics["bomb_target_kill_events_included"] == diagnostics["observed_kill_events"]
                    and diagnostics["bomb_target_self_events_included"] == diagnostics["observed_self_events"]
                )
            )
    supportive_count = sum(item["supportive"] for item in pair_results.values())
    passed = (
        supportive_count >= rule["minimum_supportive_pairs"]
        and all(pooled_checks.values())
        and all(target_checks.values())
    )
    return {
        "passed": passed,
        "decision": rule["pass_decision"] if passed else rule["fail_decision"],
        "supportive_pairs": supportive_count,
        "pair_results": pair_results,
        "pooled_task4c": {"control": pooled_control, "credit": pooled_credit, "source": source["task4c_three_rule"]},
        "pooled_checks": pooled_checks,
        "target_integrity_checks": target_checks,
        "task4b_started": False,
        "distillation_started": False,
        "automatic_followup_started": False,
        "task4_complete": False,
    }


def write_report(protocol_path: Path, protocol: dict, protocol_hash: str, training: dict, rows: dict, manifests: dict, result: dict) -> tuple[dict, str]:
    path = report_path(protocol)
    existing = load_completed(path, REPORT_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("completed BOMB-credit report protocol drift")
        return existing, sha256_file(path)
    record = {
        "schema_version": 1, "kind": REPORT_KIND, "status": "completed",
        "completed_at_utc": utc_now(), "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path), "protocol_sha256": protocol_hash,
        "active_subcourse": "task4c_three_rule",
        "training_manifests": {
            arm: {replica: {"path": relative(training_manifest_path(protocol, arm, replica)),
                            "sha256": sha256_file(training_manifest_path(protocol, arm, replica))}
                  for replica in REPLICAS} for arm in ARMS
        },
        "evaluation_rows": rows, "evaluation_manifests": manifests, "result": result,
        "awaiting_user_instruction": True, "source_checkpoint_modified": False,
        "default_checkpoint_modified": False,
    }
    atomic_json(path, record)
    return record, sha256_file(path)


def dry_run(protocol: dict) -> dict:
    training_rounds = len(ARMS) * len(REPLICAS) * int(protocol["training"]["rounds_per_arm"])
    identities = 2 + len(ARMS) * len(REPLICAS)
    evaluation_rounds = identities * sum(
        len(spec["cases"]) * int(spec["rounds_per_case"])
        for spec in protocol["evaluation"]["strata"].values()
    )
    return {
        "mode": "dry-run", "protocol_id": protocol["protocol_id"], "parent": "source-r2",
        "active_subcourse": "task4c_three_rule", "single_variable": protocol["single_training_variable"],
        "arms": list(ARMS), "replicas": list(REPLICAS), "rounds_per_arm": protocol["training"]["rounds_per_arm"],
        "total_training_rounds": training_rounds, "fixed_endpoint_round": 100,
        "diagnostic_snapshots": len(ARMS) * len(REPLICAS) * len(SNAPSHOT_ROUNDS),
        "evaluation_rounds": evaluation_rounds, "maximum_total_game_rounds": training_rounds + evaluation_rounds,
        "control_bomb_return_steps": 1, "credit_bomb_return_steps": 5, "non_bomb_return_steps": 1,
        "formal_training_started": False, "formal_evaluation_started": False,
        "task4b_started": False, "distillation_started": False,
        "automatic_followup_started": False, "writes_terminal_report_and_stops": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    protocol, protocol_hash = load_protocol(protocol_path)
    assert_registered_seeds_untouched(protocol_path, protocol)
    if not args.execute:
        print(json.dumps(dry_run(protocol), indent=2, sort_keys=True))
        return 0
    existing = load_completed(report_path(protocol), REPORT_KIND)
    if existing is not None:
        print(json.dumps({"status": existing["status"], "report": relative(report_path(protocol)),
                          "report_sha256": sha256_file(report_path(protocol)), **existing["result"]}, indent=2, sort_keys=True))
        return 0
    source_path = ROOT / protocol["source_parent"]["path"]
    source_before = sha256_file(source_path)
    training = {
        arm: {replica: train_one(protocol_path.resolve(), protocol, protocol_hash, arm, replica) for replica in REPLICAS}
        for arm in ARMS
    }
    if sha256_file(source_path) != source_before:
        raise RuntimeError("source-r2 changed during BOMB-credit training")
    rows, manifests, _ = run_evaluations(protocol_path.resolve(), protocol, protocol_hash)
    result = decide(protocol, rows, training)
    report, report_hash = write_report(protocol_path.resolve(), protocol, protocol_hash, training, rows, manifests, result)
    print(json.dumps({"status": report["status"], "report": relative(report_path(protocol)),
                      "report_sha256": report_hash, **report["result"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
