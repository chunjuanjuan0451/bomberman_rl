"""Run the terminal 25-round Task4C six-step BOMB reward-scale signal gate."""

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

from agent_code.model_a_v4_bomb_reward80.config import (  # noqa: E402
    ARMS, EVALUATION_STRATA, REPLICAS, SNAPSHOT_ROUNDS, final_checkpoint_path,
    load_protocol, sha256_file, snapshot_path,
)
from tools.v4_task4_duel_training import (  # noqa: E402
    aggregate, atomic_json, load_completed, metrics_by_agent, opponent_summary,
    relative, seed_values,
)


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-task4-bomb-reward80-s132000.json"
TRAINING_KIND = "model-a-v4-task4-bomb-reward80-training"
EVALUATION_KIND = "model-a-v4-task4-bomb-reward80-evaluation"
REPORT_KIND = "model-a-v4-task4-bomb-reward80-report"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def training_manifest_path(protocol: dict, arm: str, replica: str) -> Path:
    return ROOT / protocol["training_manifest_directory"] / arm / f"{replica}.json"


def evaluation_manifest_path(protocol: dict, stratum: str, label: str, world_seed: int) -> Path:
    return ROOT / protocol["evaluation_manifest_directory"] / stratum / f"{label}-s{world_seed}.json"


def report_path(protocol: dict) -> Path:
    return ROOT / protocol["report_path"]


def registered_seeds(protocol: dict) -> set[int]:
    values: set[int] = set()
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
        raise ValueError(f"registered s132000 seeds were already used: {conflicts}")


def _torch_load(path: Path) -> dict:
    from agent_code.model_a_v4_bomb_reward80.callbacks import _torch_load as load
    return load(path)


def validate_checkpoint(protocol: dict, protocol_hash: str, arm: str, replica: str, path: Path, stage_round: int) -> dict:
    if not path.is_file():
        raise RuntimeError(f"BOMB-reward80 checkpoint missing: {path}")
    payload = _torch_load(path)
    expected = {
        "architecture": "model-a-mlp-v4-global7",
        "protocol_sha256": protocol_hash,
        "arm": arm,
        "replica": replica,
        "stage_id": "task4c_bomb_reward80_gate",
        "stage_completed_rounds": stage_round,
        "parent_sha256": protocol["source_parent"]["sha256"],
        "training_variable": "killed_opponent_reward_12_vs_80",
        "kill_reward": 12.0 if arm == "reward12" else 80.0,
        "reward_profile": protocol["reward_profiles"][arm],
        "bomb_return_steps": 6,
        "non_bomb_return_steps": 1,
        "bomb_maturation_steps": 6,
        "allowed_outcome_lags": [4, 5],
        "terminal_event_delivery": "posthumous owned-bomb kills attach to the last terminal transition",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"BOMB-reward80 checkpoint {key} mismatch: {path}")
    if abs(float(payload.get("epsilon", -1.0)) - 0.05) > 1e-12:
        raise RuntimeError(f"BOMB-reward80 epsilon changed: {path}")
    diagnostics = payload.get("target_diagnostics", {})
    if diagnostics.get("raw_transitions") != diagnostics.get("non_bomb_one_step_targets", -1) + diagnostics.get("bomb_targets_matured", -1):
        raise RuntimeError(f"BOMB-reward80 transition accounting mismatch: {path}")
    if diagnostics.get("attribution_errors") != 0:
        raise RuntimeError(f"BOMB-reward80 outcome attribution failed: {path}")
    if diagnostics.get("observed_kill_events") != diagnostics.get("uniquely_attributed_kill_events"):
        raise RuntimeError(f"BOMB-reward80 kill attribution mismatch: {path}")
    if diagnostics.get("observed_self_events") != diagnostics.get("uniquely_attributed_self_events"):
        raise RuntimeError(f"BOMB-reward80 self attribution mismatch: {path}")
    if diagnostics.get("bomb_target_kill_events_included") != diagnostics.get("observed_kill_events"):
        raise RuntimeError(f"BOMB-reward80 kill target coverage mismatch: {path}")
    if diagnostics.get("bomb_target_self_events_included") != diagnostics.get("observed_self_events"):
        raise RuntimeError(f"BOMB-reward80 self target coverage mismatch: {path}")
    histogram = diagnostics.get("bomb_target_return_steps_histogram", {})
    if set(histogram) != {str(i) for i in range(1, 7)} or sum(histogram.values()) != diagnostics["bomb_targets_matured"]:
        raise RuntimeError(f"BOMB-reward80 return histogram mismatch: {path}")
    if histogram["6"] != diagnostics.get("bomb_targets_full_six_step_observed"):
        raise RuntimeError(f"BOMB-reward80 full-horizon mismatch: {path}")
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
        raise RuntimeError(f"BOMB-reward80 final is not immutable round-25: {arm}/{replica}")
    return result


def train_one(protocol_path: Path, protocol: dict, protocol_hash: str, arm: str, replica: str) -> dict:
    manifest = training_manifest_path(protocol, arm, replica)
    existing = load_completed(manifest, TRAINING_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError(f"completed BOMB-reward80 training protocol drift: {arm}/{replica}")
        validate_training_artifacts(protocol, protocol_hash, arm, replica)
        return existing
    output = final_checkpoint_path(protocol, arm, replica)
    stats_path = manifest.with_suffix(".stats.json")
    if output.parent.exists() or stats_path.exists() or manifest.exists():
        raise RuntimeError(f"refusing orphaned BOMB-reward80 artifacts: {arm}/{replica}")
    parent = (ROOT / protocol["source_parent"]["path"]).resolve()
    spec = protocol["training"]
    seeds = spec["seeds_by_replica"][replica]
    command = [
        sys.executable, "main.py", "play", "--agents", "model_a_v4_bomb_reward80", *spec["opponents"],
        "--train", "1", "--continue-without-training", "--no-gui", "--scenario", spec["scenario"],
        "--n-rounds", str(spec["rounds_per_arm"]), "--seed", str(seeds["world_seed"]),
        "--save-stats", str(stats_path),
    ]
    overrides = {
        "MODEL_A_BOMB_REWARD80_PROTOCOL_PATH": str(protocol_path),
        "MODEL_A_BOMB_REWARD80_CHECKPOINT_PATH": str(output),
        "MODEL_A_BOMB_REWARD80_PARENT_PATH": str(parent),
        "MODEL_A_BOMB_REWARD80_ARM": arm,
        "MODEL_A_BOMB_REWARD80_REPLICA": replica,
        "MODEL_A_BOMB_REWARD80_SEED": str(seeds["agent_seed"]),
        "TASK4_RULE_SEED": str(seeds["opponent_seed"]),
    }
    record = {
        "schema_version": 1, "kind": TRAINING_KIND, "status": "running", "started_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"], "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash, "arm": arm, "replica": replica,
        "training": {**seeds, "scenario": spec["scenario"], "opponents": spec["opponents"], "rounds": spec["rounds_per_arm"]},
        "reward_profile": protocol["reward_profiles"][arm],
        "target_contract": {"bomb_maturation_steps": 6, "bomb_return_steps": 6, "non_bomb_return_steps": 1,
                            "allowed_outcome_lags": [4, 5]},
        "parent_checkpoint": {"path": relative(parent), "sha256": sha256_file(parent)},
        "checkpoint": {"path": relative(output), "sha256": None}, "command": command,
        "environment_overrides": overrides, "artifacts": {"raw_stats": relative(stats_path)},
    }
    atomic_json(manifest, record)
    child = os.environ.copy()
    child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = utc_now()
    record["exit_code"] = completed.returncode
    try:
        if completed.returncode != 0 or not stats_path.is_file():
            raise RuntimeError("BOMB-reward80 training process or stats failed")
        record["snapshots"] = validate_training_artifacts(protocol, protocol_hash, arm, replica)
        record["metrics_by_agent"] = metrics_by_agent(json.loads(stats_path.read_text(encoding="utf-8")))
        record["checkpoint"]["sha256"] = sha256_file(output)
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"BOMB-reward80 training failed: {arm}/{replica}: {record.get('error')}")
    return record


def checkpoint_environment(protocol_path: Path, protocol: dict, label: str, case: dict) -> tuple[str, Path, dict[str, str]]:
    if label == "v4":
        checkpoint = (ROOT / protocol["frozen_v4_baseline"]["path"]).resolve()
        return "model_a_dqn", checkpoint, {"MODEL_A_CHECKPOINT_PATH": str(checkpoint), "MODEL_A_SEED": str(case["agent_seed"])}
    if label == "source-r2":
        checkpoint = (ROOT / protocol["source_parent"]["path"]).resolve()
        lineage = protocol["source_parent"]["lineage"]
        return "model_a_v4_curriculum", checkpoint, {
            "MODEL_A_V4C_PROTOCOL_PATH": str(ROOT / lineage["protocol_path"]),
            "MODEL_A_V4C_CHECKPOINT_PATH": str(checkpoint), "MODEL_A_V4C_ARM": "curriculum",
            "MODEL_A_V4C_REPLICA": "r2", "MODEL_A_V4C_STAGE": "task2", "MODEL_A_V4C_SEED": str(case["agent_seed"]),
        }
    arm, replica = label.split("-", 1)
    checkpoint = final_checkpoint_path(protocol, arm, replica)
    return "model_a_v4_bomb_reward80", checkpoint, {
        "MODEL_A_BOMB_REWARD80_PROTOCOL_PATH": str(protocol_path),
        "MODEL_A_BOMB_REWARD80_CHECKPOINT_PATH": str(checkpoint), "MODEL_A_BOMB_REWARD80_ARM": arm,
        "MODEL_A_BOMB_REWARD80_REPLICA": replica, "MODEL_A_BOMB_REWARD80_SEED": str(case["agent_seed"]),
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
            raise RuntimeError(f"completed BOMB-reward80 evaluation drift: {output}")
        if existing.get("checkpoint", {}).get("sha256") != checkpoint_hash:
            raise RuntimeError(f"completed BOMB-reward80 checkpoint drift: {output}")
        return existing
    stats_path = output.with_suffix(".stats.json")
    if output.exists() or stats_path.exists():
        raise RuntimeError(f"refusing orphaned BOMB-reward80 evaluation: {output}")
    command = [
        sys.executable, "main.py", "play", "--agents", target, *spec["opponents"], "--train", "0",
        "--continue-without-training", "--no-gui", "--scenario", spec["scenario"],
        "--n-rounds", str(spec["rounds_per_case"]), "--seed", str(case["world_seed"]), "--save-stats", str(stats_path),
    ]
    if spec["opponents"]:
        overrides["TASK4_RULE_SEED"] = str(case["opponent_seed"])
    record = {
        "schema_version": 1, "kind": EVALUATION_KIND, "status": "running", "started_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"], "protocol_path": relative(protocol_path), "protocol_sha256": protocol_hash,
        "stratum": stratum, "label": label, "evaluation": expected,
        "checkpoint": {"path": relative(checkpoint), "sha256": checkpoint_hash}, "command": command,
        "environment_overrides": overrides, "artifacts": {"raw_stats": relative(stats_path)},
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
        raise RuntimeError(f"BOMB-reward80 evaluation failed: {stratum}/{label}")
    return record


def run_evaluations(protocol_path: Path, protocol: dict, protocol_hash: str) -> tuple[dict, dict]:
    labels = ["v4", "source-r2", *[f"{arm}-{replica}" for replica in REPLICAS for arm in ARMS]]
    rows, manifests = {}, {}
    for label in labels:
        rows[label], manifests[label] = {}, {}
        for stratum in EVALUATION_STRATA:
            spec = protocol["evaluation"]["strata"][stratum]
            runs = [evaluate_one(protocol_path, protocol, protocol_hash, stratum, label, case) for case in spec["cases"]]
            target, _, _ = checkpoint_environment(protocol_path, protocol, label, spec["cases"][0])
            rows[label][stratum] = {"target": aggregate(runs), "opponents": opponent_summary(runs, target)}
            manifests[label][stratum] = [relative(evaluation_manifest_path(protocol, stratum, label, int(case["world_seed"]))) for case in spec["cases"]]
    return rows, manifests


def _mean_metrics(rows: dict, labels: list[str], stratum: str) -> dict:
    fields = ("score_per_round", "kills_per_round", "suicides_per_round", "bombs_per_round", "invalid_actions_per_round")
    return {field: sum(rows[label][stratum]["target"][field] for label in labels) / len(labels) for field in fields}


def _bomb_intervals(run: dict) -> dict:
    cumulative = [item["target_diagnostics"]["bomb_targets_matured"] for item in run["snapshots"]]
    counts = [cumulative[0], cumulative[1] - cumulative[0], cumulative[2] - cumulative[1], cumulative[3] - cumulative[2]]
    widths = [5, 5, 5, 10]
    return {"counts": counts, "rates_per_round": [count / width for count, width in zip(counts, widths)],
            "first_rate": counts[0] / 5, "last_rate": counts[-1] / 10}


def decide(protocol: dict, rows: dict, training: dict) -> dict:
    rule = protocol["evaluation"]["decision_rule"]
    source = {stratum: rows["source-r2"][stratum]["target"] for stratum in EVALUATION_STRATA}
    pair_results = {}
    for replica in REPLICAS:
        control = rows[f"reward12-{replica}"]["task4c_three_rule"]["target"]
        candidate = rows[f"reward80-{replica}"]["task4c_three_rule"]["target"]
        control_bombs = _bomb_intervals(training["reward12"][replica])
        candidate_bombs = _bomb_intervals(training["reward80"][replica])
        checks = {
            "kills_gain": candidate["kills_per_round"] >= control["kills_per_round"] + rule["pair_minimum_task4c_kills_gain"] - 1e-12,
            "score_safe": candidate["score_per_round"] >= control["score_per_round"] + rule["pair_minimum_task4c_score_delta"] - 1e-12,
            "suicide_safe": candidate["suicides_per_round"] <= control["suicides_per_round"] + rule["pair_maximum_task4c_suicide_delta"] + 1e-12,
            "last_bomb_rate_vs_control": candidate_bombs["last_rate"] >= rule["pair_minimum_last_interval_bomb_fraction_of_control"] * control_bombs["last_rate"] - 1e-12,
            "candidate_bomb_rate_not_collapsing": candidate_bombs["last_rate"] >= rule["candidate_minimum_last_interval_bomb_fraction_of_first"] * candidate_bombs["first_rate"] - 1e-12,
        }
        pair_results[replica] = {
            "supportive": all(checks.values()), "checks": checks,
            "task4c_deltas": {field: candidate[field] - control[field] for field in ("score_per_round", "kills_per_round", "suicides_per_round", "bombs_per_round")},
            "training_bomb_intervals": {"reward12": control_bombs, "reward80": candidate_bombs},
        }
    controls = [f"reward12-{replica}" for replica in REPLICAS]
    candidates = [f"reward80-{replica}" for replica in REPLICAS]
    pooled_control = _mean_metrics(rows, controls, "task4c_three_rule")
    pooled_candidate = _mean_metrics(rows, candidates, "task4c_three_rule")
    candidate_task1 = _mean_metrics(rows, candidates, "task1")
    candidate_task2 = _mean_metrics(rows, candidates, "task2")
    pooled_checks = {
        "kills_gain": pooled_candidate["kills_per_round"] >= pooled_control["kills_per_round"] + rule["pooled_minimum_task4c_kills_gain"] - 1e-12,
        "score_safe": pooled_candidate["score_per_round"] >= pooled_control["score_per_round"] + rule["pooled_minimum_task4c_score_delta"] - 1e-12,
        "suicide_safe": pooled_candidate["suicides_per_round"] <= pooled_control["suicides_per_round"] + rule["pooled_maximum_task4c_suicide_delta"] + 1e-12,
        "bomb_rate_safe": pooled_candidate["bombs_per_round"] >= rule["pooled_minimum_task4c_bomb_fraction_of_control"] * pooled_control["bombs_per_round"] - 1e-12,
        "task1_retention": candidate_task1["score_per_round"] >= rule["minimum_task1_fraction_of_source"] * source["task1"]["score_per_round"] - 1e-12,
        "task2_retention": candidate_task2["score_per_round"] >= source["task2"]["score_per_round"] - rule["maximum_task2_score_drop_from_source"] - 1e-12,
    }
    target_checks = {}
    for arm in ARMS:
        for replica in REPLICAS:
            payload = training[arm][replica]["snapshots"][-1]
            diagnostics = payload["target_diagnostics"]
            target_checks[f"{arm}-{replica}"] = (
                diagnostics["attribution_errors"] == 0
                and diagnostics["observed_kill_events"] >= rule["minimum_observed_kill_events_per_replica"]
                and diagnostics["observed_kill_events"] == diagnostics["uniquely_attributed_kill_events"] == diagnostics["bomb_target_kill_events_included"]
                and diagnostics["observed_self_events"] == diagnostics["uniquely_attributed_self_events"] == diagnostics["bomb_target_self_events_included"]
            )
    supportive_count = sum(item["supportive"] for item in pair_results.values())
    passed = supportive_count >= rule["minimum_supportive_pairs"] and all(pooled_checks.values()) and all(target_checks.values())
    return {
        "passed": passed, "decision": rule["pass_decision"] if passed else rule["fail_decision"],
        "supportive_pairs": supportive_count, "pair_results": pair_results,
        "pooled_task4c": {"reward12": pooled_control, "reward80": pooled_candidate, "source-r2": source["task4c_three_rule"]},
        "pooled_retention": {"reward80_task1": candidate_task1, "reward80_task2": candidate_task2,
                             "source_task1": source["task1"], "source_task2": source["task2"]},
        "pooled_checks": pooled_checks, "target_integrity_checks": target_checks,
        "expansion_started": False, "task4b_started": False, "sampling_experiment_started": False,
        "automatic_followup_started": False, "task4_complete": False,
    }


def write_report(protocol_path: Path, protocol: dict, protocol_hash: str, training: dict, rows: dict, manifests: dict, result: dict) -> tuple[dict, str]:
    path = report_path(protocol)
    existing = load_completed(path, REPORT_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("completed BOMB-reward80 report protocol drift")
        return existing, sha256_file(path)
    record = {
        "schema_version": 1, "kind": REPORT_KIND, "status": "completed", "completed_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"], "protocol_path": relative(protocol_path), "protocol_sha256": protocol_hash,
        "active_subcourse": "task4c_three_rule", "training_manifests": {
            arm: {replica: {"path": relative(training_manifest_path(protocol, arm, replica)),
                            "sha256": sha256_file(training_manifest_path(protocol, arm, replica))} for replica in REPLICAS}
            for arm in ARMS
        },
        "evaluation_rows": rows, "evaluation_manifests": manifests, "result": result,
        "awaiting_user_instruction": True, "source_checkpoint_modified": False,
        "default_checkpoint_modified": False, "prior_s131_artifacts_reused": False,
    }
    atomic_json(path, record)
    return record, sha256_file(path)


def dry_run(protocol: dict) -> dict:
    training_rounds = len(ARMS) * len(REPLICAS) * int(protocol["training"]["rounds_per_arm"])
    identities = 2 + len(ARMS) * len(REPLICAS)
    evaluation_rounds = identities * sum(len(spec["cases"]) * int(spec["rounds_per_case"]) for spec in protocol["evaluation"]["strata"].values())
    return {
        "mode": "dry-run", "protocol_id": protocol["protocol_id"], "parent": "source-r2",
        "active_subcourse": "task4c_three_rule", "single_variable": protocol["single_training_variable"],
        "arms": list(ARMS), "replicas": list(REPLICAS), "rounds_per_arm": protocol["training"]["rounds_per_arm"],
        "snapshot_rounds": list(SNAPSHOT_ROUNDS), "fixed_endpoint_round": 25,
        "total_training_rounds": training_rounds, "evaluation_rounds": evaluation_rounds,
        "maximum_total_game_rounds": training_rounds + evaluation_rounds,
        "reward12_kill_reward": 12.0, "reward80_kill_reward": 80.0,
        "bomb_return_steps_both_arms": 6, "non_bomb_return_steps_both_arms": 1,
        "bomb_maturation_steps_both_arms": 6, "allowed_outcome_lags": [4, 5],
        "fresh_s132000_seeds": True, "prior_s131_artifacts_reused": False,
        "formal_training_started": False, "formal_evaluation_started": False,
        "expansion_started": False, "task4b_started": False, "sampling_experiment_started": False,
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
    training = {arm: {replica: train_one(protocol_path.resolve(), protocol, protocol_hash, arm, replica) for replica in REPLICAS} for arm in ARMS}
    if sha256_file(source_path) != source_before:
        raise RuntimeError("source-r2 changed during BOMB-reward80 training")
    rows, manifests = run_evaluations(protocol_path.resolve(), protocol, protocol_hash)
    result = decide(protocol, rows, training)
    report, report_hash = write_report(protocol_path.resolve(), protocol, protocol_hash, training, rows, manifests, result)
    print(json.dumps({"status": report["status"], "report": relative(report_path(protocol)),
                      "report_sha256": report_hash, **report["result"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
