"""Task4-only confirmation of the post-hoc rehearsal25-r3 endpoint.

The fixed candidate, source-r2, and frozen v4 are evaluated on untouched
Task4B/Task4C seeds.  This runner is evaluation-only and always stops after
writing one terminal report.
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

from agent_code.model_a_dqn.callbacks import MODEL_ARCHITECTURE  # noqa: E402
from agent_code.model_a_dqn.network import torch  # noqa: E402
from tools.v4_task4_duel_training import (  # noqa: E402
    aggregate, atomic_json, load_completed, metrics_by_agent, opponent_summary,
    relative, seed_values,
)


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-rehearsal-r3-task4-confirmation-s138000.json"
LABELS = ("v4", "source-r2", "candidate-r3")
STRATA = ("task4b_duel", "task4c_three_rule")
EVALUATION_KIND = "model-a-v4-rehearsal-r3-task4-confirmation-evaluation"
REPORT_KIND = "model-a-v4-rehearsal-r3-task4-confirmation-report"
EXPECTED_CASES = {
    "task4b_duel": tuple(
        {"world_seed": 138400 + i, "agent_seed": 238400 + i, "opponent_seed": 338400 + i}
        for i in range(1, 5)
    ),
    "task4c_three_rule": tuple(
        {"world_seed": 138410 + i, "agent_seed": 238410 + i, "opponent_seed": 338410 + i}
        for i in range(1, 5)
    ),
}
EXPECTED_DISTRIBUTIONS = {
    "task4b_duel": ("classic", ["seeded_rule_based_agent"]),
    "task4c_three_rule": ("classic", ["seeded_rule_based_agent"] * 3),
}
EXPECTED_RULE = {
    "maximum_pooled_score_drop_from_each_baseline": 0.10,
    "maximum_pooled_score_margin_drop_from_each_baseline": 0.15,
    "minimum_pooled_kills_gain_over_each_baseline": 0.03,
    "maximum_pooled_suicide_increase_over_each_baseline": 0.03,
    "maximum_stratum_score_drop_from_each_baseline": 0.25,
    "maximum_stratum_score_margin_drop_from_each_baseline": 0.25,
    "maximum_stratum_kills_drop_from_each_baseline": 0.02,
    "maximum_stratum_suicide_increase_over_each_baseline": 0.05,
    "case_support_score_tolerance": 0.50,
    "case_support_margin_tolerance": 0.50,
    "case_support_suicide_tolerance": 0.10,
    "minimum_supportive_case_blocks_report_only": 5,
    "pass_competitive_decision": "candidate_confirmed_rule_competitive_stop_before_promotion",
    "pass_relative_only_decision": "candidate_confirmed_relative_only_stop_before_promotion",
    "fail_decision": "candidate_not_confirmed_retain_incumbent_stop",
}
SOURCE_PATHS = {
    "agent_code/model_a_dqn/callbacks.py",
    "agent_code/model_a_dqn/features.py",
    "agent_code/model_a_dqn/network.py",
    "agent_code/seeded_rule_based_agent/callbacks.py",
    "tools/v4_task4_duel_training.py",
    "tools/v4_rehearsal_r3_task4_confirmation.py",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    import hashlib
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
        raise ValueError(f"invalid confirmation checkpoint: {path}")
    return payload


def registered_seeds(protocol: dict) -> set[int]:
    return {
        int(value)
        for spec in protocol["evaluation"]["strata"].values()
        for case in spec["cases"]
        for value in case.values()
    }


def assert_registered_seeds_untouched(protocol_path: Path, protocol: dict) -> None:
    registered = registered_seeds(protocol)
    output_root = (ROOT / protocol["evaluation_manifest_directory"]).resolve()
    excluded_files = {protocol_path.resolve(), (ROOT / protocol["report_path"]).resolve()}
    conflicts = []
    for path in (ROOT / "experiments").rglob("*.json"):
        resolved = path.resolve()
        if resolved in excluded_files or output_root == resolved or output_root in resolved.parents:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        overlap = sorted(registered & seed_values(payload))
        if overlap:
            conflicts.append({"path": relative(path), "seeds": overlap})
    if conflicts:
        raise ValueError(f"registered s138000 seeds were already used: {conflicts}")


def _validate_checkpoint(protocol: dict, label: str) -> None:
    item = protocol["checkpoint_inventory"][label]
    path = ROOT / item["path"]
    if not path.is_file() or sha256_file(path) != item["sha256"]:
        raise ValueError(f"confirmation checkpoint mismatch: {label}")
    payload = _torch_load(path)
    if payload.get("architecture") not in (None, MODEL_ARCHITECTURE):
        raise ValueError(f"confirmation checkpoint architecture mismatch: {label}")
    if label == "candidate-r3":
        expected = {
            "architecture": MODEL_ARCHITECTURE,
            "protocol_sha256": protocol["candidate_lineage"]["training_protocol"]["sha256"],
            "arm": "rehearsal25",
            "replica": "r3",
            "stage_id": "task4c_rehearsal_pilot",
            "stage_completed_rounds": 100,
            "parent_sha256": protocol["checkpoint_inventory"]["source-r2"]["sha256"],
            "all_action_return_horizon": 8,
            "rehearsal_fraction": 0.25,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"candidate lineage mismatch: {key}")
    elif label == "source-r2":
        expected = protocol["source_lineage"]["checkpoint_metadata"]
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"source-r2 lineage mismatch: {key}")


def _combine(values: list[dict]) -> dict:
    additive = (
        "rounds", "score", "coins", "kills", "crates", "bombs", "moves",
        "waits", "suicides", "invalid_actions", "steps",
    )
    totals = {key: sum(int(value[key]) for value in values) for key in additive}
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


def _validate_discovery_lineage(protocol: dict) -> None:
    lineage = protocol["candidate_lineage"]
    for key in ("training_protocol", "terminal_report"):
        item = lineage[key]
        path = ROOT / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"candidate lineage artifact mismatch: {key}")
    report = json.loads((ROOT / lineage["terminal_report"]["path"]).read_text(encoding="utf-8"))
    if (
        report.get("status") != "completed"
        or report.get("result", {}).get("decision") != "rehearsal_not_supported_stop"
        or report.get("confirmation", {}).get("decision") is not None
    ):
        raise ValueError("s137000 terminal result changed")
    observed = lineage["revealed_validation_observation"]
    for label in ("source-r2", "rehearsal25-r3"):
        values = []
        for stratum in STRATA:
            row = report["validation"]["rows"][label][stratum]
            target = row["target"]
            margin = target["score_per_round"] - row["opponents"]["mean_score_per_agent_round"]
            expected = observed[label][stratum]
            actual = {
                "score_per_round": target["score_per_round"],
                "kills_per_round": target["kills_per_round"],
                "suicides_per_round": target["suicides_per_round"],
                "score_margin": margin,
            }
            if actual != expected:
                raise ValueError(f"revealed s137000 observation changed: {label}/{stratum}")
            values.append(target)
        pooled = _combine(values)
        expected_pooled = observed[label]["pooled"]
        actual_pooled = {
            "score_per_round": pooled["score_per_round"],
            "kills_per_round": pooled["kills_per_round"],
            "suicides_per_round": pooled["suicides_per_round"],
        }
        if actual_pooled != expected_pooled:
            raise ValueError(f"revealed s137000 pooled observation changed: {label}")


def load_protocol(path: Path) -> tuple[dict, str]:
    path = path.resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or protocol.get("kind") != "model-a-v4-rehearsal-r3-task4-confirmation":
        raise ValueError("wrong rehearsal-r3 confirmation protocol")
    if tuple(protocol.get("labels", ())) != LABELS or protocol.get("fixed_candidate") != "candidate-r3":
        raise ValueError("confirmation identities changed")
    if protocol.get("selection_free") is not True or protocol.get("training_allowed") is not False:
        raise ValueError("confirmation must be selection-free and evaluation-only")
    if protocol.get("task1_to_task3_are_diagnostic_only") is not True or protocol.get("automatic_followup") is not False:
        raise ValueError("confirmation scope or stop contract changed")
    strata = protocol.get("evaluation", {}).get("strata", {})
    if tuple(strata) != STRATA:
        raise ValueError("Task4 confirmation strata changed")
    for stratum in STRATA:
        scenario, opponents = EXPECTED_DISTRIBUTIONS[stratum]
        spec = strata[stratum]
        if (
            spec.get("scenario") != scenario
            or spec.get("opponents") != opponents
            or int(spec.get("rounds_per_case", 0)) != 50
            or tuple(spec.get("cases", ())) != EXPECTED_CASES[stratum]
        ):
            raise ValueError(f"Task4 confirmation distribution changed: {stratum}")
    seeds = registered_seeds(protocol)
    if len(seeds) != 24:
        raise ValueError("Task4 confirmation requires 24 globally unique seeds")
    if protocol.get("decision_rule") != EXPECTED_RULE:
        raise ValueError("Task4 confirmation decision rule changed")
    if set(protocol.get("checkpoint_inventory", {})) != set(LABELS):
        raise ValueError("Task4 confirmation checkpoint inventory changed")
    for label in LABELS:
        _validate_checkpoint(protocol, label)
    _validate_discovery_lineage(protocol)
    if set(protocol.get("source_bindings", {})) != SOURCE_PATHS:
        raise ValueError("Task4 confirmation source binding set changed")
    for source, expected in protocol["source_bindings"].items():
        path_to_source = ROOT / source
        if not path_to_source.is_file() or sha256_file(path_to_source) != expected:
            raise ValueError(f"Task4 confirmation source binding mismatch: {source}")
    assert_registered_seeds_untouched(path, protocol)
    return protocol, sha256_file(path)


def manifest_path(protocol: dict, stratum: str, label: str, world_seed: int) -> Path:
    return ROOT / protocol["evaluation_manifest_directory"] / stratum / f"{label}-s{world_seed}.json"


def report_path(protocol: dict) -> Path:
    return ROOT / protocol["report_path"]


def evaluate_one(protocol_path: Path, protocol: dict, protocol_hash: str, stratum: str, label: str, case: dict) -> dict:
    spec = protocol["evaluation"]["strata"][stratum]
    item = protocol["checkpoint_inventory"][label]
    checkpoint = (ROOT / item["path"]).resolve()
    if sha256_file(checkpoint) != item["sha256"]:
        raise RuntimeError(f"confirmation checkpoint drift: {label}")
    output = manifest_path(protocol, stratum, label, int(case["world_seed"]))
    expected = {
        **case,
        "scenario": spec["scenario"],
        "opponents": spec["opponents"],
        "rounds": int(spec["rounds_per_case"]),
        "target_agent": "model_a_dqn",
    }
    existing = load_completed(output, EVALUATION_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash or existing.get("evaluation") != expected:
            raise RuntimeError(f"completed confirmation evaluation drift: {output}")
        return existing
    stats = output.with_suffix(".stats.json")
    if output.exists() or stats.exists():
        raise RuntimeError(f"refusing orphaned confirmation output: {output}")
    command = [
        sys.executable, "main.py", "play", "--agents", "model_a_dqn", *spec["opponents"],
        "--train", "0", "--continue-without-training", "--no-gui",
        "--scenario", spec["scenario"], "--n-rounds", str(spec["rounds_per_case"]),
        "--seed", str(case["world_seed"]), "--save-stats", str(stats),
    ]
    overrides = {
        "MODEL_A_CHECKPOINT_PATH": str(checkpoint),
        "MODEL_A_SEED": str(case["agent_seed"]),
        "TASK4_RULE_SEED": str(case["opponent_seed"]),
    }
    record = {
        "schema_version": 1, "kind": EVALUATION_KIND, "status": "running",
        "started_at_utc": utc_now(), "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path), "protocol_sha256": protocol_hash,
        "stratum": stratum, "label": label, "evaluation": expected,
        "checkpoint": {"path": relative(checkpoint), "sha256": item["sha256"]},
        "command": command, "environment_overrides": overrides,
        "raw_stats": relative(stats),
    }
    atomic_json(output, record)
    child = os.environ.copy()
    child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = utc_now()
    record["exit_code"] = completed.returncode
    try:
        if completed.returncode != 0 or not stats.is_file() or sha256_file(checkpoint) != item["sha256"]:
            raise RuntimeError("evaluation process, stats, or checkpoint failed")
        record["metrics_by_agent"] = metrics_by_agent(json.loads(stats.read_text(encoding="utf-8")))
        record["target_metrics"] = record["metrics_by_agent"].get("model_a_dqn")
        if not record["target_metrics"]:
            raise RuntimeError("target metrics missing")
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(output, record)
    if record["status"] != "completed":
        raise RuntimeError(f"Task4 confirmation failed: {stratum}/{label}/{case['world_seed']}")
    return record


def run_evaluations(protocol_path: Path, protocol: dict, protocol_hash: str) -> tuple[dict, dict, dict]:
    rows: dict[str, dict] = {label: {} for label in LABELS}
    manifests: dict[str, dict] = {label: {} for label in LABELS}
    case_rows: dict[str, dict] = {stratum: {} for stratum in STRATA}
    for stratum in STRATA:
        spec = protocol["evaluation"]["strata"][stratum]
        runs_by_label = {}
        for label in LABELS:
            runs = [evaluate_one(protocol_path, protocol, protocol_hash, stratum, label, case) for case in spec["cases"]]
            runs_by_label[label] = runs
            target = aggregate(runs)
            opponents = opponent_summary(runs, "model_a_dqn")
            rows[label][stratum] = {
                "target": target,
                "opponents": opponents,
                "score_margin": target["score_per_round"] - opponents["mean_score_per_agent_round"],
            }
            manifests[label][stratum] = [relative(manifest_path(protocol, stratum, label, int(case["world_seed"]))) for case in spec["cases"]]
        for index, case in enumerate(spec["cases"]):
            block = {}
            for label in LABELS:
                run = runs_by_label[label][index]
                opponents = opponent_summary([run], "model_a_dqn")
                block[label] = {
                    "target": run["target_metrics"],
                    "opponents": opponents,
                    "score_margin": run["target_metrics"]["score_per_round"] - opponents["mean_score_per_agent_round"],
                }
            case_rows[stratum][str(case["world_seed"])] = block
    for label in LABELS:
        target = _combine([rows[label][stratum]["target"] for stratum in STRATA])
        total_rounds = sum(rows[label][stratum]["target"]["rounds"] for stratum in STRATA)
        margin = sum(rows[label][stratum]["score_margin"] * rows[label][stratum]["target"]["rounds"] for stratum in STRATA) / total_rounds
        rows[label]["pooled_task4"] = {"target": target, "score_margin": margin}
    return rows, case_rows, manifests


def decide(protocol: dict, rows: dict, case_rows: dict) -> dict:
    rule = protocol["decision_rule"]
    candidate = rows["candidate-r3"]
    baselines = ("source-r2", "v4")
    gates = {}
    c4 = candidate["pooled_task4"]
    for baseline in baselines:
        b4 = rows[baseline]["pooled_task4"]
        gates[f"pooled_score_vs_{baseline}"] = c4["target"]["score_per_round"] >= b4["target"]["score_per_round"] - rule["maximum_pooled_score_drop_from_each_baseline"] - 1e-12
        gates[f"pooled_margin_vs_{baseline}"] = c4["score_margin"] >= b4["score_margin"] - rule["maximum_pooled_score_margin_drop_from_each_baseline"] - 1e-12
        gates[f"pooled_kills_vs_{baseline}"] = c4["target"]["kills_per_round"] >= b4["target"]["kills_per_round"] + rule["minimum_pooled_kills_gain_over_each_baseline"] - 1e-12
        gates[f"pooled_suicide_vs_{baseline}"] = c4["target"]["suicides_per_round"] <= b4["target"]["suicides_per_round"] + rule["maximum_pooled_suicide_increase_over_each_baseline"] + 1e-12
        for stratum in STRATA:
            c = candidate[stratum]
            b = rows[baseline][stratum]
            prefix = f"{stratum}_vs_{baseline}"
            gates[f"{prefix}_score"] = c["target"]["score_per_round"] >= b["target"]["score_per_round"] - rule["maximum_stratum_score_drop_from_each_baseline"] - 1e-12
            gates[f"{prefix}_margin"] = c["score_margin"] >= b["score_margin"] - rule["maximum_stratum_score_margin_drop_from_each_baseline"] - 1e-12
            gates[f"{prefix}_kills"] = c["target"]["kills_per_round"] >= b["target"]["kills_per_round"] - rule["maximum_stratum_kills_drop_from_each_baseline"] - 1e-12
            gates[f"{prefix}_suicide"] = c["target"]["suicides_per_round"] <= b["target"]["suicides_per_round"] + rule["maximum_stratum_suicide_increase_over_each_baseline"] + 1e-12

    supportive_blocks = {}
    for stratum, blocks in case_rows.items():
        for seed, data in blocks.items():
            c = data["candidate-r3"]
            best_score = max(data[label]["target"]["score_per_round"] for label in baselines)
            best_margin = max(data[label]["score_margin"] for label in baselines)
            best_kills = max(data[label]["target"]["kills_per_round"] for label in baselines)
            best_suicide = min(data[label]["target"]["suicides_per_round"] for label in baselines)
            supportive_blocks[f"{stratum}/s{seed}"] = (
                c["target"]["score_per_round"] >= best_score - rule["case_support_score_tolerance"] - 1e-12
                and c["score_margin"] >= best_margin - rule["case_support_margin_tolerance"] - 1e-12
                and c["target"]["kills_per_round"] >= best_kills - 1e-12
                and c["target"]["suicides_per_round"] <= best_suicide + rule["case_support_suicide_tolerance"] + 1e-12
            )
    relative_passed = all(gates.values())
    competitive_checks = {
        f"{stratum}_beats_rule_mean": candidate[stratum]["score_margin"] >= -1e-12
        for stratum in STRATA
    }
    rule_competitive = all(competitive_checks.values())
    if relative_passed:
        decision = rule["pass_competitive_decision"] if rule_competitive else rule["pass_relative_only_decision"]
    else:
        decision = rule["fail_decision"]

    ranking = sorted(
        LABELS,
        key=lambda label: (
            rows[label]["pooled_task4"]["target"]["score_per_round"],
            rows[label]["pooled_task4"]["score_margin"],
            rows[label]["pooled_task4"]["target"]["kills_per_round"],
            -rows[label]["pooled_task4"]["target"]["suicides_per_round"],
            -LABELS.index(label),
        ),
        reverse=True,
    )
    deltas = {
        baseline: {
            "score_per_round": c4["target"]["score_per_round"] - rows[baseline]["pooled_task4"]["target"]["score_per_round"],
            "score_margin": c4["score_margin"] - rows[baseline]["pooled_task4"]["score_margin"],
            "kills_per_round": c4["target"]["kills_per_round"] - rows[baseline]["pooled_task4"]["target"]["kills_per_round"],
            "suicides_per_round": c4["target"]["suicides_per_round"] - rows[baseline]["pooled_task4"]["target"]["suicides_per_round"],
        }
        for baseline in baselines
    }
    return {
        "decision": decision,
        "relative_candidate_confirmed": relative_passed,
        "rule_competitive": rule_competitive,
        "relative_gates": gates,
        "rule_competitive_checks": competitive_checks,
        "candidate_minus_baselines": deltas,
        "supportive_case_blocks_report_only": sum(supportive_blocks.values()),
        "minimum_supportive_case_blocks_report_only": rule["minimum_supportive_case_blocks_report_only"],
        "case_block_support_report_only": supportive_blocks,
        "observed_ranking_report_only": ranking,
        "fixed_candidate": "candidate-r3",
        "selection_free": True,
        "training_started": False,
        "checkpoint_copied": False,
        "incumbent_replaced": False,
        "automatic_followup_started": False,
        "awaiting_user_instruction": True,
    }


def write_report(protocol_path: Path, protocol: dict, protocol_hash: str, rows: dict, case_rows: dict, manifests: dict) -> tuple[dict, str]:
    path = report_path(protocol)
    existing = load_completed(path, REPORT_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("completed Task4 confirmation report protocol drift")
        return existing, sha256_file(path)
    for label, item in protocol["checkpoint_inventory"].items():
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"checkpoint changed during confirmation: {label}")
    report = {
        "schema_version": 1, "kind": REPORT_KIND, "status": "completed",
        "completed_at_utc": utc_now(), "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path), "protocol_sha256": protocol_hash,
        "rows": rows, "case_rows": case_rows, "evaluation_manifests": manifests,
        "result": decide(protocol, rows, case_rows),
        "source_artifacts_modified": False, "default_checkpoint_modified": False,
        "automatic_followup_started": False, "awaiting_user_instruction": True,
    }
    atomic_json(path, report)
    return report, sha256_file(path)


def dry_run(protocol: dict) -> dict:
    per_label = sum(len(protocol["evaluation"]["strata"][s]["cases"]) * int(protocol["evaluation"]["strata"][s]["rounds_per_case"]) for s in STRATA)
    return {
        "mode": "dry-run", "protocol_id": protocol["protocol_id"],
        "fixed_candidate": protocol["fixed_candidate"], "labels": list(LABELS),
        "strata": list(STRATA), "rounds_per_label": per_label,
        "total_evaluation_rounds": per_label * len(LABELS),
        "task1_to_task3_are_diagnostic_only": True,
        "selection_free": True, "formal_evaluation_started": False,
        "training_started": False, "checkpoint_copy_allowed": False,
        "automatic_followup_started": False, "writes_terminal_report_and_stops": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    protocol, protocol_hash = load_protocol(protocol_path)
    if not args.execute:
        print(json.dumps(dry_run(protocol), indent=2, sort_keys=True))
        return 0
    existing = load_completed(report_path(protocol), REPORT_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("completed Task4 confirmation report protocol drift")
        report, report_hash = existing, sha256_file(report_path(protocol))
    else:
        rows, case_rows, manifests = run_evaluations(protocol_path.resolve(), protocol, protocol_hash)
        report, report_hash = write_report(protocol_path.resolve(), protocol, protocol_hash, rows, case_rows, manifests)
    print(json.dumps({
        "status": report["status"], "report": relative(report_path(protocol)),
        "report_sha256": report_hash, **report["result"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
