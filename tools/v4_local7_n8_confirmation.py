"""Selection-free confirmation of the post-hoc s141000 local7-n8 family signal.

The three immutable round-200 local7 endpoints are evaluated as a family with
frozen-v4 and source-r2 baselines.  No training, individual replica selection,
checkpoint copying, or automatic follow-up is permitted.
"""

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
from agent_code.model_a_v4_global_resource_n8.config import architecture_name  # noqa: E402
from agent_code.model_a_v4_global_resource_n8.network import torch  # noqa: E402
from tools.v4_task4_duel_training import (  # noqa: E402
    aggregate, atomic_json, load_completed, metrics_by_agent, opponent_summary,
    relative, seed_values,
)


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-local7-n8-confirmation-s142000.json"
LABELS = ("frozen-v4", "source-r2", "local7-r1", "local7-r2", "local7-r3")
LOCAL_LABELS = ("local7-r1", "local7-r2", "local7-r3")
STRATA = ("task1", "task2", "task4b_duel", "task4c_three_rule")
TASK4_STRATA = ("task4b_duel", "task4c_three_rule")
EVALUATION_KIND = "model-a-v4-local7-n8-confirmation-evaluation"
REPORT_KIND = "model-a-v4-local7-n8-confirmation-report"
EXPECTED_CASES = {
    "task1": ({"world_seed": 142101, "agent_seed": 242101},),
    "task2": tuple(
        {"world_seed": 142200 + index, "agent_seed": 242200 + index}
        for index in range(1, 5)
    ),
    "task4b_duel": tuple(
        {
            "world_seed": 142300 + index,
            "agent_seed": 242300 + index,
            "opponent_seed": 342300 + index,
        }
        for index in range(1, 5)
    ),
    "task4c_three_rule": tuple(
        {
            "world_seed": 142400 + index,
            "agent_seed": 242400 + index,
            "opponent_seed": 342400 + index,
        }
        for index in range(1, 5)
    ),
}
EXPECTED_SPECS = {
    "task1": ("coin-heaven", [], 25),
    "task2": ("classic", [], 25),
    "task4b_duel": ("classic", ["seeded_rule_based_agent"], 50),
    "task4c_three_rule": ("classic", ["seeded_rule_based_agent"] * 3, 50),
}


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
        raise ValueError(f"invalid s142000 checkpoint: {path}")
    return payload


def registered_seeds(protocol: dict) -> set[int]:
    return {
        int(value)
        for spec in protocol["evaluation"]["strata"].values()
        for case in spec["cases"]
        for value in case.values()
    }


def _validate_protocol_shape(protocol: dict) -> None:
    if protocol.get("schema_version") != 1 or protocol.get("kind") != "model-a-v4-local7-n8-confirmation":
        raise ValueError("wrong s142000 protocol kind")
    if protocol.get("protocol_id") != "model-a-v4-local7-n8-confirmation-s142000":
        raise ValueError("wrong s142000 protocol id")
    if tuple(protocol.get("labels", ())) != LABELS or tuple(protocol.get("fixed_candidate_family", ())) != LOCAL_LABELS:
        raise ValueError("s142000 identity set changed")
    if not protocol.get("selection_free") or protocol.get("training_allowed") is not False:
        raise ValueError("s142000 must remain selection-free and evaluation-only")
    if protocol.get("individual_replica_selection_allowed") is not False:
        raise ValueError("s142000 cannot select an individual local7 replica")
    if protocol.get("checkpoint_copy_allowed") is not False or protocol.get("automatic_followup") is not False:
        raise ValueError("s142000 cannot copy checkpoints or start a follow-up")
    strata = protocol.get("evaluation", {}).get("strata", {})
    if tuple(strata) != STRATA:
        raise ValueError("s142000 stratum order changed")
    for stratum in STRATA:
        spec = strata[stratum]
        scenario, opponents, rounds = EXPECTED_SPECS[stratum]
        if (
            spec.get("scenario") != scenario
            or spec.get("opponents") != opponents
            or int(spec.get("rounds_per_case", -1)) != rounds
            or tuple(spec.get("cases", ())) != EXPECTED_CASES[stratum]
        ):
            raise ValueError(f"s142000 {stratum} contract changed")
    if protocol["evaluation"].get("total_manifests") != 65:
        raise ValueError("s142000 manifest budget changed")
    if protocol["evaluation"].get("total_observed_rounds") != 2625:
        raise ValueError("s142000 evaluation budget changed")
    if protocol["confirmation_gate"].get("kills_used_for_gate") is not False:
        raise ValueError("kills must remain diagnostic-only in s142000")


def _validate_history(protocol: dict) -> None:
    discovery = protocol["historical_discovery"]
    for name in ("protocol", "terminal_report"):
        item = discovery[name]
        path = ROOT / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"s141000 {name} lineage mismatch")
    report = json.loads((ROOT / discovery["terminal_report"]["path"]).read_text(encoding="utf-8"))
    if (
        report.get("status") != "completed"
        or report.get("decision") != discovery["terminal_report"]["decision"]
        or report.get("checkpoint_selected") is not False
        or report.get("checkpoint_copied") is not False
    ):
        raise ValueError("s141000 terminal state changed")
    rows = report["decision_details"]["by_label_and_stratum"]
    observed = discovery["post_hoc_observation"]
    if abs(rows["frozen-v4/task2"]["score_per_round"] - observed["frozen_v4_task2_score_per_round"]) > 1e-12:
        raise ValueError("s141000 frozen-v4 observation changed")
    for replica, expected in observed["local7_replica_task2_score_per_round"].items():
        if abs(rows[f"local7-{replica}/task2"]["score_per_round"] - expected) > 1e-12:
            raise ValueError(f"s141000 local7-{replica} observation changed")


def _validate_checkpoint(protocol: dict, label: str) -> None:
    item = protocol["checkpoint_inventory"][label]
    path = ROOT / item["path"]
    if not path.is_file() or sha256_file(path) != item["sha256"]:
        raise ValueError(f"s142000 checkpoint mismatch: {label}")
    payload = _torch_load(path)
    if item["kind"] == "v4":
        if payload.get("architecture") not in (None, MODEL_ARCHITECTURE):
            raise ValueError("frozen-v4 architecture mismatch")
        return
    if item["kind"] == "source":
        for key, value in item["required_metadata"].items():
            if payload.get(key) != value:
                raise ValueError(f"source-r2 metadata mismatch: {key}")
        source_protocol = ROOT / item["protocol_path"]
        if not source_protocol.is_file() or sha256_file(source_protocol) != item["required_metadata"]["protocol_sha256"]:
            raise ValueError("source-r2 protocol mismatch")
        return
    replica = item["replica"]
    expected = {
        "architecture": architecture_name(),
        "protocol_sha256": protocol["historical_discovery"]["protocol"]["sha256"],
        "parent_sha256": protocol["checkpoint_inventory"]["frozen-v4"]["sha256"],
        "arm": "local7",
        "replica": replica,
        "completed_rounds": 200,
        "n_step": 8,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"local7-{replica} metadata mismatch: {key}")
    parent = _torch_load(ROOT / protocol["checkpoint_inventory"]["frozen-v4"]["path"])["online_net"]
    online = payload["online_net"]
    if any(not torch.equal(online[f"base.{key}"], value) for key, value in parent.items()):
        raise ValueError(f"local7-{replica} frozen base changed")
    if not any(bool((value != 0).any()) for key, value in online.items() if key.startswith("resource_head.")):
        raise ValueError(f"local7-{replica} resource head is zero")


def load_protocol(path: Path) -> tuple[dict, Path, str]:
    resolved = path if path.is_absolute() else ROOT / path
    resolved = resolved.resolve()
    raw = resolved.read_bytes()
    protocol = json.loads(raw)
    _validate_protocol_shape(protocol)
    _validate_history(protocol)
    for label in LABELS:
        _validate_checkpoint(protocol, label)
    return protocol, resolved, hashlib.sha256(raw).hexdigest()


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
        raise ValueError(f"registered s142000 seeds were already used: {conflicts}")


def manifest_path(protocol: dict, stratum: str, label: str, world_seed: int) -> Path:
    return ROOT / protocol["evaluation_manifest_directory"] / stratum / f"{label}-s{world_seed}.json"


def report_path(protocol: dict) -> Path:
    return ROOT / protocol["report_path"]


def _agent_environment(protocol: dict, label: str, case: dict) -> tuple[str, Path, dict[str, str]]:
    item = protocol["checkpoint_inventory"][label]
    checkpoint = (ROOT / item["path"]).resolve()
    if item["kind"] == "v4":
        return "model_a_dqn", checkpoint, {
            "MODEL_A_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_SEED": str(case["agent_seed"]),
        }
    if item["kind"] == "source":
        return "model_a_v4_curriculum", checkpoint, {
            "MODEL_A_V4C_PROTOCOL_PATH": str((ROOT / item["protocol_path"]).resolve()),
            "MODEL_A_V4C_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_V4C_ARM": "curriculum",
            "MODEL_A_V4C_REPLICA": "r2",
            "MODEL_A_V4C_STAGE": "task2",
            "MODEL_A_V4C_SEED": str(case["agent_seed"]),
        }
    return "model_a_v4_global_resource_n8", checkpoint, {
        "MODEL_A_GLOBAL_PROTOCOL_PATH": str((ROOT / protocol["historical_discovery"]["protocol"]["path"]).resolve()),
        "MODEL_A_GLOBAL_RUN_MODE": "evaluate",
        "MODEL_A_GLOBAL_ARM": "local7",
        "MODEL_A_GLOBAL_REPLICA": item["replica"],
        "MODEL_A_GLOBAL_SEED": str(case["agent_seed"]),
        "MODEL_A_GLOBAL_CHECKPOINT_PATH": str(checkpoint),
    }


def evaluate_one(
    protocol_path: Path, protocol: dict, protocol_hash: str, stratum: str, label: str, case: dict,
) -> dict:
    spec = protocol["evaluation"]["strata"][stratum]
    target, checkpoint, overrides = _agent_environment(protocol, label, case)
    output = manifest_path(protocol, stratum, label, int(case["world_seed"]))
    expected = {
        **case,
        "scenario": spec["scenario"],
        "opponents": spec["opponents"],
        "rounds": spec["rounds_per_case"],
        "target_agent": target,
    }
    existing = load_completed(output, EVALUATION_KIND)
    checkpoint_hash = sha256_file(checkpoint)
    if existing is not None:
        if (
            existing.get("protocol_sha256") != protocol_hash
            or existing.get("evaluation") != expected
            or existing.get("checkpoint", {}).get("sha256") != checkpoint_hash
        ):
            raise RuntimeError(f"completed s142000 evaluation drift: {output}")
        return existing
    stats = output.with_suffix(".stats.json")
    if output.exists() or stats.exists():
        raise RuntimeError(f"refusing orphaned s142000 evaluation: {output}")
    command = [
        sys.executable, "main.py", "play", "--agents", target, *spec["opponents"],
        "--train", "0", "--continue-without-training", "--no-gui",
        "--scenario", spec["scenario"], "--n-rounds", str(spec["rounds_per_case"]),
        "--seed", str(case["world_seed"]), "--save-stats", str(stats),
    ]
    if spec["opponents"]:
        overrides["TASK4_RULE_SEED"] = str(case["opponent_seed"])
    record = {
        "schema_version": 1,
        "kind": EVALUATION_KIND,
        "status": "running",
        "started_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "stratum": stratum,
        "label": label,
        "evaluation": expected,
        "checkpoint": {"path": relative(checkpoint), "sha256": checkpoint_hash},
        "command": command,
        "environment_overrides": overrides,
        "raw_stats": relative(stats),
        "policy_updates": 0,
    }
    atomic_json(output, record)
    child = os.environ.copy()
    child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = utc_now()
    record["exit_code"] = completed.returncode
    try:
        if completed.returncode != 0 or not stats.is_file() or sha256_file(checkpoint) != checkpoint_hash:
            raise RuntimeError("s142000 process, stats, or checkpoint immutability failed")
        by_agent = metrics_by_agent(json.loads(stats.read_text(encoding="utf-8")))
        if target not in by_agent:
            raise RuntimeError("s142000 target missing from stats")
        record["metrics_by_agent"] = by_agent
        record["target_metrics"] = by_agent[target]
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(output, record)
    if record["status"] != "completed":
        raise RuntimeError(f"s142000 evaluation failed: {label}/{stratum}: {record.get('error')}")
    return record


def _entry(runs: list[dict], target: str) -> dict:
    target_metrics = aggregate(runs)
    opponents = opponent_summary(runs, target)
    return {
        "target": target_metrics,
        "opponents": opponents,
        "score_margin": target_metrics["score_per_round"] - opponents["mean_score_per_agent_round"],
    }


def summarize(protocol: dict, runs: list[dict]) -> tuple[dict, dict, dict]:
    rows: dict[str, dict] = {}
    for label in LABELS:
        target, _, _ = _agent_environment(protocol, label, EXPECTED_CASES["task1"][0])
        rows[label] = {}
        for stratum in STRATA:
            selected = [run for run in runs if run["label"] == label and run["stratum"] == stratum]
            rows[label][stratum] = _entry(selected, target)
        selected = [run for run in runs if run["label"] == label and run["stratum"] in TASK4_STRATA]
        rows[label]["pooled_task4"] = _entry(selected, target)

    family: dict[str, dict] = {}
    family_target = "model_a_v4_global_resource_n8"
    for stratum in STRATA:
        selected = [run for run in runs if run["label"] in LOCAL_LABELS and run["stratum"] == stratum]
        family[stratum] = _entry(selected, family_target)
    selected = [run for run in runs if run["label"] in LOCAL_LABELS and run["stratum"] in TASK4_STRATA]
    family["pooled_task4"] = _entry(selected, family_target)

    cases: dict[str, dict] = {}
    for stratum in STRATA:
        cases[stratum] = {}
        for case in EXPECTED_CASES[stratum]:
            world_seed = str(case["world_seed"])
            cases[stratum][world_seed] = {}
            for label in LABELS:
                selected = [
                    run for run in runs
                    if run["label"] == label
                    and run["stratum"] == stratum
                    and run["evaluation"]["world_seed"] == case["world_seed"]
                ]
                target, _, _ = _agent_environment(protocol, label, case)
                cases[stratum][world_seed][label] = _entry(selected, target)
    return rows, family, cases


def decide(protocol: dict, rows: dict, family: dict) -> dict:
    gate = protocol["confirmation_gate"]
    v4 = rows["frozen-v4"]
    task2_pairs = {}
    for label in LOCAL_LABELS:
        gain = rows[label]["task2"]["target"]["score_per_round"] - v4["task2"]["target"]["score_per_round"]
        task2_pairs[label] = {
            "score_gain_over_v4": gain,
            "positive": gain >= float(gate["minimum_task2_replica_score_gain_over_v4"]) - 1e-12,
        }
    positive_task2 = sum(item["positive"] for item in task2_pairs.values())
    task2_checks = {
        "replica_support": positive_task2 >= int(gate["minimum_positive_task2_replicas"]),
        "family_score": family["task2"]["target"]["score_per_round"] >= float(gate["minimum_family_task2_score_ratio_vs_v4"]) * v4["task2"]["target"]["score_per_round"] - 1e-12,
        "family_coins": family["task2"]["target"]["coins_per_round"] >= float(gate["minimum_family_task2_coin_ratio_vs_v4"]) * v4["task2"]["target"]["coins_per_round"] - 1e-12,
        "task1_retention": family["task1"]["target"]["score_per_round"] >= float(gate["minimum_family_task1_score_ratio_vs_v4"]) * v4["task1"]["target"]["score_per_round"] - 1e-12,
        "task2_suicide_guard": family["task2"]["target"]["suicides_per_round"] <= v4["task2"]["target"]["suicides_per_round"] + float(gate["maximum_family_task2_suicide_increase_vs_v4"]) + 1e-12,
        "task2_invalid_guard": family["task2"]["target"]["invalid_actions_per_round"] <= v4["task2"]["target"]["invalid_actions_per_round"] + float(gate["maximum_family_task2_invalid_increase_vs_v4"]) + 1e-12,
    }

    safe_replicas = sum(
        rows[label]["pooled_task4"]["target"]["score_per_round"]
        >= v4["pooled_task4"]["target"]["score_per_round"]
        - float(gate["replica_maximum_pooled_task4_score_drop_vs_v4"]) - 1e-12
        for label in LOCAL_LABELS
    )
    competition_checks = {
        "pooled_score": family["pooled_task4"]["target"]["score_per_round"] >= v4["pooled_task4"]["target"]["score_per_round"] - float(gate["maximum_family_pooled_task4_score_drop_vs_v4"]) - 1e-12,
        "pooled_margin": family["pooled_task4"]["score_margin"] >= v4["pooled_task4"]["score_margin"] - float(gate["maximum_family_pooled_task4_margin_drop_vs_v4"]) - 1e-12,
        "pooled_suicide": family["pooled_task4"]["target"]["suicides_per_round"] <= v4["pooled_task4"]["target"]["suicides_per_round"] + float(gate["maximum_family_pooled_task4_suicide_increase_vs_v4"]) + 1e-12,
        "replica_support": safe_replicas >= int(gate["minimum_task4_safe_replicas"]),
    }
    for stratum in TASK4_STRATA:
        competition_checks[f"{stratum}_score"] = family[stratum]["target"]["score_per_round"] >= v4[stratum]["target"]["score_per_round"] - float(gate["maximum_family_task4_stratum_score_drop_vs_v4"]) - 1e-12
        competition_checks[f"{stratum}_suicide"] = family[stratum]["target"]["suicides_per_round"] <= v4[stratum]["target"]["suicides_per_round"] + float(gate["maximum_family_task4_stratum_suicide_increase_vs_v4"]) + 1e-12

    task2_confirmed = all(task2_checks.values())
    competition_safe = all(competition_checks.values())
    if task2_confirmed and competition_safe:
        decision = protocol["decision_branches"]["pass"]
    elif task2_confirmed:
        decision = protocol["decision_branches"]["task2_only"]
    else:
        decision = protocol["decision_branches"]["fail"]
    return {
        "decision": decision,
        "passed": task2_confirmed and competition_safe,
        "task2_family_confirmed": task2_confirmed,
        "competition_safe": competition_safe,
        "task2_checks": task2_checks,
        "competition_checks": competition_checks,
        "task2_replica_results": task2_pairs,
        "positive_task2_replicas": positive_task2,
        "task4_safe_replicas": safe_replicas,
        "kills_used_for_gate": False,
        "individual_replica_selected": False,
        "training_started": False,
        "checkpoint_copied": False,
        "automatic_followup_started": False,
        "awaiting_user_instruction": True,
    }


def dry_run(protocol: dict) -> dict:
    rounds_per_label = sum(
        len(spec["cases"]) * int(spec["rounds_per_case"])
        for spec in protocol["evaluation"]["strata"].values()
    )
    manifests = len(LABELS) * sum(len(spec["cases"]) for spec in protocol["evaluation"]["strata"].values())
    return {
        "mode": "dry-run",
        "protocol_id": protocol["protocol_id"],
        "labels": list(LABELS),
        "candidate_family": list(LOCAL_LABELS),
        "strata": list(STRATA),
        "evaluation_manifests": manifests,
        "rounds_per_label": rounds_per_label,
        "total_observed_rounds": rounds_per_label * len(LABELS),
        "training_started": False,
        "individual_replica_selection_allowed": False,
        "checkpoint_copy_allowed": False,
        "kills_used_for_gate": False,
        "formal_evaluation_started": False,
        "automatic_followup_started": False,
    }


def execute(protocol_path: Path, protocol: dict, protocol_hash: str) -> dict:
    runs = []
    for stratum in STRATA:
        for label in LABELS:
            for case in protocol["evaluation"]["strata"][stratum]["cases"]:
                runs.append(evaluate_one(protocol_path, protocol, protocol_hash, stratum, label, case))
    rows, family, cases = summarize(protocol, runs)
    result = decide(protocol, rows, family)
    report = {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "evaluation_manifests": [
            {
                "path": relative(manifest_path(protocol, run["stratum"], run["label"], run["evaluation"]["world_seed"])),
                "sha256": sha256_file(manifest_path(protocol, run["stratum"], run["label"], run["evaluation"]["world_seed"])),
            }
            for run in runs
        ],
        "rows": rows,
        "local7_family": family,
        "case_rows": cases,
        "result": result,
        "source_artifacts_modified": False,
        "default_checkpoint_modified": False,
        "checkpoint_selected": False,
        "checkpoint_copied": False,
        "training_started": False,
        "automatic_followup_started": False,
        "awaiting_user_instruction": True,
    }
    atomic_json(report_path(protocol), report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol, protocol_path, protocol_hash = load_protocol(args.protocol)
    assert_registered_seeds_untouched(protocol_path, protocol)
    if not args.execute:
        print(json.dumps({**dry_run(protocol), "protocol_sha256": protocol_hash}, indent=2, sort_keys=True))
        return 0
    terminal = report_path(protocol)
    if terminal.exists():
        raise SystemExit(f"terminal report already exists: {terminal}")
    report = execute(protocol_path, protocol, protocol_hash)
    print(json.dumps({
        "status": report["status"],
        "report": relative(terminal),
        "report_sha256": sha256_file(terminal),
        **report["result"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
