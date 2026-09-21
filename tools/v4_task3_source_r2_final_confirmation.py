"""Confirm source-r2 as the sole Task3 candidate on fresh outer seeds.

Dry-run is the default. ``--execute`` performs exactly 400 evaluation rounds
for source-r2 and frozen v4.  It never trains or starts Task4.  A passing run
copies source-r2 byte-for-byte to the dedicated frozen Task3 checkpoint and
writes a terminal selection manifest; a failing run writes only its report.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_v4_curriculum.config import (  # noqa: E402
    load_protocol as load_clean_protocol,
    sha256_file,
)
from tools.v4_clean_curriculum import validate_checkpoint as validate_clean_checkpoint  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-task3-source-r2-final-confirmation-s122000.json"
LABELS = ("v4", "source-r2")
STRATA = ("task1", "task2", "task3_peaceful", "task3_coin")
REPORT_KIND = "model-a-v4-task3-source-r2-final-confirmation-report"
EVALUATION_KIND = "model-a-v4-task3-source-r2-final-confirmation-evaluation"
SELECTION_KIND = "model-a-v4-task3-confirmed-checkpoint"
EXPECTED_CASES = {
    "task1": (
        {"world_seed": 122101, "agent_seed": 222101, "opponent_seed": 322101},
        {"world_seed": 122102, "agent_seed": 222102, "opponent_seed": 322102},
    ),
    "task2": (
        {"world_seed": 122201, "agent_seed": 222201, "opponent_seed": 322201},
        {"world_seed": 122202, "agent_seed": 222202, "opponent_seed": 322202},
    ),
    "task3_peaceful": (
        {"world_seed": 122301, "agent_seed": 222301, "opponent_seed": 322301},
        {"world_seed": 122302, "agent_seed": 222302, "opponent_seed": 322302},
    ),
    "task3_coin": (
        {"world_seed": 122401, "agent_seed": 222401, "opponent_seed": 322401},
        {"world_seed": 122402, "agent_seed": 222402, "opponent_seed": 322402},
    ),
}
EXPECTED_RULE = {
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
    if temporary.exists():
        raise RuntimeError(f"refusing stale temporary artifact: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_completed(path: Path, kind: str) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed" or payload.get("kind") != kind:
        raise RuntimeError(f"refusing incomplete/incompatible artifact: {path}")
    return payload


def metrics_by_agent(stats: dict) -> dict[str, dict]:
    result = {}
    for name, raw in stats.get("by_agent", {}).items():
        rounds = int(raw.get("rounds", 0))
        steps = int(raw.get("steps", 0))
        moves = int(raw.get("moves", 0))
        bombs = int(raw.get("bombs", 0))
        invalid = int(raw.get("invalid", 0))
        waits = max(0, steps - moves - bombs - invalid)
        result[name] = {
            "rounds": rounds,
            "score": int(raw.get("score", 0)),
            "score_per_round": int(raw.get("score", 0)) / rounds if rounds else 0.0,
            "coins": int(raw.get("coins", 0)),
            "coins_per_round": int(raw.get("coins", 0)) / rounds if rounds else 0.0,
            "kills": int(raw.get("kills", 0)),
            "kills_per_round": int(raw.get("kills", 0)) / rounds if rounds else 0.0,
            "crates": int(raw.get("crates", 0)),
            "crates_per_round": int(raw.get("crates", 0)) / rounds if rounds else 0.0,
            "bombs": bombs,
            "bombs_per_round": bombs / rounds if rounds else 0.0,
            "moves": moves,
            "move_fraction": moves / steps if steps else 0.0,
            "waits": waits,
            "wait_fraction": waits / steps if steps else 0.0,
            "suicides": int(raw.get("suicides", 0)),
            "suicides_per_round": int(raw.get("suicides", 0)) / rounds if rounds else 0.0,
            "invalid_actions": invalid,
            "invalid_actions_per_round": invalid / rounds if rounds else 0.0,
            "steps": steps,
        }
    return result


def aggregate(manifests: list[dict]) -> dict:
    additive = (
        "rounds", "score", "coins", "kills", "crates", "bombs", "moves",
        "waits", "suicides", "invalid_actions", "steps",
    )
    totals = {key: sum(int(item["target_metrics"][key]) for item in manifests) for key in additive}
    rounds = totals["rounds"]
    steps = totals["steps"]
    return {
        **totals,
        "score_per_round": totals["score"] / rounds,
        "coins_per_round": totals["coins"] / rounds,
        "kills_per_round": totals["kills"] / rounds,
        "crates_per_round": totals["crates"] / rounds,
        "bombs_per_round": totals["bombs"] / rounds,
        "move_fraction": totals["moves"] / steps if steps else 0.0,
        "wait_fraction": totals["waits"] / steps if steps else 0.0,
        "suicides_per_round": totals["suicides"] / rounds,
        "invalid_actions_per_round": totals["invalid_actions"] / rounds,
    }


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
    registered = {
        int(value)
        for cases in EXPECTED_CASES.values()
        for case in cases
        for value in case.values()
    }
    excluded_roots = {
        (ROOT / protocol["evaluation_manifest_directory"]).resolve(),
    }
    excluded_files = {
        protocol_path.resolve(),
        (ROOT / protocol["report_path"]).resolve(),
        (ROOT / protocol["confirmed_checkpoint"]["selection_manifest_path"]).resolve(),
    }
    conflicts = []
    for path in (ROOT / "experiments").rglob("*.json"):
        resolved = path.resolve()
        if resolved in excluded_files or any(root in resolved.parents for root in excluded_roots):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        overlap = sorted(registered & seed_values(payload))
        if overlap:
            conflicts.append({"path": relative(path), "seeds": overlap})
    if conflicts:
        raise ValueError(f"registered s122000 seeds were already used: {conflicts}")


def load_protocol(path: Path) -> tuple[dict, str, dict, str]:
    path = path.resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or protocol.get("kind") != "model-a-v4-task3-source-r2-final-confirmation":
        raise ValueError("wrong source-r2 final-confirmation protocol")
    if tuple(protocol.get("labels", ())) != LABELS or protocol.get("fixed_candidate") != "source-r2":
        raise ValueError("source-r2 must be the sole candidate and v4 the sole baseline")
    if protocol.get("candidate_count") != 1 or protocol.get("selection_free") is not True:
        raise ValueError("final confirmation must remain selection-free")
    if protocol.get("training_allowed") is not False or protocol.get("automatic_task4") is not False:
        raise ValueError("training and Task4 must remain disabled")

    strata = protocol.get("evaluation", {}).get("strata", {})
    if tuple(strata) != STRATA:
        raise ValueError("four-stratum order changed")
    expected_distributions = {
        "task1": ("coin-heaven", []),
        "task2": ("classic", []),
        "task3_peaceful": ("classic", ["seeded_peaceful_agent"]),
        "task3_coin": ("classic", ["seeded_coin_collector_agent"]),
    }
    all_seeds = []
    for stratum, spec in strata.items():
        scenario, opponents = expected_distributions[stratum]
        if spec.get("scenario") != scenario or spec.get("opponents") != opponents:
            raise ValueError(f"evaluation distribution changed: {stratum}")
        if int(spec.get("rounds_per_case", 0)) != 25:
            raise ValueError("each final-confirmation case must contain 25 rounds")
        if tuple(spec.get("cases", ())) != EXPECTED_CASES[stratum]:
            raise ValueError(f"fresh seed registration changed: {stratum}")
        for case in spec["cases"]:
            if set(case) != {"world_seed", "agent_seed", "opponent_seed"}:
                raise ValueError("incomplete final-confirmation seed tuple")
            all_seeds.extend(int(value) for value in case.values())
    if len(all_seeds) != 24 or len(set(all_seeds)) != 24:
        raise ValueError("final-confirmation seeds must be globally unique")
    assert_registered_seeds_untouched(path, protocol)

    bindings = protocol.get("source_bindings", {})
    if set(bindings) != {"tools/v4_task3_source_r2_final_confirmation.py"}:
        raise ValueError("final-confirmation source binding set mismatch")
    for name, expected_hash in bindings.items():
        source = ROOT / name
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise ValueError(f"final-confirmation source binding mismatch: {name}")

    evidence = protocol["source_evidence"]
    b1_path = ROOT / evidence["rejected_b1_outer_report"]["path"]
    if not b1_path.is_file() or sha256_file(b1_path) != evidence["rejected_b1_outer_report"]["sha256"]:
        raise ValueError("rejected b1 outer report binding mismatch")
    b1 = json.loads(b1_path.read_text(encoding="utf-8"))
    if (
        b1.get("status") != "completed"
        or b1.get("result", {}).get("passed") is not False
        or b1.get("result", {}).get("decision") != "b1_peaceful_outer_rejected"
    ):
        raise ValueError("b1 is not formally rejected by the bound outer report")

    clean_item = evidence["clean_protocol"]
    clean_path = ROOT / clean_item["path"]
    clean_protocol, clean_hash = load_clean_protocol(clean_path)
    if clean_hash != clean_item["sha256"]:
        raise ValueError("clean curriculum protocol binding mismatch")
    inventory = protocol.get("checkpoint_inventory", {})
    expected_v4 = {
        "path": clean_protocol["frozen_v4_baseline"]["checkpoint_path"],
        "sha256": clean_protocol["frozen_v4_baseline"]["checkpoint_sha256"],
    }
    if inventory.get("v4") != expected_v4:
        raise ValueError("frozen-v4 checkpoint inventory mismatch")
    if inventory.get("source_r2") != evidence["source_r2_checkpoint"]:
        raise ValueError("source-r2 checkpoint inventory mismatch")
    for label, key in (("v4", "v4"), ("source-r2", "source_r2")):
        item = inventory[key]
        checkpoint = ROOT / item["path"]
        if not checkpoint.is_file() or sha256_file(checkpoint) != item["sha256"]:
            raise ValueError(f"checkpoint binding mismatch: {label}")
    validate_clean_checkpoint(clean_protocol, clean_hash, "curriculum", "r2", "task2")

    run_item = evidence["source_r2_training_manifest"]
    run_path = ROOT / run_item["path"]
    if not run_path.is_file() or sha256_file(run_path) != run_item["sha256"]:
        raise ValueError("source-r2 training-manifest binding mismatch")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if (
        run.get("status") != "completed"
        or run.get("arm") != "curriculum"
        or run.get("replica") != "r2"
        or run.get("stage_id") != "task2"
        or run.get("checkpoint") != inventory["source_r2"]
    ):
        raise ValueError("source-r2 training lineage mismatch")

    task2_item = evidence["source_r2_task2_report"]
    task2_path = ROOT / task2_item["path"]
    if not task2_path.is_file() or sha256_file(task2_path) != task2_item["sha256"]:
        raise ValueError("source-r2 Task2 report binding mismatch")
    task2_report = json.loads(task2_path.read_text(encoding="utf-8"))
    if (
        task2_report.get("status") != "completed"
        or task2_report.get("course") != "task2"
        or task2_report.get("protocol_sha256") != clean_hash
        or task2_report.get("training_manifests", {}).get("task2", {}).get("curriculum", {}).get("r2") != run_item
        or task2_report.get("checkpoints", {}).get("task2", {}).get("curriculum", {}).get("r2", {}).get("path")
        != inventory["source_r2"]["path"]
        or task2_report.get("checkpoints", {}).get("task2", {}).get("curriculum", {}).get("r2", {}).get("sha256")
        != inventory["source_r2"]["sha256"]
    ):
        raise ValueError("source-r2 Task2 report lineage mismatch")

    rule = protocol.get("decision_rule", {})
    for key, value in EXPECTED_RULE.items():
        if rule.get(key) != value:
            raise ValueError(f"final-confirmation rule changed: {key}")
    if (
        rule.get("pass_decision") != "source_r2_confirmed_task3_complete"
        or rule.get("fail_decision") != "source_r2_rejected_task3_unresolved_stop"
    ):
        raise ValueError("final-confirmation decision labels changed")
    freeze = protocol.get("confirmed_checkpoint", {})
    if (
        freeze.get("copy_on_pass") is not True
        or freeze.get("source_label") != "source-r2"
        or freeze.get("task3_terminal") is not True
    ):
        raise ValueError("passing source-r2 must freeze Task3 and terminate its search")
    return protocol, sha256_file(path), clean_protocol, clean_hash


def checkpoint_identity(protocol: dict, label: str) -> tuple[str, Path, dict[str, str]]:
    inventory = protocol["checkpoint_inventory"]
    if label == "v4":
        return "model_a_dqn", (ROOT / inventory["v4"]["path"]).resolve(), {}
    if label == "source-r2":
        clean_path = ROOT / protocol["source_evidence"]["clean_protocol"]["path"]
        return "model_a_v4_curriculum", (ROOT / inventory["source_r2"]["path"]).resolve(), {
            "MODEL_A_V4C_PROTOCOL_PATH": str(clean_path),
            "MODEL_A_V4C_ARM": "curriculum",
            "MODEL_A_V4C_REPLICA": "r2",
            "MODEL_A_V4C_STAGE": "task2",
        }
    raise ValueError(f"invalid final-confirmation label: {label}")


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
    inventory_key = "v4" if label == "v4" else "source_r2"
    if checkpoint_hash != protocol["checkpoint_inventory"][inventory_key]["sha256"]:
        raise RuntimeError(f"checkpoint drift before final confirmation: {label}")
    output = manifest_path(protocol, stratum, label, int(case["world_seed"]))
    expected = {
        **case,
        "scenario": spec["scenario"],
        "opponents": spec["opponents"],
        "rounds": int(spec["rounds_per_case"]),
        "target_agent": target,
    }
    existing = load_completed(output, EVALUATION_KIND)
    if existing is not None:
        if (
            existing.get("protocol_sha256") != protocol_hash
            or existing.get("evaluation") != expected
            or existing.get("checkpoint", {}).get("sha256") != checkpoint_hash
        ):
            raise RuntimeError(f"completed final-confirmation evaluation drift: {output}")
        return existing
    stats_path = output.with_suffix(".stats.json")
    if stats_path.exists():
        raise RuntimeError(f"refusing orphaned final-confirmation stats: {stats_path}")
    agents = [target, *spec["opponents"]]
    command = [
        sys.executable, "main.py", "play", "--agents", *agents,
        "--train", "0", "--continue-without-training", "--no-gui",
        "--scenario", spec["scenario"], "--n-rounds", str(spec["rounds_per_case"]),
        "--seed", str(case["world_seed"]), "--save-stats", str(stats_path),
    ]
    overrides.update({
        "MODEL_A_CHECKPOINT_PATH" if label == "v4" else "MODEL_A_V4C_CHECKPOINT_PATH": str(checkpoint),
        "MODEL_A_SEED" if label == "v4" else "MODEL_A_V4C_SEED": str(case["agent_seed"]),
    })
    if spec["opponents"]:
        overrides["TASK3_OPPONENT_SEED"] = str(case["opponent_seed"])
    record = {
        "schema_version": 1,
        "kind": EVALUATION_KIND,
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
        raise RuntimeError(f"final confirmation failed: {stratum}/{label}/{case['world_seed']}")
    return record


def run_evaluations(protocol_path: Path, protocol: dict, protocol_hash: str) -> tuple[dict, dict]:
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
    candidate = "source-r2"
    rule = protocol["decision_rule"]
    v4 = {stratum: rows[stratum]["v4"] for stratum in STRATA}
    item = {stratum: rows[stratum][candidate] for stratum in STRATA}
    task1_fraction = (
        item["task1"]["score_per_round"] / v4["task1"]["score_per_round"]
        if v4["task1"]["score_per_round"] > 0 else 1.0
    )
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
        "decision": rule["pass_decision"] if passed else rule["fail_decision"],
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
        "training_started": False,
        "task4_started": False,
        "task3_complete": passed,
        "task3_search_terminated": passed,
        "automatic_checkpoint_copied": False,
        "confirmed_checkpoint": None,
    }


def freeze_confirmed_checkpoint(protocol: dict, protocol_hash: str, result: dict) -> dict:
    source_item = protocol["checkpoint_inventory"]["source_r2"]
    source = ROOT / source_item["path"]
    target = ROOT / protocol["confirmed_checkpoint"]["path"]
    selection_path = ROOT / protocol["confirmed_checkpoint"]["selection_manifest_path"]
    if sha256_file(source) != source_item["sha256"]:
        raise RuntimeError("source-r2 drift before Task3 freeze")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != source_item["sha256"]:
            raise RuntimeError(f"refusing to overwrite incompatible frozen checkpoint: {target}")
    else:
        temporary = target.with_suffix(target.suffix + ".tmp")
        if temporary.exists():
            raise RuntimeError(f"refusing stale checkpoint temporary file: {temporary}")
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != source_item["sha256"]:
            raise RuntimeError("frozen Task3 checkpoint copy hash mismatch")
        temporary.replace(target)
    selection = {
        "schema_version": 1,
        "kind": SELECTION_KIND,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "decision": result["decision"],
        "task3_complete": True,
        "task3_search_terminated": True,
        "source_checkpoint": {"path": relative(source), "sha256": sha256_file(source)},
        "confirmed_checkpoint": {"path": relative(target), "sha256": sha256_file(target)},
        "copy_semantics": "byte-for-byte freeze of source-r2; no training and no default checkpoint replacement",
        "task4_started": False,
    }
    existing = load_completed(selection_path, SELECTION_KIND)
    if existing is not None:
        comparable = {key: value for key, value in selection.items() if key != "completed_at_utc"}
        actual = {key: value for key, value in existing.items() if key != "completed_at_utc"}
        if actual != comparable:
            raise RuntimeError("completed Task3 selection manifest drift")
    else:
        atomic_json(selection_path, selection)
    return {
        "path": relative(target),
        "sha256": sha256_file(target),
        "source_path": relative(source),
        "source_sha256": sha256_file(source),
        "selection_manifest_path": relative(selection_path),
        "selection_manifest_sha256": sha256_file(selection_path),
    }


def report_path(protocol: dict) -> Path:
    return ROOT / protocol["report_path"]


def validate_completed_report(protocol: dict, protocol_hash: str, report: dict) -> None:
    if report.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("completed final report protocol drift")
    result = report.get("result", {})
    if result.get("passed"):
        frozen = result.get("confirmed_checkpoint") or {}
        target = ROOT / frozen.get("path", "")
        selection = ROOT / frozen.get("selection_manifest_path", "")
        if (
            not target.is_file()
            or sha256_file(target) != frozen.get("sha256")
            or not selection.is_file()
            or sha256_file(selection) != frozen.get("selection_manifest_sha256")
        ):
            raise RuntimeError("completed report's frozen Task3 artifacts drifted")


def write_report(
    protocol_path: Path, protocol: dict, protocol_hash: str, rows: dict, manifests: dict,
) -> tuple[dict, str]:
    path = report_path(protocol)
    existing = load_completed(path, REPORT_KIND)
    if existing is not None:
        validate_completed_report(protocol, protocol_hash, existing)
        return existing, sha256_file(path)
    source_hash_before = sha256_file(ROOT / protocol["checkpoint_inventory"]["source_r2"]["path"])
    v4_hash_before = sha256_file(ROOT / protocol["checkpoint_inventory"]["v4"]["path"])
    result = decide(protocol, rows)
    if result["passed"]:
        result["confirmed_checkpoint"] = freeze_confirmed_checkpoint(protocol, protocol_hash, result)
        result["automatic_checkpoint_copied"] = True
    source_hash_after = sha256_file(ROOT / protocol["checkpoint_inventory"]["source_r2"]["path"])
    v4_hash_after = sha256_file(ROOT / protocol["checkpoint_inventory"]["v4"]["path"])
    report = {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "rows": rows,
        "evaluation_manifests": manifests,
        "result": result,
        "awaiting_user_instruction": True,
        "source_artifacts_modified": source_hash_before != source_hash_after,
        "default_checkpoint_modified": v4_hash_before != v4_hash_after,
    }
    if report["source_artifacts_modified"] or report["default_checkpoint_modified"]:
        raise RuntimeError("source or default checkpoint changed during final confirmation")
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
        "fixed_candidate": "source-r2",
        "candidate_count": 1,
        "strata": list(STRATA),
        "total_evaluation_rounds": rounds,
        "selection_free": True,
        "uses_fresh_s122000_seeds": True,
        "formal_evaluation_started": False,
        "training_started": False,
        "checkpoint_copy_started": False,
        "conditional_checkpoint_freeze_on_pass": True,
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
        existing = load_completed(report_path(protocol), REPORT_KIND)
        if existing is not None:
            validate_completed_report(protocol, protocol_hash, existing)
            print(json.dumps({
                "status": "completed",
                "report": relative(report_path(protocol)),
                "report_sha256": sha256_file(report_path(protocol)),
                "decision": existing["result"]["decision"],
                "passed": existing["result"]["passed"],
                "confirmed_checkpoint": existing["result"].get("confirmed_checkpoint"),
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
            "confirmed_checkpoint": report["result"].get("confirmed_checkpoint"),
            "awaiting_user_instruction": True,
            "training_started": False,
            "task4_started": False,
        }, indent=2, sort_keys=True))
        return 0
    except (OSError, KeyError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
