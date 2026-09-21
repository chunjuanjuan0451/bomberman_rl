"""Run the s168 frozen control-r2 productive-bomb paired A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_cnn_n8.network import ARCHITECTURE  # noqa: E402
from tools.cnn_n8_task1 import _metrics, _torch_load, atomic_json, relative, sha256_file, utc_now  # noqa: E402
from tools.cnn_n8_task4 import combine_metrics  # noqa: E402
from tools.v4_task4_duel_training import seed_values  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-cnn-n8-productive-bomb-ab-s168000.json"
KIND = "model-a-cnn-n8-productive-bomb-ab"
AGENT = "model_a_cnn_n8_productive_bomb_ab"
TRACE_KIND = "model-a-cnn-n8-productive-bomb-ab-trace"


def assert_seeds_untouched(path: Path, protocol: dict) -> None:
    registered = seed_values(protocol["collection"]["cases"])
    output_root = (ROOT / protocol["manifest_directory"]).resolve()
    trace_root = (ROOT / protocol["trace_directory"]).resolve()
    report = (ROOT / protocol["report_path"]).resolve()
    conflicts = []
    for candidate in (ROOT / "experiments").rglob("*.json"):
        resolved = candidate.resolve()
        if (resolved == path.resolve() or resolved == report
                or output_root == resolved or output_root in resolved.parents
                or trace_root == resolved or trace_root in resolved.parents):
            continue
        try:
            overlap = sorted(registered & seed_values(json.loads(candidate.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError):
            continue
        if overlap:
            conflicts.append({"path": relative(candidate), "seeds": overlap})
    if conflicts:
        raise RuntimeError(f"s168 seeds already used: {conflicts}")


def load_protocol(path: Path) -> tuple[dict, str]:
    path = path.resolve()
    raw = path.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("kind") != KIND:
        raise RuntimeError("wrong productive-bomb A/B protocol")
    if any(protocol.get(key) for key in (
        "training_allowed", "checkpoint_selection_allowed", "checkpoint_copy_allowed",
        "candidate_deployment_allowed", "automatic_followup",
    )):
        raise RuntimeError("productive-bomb A/B must remain read-only and terminal")
    collection = protocol["collection"]
    cases = collection["cases"]
    if (collection.get("rounds_per_arm_case") != 25
            or collection.get("rounds_per_arm") != 100
            or collection.get("total_environment_rounds") != 200
            or collection.get("policy_updates") != 0
            or len(cases) != 4
            or sum(case["stratum"] == "rule" for case in cases.values()) != 2
            or any(sorted(case["arm_order"]) != ["control", "treatment"] for case in cases.values())):
        raise RuntimeError("productive-bomb A/B budget or arm order drift")
    expected_gate = {
        "minimum_score_gain_per_round": 0.10,
        "minimum_case_score_wins_out_of_4": 3,
        "minimum_rule_score_delta_per_round": -0.05,
        "minimum_mixed_score_delta_per_round": -0.05,
        "maximum_suicide_increase_per_round": 0.0,
        "maximum_kill_loss_per_round": 0.02,
        "pass_decision": "productive_bomb_gate_supported_for_mask_consistent_short_training_stop",
        "fail_decision": "productive_bomb_gate_not_supported_retain_control_r2_round0150_stop",
    }
    if protocol.get("decision_rule") != expected_gate:
        raise RuntimeError("productive-bomb A/B decision rule drift")
    provenance = protocol.get("provenance", {})
    if provenance.get("external_source_imported") or provenance.get("external_weights_actions_labels_or_trajectories_used"):
        raise RuntimeError("external material is forbidden in productive-bomb A/B")
    checkpoint = ROOT / protocol["checkpoint"]["path"]
    if sha256_file(checkpoint) != protocol["checkpoint"]["sha256"]:
        raise RuntimeError("productive-bomb A/B checkpoint drift")
    payload = _torch_load(checkpoint)
    identity = {"architecture": ARCHITECTURE, "stage": "task4", "arm": "control",
                "replica": "r2", "stage_rounds": 150}
    if any(payload.get(key) != value for key, value in identity.items()):
        raise RuntimeError("productive-bomb A/B checkpoint identity mismatch")
    assert_seeds_untouched(path, protocol)
    return protocol, hashlib.sha256(raw).hexdigest()


def opponent_env(case: dict) -> dict[str, str]:
    result = {"TASK4_RULE_SEED": str(case["rule_seed"])}
    if "coin_seed" in case:
        result["TASK3_OPPONENT_SEED"] = str(case["coin_seed"])
    if "random_seed" in case:
        result["SEEDED_RANDOM_AGENT_SEED"] = str(case["random_seed"])
    return result


def trace_path(protocol: dict, case_id: str, arm: str) -> Path:
    return ROOT / protocol["trace_directory"] / f"{case_id}-{arm}.json"


def validate_trace(protocol: dict, digest: str, case_id: str, arm: str, path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "kind": TRACE_KIND,
        "protocol_sha256": digest,
        "case": case_id,
        "arm": arm,
        "checkpoint_sha256": protocol["checkpoint"]["sha256"],
        "rounds": 25,
        "policy_updates": 0,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"invalid productive-bomb trace: {case_id}/{arm}")
    counts = payload["counts"]
    if arm == "control" and any(counts[key] for key in (
        "restriction_available_decisions", "fallback_decisions", "actions_changed",
    )):
        raise RuntimeError(f"control trace contains interventions: {case_id}")
    if counts["actions_changed"] > counts["restriction_available_decisions"]:
        raise RuntimeError(f"impossible intervention counts: {case_id}/{arm}")
    return payload


def run_one(path: Path, protocol: dict, digest: str, case_id: str, arm: str, case: dict) -> dict:
    output = ROOT / protocol["manifest_directory"] / f"{case_id}-{arm}.json"
    trace = trace_path(protocol, case_id, arm)
    stats = output.with_suffix(".stats.json")
    if output.exists():
        row = json.loads(output.read_text(encoding="utf-8"))
        if row.get("status") == "completed" and row.get("protocol_sha256") == digest:
            validate_trace(protocol, digest, case_id, arm, trace)
            return row
        raise RuntimeError(f"orphaned productive-bomb manifest: {relative(output)}")
    if stats.exists() or trace.exists():
        raise RuntimeError(f"orphaned productive-bomb artifact: {case_id}/{arm}")

    checkpoint = ROOT / protocol["checkpoint"]["path"]
    checkpoint_hash = protocol["checkpoint"]["sha256"]
    command = [
        sys.executable, "main.py", "play", "--agents", AGENT, *case["opponents"],
        "--train", "1", "--continue-without-training", "--no-gui", "--scenario", "classic",
        "--n-rounds", "25", "--seed", str(case["world_seed"]), "--save-stats", str(stats),
    ]
    overrides = {
        "CNN_PRODUCTIVE_BOMB_PROTOCOL": str(path),
        "CNN_PRODUCTIVE_BOMB_CASE": case_id,
        "CNN_PRODUCTIVE_BOMB_ARM": arm,
        "CNN_PRODUCTIVE_BOMB_AGENT_SEED": str(case["agent_seed"]),
        "CNN_PRODUCTIVE_BOMB_CHECKPOINT": str(checkpoint.resolve()),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        **opponent_env(case),
    }
    row = {
        "schema_version": 1, "kind": "model-a-cnn-n8-productive-bomb-ab-case",
        "status": "running", "started_at_utc": utc_now(), "protocol_sha256": digest,
        "case_id": case_id, "arm": arm, "case": case,
        "checkpoint": protocol["checkpoint"], "command": command,
        "environment_overrides": overrides, "trace": relative(trace), "raw_stats": relative(stats),
    }
    atomic_json(output, row)
    child = os.environ.copy()
    child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    row.update({"ended_at_utc": utc_now(), "exit_code": completed.returncode})
    try:
        if completed.returncode or not stats.is_file() or not trace.is_file():
            raise RuntimeError("productive-bomb run failed")
        if sha256_file(checkpoint) != checkpoint_hash:
            raise RuntimeError("productive-bomb A/B mutated checkpoint")
        trace_payload = validate_trace(protocol, digest, case_id, arm, trace)
        raw = json.loads(stats.read_text(encoding="utf-8"))["by_agent"][AGENT]
        row.update({"status": "completed", "target_metrics": _metrics(raw),
                    "trace_counts": trace_payload["counts"], "trace_sha256": sha256_file(trace)})
    except Exception as exc:
        row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    atomic_json(output, row)
    if row["status"] != "completed":
        raise RuntimeError(row["error"])
    return row


def sum_counts(rows: list[dict]) -> dict:
    keys = rows[0]["trace_counts"]
    return {key: sum(row["trace_counts"][key] for row in rows) for key in keys}


def execute(path: Path, protocol: dict, digest: str) -> dict:
    runs = {}
    for case_id, case in protocol["collection"]["cases"].items():
        runs[case_id] = {}
        for arm in case["arm_order"]:
            runs[case_id][arm] = run_one(path, protocol, digest, case_id, arm, case)

    pooled = {
        arm: combine_metrics([runs[case_id][arm]["target_metrics"] for case_id in runs])
        for arm in ("control", "treatment")
    }
    strata = {
        stratum: {
            arm: combine_metrics([
                runs[case_id][arm]["target_metrics"] for case_id, case in protocol["collection"]["cases"].items()
                if case["stratum"] == stratum
            ])
            for arm in ("control", "treatment")
        }
        for stratum in ("rule", "mixed")
    }
    fields = ("score_per_round", "coins_per_round", "kills_per_round", "suicides_per_round",
              "invalid_actions_per_round", "wait_fraction", "mean_decision_time_ms")
    delta = {key: pooled["treatment"][key] - pooled["control"][key] for key in fields}
    stratum_delta = {
        name: values["treatment"]["score_per_round"] - values["control"]["score_per_round"]
        for name, values in strata.items()
    }
    case_delta = {
        case_id: arms["treatment"]["target_metrics"]["score_per_round"]
        - arms["control"]["target_metrics"]["score_per_round"]
        for case_id, arms in runs.items()
    }
    thresholds = protocol["decision_rule"]
    checks = {
        "score_gain": delta["score_per_round"] >= thresholds["minimum_score_gain_per_round"],
        "case_score_wins": sum(value > 0 for value in case_delta.values()) >= thresholds["minimum_case_score_wins_out_of_4"],
        "rule_noninferiority": stratum_delta["rule"] >= thresholds["minimum_rule_score_delta_per_round"],
        "mixed_noninferiority": stratum_delta["mixed"] >= thresholds["minimum_mixed_score_delta_per_round"],
        "suicide_nonincrease": delta["suicides_per_round"] <= thresholds["maximum_suicide_increase_per_round"],
        "kill_preservation": delta["kills_per_round"] >= -thresholds["maximum_kill_loss_per_round"],
    }
    passed = all(checks.values())
    treatment_rows = [runs[case_id]["treatment"] for case_id in runs]
    report = {
        "schema_version": 1, "kind": "model-a-cnn-n8-productive-bomb-ab-report",
        "status": "completed", "completed_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"], "protocol_sha256": digest,
        "checkpoint": protocol["checkpoint"], "rounds_per_arm": 100,
        "total_environment_rounds": 200, "pooled": pooled, "by_stratum": strata,
        "treatment_minus_control": delta, "stratum_score_delta": stratum_delta,
        "case_score_delta": case_delta, "treatment_trace_counts": sum_counts(treatment_rows),
        "gate": {"thresholds": thresholds, "checks": checks, "passed": passed},
        "result": {
            "decision": thresholds["pass_decision"] if passed else thresholds["fail_decision"],
            "training_started": False, "checkpoint_modified": False,
            "checkpoint_copied": False, "candidate_deployed": False,
            "automatic_followup_started": False,
        },
        "awaiting_user_instruction": True,
    }
    output = ROOT / protocol["report_path"]
    if output.exists():
        raise RuntimeError("refusing to overwrite productive-bomb A/B report")
    atomic_json(output, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    protocol, digest = load_protocol(path)
    result = execute(path.resolve(), protocol, digest) if args.execute else {
        "mode": "dry-run", "protocol_id": protocol["protocol_id"],
        "protocol_sha256": digest, "checkpoint": protocol["checkpoint"],
        "arms": ["control", "treatment"], "rounds_per_arm": 100,
        "total_environment_rounds": 200, "policy_updates": 0,
        "external_material_used": False,
        "will_stop_after": "selection-free paired A/B report",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
