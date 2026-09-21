"""Run preregistered frozen control-r2 single-view versus D4 inference A/B."""

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
from tools.experiment_utils import seed_values  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-cnn-n8-d4-symmetry-ab-s170000.json"
KIND = "model-a-cnn-n8-d4-symmetry-ab"
AGENT = "model_a_cnn_n8_symmetry_ab"
PROTOCOL_SPECS = {
    "model-a-cnn-n8-d4-symmetry-ab-s170000": {
        "rounds_per_arm_case": 20,
        "rounds_per_arm": 160,
        "total_environment_rounds": 320,
        "pass_decision": "d4_symmetry_supported_for_selection_free_confirmation_stop",
        "fail_decision": "d4_symmetry_not_supported_retain_control_r2_round0150_stop",
    },
    "model-a-cnn-n8-d4-symmetry-final-s171000": {
        "rounds_per_arm_case": 50,
        "rounds_per_arm": 400,
        "total_environment_rounds": 800,
        "pass_decision": "d4_symmetry_confirmed_as_final_internal_candidate_stop",
        "fail_decision": "d4_symmetry_not_confirmed_retain_single_view_control_r2_stop",
    },
}


def assert_seeds_untouched(path: Path, protocol: dict) -> None:
    registered = seed_values(protocol["collection"]["cases"])
    output_root = (ROOT / protocol["manifest_directory"]).resolve()
    report = (ROOT / protocol["report_path"]).resolve()
    conflicts = []
    for candidate in (ROOT / "experiments").rglob("*.json"):
        resolved = candidate.resolve()
        if (resolved == path.resolve() or resolved == report
                or output_root == resolved or output_root in resolved.parents):
            continue
        try:
            overlap = sorted(registered & seed_values(json.loads(candidate.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError):
            continue
        if overlap:
            conflicts.append({"path": relative(candidate), "seeds": overlap})
    if conflicts:
        raise RuntimeError(f"D4 symmetry protocol seeds already used: {conflicts}")


def load_protocol(path: Path) -> tuple[dict, str]:
    path = path.resolve()
    raw = path.read_bytes()
    protocol = json.loads(raw)
    protocol_id = protocol.get("protocol_id")
    if protocol.get("kind") != KIND or protocol_id not in PROTOCOL_SPECS:
        raise RuntimeError("wrong D4 symmetry A/B protocol")
    spec = PROTOCOL_SPECS[protocol_id]
    if any(protocol.get(key) for key in (
        "training_allowed", "checkpoint_selection_allowed", "checkpoint_copy_allowed",
        "candidate_deployment_allowed", "automatic_followup",
    )):
        raise RuntimeError("D4 symmetry A/B must remain read-only and terminal")
    collection = protocol["collection"]
    cases = collection["cases"]
    if (collection.get("rounds_per_arm_case") != spec["rounds_per_arm_case"]
            or collection.get("rounds_per_arm") != spec["rounds_per_arm"]
            or collection.get("total_environment_rounds") != spec["total_environment_rounds"]
            or collection.get("policy_updates") != 0
            or len(cases) != 8
            or {case["stratum"] for case in cases.values()} != {"rule", "mixed"}
            or any(sorted(case["arm_order"]) != ["control", "treatment"] for case in cases.values())
            or any(case["roster"].count("target") != 1 for case in cases.values())
            or {case["seat"] for case in cases.values() if case["stratum"] == "rule"} != set(range(4))
            or {case["seat"] for case in cases.values() if case["stratum"] == "mixed"} != set(range(4))
            or any(case["roster"][case["seat"]] != "target" for case in cases.values())):
        raise RuntimeError("D4 symmetry A/B budget, strata, or seat rotation drift")
    expected_gate = {
        "minimum_score_gain_per_round": 0.10,
        "minimum_winning_blocks_out_of_8": 5,
        "minimum_rule_score_delta_per_round": -0.05,
        "minimum_mixed_score_delta_per_round": -0.05,
        "maximum_suicide_increase_per_round": 0.0,
        "maximum_kill_loss_per_round": 0.02,
        "maximum_invalid_increase_per_round": 0.0,
        "maximum_treatment_mean_decision_time_ms": 50.0,
        "pass_decision": spec["pass_decision"],
        "fail_decision": spec["fail_decision"],
    }
    if protocol.get("decision_rule") != expected_gate:
        raise RuntimeError("D4 symmetry A/B decision rule drift")
    provenance = protocol["provenance"]
    if (provenance.get("external_source_imported")
            or provenance.get("external_weights_actions_labels_or_trajectories_used")
            or provenance.get("training_data_added")
            or provenance.get("checkpoint_or_mask_modified")):
        raise RuntimeError("D4 symmetry A/B provenance drift")
    source = protocol["evidence_source"]
    source_path = ROOT / source["path"]
    if sha256_file(source_path) != source["sha256"]:
        raise RuntimeError("D4 symmetry evidence report drift")
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    if source_payload.get("result", {}).get("decision") != source["required_decision"]:
        raise RuntimeError("D4 symmetry evidence decision mismatch")
    checkpoint = ROOT / protocol["checkpoint"]["path"]
    if sha256_file(checkpoint) != protocol["checkpoint"]["sha256"]:
        raise RuntimeError("D4 symmetry A/B checkpoint drift")
    payload = _torch_load(checkpoint)
    identity = {"architecture": ARCHITECTURE, "stage": "task4", "arm": "control",
                "replica": "r2", "stage_rounds": 150}
    if any(payload.get(key) != value for key, value in identity.items()):
        raise RuntimeError("D4 symmetry A/B checkpoint identity mismatch")
    assert_seeds_untouched(path, protocol)
    return protocol, hashlib.sha256(raw).hexdigest()


def opponent_env(case: dict) -> dict[str, str]:
    result = {"TASK4_RULE_SEED": str(case["rule_seed"])}
    if "coin_seed" in case:
        result["TASK3_OPPONENT_SEED"] = str(case["coin_seed"])
    if "random_seed" in case:
        result["SEEDED_RANDOM_AGENT_SEED"] = str(case["random_seed"])
    return result


def run_one(path: Path, protocol: dict, digest: str, case_id: str, arm: str, case: dict) -> dict:
    output = ROOT / protocol["manifest_directory"] / f"{case_id}-{arm}.json"
    stats = output.with_suffix(".stats.json")
    if output.exists():
        row = json.loads(output.read_text(encoding="utf-8"))
        if row.get("status") == "completed" and row.get("protocol_sha256") == digest:
            return row
        raise RuntimeError(f"orphaned D4 symmetry manifest: {relative(output)}")
    if stats.exists():
        raise RuntimeError(f"orphaned D4 symmetry artifact: {case_id}/{arm}")
    roster = [AGENT if item == "target" else item for item in case["roster"]]
    command = [
        sys.executable, "main.py", "play", "--agents", *roster,
        "--train", "0", "--no-gui", "--scenario", "classic",
        "--n-rounds", str(protocol["collection"]["rounds_per_arm_case"]),
        "--seed", str(case["world_seed"]), "--save-stats", str(stats),
    ]
    checkpoint = ROOT / protocol["checkpoint"]["path"]
    overrides = {
        "CNN_D4_PROTOCOL": str(path), "CNN_D4_CASE": case_id, "CNN_D4_ARM": arm,
        "CNN_D4_AGENT_SEED": str(case["agent_seed"]),
        "CNN_D4_CHECKPOINT": str(checkpoint.resolve()),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", **opponent_env(case),
    }
    row = {
        "schema_version": 1, "kind": "model-a-cnn-n8-d4-symmetry-ab-case",
        "status": "running", "started_at_utc": utc_now(), "protocol_sha256": digest,
        "case_id": case_id, "arm": arm, "case": case, "checkpoint": protocol["checkpoint"],
        "command": command, "environment_overrides": overrides,
        "raw_stats": relative(stats),
    }
    atomic_json(output, row)
    child = os.environ.copy()
    child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    row.update({"ended_at_utc": utc_now(), "exit_code": completed.returncode})
    try:
        if completed.returncode or not stats.is_file():
            raise RuntimeError("D4 symmetry run failed")
        if sha256_file(checkpoint) != protocol["checkpoint"]["sha256"]:
            raise RuntimeError("D4 symmetry A/B mutated checkpoint")
        raw = json.loads(stats.read_text(encoding="utf-8"))["by_agent"][AGENT]
        row.update({"status": "completed", "target_metrics": _metrics(raw)})
    except Exception as exc:
        row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    atomic_json(output, row)
    if row["status"] != "completed":
        raise RuntimeError(row["error"])
    return row


def execute(path: Path, protocol: dict, digest: str) -> dict:
    runs = {}
    for case_id, case in protocol["collection"]["cases"].items():
        runs[case_id] = {}
        for arm in case["arm_order"]:
            runs[case_id][arm] = run_one(path, protocol, digest, case_id, arm, case)
    pooled = {arm: combine_metrics([runs[c][arm]["target_metrics"] for c in runs])
              for arm in ("control", "treatment")}
    strata = {
        stratum: {arm: combine_metrics([
            runs[case_id][arm]["target_metrics"]
            for case_id, case in protocol["collection"]["cases"].items()
            if case["stratum"] == stratum
        ]) for arm in ("control", "treatment")}
        for stratum in ("rule", "mixed")
    }
    seats = {
        str(seat): {arm: combine_metrics([
            runs[case_id][arm]["target_metrics"]
            for case_id, case in protocol["collection"]["cases"].items()
            if case["seat"] == seat
        ]) for arm in ("control", "treatment")}
        for seat in range(4)
    }
    fields = ("score_per_round", "coins_per_round", "kills_per_round", "suicides_per_round",
              "invalid_actions_per_round", "wait_fraction", "mean_decision_time_ms")
    delta = {key: pooled["treatment"][key] - pooled["control"][key] for key in fields}
    stratum_delta = {name: values["treatment"]["score_per_round"] - values["control"]["score_per_round"]
                     for name, values in strata.items()}
    block_delta = {case_id: arms["treatment"]["target_metrics"]["score_per_round"]
                   - arms["control"]["target_metrics"]["score_per_round"]
                   for case_id, arms in runs.items()}
    seat_delta = {seat: values["treatment"]["score_per_round"] - values["control"]["score_per_round"]
                  for seat, values in seats.items()}
    thresholds = protocol["decision_rule"]
    checks = {
        "score_gain": delta["score_per_round"] >= thresholds["minimum_score_gain_per_round"],
        "block_support": sum(value > 0 for value in block_delta.values()) >= thresholds["minimum_winning_blocks_out_of_8"],
        "rule_noninferiority": stratum_delta["rule"] >= thresholds["minimum_rule_score_delta_per_round"],
        "mixed_noninferiority": stratum_delta["mixed"] >= thresholds["minimum_mixed_score_delta_per_round"],
        "suicide_nonincrease": delta["suicides_per_round"] <= thresholds["maximum_suicide_increase_per_round"],
        "kill_preservation": delta["kills_per_round"] >= -thresholds["maximum_kill_loss_per_round"],
        "invalid_nonincrease": delta["invalid_actions_per_round"] <= thresholds["maximum_invalid_increase_per_round"],
        "latency": pooled["treatment"]["mean_decision_time_ms"] <= thresholds["maximum_treatment_mean_decision_time_ms"],
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1, "kind": "model-a-cnn-n8-d4-symmetry-ab-report",
        "status": "completed", "completed_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"], "protocol_sha256": digest,
        "checkpoint": protocol["checkpoint"],
        "rounds_per_arm": protocol["collection"]["rounds_per_arm"],
        "total_environment_rounds": protocol["collection"]["total_environment_rounds"],
        "pooled": pooled, "by_stratum": strata,
        "by_seat": seats, "treatment_minus_control": delta,
        "stratum_score_delta": stratum_delta, "seat_score_delta": seat_delta,
        "block_score_delta": block_delta,
        "treatment_winning_blocks": sum(value > 0 for value in block_delta.values()),
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
        raise RuntimeError("refusing to overwrite D4 symmetry A/B report")
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
        "arms": ["control", "treatment"], "strata": ["rule", "mixed"],
        "seats": [0, 1, 2, 3],
        "rounds_per_arm": protocol["collection"]["rounds_per_arm"],
        "total_environment_rounds": protocol["collection"]["total_environment_rounds"],
        "policy_updates": 0,
        "external_material_used": False,
        "will_stop_after": "paired D4 symmetry A/B report",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
