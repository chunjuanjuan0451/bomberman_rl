"""Run the s169 selection-free productive-BOMB final confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_cnn_n8.network import ARCHITECTURE  # noqa: E402
from tools.cnn_n8_productive_bomb_ab import run_one, sum_counts  # noqa: E402
from tools.cnn_n8_task1 import _torch_load, atomic_json, relative, sha256_file, utc_now  # noqa: E402
from tools.cnn_n8_task4 import combine_metrics  # noqa: E402
from tools.v4_task4_duel_training import seed_values  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-cnn-n8-productive-bomb-final-s169000.json"
KIND = "model-a-cnn-n8-productive-bomb-ab"


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
        raise RuntimeError(f"s169 seeds already used: {conflicts}")


def load_protocol(path: Path) -> tuple[dict, str]:
    path = path.resolve()
    raw = path.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("kind") != KIND or protocol.get("protocol_id") != "model-a-cnn-n8-productive-bomb-final-s169000":
        raise RuntimeError("wrong productive-bomb final protocol")
    if any(protocol.get(key) for key in (
        "training_allowed", "checkpoint_selection_allowed", "checkpoint_copy_allowed",
        "candidate_deployment_allowed", "automatic_followup",
    )):
        raise RuntimeError("productive-bomb final must remain read-only and terminal")
    collection = protocol["collection"]
    cases = collection["cases"]
    if (collection.get("rounds_per_arm_case") != 25
            or collection.get("rounds_per_arm") != 200
            or collection.get("total_environment_rounds") != 400
            or collection.get("policy_updates") != 0
            or len(cases) != 8
            or sum(case["stratum"] == "rule" for case in cases.values()) != 4
            or any(sorted(case["arm_order"]) != ["control", "treatment"] for case in cases.values())):
        raise RuntimeError("productive-bomb final budget or arm order drift")
    expected_gate = {
        "minimum_score_gain_per_round": 0.10,
        "minimum_winning_blocks_out_of_8": 5,
        "minimum_rule_score_delta_per_round": 0.0,
        "minimum_mixed_score_delta_per_round": 0.0,
        "maximum_suicide_increase_per_round": 0.0,
        "maximum_kill_loss_per_round": 0.02,
        "maximum_invalid_increase_per_round": 0.0,
        "maximum_mean_decision_time_ms": 500.0,
        "pass_decision": "productive_bomb_mask_confirmed_as_final_internal_candidate_stop",
        "fail_decision": "productive_bomb_mask_not_confirmed_retain_control_r2_round0150_stop",
    }
    if protocol.get("decision_rule") != expected_gate:
        raise RuntimeError("productive-bomb final gate drift")

    source_spec = protocol["selection_source"]
    source_path = ROOT / source_spec["path"]
    if sha256_file(source_path) != source_spec["sha256"]:
        raise RuntimeError("s168 source report drift")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (source.get("protocol_sha256") != source_spec["required_protocol_sha256"]
            or source.get("result", {}).get("decision") != source_spec["required_decision"]
            or not source.get("gate", {}).get("passed")):
        raise RuntimeError("s168 did not authorize the fixed candidate")
    provenance = protocol["provenance"]
    if (not provenance.get("candidate_fixed_after_s168")
            or provenance.get("mask_definition_changed_after_selection")
            or provenance.get("external_source_imported")
            or provenance.get("external_weights_actions_labels_or_trajectories_used")):
        raise RuntimeError("productive-bomb final provenance drift")

    checkpoint = ROOT / protocol["checkpoint"]["path"]
    if sha256_file(checkpoint) != protocol["checkpoint"]["sha256"]:
        raise RuntimeError("productive-bomb final checkpoint drift")
    payload = _torch_load(checkpoint)
    identity = {"architecture": ARCHITECTURE, "stage": "task4", "arm": "control",
                "replica": "r2", "stage_rounds": 150}
    if any(payload.get(key) != value for key, value in identity.items()):
        raise RuntimeError("productive-bomb final checkpoint identity mismatch")
    assert_seeds_untouched(path, protocol)
    return protocol, hashlib.sha256(raw).hexdigest()


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
    block_delta = {
        case_id: arms["treatment"]["target_metrics"]["score_per_round"]
        - arms["control"]["target_metrics"]["score_per_round"]
        for case_id, arms in runs.items()
    }
    thresholds = protocol["decision_rule"]
    checks = {
        "score_gain": delta["score_per_round"] >= thresholds["minimum_score_gain_per_round"],
        "block_support": sum(value > 0 for value in block_delta.values()) >= thresholds["minimum_winning_blocks_out_of_8"],
        "rule_noninferiority": stratum_delta["rule"] >= thresholds["minimum_rule_score_delta_per_round"],
        "mixed_noninferiority": stratum_delta["mixed"] >= thresholds["minimum_mixed_score_delta_per_round"],
        "suicide_nonincrease": delta["suicides_per_round"] <= thresholds["maximum_suicide_increase_per_round"],
        "kill_preservation": delta["kills_per_round"] >= -thresholds["maximum_kill_loss_per_round"],
        "invalid_nonincrease": delta["invalid_actions_per_round"] <= thresholds["maximum_invalid_increase_per_round"],
        "latency": pooled["treatment"]["mean_decision_time_ms"] <= thresholds["maximum_mean_decision_time_ms"],
    }
    passed = all(checks.values())
    treatment_rows = [runs[case_id]["treatment"] for case_id in runs]
    report = {
        "schema_version": 1, "kind": "model-a-cnn-n8-productive-bomb-final-report",
        "status": "completed", "completed_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"], "protocol_sha256": digest,
        "selection_source": protocol["selection_source"], "checkpoint": protocol["checkpoint"],
        "rounds_per_arm": 200, "total_environment_rounds": 400,
        "pooled": pooled, "by_stratum": strata,
        "treatment_minus_control": delta, "stratum_score_delta": stratum_delta,
        "block_score_delta": block_delta,
        "treatment_winning_blocks": sum(value > 0 for value in block_delta.values()),
        "treatment_trace_counts": sum_counts(treatment_rows),
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
        raise RuntimeError("refusing to overwrite productive-bomb final report")
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
        "selection_source": protocol["selection_source"],
        "arms": ["control", "treatment"], "rounds_per_arm": 200,
        "total_environment_rounds": 400, "policy_updates": 0,
        "external_material_used": False,
        "will_stop_after": "selection-free final confirmation report",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
