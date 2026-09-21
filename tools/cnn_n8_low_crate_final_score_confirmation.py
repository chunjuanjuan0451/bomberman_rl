"""Run the s167 selection-free low-crate final-score confirmation."""

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


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-cnn-n8-low-crate-final-score-confirmation-s167000.json"
KIND = "model-a-cnn-n8-low-crate-final-score-confirmation"
AGENT = "model_a_cnn_n8_crate_confirm_eval"


def assert_seeds_untouched(path: Path, protocol: dict) -> None:
    registered = seed_values(protocol["confirmation"])
    output_root = (ROOT / protocol["manifest_directory"]).resolve()
    report = (ROOT / protocol["report_path"]).resolve()
    conflicts = []
    for candidate in (ROOT / "experiments").rglob("*.json"):
        resolved = candidate.resolve()
        if resolved == path.resolve() or resolved == report or output_root == resolved or output_root in resolved.parents:
            continue
        try:
            overlap = sorted(registered & seed_values(json.loads(candidate.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError):
            continue
        if overlap:
            conflicts.append({"path": relative(candidate), "seeds": overlap})
    if conflicts:
        raise RuntimeError(f"s167 seeds already used: {conflicts}")


def load_protocol(path: Path) -> tuple[dict, str]:
    path = path.resolve()
    raw = path.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("kind") != KIND or not protocol.get("post_hoc_origin_disclosed"):
        raise RuntimeError("wrong or undisclosed confirmation protocol")
    if any(protocol.get(key) for key in (
        "training_allowed", "checkpoint_selection_allowed", "checkpoint_copy_allowed",
        "candidate_deployment_allowed", "automatic_followup",
    )):
        raise RuntimeError("confirmation must remain selection-free, read-only, and terminal")
    confirmation = protocol["confirmation"]
    if (confirmation.get("rounds_per_identity_case") != 25
            or confirmation.get("rounds_per_identity") != 100
            or confirmation.get("total_rounds") != 200
            or len(confirmation.get("cases", [])) != 4):
        raise RuntimeError("confirmation budget drift")
    if set(protocol["identities"]) != {"candidate", "control"}:
        raise RuntimeError("confirmation identity drift")
    source_path = ROOT / protocol["selection_source"]["path"]
    if sha256_file(source_path) != protocol["selection_source"]["sha256"]:
        raise RuntimeError("s166 selection report drift")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    selected = source["selection"][protocol["selection_source"]["required_arm"]]
    if selected["selected_label"] != protocol["selection_source"]["required_selected_label"]:
        raise RuntimeError("candidate is not the disclosed s166 selection")
    if selected["checkpoint"] != protocol["identities"]["candidate"]:
        raise RuntimeError("candidate checkpoint differs from s166 selection")
    for label, item in protocol["identities"].items():
        checkpoint = ROOT / item["path"]
        if sha256_file(checkpoint) != item["sha256"]:
            raise RuntimeError(f"checkpoint drift: {label}")
        payload = _torch_load(checkpoint)
        if payload.get("architecture") != ARCHITECTURE or payload.get("stage") != "task4":
            raise RuntimeError(f"checkpoint identity mismatch: {label}")
        if label == "candidate":
            expected = {"arm": "low_crate", "replica": "r1", "stage_rounds": 25,
                        "crate_destroyed_reward": 0.05}
        else:
            expected = {"arm": "control", "replica": "r2", "stage_rounds": 150}
        if any(payload.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"checkpoint lineage mismatch: {label}")
    assert_seeds_untouched(path, protocol)
    return protocol, hashlib.sha256(raw).hexdigest()


def opponent_env(case: dict) -> dict[str, str]:
    result = {"TASK4_RULE_SEED": str(case["rule_seed"])}
    if "coin_seed" in case:
        result["TASK3_OPPONENT_SEED"] = str(case["coin_seed"])
    if "random_seed" in case:
        result["SEEDED_RANDOM_AGENT_SEED"] = str(case["random_seed"])
    return result


def run_one(path: Path, protocol: dict, digest: str, label: str, case: dict) -> dict:
    output = ROOT / protocol["manifest_directory"] / label / f"{case['case_id']}.json"
    checkpoint = ROOT / protocol["identities"][label]["path"]
    checkpoint_hash = protocol["identities"][label]["sha256"]
    if output.exists():
        row = json.loads(output.read_text(encoding="utf-8"))
        if (row.get("status") == "completed" and row.get("protocol_sha256") == digest
                and row.get("checkpoint", {}).get("sha256") == checkpoint_hash):
            return row
        raise RuntimeError(f"orphaned confirmation manifest: {relative(output)}")
    stats = output.with_suffix(".stats.json")
    command = [
        sys.executable, "main.py", "play", "--agents", AGENT, *case["opponents"],
        "--train", "0", "--no-gui", "--scenario", "classic",
        "--n-rounds", "25", "--seed", str(case["world_seed"]), "--save-stats", str(stats),
    ]
    overrides = {
        "CNN_CRATE_CONFIRM_PROTOCOL_PATH": str(path),
        "CNN_CRATE_CONFIRM_CHECKPOINT": str(checkpoint.resolve()),
        "CNN_CRATE_CONFIRM_CHECKPOINT_SHA256": checkpoint_hash,
        "CNN_CRATE_CONFIRM_AGENT_SEED": str(case["agent_seed"]),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        **opponent_env(case),
    }
    row = {
        "schema_version": 1, "kind": "model-a-cnn-n8-low-crate-final-score-confirmation-case",
        "status": "running", "started_at_utc": utc_now(), "protocol_sha256": digest,
        "identity": label, "case": case,
        "checkpoint": {"path": relative(checkpoint), "sha256": checkpoint_hash},
        "command": command, "environment_overrides": overrides, "raw_stats": relative(stats),
    }
    atomic_json(output, row)
    child = os.environ.copy()
    child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    row.update({"ended_at_utc": utc_now(), "exit_code": completed.returncode})
    try:
        if completed.returncode or not stats.is_file() or sha256_file(checkpoint) != checkpoint_hash:
            raise RuntimeError("confirmation run failed or mutated checkpoint")
        raw_stats = json.loads(stats.read_text(encoding="utf-8"))["by_agent"][AGENT]
        row.update({"status": "completed", "target_metrics": _metrics(raw_stats)})
    except Exception as exc:
        row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    atomic_json(output, row)
    if row["status"] != "completed":
        raise RuntimeError(row["error"])
    return row


def execute(path: Path, protocol: dict, digest: str) -> dict:
    runs = {
        label: {case["case_id"]: run_one(path, protocol, digest, label, case)
                for case in protocol["confirmation"]["cases"]}
        for label in ("candidate", "control")
    }
    pooled = {
        label: combine_metrics([row["target_metrics"] for row in cases.values()])
        for label, cases in runs.items()
    }
    strata = {
        label: {
            stratum: combine_metrics([
                cases[case["case_id"]]["target_metrics"]
                for case in protocol["confirmation"]["cases"] if case["stratum"] == stratum
            ])
            for stratum in ("rule", "mixed")
        }
        for label, cases in runs.items()
    }
    fields = ("score_per_round", "coins_per_round", "kills_per_round", "suicides_per_round",
              "invalid_actions_per_round", "wait_fraction")
    delta = {key: pooled["candidate"][key] - pooled["control"][key] for key in fields}
    stratum_score_delta = {
        stratum: strata["candidate"][stratum]["score_per_round"] - strata["control"][stratum]["score_per_round"]
        for stratum in ("rule", "mixed")
    }
    case_score_delta = {
        case["case_id"]: runs["candidate"][case["case_id"]]["target_metrics"]["score_per_round"]
        - runs["control"][case["case_id"]]["target_metrics"]["score_per_round"]
        for case in protocol["confirmation"]["cases"]
    }
    thresholds = protocol["decision_rule"]
    checks = {
        "score_gain": delta["score_per_round"] >= thresholds["minimum_score_gain_per_round"],
        "case_score_wins": sum(value > 0 for value in case_score_delta.values()) >= thresholds["minimum_case_score_wins_out_of_4"],
        "rule_noninferiority": stratum_score_delta["rule"] >= thresholds["minimum_rule_score_delta_per_round"],
        "mixed_noninferiority": stratum_score_delta["mixed"] >= thresholds["minimum_mixed_score_delta_per_round"],
        "suicide_guard": delta["suicides_per_round"] <= thresholds["maximum_suicide_increase_per_round"],
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1, "kind": "model-a-cnn-n8-low-crate-final-score-confirmation-report",
        "status": "completed", "completed_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"], "protocol_sha256": digest,
        "post_hoc_origin_disclosed": True, "training_started": False,
        "rounds_per_identity": 100, "total_rounds": 200,
        "pooled": pooled, "by_stratum": strata,
        "candidate_minus_control": delta, "stratum_score_delta": stratum_score_delta,
        "case_score_delta": case_score_delta,
        "gate": {"thresholds": thresholds, "checks": checks, "passed": passed},
        "result": {
            "decision": thresholds["pass_decision"] if passed else thresholds["fail_decision"],
            "candidate_deployed": False, "checkpoint_copied": False,
            "checkpoint_modified": False, "automatic_followup_started": False,
        },
        "awaiting_user_instruction": True,
    }
    output = ROOT / protocol["report_path"]
    if output.exists():
        raise RuntimeError("refusing to overwrite confirmation report")
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
        "protocol_sha256": digest, "identities": ["candidate", "control"],
        "rounds_per_identity": 100, "total_rounds": 200,
        "training_started": False, "selection_allowed": False,
        "will_stop_after": "selection-free confirmation report",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
