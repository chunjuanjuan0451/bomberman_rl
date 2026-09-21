"""Direct ten-seed match of the confirmed D4 candidate against three rules."""

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


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-cnn-n8-d4-rule-10seed-s172000.json"
PROTOCOL_ID = "model-a-cnn-n8-d4-rule-10seed-s172000"
AGENT = "model_a_cnn_n8_symmetry_ab"


def assert_seeds_untouched(path: Path, protocol: dict) -> None:
    registered = seed_values(protocol["collection"]["cases"])
    output_root = (ROOT / protocol["manifest_directory"]).resolve()
    report = (ROOT / protocol["report_path"]).resolve()
    conflicts = []
    for candidate in (ROOT / "experiments").rglob("*.json"):
        resolved = candidate.resolve()
        if (resolved in {path.resolve(), report}
                or output_root == resolved or output_root in resolved.parents):
            continue
        try:
            overlap = sorted(registered & seed_values(json.loads(candidate.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError):
            continue
        if overlap:
            conflicts.append({"path": relative(candidate), "seeds": overlap})
    if conflicts:
        raise RuntimeError(f"s172 seeds already used: {conflicts}")


def load_protocol(path: Path) -> tuple[dict, str]:
    path = path.resolve(); raw = path.read_bytes(); protocol = json.loads(raw)
    if protocol.get("kind") != "model-a-cnn-n8-d4-symmetry-ab" or protocol.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("wrong D4 versus rule protocol")
    if any(protocol.get(key) for key in (
        "training_allowed", "selection_allowed", "checkpoint_copy_allowed",
        "candidate_deployment_allowed", "automatic_followup",
    )):
        raise RuntimeError("D4 versus rule comparison must remain read-only")
    collection = protocol["collection"]; cases = collection["cases"]
    if (collection.get("scenario") != "classic"
            or collection.get("rounds_per_seed") != 50
            or collection.get("seed_count") != 10
            or collection.get("total_environment_rounds") != 500
            or collection.get("policy_updates") != 0
            or len(cases) != 10
            or any(case["roster"].count("target") != 1 for case in cases.values())
            or any(case["roster"][case["seat"]] != "target" for case in cases.values())
            or any(case["roster"].count("seeded_rule_based_agent") != 3 for case in cases.values())
            or sorted(case["seat"] for case in cases.values()) != [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]):
        raise RuntimeError("D4 versus rule collection drift")
    source = protocol["evidence_source"]; source_path = ROOT / source["path"]
    if sha256_file(source_path) != source["sha256"]:
        raise RuntimeError("D4 final evidence report drift")
    source_report = json.loads(source_path.read_text(encoding="utf-8"))
    if source_report.get("result", {}).get("decision") != source["required_decision"]:
        raise RuntimeError("D4 final evidence decision mismatch")
    checkpoint = ROOT / protocol["checkpoint"]["path"]
    if sha256_file(checkpoint) != protocol["checkpoint"]["sha256"]:
        raise RuntimeError("D4 checkpoint drift")
    payload = _torch_load(checkpoint)
    identity = {"architecture": ARCHITECTURE, "stage": "task4", "arm": "control",
                "replica": "r2", "stage_rounds": 150}
    if any(payload.get(key) != value for key, value in identity.items()):
        raise RuntimeError("D4 checkpoint identity mismatch")
    assert_seeds_untouched(path, protocol)
    return protocol, hashlib.sha256(raw).hexdigest()


def collect(path: Path, protocol: dict, digest: str, case_id: str, case: dict) -> dict:
    manifest = ROOT / protocol["manifest_directory"] / f"{case_id}.json"
    stats = manifest.with_suffix(".stats.json")
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if existing.get("status") == "completed" and existing.get("protocol_sha256") == digest:
            return existing
        raise RuntimeError(f"orphaned s172 manifest: {relative(manifest)}")
    if stats.exists():
        raise RuntimeError(f"orphaned s172 stats: {relative(stats)}")
    roster = [AGENT if name == "target" else name for name in case["roster"]]
    command = [
        sys.executable, "main.py", "play", "--agents", *roster,
        "--train", "0", "--no-gui", "--scenario", "classic",
        "--n-rounds", str(protocol["collection"]["rounds_per_seed"]),
        "--seed", str(case["world_seed"]), "--save-stats", str(stats),
    ]
    checkpoint = ROOT / protocol["checkpoint"]["path"]
    overrides = {
        "CNN_D4_PROTOCOL": str(path), "CNN_D4_CASE": case_id,
        "CNN_D4_ARM": "treatment", "CNN_D4_AGENT_SEED": str(case["agent_seed"]),
        "CNN_D4_CHECKPOINT": str(checkpoint.resolve()),
        "TASK4_RULE_SEED": str(case["rule_seed"]),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }
    row = {
        "schema_version": 1, "kind": "model-a-cnn-n8-d4-rule-10seed-case",
        "status": "running", "started_at_utc": utc_now(), "protocol_sha256": digest,
        "case_id": case_id, "case": case, "command": command,
        "environment_overrides": overrides, "raw_stats": relative(stats),
    }
    atomic_json(manifest, row)
    child = os.environ.copy(); child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    row.update({"ended_at_utc": utc_now(), "exit_code": completed.returncode})
    try:
        if completed.returncode or not stats.is_file():
            raise RuntimeError("s172 engine run failed")
        if sha256_file(checkpoint) != protocol["checkpoint"]["sha256"]:
            raise RuntimeError("s172 mutated checkpoint")
        raw = json.loads(stats.read_text(encoding="utf-8"))["by_agent"]
        metrics = {name: _metrics(values) for name, values in raw.items()}
        target = metrics[AGENT]
        rules = {name: values for name, values in metrics.items()
                 if name.startswith("seeded_rule_based_agent_")}
        if len(rules) != 3:
            raise RuntimeError("s172 expected three independent rule agents")
        rule_pooled = combine_metrics(list(rules.values()))
        scores = [target["score_per_round"], *(item["score_per_round"] for item in rules.values())]
        row.update({
            "status": "completed", "target_metrics": target,
            "rule_metrics": rules, "rule_pooled_metrics": rule_pooled,
            "target_minus_rule_mean_score_per_round": (
                target["score_per_round"] - rule_pooled["score_per_round"]
            ),
            "target_rank": 1 + sum(score > scores[0] for score in scores[1:]),
        })
    except Exception as exc:
        row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    atomic_json(manifest, row)
    if row["status"] != "completed":
        raise RuntimeError(row["error"])
    return row


def execute(path: Path, protocol: dict, digest: str) -> dict:
    runs = {case_id: collect(path, protocol, digest, case_id, case)
            for case_id, case in protocol["collection"]["cases"].items()}
    target = combine_metrics([row["target_metrics"] for row in runs.values()])
    rules = combine_metrics([
        metrics for row in runs.values() for metrics in row["rule_metrics"].values()
    ])
    by_seed = {
        case_id: {
            "seat": row["case"]["seat"], "world_seed": row["case"]["world_seed"],
            "target_score_per_round": row["target_metrics"]["score_per_round"],
            "rule_scores_per_round": {
                name: metrics["score_per_round"] for name, metrics in row["rule_metrics"].items()
            },
            "rule_mean_score_per_round": row["rule_pooled_metrics"]["score_per_round"],
            "target_minus_rule_mean_score_per_round": row["target_minus_rule_mean_score_per_round"],
            "target_rank": row["target_rank"],
        }
        for case_id, row in runs.items()
    }
    by_seat = {
        str(seat): combine_metrics([row["target_metrics"] for row in runs.values()
                                    if row["case"]["seat"] == seat])
        for seat in range(4)
    }
    report = {
        "schema_version": 1, "kind": "model-a-cnn-n8-d4-rule-10seed-report",
        "status": "completed", "completed_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"], "protocol_sha256": digest,
        "checkpoint": protocol["checkpoint"], "environment_rounds": 500,
        "target_agent_rounds": target["rounds"], "rule_agent_rounds": rules["rounds"],
        "d4_candidate": target, "rule_agents_pooled": rules,
        "d4_minus_rule_mean_score_per_round": target["score_per_round"] - rules["score_per_round"],
        "d4_seed_wins_over_rule_mean": sum(
            row["target_minus_rule_mean_score_per_round"] > 0 for row in runs.values()
        ),
        "d4_seed_top1_count": sum(row["target_rank"] == 1 for row in runs.values()),
        "d4_seed_top2_count": sum(row["target_rank"] <= 2 for row in runs.values()),
        "by_seed": by_seed, "d4_by_seat": by_seat,
        "result": {
            "diagnostic_only": True, "training_started": False,
            "checkpoint_modified": False, "candidate_deployed": False,
            "automatic_followup_started": False,
        },
        "awaiting_user_instruction": True,
    }
    output = ROOT / protocol["report_path"]
    if output.exists():
        raise RuntimeError("refusing to overwrite s172 report")
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
        "protocol_sha256": digest, "seed_count": 10,
        "rounds_per_seed": 50, "environment_rounds": 500,
        "target": "confirmed D4 candidate", "opponents": "three seeded rule-based agents",
        "policy_updates": 0, "will_stop_after": "diagnostic comparison report",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
