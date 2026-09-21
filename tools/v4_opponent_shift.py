"""Evaluate frozen v4 under a pinned non-rule opponent distribution.

The four fixed identities share every game.  No policy is trained or copied;
the runner writes one terminal diagnostic report and stops.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_dqn.callbacks import MODEL_ARCHITECTURE  # noqa: E402
from agent_code.model_a_dqn.network import torch  # noqa: E402
from tools.v4_task4_duel_training import (  # noqa: E402
    atomic_json, load_completed, metrics_by_agent, relative, seed_values,
)


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-opponent-shift-s139000.json"
LABELS = ("v4", "source-r2", "candidate-r3", "external-strong")
INTERNAL_LABELS = LABELS[:3]
AGENT_CODES = {
    "v4": "model_a_oppshift_v4",
    "source-r2": "model_a_oppshift_source",
    "candidate-r3": "model_a_oppshift_r3",
    "external-strong": "feature_is_everything_eval",
}
EXPECTED_CASES = tuple(
    {
        "case_id": f"c{index + 1}",
        "world_seed": 139001 + index,
        "agent_seeds": {
            "v4": 239001 + index,
            "source-r2": 339001 + index,
            "candidate-r3": 439001 + index,
            "external-strong": 539001 + index,
        },
        "roster": roster,
    }
    for index, roster in enumerate((
        ["v4", "source-r2", "candidate-r3", "external-strong"],
        ["source-r2", "candidate-r3", "external-strong", "v4"],
        ["candidate-r3", "external-strong", "v4", "source-r2"],
        ["external-strong", "v4", "source-r2", "candidate-r3"],
        ["v4", "source-r2", "candidate-r3", "external-strong"],
        ["source-r2", "candidate-r3", "external-strong", "v4"],
        ["candidate-r3", "external-strong", "v4", "source-r2"],
        ["external-strong", "v4", "source-r2", "candidate-r3"],
    ))
)
SOURCE_PATHS = {
    "agent_code/model_a_dqn/callbacks.py",
    "agent_code/model_a_dqn/features.py",
    "agent_code/model_a_dqn/network.py",
    "agent_code/model_a_opponent_shift/common.py",
    "agent_code/model_a_oppshift_v4/callbacks.py",
    "agent_code/model_a_oppshift_source/callbacks.py",
    "agent_code/model_a_oppshift_r3/callbacks.py",
    "agent_code/feature_is_everything_eval/callbacks.py",
    "tools/v4_task4_duel_training.py",
    "tools/v4_opponent_shift.py",
}
EVALUATION_KIND = "model-a-v4-opponent-shift-evaluation"
REPORT_KIND = "model-a-v4-opponent-shift-report"


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
        raise ValueError(f"invalid opponent-shift checkpoint: {path}")
    return payload


def registered_seeds(protocol: dict) -> set[int]:
    result = set()
    for case in protocol["evaluation"]["cases"]:
        result.add(int(case["world_seed"]))
        result.update(int(value) for value in case["agent_seeds"].values())
    return result


def assert_registered_seeds_untouched(protocol_path: Path, protocol: dict) -> None:
    registered = registered_seeds(protocol)
    output_root = (ROOT / protocol["evaluation_manifest_directory"]).resolve()
    excluded = {protocol_path.resolve(), (ROOT / protocol["report_path"]).resolve()}
    conflicts = []
    for path in (ROOT / "experiments").rglob("*.json"):
        resolved = path.resolve()
        if resolved in excluded or output_root == resolved or output_root in resolved.parents:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        overlap = sorted(registered & seed_values(payload))
        if overlap:
            conflicts.append({"path": relative(path), "seeds": overlap})
    if conflicts:
        raise ValueError(f"registered s139000 seeds were already used: {conflicts}")


def _validate_checkpoint(protocol: dict, label: str) -> None:
    item = protocol["checkpoint_inventory"][label]
    path = ROOT / item["path"]
    if not path.is_file() or sha256_file(path) != item["sha256"]:
        raise ValueError(f"opponent-shift checkpoint mismatch: {label}")
    payload = _torch_load(path)
    if payload.get("architecture") not in (None, MODEL_ARCHITECTURE):
        raise ValueError(f"opponent-shift architecture mismatch: {label}")
    for key, value in item.get("required_metadata", {}).items():
        if payload.get(key) != value:
            raise ValueError(f"opponent-shift lineage mismatch: {label}/{key}")


def load_protocol(path: Path) -> tuple[dict, str]:
    path = path.resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or protocol.get("kind") != "model-a-v4-opponent-shift":
        raise ValueError("wrong opponent-shift protocol")
    if tuple(protocol.get("labels", ())) != LABELS or protocol.get("training_allowed") is not False:
        raise ValueError("opponent-shift identities or training boundary changed")
    if protocol.get("selection_allowed") is not False or protocol.get("automatic_followup") is not False:
        raise ValueError("opponent-shift must remain diagnostic and terminal")
    evaluation = protocol.get("evaluation", {})
    if evaluation.get("scenario") != "classic" or int(evaluation.get("rounds_per_case", 0)) != 50:
        raise ValueError("opponent-shift scenario or case budget changed")
    if tuple(evaluation.get("cases", ())) != EXPECTED_CASES:
        raise ValueError("opponent-shift cases changed")
    if len(registered_seeds(protocol)) != 40:
        raise ValueError("s139000 requires forty globally unique seed values")
    if set(protocol.get("checkpoint_inventory", {})) != set(INTERNAL_LABELS):
        raise ValueError("opponent-shift checkpoint inventory changed")
    for label in INTERNAL_LABELS:
        _validate_checkpoint(protocol, label)
    external = protocol.get("external_stress_opponent", {})
    expected_external = {
        "repository": "https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman.git",
        "commit": "a7fe5041b02548ce4438e502ca3adb11576bae75",
        "agent_subdirectory": "agent_code/feature_is_everything",
        "known_previous_opponent": True,
        "training_or_teacher_use_allowed": False,
        "license_file_present": False,
        "file_sha256": {
            "callbacks.py": "cadc00f69e4b0e04a1063e6760deea1ed48e23855992201c3edcebe6284071fe",
            "features.py": "8baf96ae76f685a5dbfb74bae644e2cbb816952103859081801888ebe6b9f620",
            "model.py": "27736c5afa4ccf50582141b6aea57740699365e35879ad6d5cb4842b3fc7829c",
            "my-saved-model.pt": "0fb1df0b3c3f255db121e8ba7c6061d1293f55d3e7ae12772259643ee2060577",
            "symmetry.py": "cac8ad3d8b197d1cca6fa67ca35d3db00d45c4c81afa549d797bb8b45657f82e",
        },
    }
    if external != expected_external:
        raise ValueError("external stress opponent binding changed")
    reference = protocol["s138_reference"]
    reference_path = ROOT / reference["path"]
    if not reference_path.is_file() or sha256_file(reference_path) != reference["sha256"]:
        raise ValueError("s138000 reference drift")
    reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
    if (
        reference_payload.get("status") != "completed"
        or reference_payload.get("result", {}).get("decision") != "candidate_not_confirmed_retain_incumbent_stop"
    ):
        raise ValueError("s138000 terminal decision changed")
    if set(protocol.get("source_bindings", {})) != SOURCE_PATHS:
        raise ValueError("opponent-shift source binding set changed")
    for source, expected in protocol["source_bindings"].items():
        source_path = ROOT / source
        if not source_path.is_file() or sha256_file(source_path) != expected:
            raise ValueError(f"opponent-shift source binding mismatch: {source}")
    assert_registered_seeds_untouched(path, protocol)
    return protocol, sha256_file(path)


@contextmanager
def pinned_external_checkout(protocol: dict):
    external = protocol["external_stress_opponent"]
    with tempfile.TemporaryDirectory(prefix="bomberman-s139000-") as temporary:
        checkout = Path(temporary) / "repository"
        commands = (
            ["git", "init", "-q", str(checkout)],
            ["git", "-C", str(checkout), "remote", "add", "origin", external["repository"]],
            ["git", "-C", str(checkout), "fetch", "-q", "--depth", "1", "origin", external["commit"]],
            ["git", "-C", str(checkout), "checkout", "-q", "--detach", "FETCH_HEAD"],
        )
        for command in commands:
            completed = subprocess.run(command, text=True, capture_output=True, check=False)
            if completed.returncode:
                raise RuntimeError(f"external checkout failed: {' '.join(command)}\n{completed.stderr}")
        head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        if head != external["commit"]:
            raise RuntimeError("external stress opponent commit mismatch")
        agent_dir = checkout / external["agent_subdirectory"]
        for name, expected in external["file_sha256"].items():
            if sha256_file(agent_dir / name) != expected:
                raise RuntimeError(f"external stress opponent hash mismatch: {name}")
        yield agent_dir


def manifest_path(protocol: dict, case_id: str) -> Path:
    return ROOT / protocol["evaluation_manifest_directory"] / f"{case_id}.json"


def report_path(protocol: dict) -> Path:
    return ROOT / protocol["report_path"]


def evaluate_one(
    protocol_path: Path, protocol: dict, protocol_hash: str, case: dict, external_agent_dir: Path,
) -> dict:
    output = manifest_path(protocol, case["case_id"])
    expected = {
        "case_id": case["case_id"], "world_seed": case["world_seed"],
        "agent_seeds": case["agent_seeds"], "roster": case["roster"],
        "scenario": "classic", "rounds": 50,
    }
    existing = load_completed(output, EVALUATION_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash or existing.get("evaluation") != expected:
            raise RuntimeError(f"completed opponent-shift evaluation drift: {output}")
        return existing
    stats = output.with_suffix(".stats.json")
    if output.exists() or stats.exists():
        raise RuntimeError(f"refusing orphaned opponent-shift output: {output}")
    codes = [AGENT_CODES[label] for label in case["roster"]]
    command = [
        sys.executable, "main.py", "play", "--agents", *codes,
        "--train", "0", "--continue-without-training", "--no-gui",
        "--scenario", "classic", "--n-rounds", "50", "--seed", str(case["world_seed"]),
        "--save-stats", str(stats),
    ]
    overrides = {
        "MODEL_A_OPPONENT_SHIFT_PROTOCOL_PATH": str(protocol_path),
        "MODEL_A_OPPONENT_SHIFT_CASE_ID": case["case_id"],
        "OPPONENT_SHIFT_EXTERNAL_AGENT_DIR": str(external_agent_dir),
    }
    checkpoint_hashes_before = {
        label: sha256_file(ROOT / protocol["checkpoint_inventory"][label]["path"])
        for label in INTERNAL_LABELS
    }
    record = {
        "schema_version": 1, "kind": EVALUATION_KIND, "status": "running",
        "started_at_utc": utc_now(), "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path), "protocol_sha256": protocol_hash,
        "evaluation": expected, "command": command,
        "environment_overrides": {**overrides, "OPPONENT_SHIFT_EXTERNAL_AGENT_DIR": "<temporary-pinned-checkout>"},
        "checkpoint_sha256_before": checkpoint_hashes_before,
        "external_commit": protocol["external_stress_opponent"]["commit"],
        "raw_stats": relative(stats),
    }
    atomic_json(output, record)
    child = os.environ.copy()
    child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = utc_now()
    record["exit_code"] = completed.returncode
    try:
        hashes_after = {
            label: sha256_file(ROOT / protocol["checkpoint_inventory"][label]["path"])
            for label in INTERNAL_LABELS
        }
        if completed.returncode != 0 or not stats.is_file() or hashes_after != checkpoint_hashes_before:
            raise RuntimeError("evaluation process, stats, or checkpoint immutability failed")
        raw = metrics_by_agent(json.loads(stats.read_text(encoding="utf-8")))
        reverse = {code: label for label, code in AGENT_CODES.items()}
        if set(raw) != set(reverse):
            raise RuntimeError(f"unexpected agents in opponent-shift stats: {sorted(raw)}")
        record["metrics_by_label"] = {reverse[code]: metrics for code, metrics in raw.items()}
        record["checkpoint_sha256_after"] = hashes_after
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(output, record)
    if record["status"] != "completed":
        raise RuntimeError(f"opponent-shift evaluation failed: {case['case_id']}")
    return record


def aggregate_label(runs: list[dict], label: str) -> dict:
    additive = (
        "rounds", "score", "coins", "kills", "crates", "bombs", "moves",
        "waits", "suicides", "invalid_actions", "steps",
    )
    totals = {key: sum(int(run["metrics_by_label"][label][key]) for run in runs) for key in additive}
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


def _ranking(metrics: dict[str, dict]) -> list[str]:
    return sorted(
        LABELS,
        key=lambda label: (
            metrics[label]["score_per_round"], metrics[label]["kills_per_round"],
            -metrics[label]["suicides_per_round"], -LABELS.index(label),
        ),
        reverse=True,
    )


def summarize(protocol: dict, runs: list[dict]) -> tuple[dict, list[dict], dict]:
    pooled = {label: aggregate_label(runs, label) for label in LABELS}
    blocks = []
    for run in runs:
        metrics = run["metrics_by_label"]
        blocks.append({
            "case_id": run["evaluation"]["case_id"],
            "world_seed": run["evaluation"]["world_seed"],
            "roster": run["evaluation"]["roster"],
            "metrics": metrics, "ranking": _ranking(metrics),
            "v4_leads_internal_by_score": pooled is not None and metrics["v4"]["score_per_round"] >= max(
                metrics["source-r2"]["score_per_round"], metrics["candidate-r3"]["score_per_round"]
            ) - 1e-12,
        })
    s138 = json.loads((ROOT / protocol["s138_reference"]["path"]).read_text(encoding="utf-8"))
    historical = {
        label: s138["rows"][label]["pooled_task4"]["target"]
        for label in INTERNAL_LABELS
    }
    return pooled, blocks, historical


def decide(protocol: dict, pooled: dict, blocks: list[dict], historical: dict) -> dict:
    v4_score = pooled["v4"]["score_per_round"]
    internal_leader = v4_score >= max(
        pooled["source-r2"]["score_per_round"], pooled["candidate-r3"]["score_per_round"]
    ) - 1e-12
    supportive_blocks = sum(bool(block["v4_leads_internal_by_score"]) for block in blocks)
    stable_internal_lead = supportive_blocks >= int(protocol["interpretation_rule"]["minimum_v4_internal_lead_blocks"])
    external_gap = pooled["external-strong"]["score_per_round"] - v4_score
    large_external_gap = external_gap >= float(protocol["interpretation_rule"]["external_score_gap_effect_size"])
    if internal_leader and stable_internal_lead:
        decision = (
            "v4_internal_lead_retained_external_gap_stop"
            if large_external_gap else "v4_internal_lead_retained_no_large_external_gap_stop"
        )
    elif internal_leader:
        decision = "v4_internal_lead_unstable_stop"
    else:
        decision = "v4_not_internal_leader_under_shift_stop"
    return {
        "decision": decision,
        "v4_internal_leader": internal_leader,
        "v4_internal_lead_blocks": supportive_blocks,
        "minimum_v4_internal_lead_blocks": protocol["interpretation_rule"]["minimum_v4_internal_lead_blocks"],
        "v4_internal_lead_stable": stable_internal_lead,
        "external_minus_v4_score_per_round": external_gap,
        "large_external_gap": large_external_gap,
        "external_score_gap_effect_size": protocol["interpretation_rule"]["external_score_gap_effect_size"],
        "pooled_ranking": _ranking(pooled),
        "shifted_score_minus_s138_rule_distribution": {
            label: pooled[label]["score_per_round"] - historical[label]["score_per_round"]
            for label in INTERNAL_LABELS
        },
        "selection_made": False,
        "training_started": False,
        "checkpoint_copied": False,
        "incumbent_replaced": False,
        "external_used_for_training_or_teacher_labels": False,
        "automatic_followup_started": False,
        "awaiting_user_instruction": True,
    }


def write_report(
    protocol_path: Path, protocol: dict, protocol_hash: str, runs: list[dict],
) -> tuple[dict, str]:
    output = report_path(protocol)
    existing = load_completed(output, REPORT_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("completed opponent-shift report protocol drift")
        return existing, sha256_file(output)
    pooled, blocks, historical = summarize(protocol, runs)
    report = {
        "schema_version": 1, "kind": REPORT_KIND, "status": "completed",
        "completed_at_utc": utc_now(), "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path), "protocol_sha256": protocol_hash,
        "evaluation_manifests": [relative(manifest_path(protocol, case["case_id"])) for case in protocol["evaluation"]["cases"]],
        "pooled": pooled, "case_blocks": blocks,
        "s138_rule_distribution_reference": historical,
        "result": decide(protocol, pooled, blocks, historical),
        "source_artifacts_modified": False, "default_checkpoint_modified": False,
        "external_source_persisted_in_repository": False,
        "automatic_followup_started": False, "awaiting_user_instruction": True,
    }
    atomic_json(output, report)
    return report, sha256_file(output)


def dry_run(protocol: dict) -> dict:
    cases = len(protocol["evaluation"]["cases"])
    rounds = int(protocol["evaluation"]["rounds_per_case"])
    return {
        "mode": "dry-run", "protocol_id": protocol["protocol_id"],
        "labels": list(LABELS), "cases": cases, "rounds_per_case": rounds,
        "shared_games_total": cases * rounds, "rounds_observed_per_label": cases * rounds,
        "roster_orders": [case["roster"] for case in protocol["evaluation"]["cases"]],
        "external_commit": protocol["external_stress_opponent"]["commit"],
        "external_checkout_started": False, "formal_evaluation_started": False,
        "training_started": False, "selection_allowed": False,
        "checkpoint_copy_allowed": False, "automatic_followup_started": False,
        "writes_terminal_report_and_stops": True,
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
        report, report_hash = existing, sha256_file(report_path(protocol))
    else:
        with pinned_external_checkout(protocol) as external_agent_dir:
            runs = [
                evaluate_one(protocol_path.resolve(), protocol, protocol_hash, case, external_agent_dir)
                for case in protocol["evaluation"]["cases"]
            ]
        report, report_hash = write_report(protocol_path.resolve(), protocol, protocol_hash, runs)
    print(json.dumps({
        "status": report["status"], "report": relative(report_path(protocol)),
        "report_sha256": report_hash, **report["result"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
