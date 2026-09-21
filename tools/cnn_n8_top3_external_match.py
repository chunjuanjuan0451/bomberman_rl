"""Run the three highest validated internal agents against the pinned strong agent."""

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

from agent_code.model_a_cnn_n8.callbacks import _load as load_cnn  # noqa: E402
from agent_code.model_a_cnn_n8.network import ARCHITECTURE  # noqa: E402
from agent_code.model_a_dqn.callbacks import MODEL_ARCHITECTURE  # noqa: E402
from agent_code.model_a_dqn.network import torch  # noqa: E402
from tools.v4_task4_duel_training import atomic_json, load_completed, metrics_by_agent, relative, seed_values  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-cnn-n8-top3-external-match-s161000.json"
LABELS = ("collision-cnn", "original-cnn", "frozen-v4", "external-strong")
INTERNAL_LABELS = LABELS[:3]
AGENT_CODES = {
    "collision-cnn": "model_a_cnn_n8_top3_collision",
    "original-cnn": "model_a_cnn_n8_top3_original",
    "frozen-v4": "model_a_dqn",
    "external-strong": "feature_is_everything_top3_eval",
}
EXPECTED_CASES = tuple(
    {
        "case_id": f"c{index + 1}",
        "world_seed": 161001 + index,
        "agent_seeds": {
            "collision-cnn": 261001 + index,
            "original-cnn": 361001 + index,
            "frozen-v4": 461001 + index,
            "external-strong": 561001 + index,
        },
        "roster": roster,
    }
    for index, roster in enumerate((
        ["collision-cnn", "original-cnn", "frozen-v4", "external-strong"],
        ["original-cnn", "frozen-v4", "external-strong", "collision-cnn"],
        ["frozen-v4", "external-strong", "collision-cnn", "original-cnn"],
        ["external-strong", "collision-cnn", "original-cnn", "frozen-v4"],
    ))
)
MANIFEST_KIND = "model-a-cnn-n8-top3-external-match-case"
REPORT_KIND = "model-a-cnn-n8-top3-external-match-report"


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
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def registered_seeds(protocol: dict) -> set[int]:
    values = set()
    for case in protocol["evaluation"]["cases"]:
        values.add(int(case["world_seed"]))
        values.update(int(seed) for seed in case["agent_seeds"].values())
    return values


def assert_seeds_untouched(protocol_path: Path, protocol: dict) -> None:
    registered = registered_seeds(protocol)
    output_root = (ROOT / protocol["manifest_directory"]).resolve()
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
        raise RuntimeError(f"registered s161000 seeds were already used: {conflicts}")


def _validate_checkpoint(protocol: dict, label: str) -> None:
    item = protocol["checkpoint_inventory"][label]
    path = ROOT / item["path"]
    if not path.is_file() or sha256_file(path) != item["sha256"]:
        raise RuntimeError(f"top-three checkpoint mismatch: {label}")
    payload = load_cnn(path) if label != "frozen-v4" else _torch_load(path)
    if label == "frozen-v4":
        if payload.get("architecture") not in (None, MODEL_ARCHITECTURE):
            raise RuntimeError("frozen-v4 architecture mismatch")
    else:
        required = {"architecture": ARCHITECTURE, "stage": "task4", "replica": "r3", "stage_rounds": 800}
        if any(payload.get(key) != value for key, value in required.items()):
            raise RuntimeError(f"CNN lineage mismatch: {label}")


def load_protocol(path: Path) -> tuple[dict, str]:
    path = path.resolve()
    raw = path.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("schema_version") != 1 or protocol.get("kind") != "model-a-cnn-n8-top3-external-match":
        raise RuntimeError("wrong top-three direct-match protocol")
    if tuple(protocol.get("labels", ())) != LABELS:
        raise RuntimeError("top-three direct-match identities changed")
    if protocol.get("training_allowed") or protocol.get("checkpoint_copy_allowed") or protocol.get("automatic_followup"):
        raise RuntimeError("top-three direct match must remain evaluation-only and terminal")
    evaluation = protocol["evaluation"]
    if evaluation.get("scenario") != "classic" or evaluation.get("rounds_per_case") != 25:
        raise RuntimeError("top-three direct-match budget changed")
    if tuple(evaluation.get("cases", ())) != EXPECTED_CASES:
        raise RuntimeError("top-three direct-match cases changed")
    if len(registered_seeds(protocol)) != 20:
        raise RuntimeError("s161000 requires twenty unique fresh seeds")
    selection = protocol["selection_source"]
    source = ROOT / selection["path"]
    if not source.is_file() or sha256_file(source) != selection["sha256"]:
        raise RuntimeError("top-three selection source drift")
    source_payload = json.loads(source.read_text(encoding="utf-8"))
    if source_payload.get("status") != "completed":
        raise RuntimeError("top-three selection source is incomplete")
    observed = source_payload["pooled_metrics"]
    for label, expected_score in selection["selected_score_per_round"].items():
        if abs(float(observed[label]["score_per_round"]) - float(expected_score)) > 1e-12:
            raise RuntimeError(f"top-three selection score drift: {label}")
    if set(protocol.get("checkpoint_inventory", {})) != set(INTERNAL_LABELS):
        raise RuntimeError("top-three checkpoint inventory changed")
    for label in INTERNAL_LABELS:
        _validate_checkpoint(protocol, label)
    expected_external = {
        "repository": "https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman.git",
        "commit": "a7fe5041b02548ce4438e502ca3adb11576bae75",
        "agent_subdirectory": "agent_code/feature_is_everything",
        "training_or_teacher_use_allowed": False,
        "file_sha256": {
            "callbacks.py": "cadc00f69e4b0e04a1063e6760deea1ed48e23855992201c3edcebe6284071fe",
            "features.py": "8baf96ae76f685a5dbfb74bae644e2cbb816952103859081801888ebe6b9f620",
            "model.py": "27736c5afa4ccf50582141b6aea57740699365e35879ad6d5cb4842b3fc7829c",
            "my-saved-model.pt": "0fb1df0b3c3f255db121e8ba7c6061d1293f55d3e7ae12772259643ee2060577",
            "symmetry.py": "cac8ad3d8b197d1cca6fa67ca35d3db00d45c4c81afa549d797bb8b45657f82e",
        },
    }
    if protocol.get("external_strong_agent") != expected_external:
        raise RuntimeError("external strong-agent binding changed")
    assert_seeds_untouched(path, protocol)
    return protocol, hashlib.sha256(raw).hexdigest()


@contextmanager
def pinned_external_checkout(protocol: dict):
    external = protocol["external_strong_agent"]
    with tempfile.TemporaryDirectory(prefix="bomberman-s161000-") as temporary:
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
        if subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip() != external["commit"]:
            raise RuntimeError("external strong-agent commit mismatch")
        agent_dir = checkout / external["agent_subdirectory"]
        for name, expected in external["file_sha256"].items():
            if sha256_file(agent_dir / name) != expected:
                raise RuntimeError(f"external strong-agent hash mismatch: {name}")
        yield agent_dir


def manifest_path(protocol: dict, case_id: str) -> Path:
    return ROOT / protocol["manifest_directory"] / f"{case_id}.json"


def report_path(protocol: dict) -> Path:
    return ROOT / protocol["report_path"]


def evaluate_one(protocol_path: Path, protocol: dict, protocol_hash: str, case: dict, external_dir: Path) -> dict:
    output = manifest_path(protocol, case["case_id"])
    expected = {"case_id": case["case_id"], "world_seed": case["world_seed"],
                "agent_seeds": case["agent_seeds"], "roster": case["roster"],
                "scenario": "classic", "rounds": 25}
    existing = load_completed(output, MANIFEST_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash or existing.get("evaluation") != expected:
            raise RuntimeError(f"completed top-three case drift: {output}")
        return existing
    stats = output.with_suffix(".stats.json")
    if output.exists() or stats.exists():
        raise RuntimeError(f"refusing orphaned top-three output: {output}")
    codes = [AGENT_CODES[label] for label in case["roster"]]
    command = [sys.executable, "main.py", "play", "--agents", *codes, "--train", "0",
               "--continue-without-training", "--no-gui", "--scenario", "classic",
               "--n-rounds", "25", "--seed", str(case["world_seed"]), "--save-stats", str(stats)]
    overrides = {
        "CNN_TOP3_MATCH_PROTOCOL_PATH": str(protocol_path),
        "CNN_TOP3_MATCH_CASE_ID": case["case_id"],
        "CNN_TOP3_EXTERNAL_AGENT_DIR": str(external_dir),
        "MODEL_A_CHECKPOINT_PATH": str((ROOT / protocol["checkpoint_inventory"]["frozen-v4"]["path"]).resolve()),
        "MODEL_A_SEED": str(case["agent_seeds"]["frozen-v4"]),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }
    before = {label: sha256_file(ROOT / protocol["checkpoint_inventory"][label]["path"]) for label in INTERNAL_LABELS}
    record = {"schema_version": 1, "kind": MANIFEST_KIND, "status": "running", "started_at_utc": utc_now(),
              "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_hash, "evaluation": expected,
              "command": command, "checkpoint_sha256_before": before, "raw_stats": relative(stats)}
    atomic_json(output, record)
    child = os.environ.copy(); child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record.update({"ended_at_utc": utc_now(), "exit_code": completed.returncode})
    try:
        after = {label: sha256_file(ROOT / protocol["checkpoint_inventory"][label]["path"]) for label in INTERNAL_LABELS}
        if completed.returncode or not stats.is_file() or after != before:
            raise RuntimeError("top-three process, stats, or checkpoint immutability failed")
        raw = metrics_by_agent(json.loads(stats.read_text(encoding="utf-8")))
        reverse = {code: label for label, code in AGENT_CODES.items()}
        if set(raw) != set(reverse):
            raise RuntimeError(f"unexpected agents in top-three stats: {sorted(raw)}")
        record.update({"metrics_by_label": {reverse[code]: value for code, value in raw.items()},
                       "checkpoint_sha256_after": after, "status": "completed"})
    except Exception as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    atomic_json(output, record)
    if record["status"] != "completed":
        raise RuntimeError(f"top-three evaluation failed: {case['case_id']}: {record.get('error')}")
    return record


def aggregate(runs: list[dict], label: str) -> dict:
    additive = ("rounds", "score", "coins", "kills", "crates", "bombs", "moves", "waits",
                "suicides", "invalid_actions", "steps")
    totals = {key: sum(int(run["metrics_by_label"][label][key]) for run in runs) for key in additive}
    rounds, steps = totals["rounds"], totals["steps"]
    return {**totals, "score_per_round": totals["score"] / rounds, "coins_per_round": totals["coins"] / rounds,
            "kills_per_round": totals["kills"] / rounds, "suicides_per_round": totals["suicides"] / rounds,
            "invalid_actions_per_round": totals["invalid_actions"] / rounds,
            "wait_fraction": totals["waits"] / steps if steps else 0.0,
            "episode_length": totals["steps"] / rounds}


def ranking(metrics: dict[str, dict]) -> list[str]:
    return sorted(LABELS, key=lambda label: (metrics[label]["score_per_round"], metrics[label]["kills_per_round"],
                                             -metrics[label]["suicides_per_round"]), reverse=True)


def summarize(protocol: dict, protocol_hash: str, runs: list[dict]) -> dict:
    pooled = {label: aggregate(runs, label) for label in LABELS}
    blocks = []
    wins = {label: 0 for label in LABELS}
    for run in runs:
        block_ranking = ranking(run["metrics_by_label"])
        wins[block_ranking[0]] += 1
        blocks.append({"case_id": run["evaluation"]["case_id"], "roster": run["evaluation"]["roster"],
                       "metrics": run["metrics_by_label"], "ranking": block_ranking})
    external_score = pooled["external-strong"]["score_per_round"]
    return {"schema_version": 1, "kind": REPORT_KIND, "status": "completed", "completed_at_utc": utc_now(),
            "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_hash,
            "shared_games": 100, "rounds_observed_per_identity": 100, "pooled": pooled,
            "pooled_ranking": ranking(pooled), "case_blocks": blocks, "block_first_place_counts": wins,
            "score_gap_to_external": {label: pooled[label]["score_per_round"] - external_score for label in INTERNAL_LABELS},
            "result": {"decision": "top3_external_direct_match_completed_stop", "training_started": False,
                       "checkpoint_modified": False, "checkpoint_copied": False, "model_selected": False,
                       "automatic_followup_started": False}, "awaiting_user_instruction": True}


def execute(protocol_path: Path, protocol: dict, protocol_hash: str) -> dict:
    existing = load_completed(report_path(protocol), REPORT_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("completed top-three report protocol drift")
        return existing
    with pinned_external_checkout(protocol) as external_dir:
        runs = [evaluate_one(protocol_path, protocol, protocol_hash, case, external_dir)
                for case in protocol["evaluation"]["cases"]]
    report = summarize(protocol, protocol_hash, runs)
    if report_path(protocol).exists():
        raise RuntimeError("refusing to overwrite top-three report")
    atomic_json(report_path(protocol), report)
    return report


def dry_run(protocol: dict, protocol_hash: str) -> dict:
    return {"mode": "dry-run", "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_hash,
            "labels": list(LABELS), "cases": 4, "rounds_per_case": 25, "shared_games": 100,
            "rounds_observed_per_identity": 100, "seat_balanced": True, "external_checkout_started": False,
            "formal_evaluation_started": False, "training_started": False, "writes_terminal_report_and_stops": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    protocol, protocol_hash = load_protocol(protocol_path)
    result = execute(protocol_path.resolve(), protocol, protocol_hash) if args.execute else dry_run(protocol, protocol_hash)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
