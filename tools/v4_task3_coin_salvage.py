"""Screen and provisionally select existing exact-v4 Task3 coin snapshots.

Dry-run is the default. ``--execute`` performs evaluation only in two fixed
phases, writes one terminal diagnostic report, and never trains or starts
Task4. The original outer seeds remain untouched.
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

from agent_code.model_a_v4_task3_retention.config import (  # noqa: E402
    BRANCHES, load_protocol as load_source_protocol,
)
from tools.v4_task3_retention import (  # noqa: E402
    aggregate, metrics_by_agent, sha256_file, validate_new_checkpoint,
)


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-task3-coin-salvage-s121000.json"
ROUNDS = tuple(range(50, 601, 50))
BASELINE_LABELS = ("v4", "source-r2", "parent-b1", "parent-b2", "parent-b3")
STRATA = ("task1", "task2", "task3_peaceful", "task3_coin")


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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_completed(path: Path, kind: str) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed" or payload.get("kind") != kind:
        raise RuntimeError(f"refusing incomplete/incompatible artifact: {path}")
    return payload


def _seed_values(suite: dict) -> set[int]:
    values = set()
    for spec in suite.values():
        for case in spec["cases"]:
            for raw in case.values():
                value = int(raw)
                if value <= 0 or value in values:
                    raise ValueError("salvage evaluation seeds must be positive and globally unique")
                values.add(value)
    return values


def _source_seed_values(protocol: dict) -> set[int]:
    values = set()
    for stage in protocol["training"].values():
        for case in stage["branches"].values():
            values.update(int(value) for value in case.values())
    for suite in (protocol["inner_selection"]["strata"], protocol["outer_evaluation"]["strata"]):
        for spec in suite.values():
            for case in spec["cases"]:
                values.update(int(value) for value in case.values())
    return values


def load_protocol(path: Path) -> tuple[dict, str, dict, str]:
    path = path.resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or protocol.get("kind") != "model-a-v4-task3-coin-salvage":
        raise ValueError("wrong Task3 coin salvage protocol")
    if tuple(protocol.get("branches", ())) != BRANCHES:
        raise ValueError("coin salvage requires ordered b1/b2/b3")
    source_path = (ROOT / protocol["source_experiment"]["protocol_path"]).resolve()
    source_protocol, source_hash = load_source_protocol(source_path)
    if source_hash != protocol["source_experiment"]["protocol_sha256"]:
        raise ValueError("source Task3 protocol hash mismatch")

    bindings = protocol.get("source_bindings", {})
    if set(bindings) != {"tools/v4_task3_coin_salvage.py"}:
        raise ValueError("coin salvage source binding set mismatch")
    for name, expected_hash in bindings.items():
        source = ROOT / name
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise ValueError(f"coin salvage source binding mismatch: {name}")

    expected_inventory = {
        branch: {str(stage_round) for stage_round in ROUNDS} for branch in BRANCHES
    }
    inventory = protocol.get("checkpoint_inventory", {})
    if inventory.get("v4") != {
        "path": source_protocol["frozen_v4_baseline"]["checkpoint_path"],
        "sha256": source_protocol["frozen_v4_baseline"]["checkpoint_sha256"],
    }:
        raise ValueError("frozen-v4 inventory binding mismatch")
    if inventory.get("source_r2") != source_protocol["source_parent"]["checkpoint"]:
        raise ValueError("source-r2 inventory binding mismatch")
    for name in ("v4", "source_r2"):
        item = inventory[name]
        checkpoint = ROOT / item["path"]
        if not checkpoint.is_file() or sha256_file(checkpoint) != item["sha256"]:
            raise ValueError(f"baseline checkpoint binding mismatch: {name}")
    if set(inventory.get("coin_snapshots", {})) != set(BRANCHES):
        raise ValueError("coin snapshot branches incomplete")
    if set(inventory.get("peaceful_parents", {})) != set(BRANCHES):
        raise ValueError("peaceful parent branches incomplete")
    for branch in BRANCHES:
        items = inventory["coin_snapshots"][branch]
        if set(items) != expected_inventory[branch]:
            raise ValueError(f"coin snapshot rounds incomplete: {branch}")
        for stage_round in ROUNDS:
            item = items[str(stage_round)]
            checkpoint = (ROOT / item["path"]).resolve()
            if not checkpoint.is_file() or sha256_file(checkpoint) != item["sha256"]:
                raise ValueError(f"coin snapshot binding mismatch: {branch}/{stage_round}")
            validate_new_checkpoint(
                source_protocol, source_hash, checkpoint, branch, "task3_coin",
                stage_round=stage_round, selected=False,
            )
        parent_item = inventory["peaceful_parents"][branch]
        parent = (ROOT / parent_item["path"]).resolve()
        if not parent.is_file() or sha256_file(parent) != parent_item["sha256"]:
            raise ValueError(f"peaceful parent binding mismatch: {branch}")
        validate_new_checkpoint(
            source_protocol, source_hash, parent, branch, "task3_peaceful", selected=True,
        )

    evidence = protocol.get("source_evidence", {})
    expected_evidence = {
        f"{branch}_{kind}" for branch in BRANCHES
        for kind in ("peaceful_training", "coin_training", "peaceful_selection")
    }
    if set(evidence) != expected_evidence:
        raise ValueError("coin salvage requires six training and three peaceful-selection manifests")
    for name, item in evidence.items():
        artifact = ROOT / item["path"]
        if not artifact.is_file() or sha256_file(artifact) != item["sha256"]:
            raise ValueError(f"source evidence binding mismatch: {name}")
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        if payload.get("status") != "completed" or payload.get("protocol_sha256") != source_hash:
            raise ValueError(f"source evidence is incomplete: {name}")

    expected_distributions = {
        "task1": ("coin-heaven", []),
        "task2": ("classic", []),
        "task3_peaceful": ("classic", ["seeded_peaceful_agent"]),
        "task3_coin": ("classic", ["seeded_coin_collector_agent"]),
    }
    screening = protocol.get("screening", {})
    selection = protocol.get("selection", {})
    if (
        selection.get("provisional_only") is not True
        or selection.get("requires_later_selection_free_outer_confirmation") is not True
    ):
        raise ValueError("coin salvage must remain provisional and require outer confirmation")
    if tuple(screening.get("strata", {})) != ("task3_coin",):
        raise ValueError("screening must evaluate Task3 coin only")
    if tuple(selection.get("strata", {})) != STRATA:
        raise ValueError("selection strata changed")
    if int(screening.get("shortlist_per_branch_maximum", 0)) != 3:
        raise ValueError("shortlist maximum changed")
    if tuple(screening.get("shortlist_reasons", ())) != (
        "highest_score", "highest_kills", "highest_balanced_achievement",
    ):
        raise ValueError("shortlist reasons changed")
    for suite, rounds, cases in ((screening["strata"], 10, 2), (selection["strata"], 20, 1)):
        for name, spec in suite.items():
            scenario, opponents = expected_distributions[name]
            if (
                spec.get("scenario") != scenario
                or spec.get("opponents") != opponents
                or int(spec.get("rounds_per_case", 0)) != rounds
                or len(spec.get("cases", ())) != cases
            ):
                raise ValueError(f"salvage evaluation contract changed: {name}")
    fresh_seeds = _seed_values(screening["strata"])
    overlap = fresh_seeds.intersection(_seed_values(selection["strata"]))
    if overlap:
        raise ValueError("screening and selection seeds overlap")
    fresh_seeds.update(_seed_values(selection["strata"]))
    if fresh_seeds.intersection(_source_seed_values(source_protocol)):
        raise ValueError("salvage seeds overlap the source Task3 experiment")

    rule = selection.get("rule", {})
    expected_rule = {
        "minimum_current_utility_fraction": 0.8,
        "maximum_coin_score_gap_to_v4": 0.25,
        "minimum_coin_kills_delta_to_v4": 0.0,
        "minimum_prior_retained_fraction": 0.7,
        "prior_metrics": {
            "task1": ["score_per_round"],
            "task2": ["score_per_round"],
            "task3_peaceful": ["score_per_round", "kills_per_round"],
        },
        "selection_order": [
            "maximum worst prior retained fraction",
            "maximum mean prior retained fraction",
            "maximum balanced current utility",
            "maximum coin kills_per_round",
            "maximum coin score_per_round",
            "earlier stage round",
            "branch order b1/b2/b3",
        ],
    }
    for key, expected in expected_rule.items():
        if rule.get(key) != expected:
            raise ValueError(f"salvage selection rule changed: {key}")

    original_report = ROOT / source_protocol["report_path"]
    if original_report.exists():
        raise ValueError("source Task3 terminal report unexpectedly exists")
    original_outer = ROOT / source_protocol["evaluation_manifest_directory"] / "outer"
    if original_outer.exists() and any(original_outer.rglob("*.json")):
        raise ValueError("source Task3 outer seeds were already consumed")
    for branch in BRANCHES:
        selected_coin = ROOT / source_protocol["checkpoint_directory"] / branch / "task3_coin/selected.pt"
        if selected_coin.exists():
            raise ValueError("source Task3 coin selection unexpectedly exists")
    return protocol, sha256_file(path), source_protocol, source_hash


def candidate_label(branch: str, stage_round: int) -> str:
    return f"{branch}-r{stage_round:04d}"


def identity(protocol: dict, label: str) -> dict:
    inventory = protocol["checkpoint_inventory"]
    if label == "v4":
        return {"kind": "v4", **inventory["v4"]}
    if label == "source-r2":
        return {"kind": "source", **inventory["source_r2"]}
    if label.startswith("parent-"):
        branch = label.removeprefix("parent-")
        if branch not in BRANCHES:
            raise ValueError(f"invalid parent label: {label}")
        return {"kind": "parent", "branch": branch, **inventory["peaceful_parents"][branch]}
    branch, raw_round = label.split("-r", 1)
    stage_round = int(raw_round)
    if branch not in BRANCHES or stage_round not in ROUNDS:
        raise ValueError(f"invalid candidate label: {label}")
    return {
        "kind": "candidate", "branch": branch, "stage_round": stage_round,
        **inventory["coin_snapshots"][branch][str(stage_round)],
    }


def checkpoint_environment(
    protocol: dict, source_protocol_path: Path, item: dict, case: dict,
) -> tuple[str, Path, dict[str, str]]:
    checkpoint = (ROOT / item["path"]).resolve()
    kind = item["kind"]
    if kind == "v4":
        return "model_a_dqn", checkpoint, {
            "MODEL_A_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_SEED": str(case["agent_seed"]),
        }
    if kind == "source":
        clean_path = ROOT / protocol["source_experiment"]["clean_protocol_path"]
        return "model_a_v4_curriculum", checkpoint, {
            "MODEL_A_V4C_PROTOCOL_PATH": str(clean_path),
            "MODEL_A_V4C_CHECKPOINT_PATH": str(checkpoint),
            "MODEL_A_V4C_ARM": "curriculum",
            "MODEL_A_V4C_REPLICA": "r2",
            "MODEL_A_V4C_STAGE": "task2",
            "MODEL_A_V4C_SEED": str(case["agent_seed"]),
        }
    stage = "task3_peaceful" if kind == "parent" else "task3_coin"
    return "model_a_v4_task3_retention", checkpoint, {
        "MODEL_A_V4T3R_PROTOCOL_PATH": str(source_protocol_path),
        "MODEL_A_V4T3R_CHECKPOINT_PATH": str(checkpoint),
        "MODEL_A_V4T3R_BRANCH": item["branch"],
        "MODEL_A_V4T3R_STAGE": stage,
        "MODEL_A_V4T3R_SEED": str(case["agent_seed"]),
    }


def manifest_path(protocol: dict, phase: str, stratum: str, label: str, world_seed: int) -> Path:
    return ROOT / protocol["evaluation_manifest_directory"] / phase / stratum / f"{label}-s{world_seed}.json"


def evaluate_one(
    protocol_path: Path,
    protocol: dict,
    protocol_hash: str,
    source_protocol_path: Path,
    phase: str,
    stratum: str,
    label: str,
    spec: dict,
    case: dict,
) -> dict:
    item = identity(protocol, label)
    target, checkpoint, overrides = checkpoint_environment(protocol, source_protocol_path, item, case)
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != item["sha256"]:
        raise RuntimeError(f"checkpoint drift before salvage evaluation: {label}")
    output = manifest_path(protocol, phase, stratum, label, int(case["world_seed"]))
    expected = {
        **case,
        "scenario": spec["scenario"],
        "opponents": spec["opponents"],
        "rounds": int(spec["rounds_per_case"]),
        "target_agent": target,
    }
    existing = load_completed(output, "model-a-v4-task3-coin-salvage-evaluation")
    if existing is not None:
        if (
            existing.get("protocol_sha256") != protocol_hash
            or existing.get("evaluation") != expected
            or existing.get("checkpoint", {}).get("sha256") != checkpoint_hash
        ):
            raise RuntimeError(f"completed salvage evaluation drift: {output}")
        return existing
    stats_path = output.with_suffix(".stats.json")
    if stats_path.exists():
        raise RuntimeError(f"refusing orphaned salvage stats: {stats_path}")
    agents = [target, *spec["opponents"]]
    command = [
        sys.executable, "main.py", "play", "--agents", *agents,
        "--train", "0", "--continue-without-training", "--no-gui",
        "--scenario", spec["scenario"], "--n-rounds", str(spec["rounds_per_case"]),
        "--seed", str(case["world_seed"]), "--save-stats", str(stats_path),
    ]
    if spec["opponents"]:
        overrides["TASK3_OPPONENT_SEED"] = str(case["opponent_seed"])
    record = {
        "schema_version": 1,
        "kind": "model-a-v4-task3-coin-salvage-evaluation",
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "source_protocol_sha256": protocol["source_experiment"]["protocol_sha256"],
        "phase": phase,
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
        raise RuntimeError(f"salvage evaluation failed: {phase}/{stratum}/{label}")
    return record


def evaluate_labels(
    protocol_path: Path,
    protocol: dict,
    protocol_hash: str,
    source_protocol_path: Path,
    phase: str,
    labels: list[str],
    strata: dict,
) -> tuple[dict, dict]:
    rows = {}
    manifests = {}
    for stratum, spec in strata.items():
        rows[stratum] = {}
        manifests[stratum] = {}
        for label in labels:
            runs = [
                evaluate_one(
                    protocol_path, protocol, protocol_hash, source_protocol_path,
                    phase, stratum, label, spec, case,
                )
                for case in spec["cases"]
            ]
            rows[stratum][label] = aggregate(runs)
            manifests[stratum][label] = [
                relative(manifest_path(protocol, phase, stratum, label, int(case["world_seed"])))
                for case in spec["cases"]
            ]
    return rows, manifests


def balanced_achievement(score: float, kills: float, best_score: float, best_kills: float) -> float:
    score_ratio = score / best_score if best_score > 0.0 else 1.0
    kill_ratio = kills / best_kills if best_kills > 0.0 else 1.0
    if score_ratio + kill_ratio <= 0.0:
        return 0.0
    return 2.0 * score_ratio * kill_ratio / (score_ratio + kill_ratio)


def shortlist(protocol: dict, screening_rows: dict) -> tuple[dict, dict]:
    rows = screening_rows["task3_coin"]
    selected = {}
    diagnostics = {}
    for branch in BRANCHES:
        labels = [candidate_label(branch, stage_round) for stage_round in ROUNDS]
        best_score = max(float(rows[label]["score_per_round"]) for label in labels)
        best_kills = max(float(rows[label]["kills_per_round"]) for label in labels)

        def round_number(label: str) -> int:
            return int(identity(protocol, label)["stage_round"])

        by_score = max(labels, key=lambda label: (
            rows[label]["score_per_round"], rows[label]["kills_per_round"], -round_number(label),
        ))
        by_kills = max(labels, key=lambda label: (
            rows[label]["kills_per_round"], rows[label]["score_per_round"], -round_number(label),
        ))
        utilities = {
            label: balanced_achievement(
                float(rows[label]["score_per_round"]), float(rows[label]["kills_per_round"]),
                best_score, best_kills,
            )
            for label in labels
        }
        by_balance = max(labels, key=lambda label: (
            utilities[label], rows[label]["kills_per_round"],
            rows[label]["score_per_round"], -round_number(label),
        ))
        reasons = {
            "highest_score": by_score,
            "highest_kills": by_kills,
            "highest_balanced_achievement": by_balance,
        }
        chosen = []
        for reason in protocol["screening"]["shortlist_reasons"]:
            label = reasons[reason]
            if label not in chosen:
                chosen.append(label)
        selected[branch] = chosen
        diagnostics[branch] = {
            "best_score_per_round": best_score,
            "best_kills_per_round": best_kills,
            "balanced_achievement": utilities,
            "reason_to_label": reasons,
            "shortlist": chosen,
        }
    return selected, diagnostics


def retained_fraction(value: float, parent: float) -> float:
    if parent > 0.0:
        return value / parent
    return 1.0 if value >= parent else 0.0


def decide(protocol: dict, shortlists: dict, selection_rows: dict) -> dict:
    rule = protocol["selection"]["rule"]
    candidate_diagnostics = {}
    branch_selections = {}
    viable_labels = []
    for branch in BRANCHES:
        labels = shortlists[branch]
        coin_rows = {label: selection_rows["task3_coin"][label] for label in labels}
        best_score = max(float(row["score_per_round"]) for row in coin_rows.values())
        best_kills = max(float(row["kills_per_round"]) for row in coin_rows.values())
        utilities = {
            label: balanced_achievement(
                float(row["score_per_round"]), float(row["kills_per_round"]),
                best_score, best_kills,
            )
            for label, row in coin_rows.items()
        }
        best_utility = max(utilities.values())
        parent_label = f"parent-{branch}"
        for label in labels:
            prior_fractions = {}
            for stratum, metrics in rule["prior_metrics"].items():
                for metric in metrics:
                    prior_fractions[f"{stratum}:{metric}"] = retained_fraction(
                        float(selection_rows[stratum][label][metric]),
                        float(selection_rows[stratum][parent_label][metric]),
                    )
            coin = selection_rows["task3_coin"][label]
            v4 = selection_rows["task3_coin"]["v4"]
            gates = {
                "current_utility": utilities[label] + 1e-12 >= (
                    float(rule["minimum_current_utility_fraction"]) * best_utility
                ),
                "coin_score_vs_v4": float(coin["score_per_round"]) >= (
                    float(v4["score_per_round"]) - float(rule["maximum_coin_score_gap_to_v4"])
                ),
                "coin_kills_vs_v4": float(coin["kills_per_round"]) >= (
                    float(v4["kills_per_round"]) + float(rule["minimum_coin_kills_delta_to_v4"])
                ),
                "prior_retention": min(prior_fractions.values()) + 1e-12 >= float(
                    rule["minimum_prior_retained_fraction"]
                ),
            }
            viable = all(gates.values())
            if viable:
                viable_labels.append(label)
            candidate_diagnostics[label] = {
                "branch": branch,
                "stage_round": identity(protocol, label)["stage_round"],
                "current_utility": utilities[label],
                "best_branch_current_utility": best_utility,
                "coin_score_per_round": float(coin["score_per_round"]),
                "coin_kills_per_round": float(coin["kills_per_round"]),
                "v4_coin_score_per_round": float(v4["score_per_round"]),
                "v4_coin_kills_per_round": float(v4["kills_per_round"]),
                "prior_retained_fractions": prior_fractions,
                "worst_prior_retained_fraction": min(prior_fractions.values()),
                "mean_prior_retained_fraction": sum(prior_fractions.values()) / len(prior_fractions),
                "gates": gates,
                "viable": viable,
                "checkpoint": protocol["checkpoint_inventory"]["coin_snapshots"][branch][
                    str(identity(protocol, label)["stage_round"])
                ],
            }
        branch_viable = [label for label in labels if candidate_diagnostics[label]["viable"]]
        if branch_viable:
            branch_selections[branch] = max(branch_viable, key=lambda label: (
                candidate_diagnostics[label]["worst_prior_retained_fraction"],
                candidate_diagnostics[label]["mean_prior_retained_fraction"],
                candidate_diagnostics[label]["current_utility"],
                candidate_diagnostics[label]["coin_kills_per_round"],
                candidate_diagnostics[label]["coin_score_per_round"],
                -candidate_diagnostics[label]["stage_round"],
            ))
        else:
            branch_selections[branch] = None
    if viable_labels:
        branch_order = {branch: -index for index, branch in enumerate(BRANCHES)}
        selected = max(viable_labels, key=lambda label: (
            candidate_diagnostics[label]["worst_prior_retained_fraction"],
            candidate_diagnostics[label]["mean_prior_retained_fraction"],
            candidate_diagnostics[label]["current_utility"],
            candidate_diagnostics[label]["coin_kills_per_round"],
            candidate_diagnostics[label]["coin_score_per_round"],
            branch_order[candidate_diagnostics[label]["branch"]],
            -candidate_diagnostics[label]["stage_round"],
        ))
        decision = "provisional_coin_snapshot_selected"
    else:
        selected = None
        decision = "no_viable_coin_snapshot"
    return {
        "decision": decision,
        "candidate_diagnostics": candidate_diagnostics,
        "viable_labels": viable_labels,
        "branch_selections": branch_selections,
        "provisional_selected_label": selected,
        "provisional_selected_checkpoint": (
            candidate_diagnostics[selected]["checkpoint"] if selected is not None else None
        ),
        "requires_selection_free_outer_confirmation": selected is not None,
        "automatic_checkpoint_copied": False,
        "training_started": False,
        "outer_evaluation_started": False,
        "task4_started": False,
    }


def report_path(protocol: dict) -> Path:
    return ROOT / protocol["report_path"]


def write_report(
    protocol_path: Path,
    protocol: dict,
    protocol_hash: str,
    screening_rows: dict,
    screening_manifests: dict,
    shortlists: dict,
    shortlist_diagnostics: dict,
    selection_rows: dict,
    selection_manifests: dict,
) -> tuple[dict, str]:
    path = report_path(protocol)
    existing = load_completed(path, "model-a-v4-task3-coin-salvage-report")
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("completed coin salvage report protocol drift")
        return existing, sha256_file(path)
    result = decide(protocol, shortlists, selection_rows)
    report = {
        "schema_version": 1,
        "kind": "model-a-v4-task3-coin-salvage-report",
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "source_protocol_sha256": protocol["source_experiment"]["protocol_sha256"],
        "status": "completed",
        "completed_at_utc": utc_now(),
        "screening_rows": screening_rows,
        "screening_evaluation_manifests": screening_manifests,
        "shortlists": shortlists,
        "shortlist_diagnostics": shortlist_diagnostics,
        "selection_rows": selection_rows,
        "selection_evaluation_manifests": selection_manifests,
        "result": result,
        "awaiting_user_instruction": True,
        "source_artifacts_modified": False,
        "default_checkpoint_modified": False,
    }
    atomic_json(path, report)
    return report, sha256_file(path)


def dry_run(protocol: dict) -> dict:
    screen_spec = protocol["screening"]["strata"]["task3_coin"]
    screen_labels = len(BRANCHES) * len(ROUNDS) + len(BASELINE_LABELS)
    screening_rounds = screen_labels * len(screen_spec["cases"]) * int(screen_spec["rounds_per_case"])
    maximum_selection_labels = (
        len(BRANCHES) * int(protocol["screening"]["shortlist_per_branch_maximum"])
        + len(BASELINE_LABELS)
    )
    selection_rounds = maximum_selection_labels * sum(
        len(spec["cases"]) * int(spec["rounds_per_case"])
        for spec in protocol["selection"]["strata"].values()
    )
    return {
        "mode": "dry-run",
        "protocol_id": protocol["protocol_id"],
        "coin_snapshots": len(BRANCHES) * len(ROUNDS),
        "screening_labels": screen_labels,
        "screening_evaluation_rounds": screening_rounds,
        "maximum_shortlisted_candidates": len(BRANCHES) * int(
            protocol["screening"]["shortlist_per_branch_maximum"]
        ),
        "maximum_selection_labels": maximum_selection_labels,
        "maximum_selection_evaluation_rounds": selection_rounds,
        "maximum_total_evaluation_rounds": screening_rounds + selection_rounds,
        "source_outer_seeds_used": False,
        "formal_evaluation_started": False,
        "training_started": False,
        "checkpoint_copy_started": False,
        "outer_evaluation_started": False,
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
        existing = load_completed(report_path(protocol), "model-a-v4-task3-coin-salvage-report")
        if existing is not None:
            if existing.get("protocol_sha256") != protocol_hash:
                raise RuntimeError("completed coin salvage report protocol drift")
            print(json.dumps({
                "status": "completed",
                "report": relative(report_path(protocol)),
                "report_sha256": sha256_file(report_path(protocol)),
                "decision": existing["result"]["decision"],
                "awaiting_user_instruction": True,
                "training_started": False,
                "task4_started": False,
            }, indent=2, sort_keys=True))
            return 0
        source_protocol_path = (ROOT / protocol["source_experiment"]["protocol_path"]).resolve()
        screen_labels = [
            candidate_label(branch, stage_round)
            for branch in BRANCHES for stage_round in ROUNDS
        ] + list(BASELINE_LABELS)
        screening_rows, screening_manifests = evaluate_labels(
            protocol_path.resolve(), protocol, protocol_hash, source_protocol_path,
            "screening", screen_labels, protocol["screening"]["strata"],
        )
        shortlists, shortlist_diagnostics = shortlist(protocol, screening_rows)
        selection_labels = [
            label for branch in BRANCHES for label in shortlists[branch]
        ] + list(BASELINE_LABELS)
        selection_rows, selection_manifests = evaluate_labels(
            protocol_path.resolve(), protocol, protocol_hash, source_protocol_path,
            "selection", selection_labels, protocol["selection"]["strata"],
        )
        report, report_hash = write_report(
            protocol_path.resolve(), protocol, protocol_hash,
            screening_rows, screening_manifests, shortlists, shortlist_diagnostics,
            selection_rows, selection_manifests,
        )
        print(json.dumps({
            "status": "completed",
            "report": relative(report_path(protocol)),
            "report_sha256": report_hash,
            "shortlists": shortlists,
            "decision": report["result"]["decision"],
            "provisional_selected_label": report["result"]["provisional_selected_label"],
            "awaiting_user_instruction": True,
            "training_started": False,
            "outer_evaluation_started": False,
            "task4_started": False,
        }, indent=2, sort_keys=True))
        return 0
    except (OSError, KeyError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
