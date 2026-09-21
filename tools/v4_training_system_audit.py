"""Run Phase A passive collection and Phase B offline temporal-credit audit.

This tool never creates an optimizer, changes an action, updates a policy, or
selects a horizon for training.  It writes one terminal diagnostic report and
stops for review.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_dqn.features import ACTIONS  # noqa: E402
from agent_code.model_a_v4_system_audit.config import (  # noqa: E402
    CASES, CUSTOM_EVENTS_CONFIRMED_ABSENT, EVENT_NAMES, HORIZONS, STRATA,
    case_stratum, load_protocol, sha256_file, trace_path,
)
from tools.v4_task4_duel_training import (  # noqa: E402
    atomic_json, load_completed, metrics_by_agent, relative, seed_values,
)


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-training-system-audit-s135000.json"
COLLECTION_KIND = "model-a-v4-training-system-audit-collection"
REPORT_KIND = "model-a-v4-training-system-audit-report"
TRACE_FIELDS = {
    "round", "step", "action", "legal_mask", "online_q_values", "target_q_values",
    "global_features", "opponent_count", "nearest_opponent_distance",
    "next_nearest_opponent_distance", "approach_step", "bomb_legal", "oracle_evaluated",
    "oracle_timeout", "guaranteed_traps", "affected_opponents", "max_space_reduction",
    "own_bottleneck", "own_terminal_positions", "event_counts", "event_reward_components",
    "state_shaping", "total_reward", "metadata",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def collection_manifest_path(protocol: dict, case: str) -> Path:
    return ROOT / protocol["collection_manifest_directory"] / f"{case}.json"


def report_path(protocol: dict) -> Path:
    return ROOT / protocol["report_path"]


def registered_seeds(protocol: dict) -> set[int]:
    return {
        int(seed)
        for case in protocol["collection"]["cases"].values()
        for seed in case.values()
    }


def assert_registered_seeds_untouched(protocol_path: Path, protocol: dict) -> None:
    registered = registered_seeds(protocol)
    excluded_roots = {
        (ROOT / protocol["trace_directory"]).resolve(),
        (ROOT / protocol["collection_manifest_directory"]).resolve(),
    }
    excluded_files = {protocol_path.resolve(), report_path(protocol).resolve()}
    conflicts = []
    for path in (ROOT / "experiments").rglob("*.json"):
        resolved = path.resolve()
        if resolved in excluded_files or any(root == resolved or root in resolved.parents for root in excluded_roots):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        overlap = sorted(registered & seed_values(payload))
        if overlap:
            conflicts.append({"path": relative(path), "seeds": overlap})
    if conflicts:
        raise ValueError(f"registered s135000 seeds were already used: {conflicts}")


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise RuntimeError(f"training-system trace missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def _event_index(name: str) -> int:
    return EVENT_NAMES.index(name)


def validate_trace(protocol: dict, protocol_hash: str, case: str) -> dict[str, np.ndarray]:
    path = trace_path(protocol, case)
    payload = _load_trace(path)
    if set(payload) != TRACE_FIELDS:
        raise RuntimeError(f"training-system trace fields changed: {case}")
    metadata = json.loads(str(payload["metadata"].item()))
    expected = {
        "schema_version": 1,
        "kind": "model-a-v4-training-system-trace",
        "protocol_sha256": protocol_hash,
        "case": case,
        "stratum": case_stratum(protocol, case),
        "parent_sha256": protocol["source_parent"]["sha256"],
        "rounds": int(protocol["collection"]["rounds_per_case"]),
        "collection_epsilon": float(protocol["collection"]["epsilon"]),
        "oracle_deadline_seconds": float(protocol["collection"]["oracle_deadline_seconds"]),
        "event_names": list(EVENT_NAMES),
        "custom_events_confirmed_absent": list(CUSTOM_EVENTS_CONFIRMED_ABSENT),
        "policy_updates": 0,
        "actions_overridden": 0,
    }
    if metadata != expected:
        raise RuntimeError(f"training-system trace metadata changed: {case}")
    rows = len(payload["action"])
    matrix_shapes = {
        "legal_mask": (rows, len(ACTIONS)),
        "online_q_values": (rows, len(ACTIONS)),
        "target_q_values": (rows, len(ACTIONS)),
        "event_counts": (rows, len(EVENT_NAMES)),
        "event_reward_components": (rows, len(EVENT_NAMES)),
    }
    for field, shape in matrix_shapes.items():
        if payload[field].shape != shape:
            raise RuntimeError(f"training-system trace shape mismatch: {case}/{field}")
    if payload["global_features"].ndim != 2 or payload["global_features"].shape[0] != rows:
        raise RuntimeError(f"training-system global feature shape mismatch: {case}")
    for field, values in payload.items():
        if field not in {*matrix_shapes, "global_features", "metadata"} and len(values) != rows:
            raise RuntimeError(f"training-system row count mismatch: {case}/{field}")
    if rows <= 0 or len(np.unique(payload["round"])) != int(protocol["collection"]["rounds_per_case"]):
        raise RuntimeError(f"training-system round coverage mismatch: {case}")
    actions = payload["action"].astype(np.int64)
    if np.any(actions < 0) or np.any(actions >= len(ACTIONS)):
        raise RuntimeError(f"training-system action index mismatch: {case}")
    if not np.all(payload["legal_mask"][np.arange(rows), actions]):
        raise RuntimeError(f"training-system observed an illegal selected action: {case}")
    if np.any(payload["approach_step"] != (
        (payload["nearest_opponent_distance"] >= 0)
        & (payload["next_nearest_opponent_distance"] >= 0)
        & (payload["next_nearest_opponent_distance"] < payload["nearest_opponent_distance"])
    )):
        raise RuntimeError(f"training-system approach label mismatch: {case}")
    if not np.allclose(
        payload["total_reward"],
        payload["event_reward_components"].sum(axis=1) + payload["state_shaping"],
        atol=1e-6,
    ):
        raise RuntimeError(f"training-system reward ledger mismatch: {case}")
    if np.any(payload["oracle_evaluated"] & ~payload["bomb_legal"]):
        raise RuntimeError(f"training-system Oracle evaluated illegal BOMB: {case}")
    for round_id in np.unique(payload["round"]):
        indices = np.flatnonzero(payload["round"] == round_id)
        if not np.all(np.diff(payload["step"][indices]) == 1):
            raise RuntimeError(f"training-system trace is not a complete episode: {case}/round-{round_id}")
    return payload


def _same_round_window(payload: dict[str, np.ndarray], index: int, width: int) -> np.ndarray:
    return (
        (payload["round"] == payload["round"][index])
        & (payload["step"] >= payload["step"][index])
        & (payload["step"] < payload["step"][index] + width)
    )


def _escape_flags(payload: dict[str, np.ndarray]) -> np.ndarray:
    bomb = payload["action"] == ACTIONS.index("BOMB")
    self_event = payload["event_counts"][:, _event_index("KILLED_SELF")] > 0
    flags = np.zeros(len(bomb), dtype=np.bool_)
    for index in np.flatnonzero(bomb):
        flags[index] = not bool(self_event[_same_round_window(payload, int(index), 8)].any())
    return flags


def _linked_kills(payload: dict[str, np.ndarray]) -> list[dict[str, int]]:
    actions = payload["action"]
    kills = payload["event_counts"][:, _event_index("KILLED_OPPONENT")]
    result = []
    for kill_index in np.flatnonzero(kills > 0):
        matches = np.flatnonzero(
            (payload["round"] == payload["round"][kill_index])
            & np.isin(payload["step"][kill_index] - payload["step"], (4, 5))
            & (actions == ACTIONS.index("BOMB"))
        )
        result.append({
            "kill_row": int(kill_index),
            "kill_count": int(kills[kill_index]),
            "candidate_bombs": int(len(matches)),
            "bomb_row": int(matches[0]) if len(matches) == 1 else -1,
            "lag": int(payload["step"][kill_index] - payload["step"][matches[0]]) if len(matches) == 1 else -1,
        })
    return result


def reward_ledger(payloads: list[dict[str, np.ndarray]]) -> dict:
    counts = sum((p["event_counts"].sum(axis=0).astype(np.int64) for p in payloads), np.zeros(len(EVENT_NAMES), dtype=np.int64))
    components = sum((p["event_reward_components"].sum(axis=0).astype(np.float64) for p in payloads), np.zeros(len(EVENT_NAMES)))
    shaping_values = np.concatenate([p["state_shaping"].astype(np.float64) for p in payloads])
    rewards = np.concatenate([p["total_reward"].astype(np.float64) for p in payloads])
    rounds = sum(len(np.unique(p["round"])) for p in payloads)
    abs_total = float(np.abs(components).sum() + np.abs(shaping_values).sum())
    by_event = {}
    for index, name in enumerate(EVENT_NAMES):
        by_event[name] = {
            "count": int(counts[index]),
            "count_per_round": float(counts[index] / rounds),
            "signed_contribution": float(components[index]),
            "signed_contribution_per_round": float(components[index] / rounds),
            "absolute_ledger_share": float(abs(components[index]) / abs_total) if abs_total else 0.0,
        }
    return {
        "rounds": rounds,
        "rows": int(sum(len(p["action"]) for p in payloads)),
        "by_official_event": by_event,
        "danger_state_shaping": {
            "positive_count": int((shaping_values > 0).sum()),
            "negative_count": int((shaping_values < 0).sum()),
            "signed_contribution": float(shaping_values.sum()),
            "absolute_ledger_share": float(np.abs(shaping_values).sum() / abs_total) if abs_total else 0.0,
        },
        "custom_events": {name: {"implemented": False, "count": 0, "signed_contribution": 0.0} for name in CUSTOM_EVENTS_CONFIRMED_ABSENT},
        "total_signed_reward": float(rewards.sum()),
        "reward_per_round": float(rewards.sum() / rounds),
    }


def attack_funnel(payloads: list[dict[str, np.ndarray]]) -> dict:
    result = {stage: 0 for stage in ("approach", "positioning", "threat", "bomb", "trap", "escape", "kill")}
    result.update({"rows": 0, "bomb_legal": 0, "oracle_timeouts": 0, "unique_linked_kills": 0, "ambiguous_or_unlinked_kills": 0})
    for payload in payloads:
        actions = payload["action"]
        bomb = actions == ACTIONS.index("BOMB")
        positioning = payload["bomb_legal"] & (payload["max_space_reduction"] >= 0.5)
        threat = bomb & (payload["affected_opponents"] >= 1)
        trap = bomb & (payload["max_space_reduction"] >= 0.5)
        escape = _escape_flags(payload)
        kills = payload["event_counts"][:, _event_index("KILLED_OPPONENT")]
        links = _linked_kills(payload)
        result["rows"] += len(actions)
        result["approach"] += int(payload["approach_step"].sum())
        result["positioning"] += int(positioning.sum())
        result["threat"] += int(threat.sum())
        result["bomb"] += int(bomb.sum())
        result["trap"] += int(trap.sum())
        result["escape"] += int(escape.sum())
        result["kill"] += int(kills.sum())
        result["bomb_legal"] += int(payload["bomb_legal"].sum())
        result["oracle_timeouts"] += int(payload["oracle_timeout"].sum())
        result["unique_linked_kills"] += sum(item["kill_count"] for item in links if item["candidate_bombs"] == 1)
        result["ambiguous_or_unlinked_kills"] += sum(item["kill_count"] for item in links if item["candidate_bombs"] != 1)
    denominators = {
        "approach_per_row": result["rows"],
        "positioning_per_bomb_legal": result["bomb_legal"],
        "threat_per_bomb": result["bomb"],
        "trap_per_bomb": result["bomb"],
        "escape_per_bomb": result["bomb"],
        "kill_per_bomb": result["bomb"],
    }
    result["rates"] = {
        name: float(result[name.split("_per_")[0]] / denominator) if denominator else 0.0
        for name, denominator in denominators.items()
    }
    return result


def _episode_ends(payload: dict[str, np.ndarray]) -> np.ndarray:
    rounds = payload["round"]
    ends = np.empty(len(rounds), dtype=np.int64)
    for round_id in np.unique(rounds):
        indices = np.flatnonzero(rounds == round_id)
        ends[indices] = indices[-1]
    return ends


def _horizon_arrays(payload: dict[str, np.ndarray], horizon: int, gamma: float) -> dict[str, np.ndarray]:
    rows = len(payload["action"])
    episode_ends = _episode_ends(payload)
    targets = np.zeros(rows, dtype=np.float64)
    q_selected = payload["online_q_values"][np.arange(rows), payload["action"].astype(np.int64)].astype(np.float64)
    used_steps = np.zeros(rows, dtype=np.int16)
    event_discounted = np.zeros((rows, len(EVENT_NAMES)), dtype=np.float64)
    shaping_discounted = np.zeros(rows, dtype=np.float64)
    included_counts = np.zeros((rows, len(EVENT_NAMES)), dtype=np.int64)
    for start in range(rows):
        stop = min(start + horizon - 1, int(episode_ends[start]))
        indices = np.arange(start, stop + 1)
        discounts = gamma ** np.arange(len(indices))
        event_discounted[start] = (payload["event_reward_components"][indices] * discounts[:, None]).sum(axis=0)
        shaping_discounted[start] = float((payload["state_shaping"][indices] * discounts).sum())
        included_counts[start] = payload["event_counts"][indices].sum(axis=0)
        targets[start] = event_discounted[start].sum() + shaping_discounted[start]
        used_steps[start] = len(indices)
        if stop < episode_ends[start]:
            next_index = stop + 1
            mask = payload["legal_mask"][next_index].astype(np.bool_)
            legal = np.flatnonzero(mask)
            if legal.size:
                online_next = payload["online_q_values"][next_index]
                best = online_next[legal].max()
                choices = legal[np.isclose(online_next[legal], best)]
                next_action = int(choices[0])
                targets[start] += (gamma ** len(indices)) * float(payload["target_q_values"][next_index, next_action])
    delta = targets - q_selected
    return {
        "targets": targets,
        "delta": delta,
        "smooth_l1_abs_gradient": np.minimum(np.abs(delta), 1.0),
        "used_steps": used_steps,
        "event_discounted": event_discounted,
        "shaping_discounted": shaping_discounted,
        "included_counts": included_counts,
    }


def horizon_audit(payloads: list[dict[str, np.ndarray]], horizon: int, gamma: float, batch_size: int) -> dict:
    arrays = [_horizon_arrays(payload, horizon, gamma) for payload in payloads]
    deltas = np.concatenate([a["delta"] for a in arrays])
    gradients = np.concatenate([a["smooth_l1_abs_gradient"] for a in arrays])
    used = np.concatenate([a["used_steps"] for a in arrays])
    included = np.concatenate([a["included_counts"] for a in arrays])
    event_discounted = sum((a["event_discounted"].sum(axis=0) for a in arrays), np.zeros(len(EVENT_NAMES)))
    shaping_discounted = float(sum(a["shaping_discounted"].sum() for a in arrays))
    kill_index = _event_index("KILLED_OPPONENT")
    kill_targets = included[:, kill_index] > 0
    kill_gradient = float(gradients[kill_targets].sum())
    total_gradient = float(gradients.sum())
    action_counts = {name: 0 for name in ACTIONS}
    stage_counts = {stage: 0 for stage in ("approach", "positioning", "threat", "bomb", "trap", "escape", "kill")}
    contamination = {name: 0 for name in ("COIN_COLLECTED", "CRATE_DESTROYED", "KILLED_SELF", "GOT_KILLED")}
    linked_total = 0
    linked_covered = 0
    offset = 0
    for payload, values in zip(payloads, arrays):
        local_kill = values["included_counts"][:, kill_index] > 0
        actions = payload["action"].astype(np.int64)
        bomb = actions == ACTIONS.index("BOMB")
        escape = _escape_flags(payload)
        stages = {
            "approach": payload["approach_step"],
            "positioning": payload["bomb_legal"] & (payload["max_space_reduction"] >= 0.5),
            "threat": bomb & (payload["affected_opponents"] >= 1),
            "bomb": bomb,
            "trap": bomb & (payload["max_space_reduction"] >= 0.5),
            "escape": escape,
            "kill": payload["event_counts"][:, kill_index] > 0,
        }
        for action_index, name in enumerate(ACTIONS):
            action_counts[name] += int((local_kill & (actions == action_index)).sum())
        for name, flags in stages.items():
            stage_counts[name] += int((local_kill & flags).sum())
        for name in contamination:
            contamination[name] += int((local_kill & (values["included_counts"][:, _event_index(name)] > 0)).sum())
        for link in _linked_kills(payload):
            if link["candidate_bombs"] != 1:
                continue
            linked_total += link["kill_count"]
            if link["kill_row"] - link["bomb_row"] < horizon:
                linked_covered += link["kill_count"]
        offset += len(actions)
    probability = float(kill_targets.mean()) if len(kill_targets) else 0.0
    return {
        "horizon": horizon,
        "rows": len(deltas),
        "mean_return_steps": float(used.mean()),
        "terminal_truncated_fraction": float((used < horizon).mean()),
        "kill_bearing_targets": int(kill_targets.sum()),
        "kill_target_frequency": probability,
        "expected_kill_targets_per_uniform_batch": float(batch_size * probability),
        "probability_uniform_batch_has_no_kill_target": float((1.0 - probability) ** batch_size),
        "unique_linked_bomb_kills": linked_total,
        "linked_bomb_kills_covered": linked_covered,
        "linked_bomb_kill_coverage": float(linked_covered / linked_total) if linked_total else 0.0,
        "mean_abs_td_error": float(np.abs(deltas).mean()),
        "td_error_percentiles": {str(q): float(np.percentile(np.abs(deltas), q)) for q in (50, 90, 99)},
        "smooth_l1_gradient_mass": total_gradient,
        "kill_target_gradient_mass": kill_gradient,
        "kill_target_gradient_mass_fraction": float(kill_gradient / total_gradient) if total_gradient else 0.0,
        "kill_target_start_actions": action_counts,
        "kill_target_start_stages": stage_counts,
        "kill_target_contamination": contamination,
        "discounted_reward_components": {
            **{name: float(event_discounted[index]) for index, name in enumerate(EVENT_NAMES)},
            "DANGER_STATE_SHAPING": shaping_discounted,
        },
    }


def analyze(protocol: dict, payload_by_case: dict[str, dict[str, np.ndarray]]) -> dict:
    groups = {
        **{case: [payload_by_case[case]] for case in CASES},
        **{
            stratum: [payload_by_case[case] for case in protocol["collection"]["strata"][stratum]["cases"]]
            for stratum in STRATA
        },
        "pooled": [payload_by_case[case] for case in CASES],
    }
    gamma = float(protocol["audit_contract"]["gamma"])
    batch_size = int(protocol["audit_contract"]["batch_size"])
    result = {}
    for group, payloads in groups.items():
        result[group] = {
            "reward_ledger": reward_ledger(payloads),
            "attack_funnel": attack_funnel(payloads),
            "horizons": {
                str(horizon): horizon_audit(payloads, horizon, gamma, batch_size)
                for horizon in HORIZONS
            },
        }
    pooled_horizons = result["pooled"]["horizons"]
    covering = [h for h in HORIZONS if pooled_horizons[str(h)]["linked_bomb_kill_coverage"] >= 0.95]
    result["conclusions"] = {
        "minimum_tested_horizon_covering_95pct_unique_linked_bomb_kills": min(covering) if covering else None,
        "custom_attack_shaping_present": False,
        "new_reward_added": False,
        "policy_updates": 0,
        "actions_overridden": 0,
        "optimizer_created": False,
        "horizon_selected_for_training": None,
        "decision": protocol["decision"],
        "automatic_followup_started": False,
    }
    return result


def collect_one(protocol_path: Path, protocol: dict, protocol_hash: str, case: str) -> dict:
    manifest = collection_manifest_path(protocol, case)
    existing = load_completed(manifest, COLLECTION_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError(f"completed training-system collection drift: {case}")
        payload = validate_trace(protocol, protocol_hash, case)
        if sha256_file(trace_path(protocol, case)) != existing["trace"]["sha256"]:
            raise RuntimeError(f"completed training-system trace drift: {case}")
        return existing
    output = trace_path(protocol, case)
    stats_path = manifest.with_suffix(".stats.json")
    if output.exists() or stats_path.exists() or manifest.exists():
        raise RuntimeError(f"refusing orphaned training-system artifacts: {case}")
    collection = protocol["collection"]
    stratum = case_stratum(protocol, case)
    opponents = collection["strata"][stratum]["opponents"]
    seeds = collection["cases"][case]
    parent = (ROOT / protocol["source_parent"]["path"]).resolve()
    command = [
        sys.executable, "main.py", "play", "--agents", "model_a_v4_system_audit", *opponents,
        "--train", "1", "--continue-without-training", "--no-gui",
        "--scenario", collection["scenario"], "--n-rounds", str(collection["rounds_per_case"]),
        "--seed", str(seeds["world_seed"]), "--save-stats", str(stats_path),
    ]
    overrides = {
        "MODEL_A_SYSTEM_AUDIT_PROTOCOL_PATH": str(protocol_path),
        "MODEL_A_SYSTEM_AUDIT_PARENT_PATH": str(parent),
        "MODEL_A_SYSTEM_AUDIT_CASE": case,
        "MODEL_A_SYSTEM_AUDIT_SEED": str(seeds["agent_seed"]),
        "TASK4_RULE_SEED": str(seeds["opponent_seed"]),
    }
    record = {
        "schema_version": 1, "kind": COLLECTION_KIND, "status": "running",
        "started_at_utc": utc_now(), "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path), "protocol_sha256": protocol_hash,
        "phase": "A_passive_collection", "case": case, "stratum": stratum,
        "collection": {**seeds, "scenario": collection["scenario"], "opponents": opponents,
                       "rounds": collection["rounds_per_case"], "epsilon": collection["epsilon"]},
        "parent_checkpoint": {"path": relative(parent), "sha256": sha256_file(parent)},
        "command": command, "environment_overrides": overrides,
        "trace": {"path": relative(output), "sha256": None},
        "artifacts": {"raw_stats": relative(stats_path)},
        "policy_updates": 0, "actions_overridden": 0, "optimizer_created": False,
    }
    atomic_json(manifest, record)
    child = os.environ.copy(); child.update(overrides)
    completed = subprocess.run(command, cwd=ROOT, env=child, check=False)
    record["ended_at_utc"] = utc_now(); record["exit_code"] = completed.returncode
    try:
        if completed.returncode != 0 or not stats_path.is_file():
            raise RuntimeError("training-system collection process or stats failed")
        if sha256_file(parent) != protocol["source_parent"]["sha256"]:
            raise RuntimeError("source-r2 changed during passive collection")
        validate_trace(protocol, protocol_hash, case)
        record["trace"]["sha256"] = sha256_file(output)
        record["metrics_by_agent"] = metrics_by_agent(json.loads(stats_path.read_text(encoding="utf-8")))
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"; record["error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(manifest, record)
    if record["status"] != "completed":
        raise RuntimeError(f"training-system collection failed: {case}: {record.get('error')}")
    return record


def write_report(protocol_path: Path, protocol: dict, protocol_hash: str, collections: dict, result: dict) -> tuple[dict, str]:
    path = report_path(protocol)
    existing = load_completed(path, REPORT_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("completed training-system report drift")
        return existing, sha256_file(path)
    record = {
        "schema_version": 1, "kind": REPORT_KIND, "status": "completed",
        "completed_at_utc": utc_now(), "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path), "protocol_sha256": protocol_hash,
        "scope": protocol["scope"], "source_parent": protocol["source_parent"],
        "phase_a_collections": {
            case: {
                "manifest_path": relative(collection_manifest_path(protocol, case)),
                "manifest_sha256": sha256_file(collection_manifest_path(protocol, case)),
                "trace_path": relative(trace_path(protocol, case)),
                "trace_sha256": collections[case]["trace"]["sha256"],
            }
            for case in CASES
        },
        "phase_b_offline_analysis": result,
        "awaiting_user_instruction": True,
        "training_started": False, "policy_checkpoint_created": False,
        "source_checkpoint_modified": False, "default_checkpoint_modified": False,
        "automatic_followup_started": False,
    }
    atomic_json(path, record)
    return record, sha256_file(path)


def dry_run(protocol: dict) -> dict:
    return {
        "mode": "dry-run", "protocol_id": protocol["protocol_id"],
        "scope": protocol["scope"], "parent": "source-r2", "architecture": "exact-v4",
        "phase_a": {
            "cases": len(CASES), "strata": list(STRATA),
            "rounds_per_case": int(protocol["collection"]["rounds_per_case"]),
            "total_rounds": len(CASES) * int(protocol["collection"]["rounds_per_case"]),
            "epsilon": float(protocol["collection"]["epsilon"]),
        },
        "phase_b": {"horizons": list(HORIZONS), "same_trajectories": True, "offline_only": True},
        "policy_updates": 0, "actions_overridden": 0, "optimizer_created": False,
        "training_started": False, "formal_collection_started": False,
        "writes_terminal_report_and_stops": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    protocol, protocol_hash = load_protocol(protocol_path)
    assert_registered_seeds_untouched(protocol_path, protocol)
    if not args.execute:
        print(json.dumps(dry_run(protocol), indent=2, sort_keys=True))
        return 0
    existing = load_completed(report_path(protocol), REPORT_KIND)
    if existing is not None:
        print(json.dumps({"status": existing["status"], "report": relative(report_path(protocol)),
                          "report_sha256": sha256_file(report_path(protocol)),
                          "decision": existing["phase_b_offline_analysis"]["conclusions"]["decision"]},
                         indent=2, sort_keys=True))
        return 0
    collections = {case: collect_one(protocol_path, protocol, protocol_hash, case) for case in CASES}
    payloads = {case: validate_trace(protocol, protocol_hash, case) for case in CASES}
    result = analyze(protocol, payloads)
    report, digest = write_report(protocol_path, protocol, protocol_hash, collections, result)
    print(json.dumps({
        "status": report["status"], "report": relative(report_path(protocol)),
        "report_sha256": digest, "decision": result["conclusions"]["decision"],
        "training_started": False, "policy_checkpoint_created": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
