"""Audit the fixed lag between source-r2 BOMB actions and official outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPORT = ROOT / "experiments/logs/diagnostics/model-a-v4-counterfactual-signal-s129000/report.json"
SOURCE_REPORT_SHA256 = "ae9b39e624f5d6239a5bf665d84f664b3684d13e42231db78a6a63cba60f529a"
SOURCE_CHECKPOINT = ROOT / "experiments/checkpoints/model-a-v4-clean-curriculum-s114000/curriculum/r2/task2.pt"
SOURCE_CHECKPOINT_SHA256 = "6ab7b65aced2e3e2edeae68668ffa7f713cf7fd9f495c5c89201cdbfbaf35d40"
OUTPUT = ROOT / "experiments/logs/diagnostics/model-a-v4-bomb-credit-attribution-s130000.json"
BOMB_ACTION_INDEX = 5
EVENT_LAG_STEPS = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def attribute_events(
    rounds: np.ndarray,
    steps: np.ndarray,
    actions: np.ndarray,
    events: np.ndarray,
    lag: int = EVENT_LAG_STEPS,
) -> dict:
    """Match every event to a unique same-episode BOMB exactly ``lag`` steps earlier."""
    rounds = np.asarray(rounds)
    steps = np.asarray(steps)
    actions = np.asarray(actions)
    events = np.asarray(events, dtype=np.bool_)
    matches = []
    unmatched = []
    ambiguous = []
    for event_index in np.flatnonzero(events):
        candidates = np.flatnonzero(
            (rounds == rounds[event_index])
            & (steps == steps[event_index] - int(lag))
            & (actions == BOMB_ACTION_INDEX)
        )
        record = {
            "round": int(rounds[event_index]),
            "event_step": int(steps[event_index]),
            "expected_bomb_step": int(steps[event_index] - int(lag)),
        }
        if len(candidates) == 1:
            record["bomb_row"] = int(candidates[0])
            matches.append(record)
        elif len(candidates) == 0:
            unmatched.append(record)
        else:
            record["candidate_rows"] = [int(value) for value in candidates]
            ambiguous.append(record)
    total = int(events.sum())
    return {
        "events": total,
        "unique_matches": len(matches),
        "unique_match_fraction": len(matches) / total if total else 1.0,
        "unmatched": unmatched,
        "ambiguous": ambiguous,
        "matches": matches,
    }


def build_report() -> dict:
    if sha256_file(SOURCE_REPORT) != SOURCE_REPORT_SHA256:
        raise RuntimeError("s129000 terminal report hash changed")
    if sha256_file(SOURCE_CHECKPOINT) != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("source-r2 checkpoint hash changed")
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    if source.get("status") != "completed":
        raise RuntimeError("s129000 report is not completed")
    if source.get("result", {}).get("decision") != "counterfactual_causal_signal_not_supported_stop":
        raise RuntimeError("s129000 terminal decision changed")
    cases = {}
    pooled = {"kill_events": 0, "kill_matches": 0, "self_events": 0, "self_matches": 0}
    for case, item in sorted(source["collections"].items()):
        trace_path = ROOT / item["trace"]["path"]
        if not trace_path.is_file() or sha256_file(trace_path) != item["trace"]["sha256"]:
            raise RuntimeError(f"s129000 trace missing or changed: {case}")
        with np.load(trace_path, allow_pickle=False) as trace:
            kill = attribute_events(trace["round"], trace["step"], trace["action"], trace["kill_event"])
            self_kill = attribute_events(trace["round"], trace["step"], trace["action"], trace["self_event"])
        cases[case] = {
            "trace_path": item["trace"]["path"],
            "trace_sha256": item["trace"]["sha256"],
            "kill": kill,
            "self": self_kill,
        }
        pooled["kill_events"] += kill["events"]
        pooled["kill_matches"] += kill["unique_matches"]
        pooled["self_events"] += self_kill["events"]
        pooled["self_matches"] += self_kill["unique_matches"]
    passed = (
        len(cases) == 8
        and pooled == {"kill_events": 44, "kill_matches": 44, "self_events": 123, "self_matches": 123}
        and all(
            not outcome["unmatched"] and not outcome["ambiguous"]
            for case in cases.values() for outcome in (case["kill"], case["self"])
        )
    )
    if not passed:
        raise RuntimeError(f"fixed-lag attribution audit failed: {pooled}")
    return {
        "schema_version": 1,
        "kind": "model-a-v4-bomb-credit-attribution-audit",
        "status": "completed",
        "source_report": {
            "path": str(SOURCE_REPORT.relative_to(ROOT)),
            "sha256": SOURCE_REPORT_SHA256,
        },
        "source_parent": {
            "path": str(SOURCE_CHECKPOINT.relative_to(ROOT)),
            "sha256": SOURCE_CHECKPOINT_SHA256,
        },
        "environment_semantics": {
            "bomb_timer": 4,
            "event_lag_steps": EVENT_LAG_STEPS,
            "minimum_return_horizon_from_bomb_transition": EVENT_LAG_STEPS + 1,
            "chain_reactions": False,
        },
        "pooled": pooled,
        "cases": cases,
        "checks": {
            "eight_cases_complete": len(cases) == 8,
            "all_kills_uniquely_match_step_minus_four_bomb": True,
            "all_self_kills_uniquely_match_step_minus_four_bomb": True,
            "minimum_bomb_return_horizon_is_five": True,
        },
        "decision": "fixed_four_step_bomb_outcome_lag_confirmed",
        "passed": True,
        "new_game_rounds": 0,
        "policy_updates": 0,
    }


def dry_run() -> dict:
    return {
        "mode": "dry-run",
        "source_report_sha256": sha256_file(SOURCE_REPORT),
        "source_parent_sha256": sha256_file(SOURCE_CHECKPOINT),
        "expected_kill_matches": "44/44",
        "expected_self_matches": "123/123",
        "event_lag_steps": EVENT_LAG_STEPS,
        "minimum_return_horizon": EVENT_LAG_STEPS + 1,
        "new_game_rounds": 0,
        "policy_updates": 0,
        "output_written": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dry_run(), indent=2, sort_keys=True))
        return 0
    report = build_report()
    if OUTPUT.exists():
        existing = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if existing != report:
            raise RuntimeError("refusing to overwrite changed attribution audit")
    else:
        atomic_json(OUTPUT, report)
    print(json.dumps({
        "status": report["status"],
        "decision": report["decision"],
        "output": str(OUTPUT.relative_to(ROOT)),
        "sha256": sha256_file(OUTPUT),
        "pooled": report["pooled"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
