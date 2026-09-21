"""Trace the s122000 invalid-action boundary without generating new efficacy data.

The audit deterministically replays the two already revealed Task3 coin cases
for source-r2 and frozen v4.  It classifies every INVALID_ACTION according to
whether the action was executable in the decision state but became blocked by
another agent's same-step movement.  This diagnostic never changes the formal
s122000 decision.  If and only if all source-r2 invalids reproduce and are
same-step collisions, it creates an explicitly adjudicated Task3 checkpoint.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import events as e  # noqa: E402
from environment import BombeRLeWorld, GenericWorld, WorldArgs  # noqa: E402
from agent_code.model_a_v4_curriculum.config import sha256_file  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-v4-task3-invalid-boundary-audit-s122000.json"
AUDIT_KIND = "model-a-v4-task3-invalid-boundary-audit-report"
ADJUDICATION_KIND = "model-a-v4-task3-boundary-adjudication"
MOVE_DELTAS = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}


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
    if temporary.exists():
        raise RuntimeError(f"refusing stale temporary artifact: {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_completed(path: Path, kind: str) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed" or payload.get("kind") != kind:
        raise RuntimeError(f"refusing incomplete/incompatible artifact: {path}")
    return payload


def load_protocol(path: Path) -> tuple[dict, str, dict]:
    path = path.resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or protocol.get("kind") != "model-a-v4-task3-invalid-boundary-audit":
        raise ValueError("wrong invalid-boundary audit protocol")
    if protocol.get("efficacy_reselection_allowed") is not False or protocol.get("formal_decision_override") is not False:
        raise ValueError("boundary audit must not reselect efficacy or rewrite s122000")
    if protocol.get("training_allowed") is not False or protocol.get("task4_execution_allowed") is not False:
        raise ValueError("boundary audit must not train or execute Task4")
    expected_cases = (
        {"world_seed": 122401, "agent_seed": 222401, "opponent_seed": 322401, "rounds": 25},
        {"world_seed": 122402, "agent_seed": 222402, "opponent_seed": 322402, "rounds": 25},
    )
    if tuple(protocol.get("cases", ())) != expected_cases or tuple(protocol.get("labels", ())) != ("v4", "source-r2"):
        raise ValueError("boundary audit must replay the exact revealed coin cases and labels")
    binding = protocol.get("source_bindings", {})
    if set(binding) != {"tools/v4_task3_invalid_boundary_audit.py"}:
        raise ValueError("boundary audit source binding set mismatch")
    if sha256_file(ROOT / "tools/v4_task3_invalid_boundary_audit.py") != binding["tools/v4_task3_invalid_boundary_audit.py"]:
        raise ValueError("boundary audit runner hash mismatch")
    report_item = protocol["source_report"]
    report_path = ROOT / report_item["path"]
    if not report_path.is_file() or sha256_file(report_path) != report_item["sha256"]:
        raise ValueError("s122000 source report binding mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "completed"
        or report.get("result", {}).get("decision") != "source_r2_rejected_task3_unresolved_stop"
        or report.get("result", {}).get("gates", {}).get("task3_safety") is not False
        or sum(not value for key, value in report["result"]["gates"].items() if key != "task3_safety") != 0
    ):
        raise ValueError("s122000 is not the expected safety-only boundary failure")
    for label, item in protocol["checkpoint_inventory"].items():
        checkpoint = ROOT / item["path"]
        if not checkpoint.is_file() or sha256_file(checkpoint) != item["sha256"]:
            raise ValueError(f"boundary checkpoint binding mismatch: {label}")
    return protocol, sha256_file(path), report


def decision_legality(state: dict, action: str) -> tuple[bool, tuple[int, int] | None, str]:
    if action == "WAIT":
        return True, tuple(state["self"][3]), "wait"
    if action == "BOMB":
        return bool(state["self"][2]), tuple(state["self"][3]), "bomb_available" if state["self"][2] else "bomb_unavailable"
    if action not in MOVE_DELTAS:
        return False, None, "unknown_action"
    x, y = state["self"][3]
    dx, dy = MOVE_DELTAS[action]
    destination = (x + dx, y + dy)
    field = state["field"]
    if not (0 <= destination[0] < field.shape[0] and 0 <= destination[1] < field.shape[1]):
        return False, destination, "out_of_bounds"
    if int(field[destination]) != 0:
        return False, destination, "wall_or_crate"
    if destination in {tuple(position) for position, _ in state["bombs"]}:
        return False, destination, "bomb_occupied"
    if destination in {tuple(other[3]) for other in state["others"]}:
        return False, destination, "agent_occupied"
    return True, destination, "free_at_decision"


class CollisionAuditWorld(BombeRLeWorld):
    def __init__(self, args: WorldArgs, agents, target_name: str):
        self.invalid_trace: list[dict] = []
        self.target_name = target_name
        super().__init__(args, agents)

    def perform_agent_action(self, agent, action: str):
        state = agent.last_game_state
        legal, destination, decision_reason = decision_legality(state, action)
        prior_event_count = len(agent.events)
        decision_positions = {
            other[0]: tuple(other[3]) for other in [state["self"], *state["others"]]
        }
        live_positions = {
            other.name: (other.x, other.y) for other in self.active_agents if other is not agent
        }
        GenericWorld.perform_agent_action(self, agent, action)
        if e.INVALID_ACTION not in agent.events[prior_event_count:]:
            return
        occupants = [name for name, position in live_positions.items() if position == destination]
        bombs_now = [tuple(bomb.get_state()[0]) for bomb in self.bombs]
        if legal and action in MOVE_DELTAS and occupants:
            category = "simultaneous_agent_collision"
        elif legal and action in MOVE_DELTAS and destination in bombs_now:
            category = "same_step_bomb_block"
        elif not legal:
            category = "illegal_in_decision_state"
        else:
            category = "unexplained_execution_invalid"
        occupant_details = []
        for name in occupants:
            occupant = next(other for other in self.active_agents if other.name == name)
            occupant_details.append({
                "name": name,
                "decision_position": list(decision_positions.get(name, ())),
                "execution_position": [occupant.x, occupant.y],
                "chosen_action": occupant.last_action,
            })
        self.invalid_trace.append({
            "round": self.round,
            "step": self.step,
            "agent": agent.name,
            "is_target": agent.name == self.target_name,
            "action": action,
            "decision_position": list(state["self"][3]),
            "destination": None if destination is None else list(destination),
            "decision_legal": legal,
            "decision_reason": decision_reason,
            "category": category,
            "occupants_at_execution": occupant_details,
        })


@contextmanager
def environment(overrides: dict[str, str]):
    previous = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_case(protocol: dict, label: str, case: dict) -> dict:
    inventory = protocol["checkpoint_inventory"]
    if label == "v4":
        target_code = "model_a_dqn"
        target_name = "model_a_dqn"
        overrides = {
            "MODEL_A_CHECKPOINT_PATH": str(ROOT / inventory["v4"]["path"]),
            "MODEL_A_SEED": str(case["agent_seed"]),
        }
    else:
        target_code = "model_a_v4_curriculum"
        target_name = "model_a_v4_curriculum"
        overrides = {
            "MODEL_A_V4C_PROTOCOL_PATH": str(ROOT / protocol["clean_protocol_path"]),
            "MODEL_A_V4C_CHECKPOINT_PATH": str(ROOT / inventory["source-r2"]["path"]),
            "MODEL_A_V4C_ARM": "curriculum",
            "MODEL_A_V4C_REPLICA": "r2",
            "MODEL_A_V4C_STAGE": "task2",
            "MODEL_A_V4C_SEED": str(case["agent_seed"]),
        }
    overrides["TASK3_OPPONENT_SEED"] = str(case["opponent_seed"])
    log_dir = ROOT / protocol["log_directory"] / label / f"s{case['world_seed']}"
    log_dir.mkdir(parents=True, exist_ok=True)
    args = WorldArgs(
        no_gui=True,
        fps=15,
        turn_based=False,
        update_interval=0.1,
        save_replay=False,
        replay=None,
        make_video=False,
        continue_without_training=True,
        log_dir=str(log_dir),
        save_stats=False,
        match_name=None,
        seed=int(case["world_seed"]),
        silence_errors=False,
        scenario="classic",
    )
    with environment(overrides):
        world = CollisionAuditWorld(
            args,
            [(target_code, False), ("seeded_coin_collector_agent", False)],
            target_name,
        )
        for _ in range(int(case["rounds"])):
            world.new_round()
            while world.running:
                world.do_step()
        world.end()
    target = next(agent for agent in world.agents if agent.name == target_name)
    trace = [item for item in world.invalid_trace if item["is_target"]]
    categories = Counter(item["category"] for item in trace)
    for handler in list(world.logger.handlers):
        handler.close()
        world.logger.removeHandler(handler)
    return {
        "label": label,
        "case": case,
        "observed_score": int(target.total_score),
        "observed_invalid_actions": int(target.lifetime_statistics["invalid"]),
        "trace_count": len(trace),
        "categories": dict(sorted(categories.items())),
        "all_invalid_trace": trace,
    }


def freeze_adjudicated_checkpoint(protocol: dict, protocol_hash: str, source_report_hash: str) -> dict:
    source = ROOT / protocol["checkpoint_inventory"]["source-r2"]["path"]
    target = ROOT / protocol["adjudicated_checkpoint"]["path"]
    manifest_path = ROOT / protocol["adjudicated_checkpoint"]["manifest_path"]
    source_hash = sha256_file(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != source_hash:
            raise RuntimeError(f"refusing incompatible adjudicated checkpoint: {target}")
    else:
        temporary = target.with_suffix(target.suffix + ".tmp")
        if temporary.exists():
            raise RuntimeError(f"refusing stale checkpoint temporary file: {temporary}")
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != source_hash:
            raise RuntimeError("adjudicated Task3 checkpoint copy hash mismatch")
        temporary.replace(target)
    manifest = {
        "schema_version": 1,
        "kind": ADJUDICATION_KIND,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_hash,
        "formal_s122000_confirmation_passed": False,
        "formal_s122000_report_sha256": source_report_hash,
        "adjudication": "accept_source_r2_for_task4_progression_after_dynamic_collision_boundary_audit",
        "waiver_scope": "only the +0.05 invalid-actions-per-round safety delta; all efficacy and other safety gates passed",
        "source_checkpoint": {"path": relative(source), "sha256": source_hash},
        "adjudicated_checkpoint": {"path": relative(target), "sha256": sha256_file(target)},
        "training_started": False,
        "task4_started": False,
    }
    existing = load_completed(manifest_path, ADJUDICATION_KIND)
    if existing is None:
        atomic_json(manifest_path, manifest)
    else:
        left = {key: value for key, value in existing.items() if key != "completed_at_utc"}
        right = {key: value for key, value in manifest.items() if key != "completed_at_utc"}
        if left != right:
            raise RuntimeError("adjudication manifest drift")
    return {
        "path": relative(target),
        "sha256": sha256_file(target),
        "manifest_path": relative(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def execute(protocol_path: Path, protocol: dict, protocol_hash: str, source_report: dict) -> dict:
    output = ROOT / protocol["report_path"]
    existing = load_completed(output, AUDIT_KIND)
    if existing is not None:
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("completed boundary audit protocol drift")
        return existing
    runs = [run_case(protocol, label, case) for label in protocol["labels"] for case in protocol["cases"]]
    expected = protocol["expected_reproduction"]
    reproduction = all(
        run["observed_invalid_actions"] == int(expected[run["label"]][str(run["case"]["world_seed"])])
        and run["trace_count"] == run["observed_invalid_actions"]
        for run in runs
    )
    source_events = [event for run in runs if run["label"] == "source-r2" for event in run["all_invalid_trace"]]
    all_source_dynamic_collisions = bool(source_events) and all(
        event["decision_legal"] and event["category"] == "simultaneous_agent_collision"
        for event in source_events
    )
    confirmed = reproduction and len(source_events) == 10 and all_source_dynamic_collisions
    result = {
        "decision": "dynamic_collision_boundary_confirmed" if confirmed else "invalid_boundary_not_explained_stop",
        "exact_formal_count_reproduction": reproduction,
        "source_invalid_count": len(source_events),
        "all_source_invalids_legal_at_decision": bool(source_events) and all(event["decision_legal"] for event in source_events),
        "all_source_invalids_same_step_agent_collisions": all_source_dynamic_collisions,
        "formal_s122000_confirmation_passed": False,
        "formal_s122000_decision_unchanged": source_report["result"]["decision"],
        "efficacy_data_added": False,
        "training_started": False,
        "task4_started": False,
        "task3_adjudicated_for_task4_progression": confirmed,
        "adjudicated_checkpoint": None,
    }
    if confirmed:
        result["adjudicated_checkpoint"] = freeze_adjudicated_checkpoint(
            protocol, protocol_hash, protocol["source_report"]["sha256"],
        )
    report = {
        "schema_version": 1,
        "kind": AUDIT_KIND,
        "status": "completed",
        "completed_at_utc": utc_now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_path": relative(protocol_path),
        "protocol_sha256": protocol_hash,
        "source_report": protocol["source_report"],
        "runs": runs,
        "result": result,
        "awaiting_task4_design": True,
    }
    atomic_json(output, report)
    return report


def dry_run(protocol: dict) -> dict:
    return {
        "mode": "dry-run",
        "protocol_id": protocol["protocol_id"],
        "labels": protocol["labels"],
        "revealed_cases_only": [case["world_seed"] for case in protocol["cases"]],
        "total_diagnostic_rounds": len(protocol["labels"]) * sum(case["rounds"] for case in protocol["cases"]),
        "formal_decision_override": False,
        "efficacy_reselection_allowed": False,
        "training_started": False,
        "task4_started": False,
        "conditional_adjudicated_freeze": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    protocol, protocol_hash, source_report = load_protocol(protocol_path)
    if not args.execute:
        print(json.dumps(dry_run(protocol), indent=2, sort_keys=True))
        return 0
    report = execute(protocol_path.resolve(), protocol, protocol_hash, source_report)
    print(json.dumps({
        "status": report["status"],
        "report": relative(ROOT / protocol["report_path"]),
        "report_sha256": sha256_file(ROOT / protocol["report_path"]),
        **report["result"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
