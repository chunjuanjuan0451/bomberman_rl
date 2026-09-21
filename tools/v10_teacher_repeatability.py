"""Pre-registered Task 2 teacher repeatability diagnostic; dry-run by default."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_v10.interfaces import ACTIONS, MOVES  # noqa: E402
from agent_code.model_a_v10.simulator import SimState, step  # noqa: E402
from tools.v10_classic_residual_train import (  # noqa: E402
    build_teacher,
    initial_state,
    observer_game_state,
    policy_array,
    state_from_game_state,
    teacher_visit_policy,
    visit_policy,
)


PREREGISTRATION_KIND = "v10-task2-teacher-repeatability-preregistration-v2"
REPORT_KIND = "v10-task2-teacher-repeatability-report-v2"
EXPECTED_RUN_ID = "v10.8.1-task2-teacher-repeatability-s106120"
EXPECTED_SEEDS = list(range(106120, 106132))
EXPECTED_STRATA = (
    "wait_target", "bomb_target", "confident_correction_move", "preservation_move")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_default(value):
    """Canonicalize NumPy values produced by the simulator and diagnostics."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(
        payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    temporary.replace(path)


def _policy(visits: dict[str, int]) -> np.ndarray:
    return visit_policy(visits).astype(np.float64)


def _top_action(values: np.ndarray) -> str:
    return ACTIONS[int(np.argmax(values))]


def _top_margin(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    return float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else 1.0


def _mode_fraction(values: list[str]) -> tuple[str, float]:
    counts = Counter(values)
    mode = max(ACTIONS, key=lambda action: (counts[action], -ACTIONS.index(action)))
    return mode, counts[mode] / max(len(values), 1)


def jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    """Natural-log Jensen-Shannon divergence with exact zero handling."""
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("Jensen-Shannon inputs must be same-shape vectors")
    if np.any(p < 0.0) or np.any(q < 0.0) or p.sum() <= 0.0 or q.sum() <= 0.0:
        raise ValueError("Jensen-Shannon inputs must be non-negative distributions")
    p, q = p / p.sum(), q / q.sum()
    midpoint = 0.5 * (p + q)

    def kl(values: np.ndarray) -> float:
        mask = values > 0.0
        return float(np.sum(values[mask] * np.log(values[mask] / midpoint[mask])))

    return 0.5 * (kl(p) + kl(q))


def classify_stratum(target_action: str, planner_action: str,
                     search_margin: float, confidence_margin: float) -> str | None:
    """Apply the frozen, mutually exclusive state-stratification rule."""
    if target_action == "WAIT":
        return "wait_target"
    if target_action == "BOMB":
        return "bomb_target"
    if (target_action in MOVES and target_action != planner_action
            and search_margin >= confidence_margin):
        return "confident_correction_move"
    if target_action in MOVES and target_action == planner_action:
        return "preservation_move"
    return None


def observer_state_sha256(state: SimState) -> str:
    public = observer_game_state(state)
    serializable = {
        "round": public["round"], "step": public["step"],
        "field": public["field"].tolist(), "self": public["self"],
        "others": public["others"], "bombs": public["bombs"],
        "coins": public["coins"],
        "explosion_map": public["explosion_map"].tolist(),
    }
    encoded = json.dumps(
        serializable, sort_keys=True, separators=(",", ":"),
        default=_json_default)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _candidate(seed: int, state: SimState, target, result, transform,
               confidence_margin: float) -> dict:
    raw_target = _policy(result.root_visits)
    protected_action, protected = transform.protect_action(result.action)
    supervised = teacher_visit_policy(result.root_visits, protected_action, protected)
    planner = policy_array(result.raw_prior).astype(np.float64)
    target_action = _top_action(supervised)
    planner_action = _top_action(planner)
    return {
        "seed": seed,
        "step": state.step_count,
        "state": state.copy(),
        "observer_state_sha256": observer_state_sha256(state),
        "crate_target": transform.crate_target,
        "reference_search_seed": seed * 1000 + state.step_count,
        "reference_raw_visit_policy": raw_target,
        "reference_supervised_policy": supervised.astype(np.float64),
        "reference_search_action": result.action,
        "reference_protected_action": protected_action,
        "reference_target_action": target_action,
        "reference_protected": bool(protected),
        "planner_policy": planner,
        "planner_action": planner_action,
        "search_margin": _top_margin(raw_target),
        "stratum": classify_stratum(
            target_action, planner_action, _top_margin(raw_target), confidence_margin),
    }


def collect_candidates(config: dict, base: dict) -> tuple[list[dict], list[dict]]:
    """Roll out the frozen teacher and retain exact labeled observer states."""
    candidates, episodes = [], []
    for seed in config["episode_seeds"]:
        network, planner, transform, searcher = build_teacher(base)
        state = initial_state(seed, int(base["max_steps"]), 0)
        episode_candidates = 0
        while not state.ended and state.agents[0].alive:
            observed = state_from_game_state(observer_game_state(state))
            base_policy = network.evaluate(observed, 0).policy
            target = transform.prepare(observed, 0, round_id=1)
            plan = planner.plan(observed, 0, base_policy, crate_target=target)
            if state.step_count % int(config["state_sample_stride"]) == 0:
                result = searcher.search(
                    observed, 0, simulations=int(config["reference_simulations"]),
                    seed=seed * 1000 + state.step_count, deadline_ms=None)
                if result.simulations != int(config["reference_simulations"]):
                    raise RuntimeError("reference teacher did not finish its fixed budget")
                row = _candidate(
                    seed, observed, target, result, transform,
                    float(config["confidence_margin"]))
                if row["stratum"] is not None:
                    candidates.append(row)
                    episode_candidates += 1
                # Advance the episode with the actual protected teacher action.
                # The visit-policy argmax can differ under a visit-count tie.
                action = row["reference_protected_action"]
            else:
                action = max(plan.prior, key=plan.prior.get)
            state = step(state, [action], (0,))
        episodes.append({
            "seed": seed, "steps": state.step_count,
            "score": state.agents[0].score, "survived": state.agents[0].alive,
            "crates_remaining": int(np.count_nonzero(state.field == 1)),
            "eligible_states": episode_candidates,
        })
    return candidates, episodes


def select_states(candidates: list[dict], config: dict) -> tuple[list[dict], dict[str, int]]:
    quota = int(config["states_per_stratum"])
    selected, counts = [], {}
    for stratum in EXPECTED_STRATA:
        matches = [row for row in candidates if row["stratum"] == stratum]
        counts[stratum] = len(matches)
        selected.extend(matches[:quota])
    return selected, counts


def _run_search(searcher, transform, row: dict, simulations: int, seed: int) -> dict:
    transform.set_crate_target(row["crate_target"])
    result = searcher.search(
        row["state"], 0, simulations=simulations, seed=seed, deadline_ms=None)
    if result.simulations != simulations:
        raise RuntimeError("repeatability search did not finish its fixed budget")
    raw = _policy(result.root_visits)
    protected_action, protected = transform.protect_action(result.action)
    supervised = teacher_visit_policy(
        result.root_visits, protected_action, protected).astype(np.float64)
    return {
        "raw_policy": raw,
        "supervised_policy": supervised,
        "search_action": result.action,
        "protected_action": protected_action,
        "protected": bool(protected),
        "margin": _top_margin(raw),
    }


def diagnose_state(row: dict, ordinal: int, config: dict, base: dict) -> dict:
    _, _, transform, searcher = build_teacher(base)
    repeat_results = []
    seed_base = int(config["repeat_search_seed_base"]) + ordinal * 100
    for repeat in range(int(config["repeat_count"])):
        repeat_results.append(_run_search(
            searcher, transform, row, int(config["reference_simulations"]),
            seed_base + repeat))

    raw_policies = np.stack([item["raw_policy"] for item in repeat_results])
    mean_raw = raw_policies.mean(axis=0)
    search_actions = [item["search_action"] for item in repeat_results]
    protected_actions = [item["protected_action"] for item in repeat_results]
    target_actions = [_top_action(item["supervised_policy"])
                      for item in repeat_results]
    search_mode, search_modal_fraction = _mode_fraction(search_actions)
    protected_mode, protected_modal_fraction = _mode_fraction(protected_actions)
    target_mode, target_modal_fraction = _mode_fraction(target_actions)
    planner_action = row["planner_action"]
    confidence_margin = float(config["confidence_margin"])
    correction_labels = [
        item["protected_action"] != planner_action
        and item["margin"] >= confidence_margin
        for item in repeat_results
    ]
    correction_modal_fraction = max(
        sum(correction_labels), len(correction_labels) - sum(correction_labels)
    ) / len(correction_labels)

    budget_results = {}
    for simulations in config["budget_probes"]:
        if simulations == int(config["reference_simulations"]):
            probe = repeat_results[0]
        else:
            probe = _run_search(
                searcher, transform, row, int(simulations),
                seed_base + 50 + int(simulations))
        budget_results[str(simulations)] = {
            "raw_policy": probe["raw_policy"].tolist(),
            "search_action": probe["search_action"],
            "protected_action": probe["protected_action"],
            "margin": probe["margin"],
            "js_to_mean_128": jensen_shannon(probe["raw_policy"], mean_raw),
            "protected_action_matches_128_mode": probe["protected_action"] == protected_mode,
            "confident_correction_matches_128_mode": (
                (probe["protected_action"] != planner_action
                 and probe["margin"] >= confidence_margin)
                == (sum(correction_labels) >= len(correction_labels) / 2)
            ),
        }

    return {
        "seed": row["seed"], "step": row["step"], "stratum": row["stratum"],
        "observer_state_sha256": row["observer_state_sha256"],
        "crate_target": (None if row["crate_target"] is None else list(row["crate_target"])),
        "planner_action": planner_action,
        "reference_search_seed": row["reference_search_seed"],
        "reference_target_action": row["reference_target_action"],
        "reference_search_margin": row["search_margin"],
        "repeat_search_seeds": list(range(seed_base, seed_base + len(repeat_results))),
        "repeat_raw_policies": [item["raw_policy"].tolist() for item in repeat_results],
        "repeat_search_actions": search_actions,
        "repeat_protected_actions": protected_actions,
        "repeat_target_actions": target_actions,
        "mean_raw_policy": mean_raw.tolist(),
        "mean_js_to_consensus": float(np.mean([
            jensen_shannon(policy, mean_raw) for policy in raw_policies])),
        "max_js_to_consensus": float(np.max([
            jensen_shannon(policy, mean_raw) for policy in raw_policies])),
        "search_action_mode": search_mode,
        "search_action_modal_fraction": search_modal_fraction,
        "protected_action_mode": protected_mode,
        "protected_action_modal_fraction": protected_modal_fraction,
        "target_action_mode": target_mode,
        "target_action_modal_fraction": target_modal_fraction,
        "reference_target_matches_repeat_mode": (
            row["reference_target_action"] == target_mode),
        "confident_correction_label_modal_fraction": correction_modal_fraction,
        "budget_probes": budget_results,
    }


def _summary(rows: list[dict]) -> dict:
    if not rows:
        return {"states": 0}

    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows]))

    divergences = [row["mean_js_to_consensus"] for row in rows]
    return {
        "states": len(rows),
        "mean_js_to_consensus": float(np.mean(divergences)),
        "p90_js_to_consensus": float(np.quantile(divergences, 0.9)),
        "maximum_state_js_to_consensus": float(np.max(divergences)),
        "mean_search_action_modal_fraction": mean("search_action_modal_fraction"),
        "mean_protected_action_modal_fraction": mean("protected_action_modal_fraction"),
        "mean_target_action_modal_fraction": mean("target_action_modal_fraction"),
        "reference_target_consensus_rate": float(np.mean([
            row["reference_target_matches_repeat_mode"] for row in rows])),
        "mean_confident_correction_label_modal_fraction": mean(
            "confident_correction_label_modal_fraction"),
    }


def summarize(rows: list[dict], config: dict, available: dict[str, int]) -> tuple[dict, dict]:
    by_stratum = {
        name: _summary([row for row in rows if row["stratum"] == name])
        for name in EXPECTED_STRATA
    }
    overall = _summary(rows)
    if not rows:
        summary = {
            "available_states_by_stratum": available,
            "selected_states": 0,
            "overall": overall,
            "by_stratum": by_stratum,
            "budget_stability": {},
        }
        decision = "insufficient_stratum_coverage"
        return summary, {
            "passed": False, "decision": decision, "coverage_passed": False,
            "seed_repeatability_checks": {}, "budget_convergence_checks": {},
            "next_step": config["decision_branches"][decision],
        }
    budget = {}
    for simulations in config["budget_probes"]:
        key = str(simulations)
        budget[key] = {
            "mean_js_to_mean_128": float(np.mean([
                row["budget_probes"][key]["js_to_mean_128"] for row in rows])),
            "p90_js_to_mean_128": float(np.quantile([
                row["budget_probes"][key]["js_to_mean_128"] for row in rows], 0.9)),
            "protected_action_agreement_with_128": float(np.mean([
                row["budget_probes"][key]["protected_action_matches_128_mode"]
                for row in rows])),
            "confident_correction_agreement_with_128": float(np.mean([
                row["budget_probes"][key]["confident_correction_matches_128_mode"]
                for row in rows])),
        }
    summary = {
        "available_states_by_stratum": available,
        "selected_states": len(rows),
        "overall": overall,
        "by_stratum": by_stratum,
        "budget_stability": budget,
    }
    gates = config["decision_gate"]
    coverage = all(
        by_stratum[name]["states"] >= int(gates["minimum_states_per_stratum"])
        for name in EXPECTED_STRATA)
    seed_checks = {
        "mean_js": overall["mean_js_to_consensus"] <= gates["maximum_mean_js"],
        "p90_js": overall["p90_js_to_consensus"] <= gates["maximum_p90_js"],
        "search_action_repeatability": overall["mean_search_action_modal_fraction"]
        >= gates["minimum_search_action_modal_fraction"],
        "protected_action_repeatability": overall["mean_protected_action_modal_fraction"]
        >= gates["minimum_protected_action_modal_fraction"],
        "target_action_repeatability": overall["mean_target_action_modal_fraction"]
        >= gates["minimum_target_action_modal_fraction"],
        "correction_label_repeatability": overall[
            "mean_confident_correction_label_modal_fraction"]
        >= gates["minimum_correction_label_modal_fraction"],
        "wait_stability": by_stratum["wait_target"].get(
            "reference_target_consensus_rate", 0.0) >= gates["minimum_wait_stability"],
        "bomb_stability": by_stratum["bomb_target"].get(
            "reference_target_consensus_rate", 0.0) >= gates["minimum_bomb_stability"],
        "confident_correction_stability": by_stratum["confident_correction_move"].get(
            "reference_target_consensus_rate", 0.0)
        >= gates["minimum_confident_correction_stability"],
    }
    convergence_budget = str(gates["convergence_probe_simulations"])
    convergence = budget[convergence_budget]
    budget_checks = {
        "p90_js": convergence["p90_js_to_mean_128"]
        <= gates["maximum_convergence_p90_js"],
        "protected_action_agreement": convergence[
            "protected_action_agreement_with_128"]
        >= gates["minimum_convergence_action_agreement"],
        "correction_label_agreement": convergence[
            "confident_correction_agreement_with_128"]
        >= gates["minimum_convergence_correction_agreement"],
    }
    if not coverage:
        decision = "insufficient_stratum_coverage"
    elif not all(seed_checks.values()):
        decision = "search_seed_instability"
    elif not all(budget_checks.values()):
        decision = "search_budget_instability"
    else:
        decision = "teacher_targets_repeatable"
    gate = {
        "passed": decision == "teacher_targets_repeatable",
        "decision": decision,
        "coverage_passed": coverage,
        "seed_repeatability_checks": seed_checks,
        "budget_convergence_checks": budget_checks,
        "next_step": config["decision_branches"][decision],
    }
    return summary, gate


def load_preregistration(path: Path, *, allow_existing_output: bool = False
                         ) -> tuple[dict, dict]:
    protocol = json.loads(path.read_text())
    if protocol.get("kind") != PREREGISTRATION_KIND:
        raise ValueError("not a teacher repeatability preregistration")
    exact = {
        "run_id": EXPECTED_RUN_ID,
        "supersedes_failed_run": "v10.8-task2-teacher-repeatability-s106100",
        "episode_seeds": EXPECTED_SEEDS,
        "opponent_count": 0,
        "state_sample_stride": 4,
        "reference_simulations": 128,
        "repeat_count": 8,
        "budget_probes": [64, 128, 256],
        "states_per_stratum": 24,
        "confidence_margin": 0.15,
        "automatic_training": False,
    }
    for key, expected in exact.items():
        if protocol.get(key) != expected:
            raise ValueError(f"repeatability protocol requires {key}={expected!r}")
    if tuple(protocol.get("strata", ())) != EXPECTED_STRATA:
        raise ValueError("repeatability strata do not match the frozen protocol")
    for relative, expected in protocol.get("source_sha256", {}).items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"source hash mismatch: {relative}")
    base_path = ROOT / protocol["base_teacher_config"]
    if sha256(base_path) != protocol["base_teacher_config_sha256"]:
        raise ValueError("base teacher config hash mismatch")
    base = json.loads(base_path.read_text())
    required_base = {
        "opponent_count": 0, "teacher_search_simulations": 128,
        "search_horizon_plies": 8, "teacher_productive_bomb_protection": True,
        "teacher_persistent_target": True, "teacher_crate_progress_weight": 0.08,
        "teacher_crate_frontier_weight": 0.32,
    }
    for key, expected in required_base.items():
        if base.get(key) != expected:
            raise ValueError(f"base teacher config requires {key}={expected!r}")
    output = ROOT / protocol["output_path"]
    if output.exists() and not allow_existing_output:
        raise FileExistsError(f"refusing to overwrite diagnostic report: {output}")
    return protocol, base


def run(path: Path, execute: bool) -> int:
    protocol, base = load_preregistration(path)
    print(json.dumps({
        "mode": "execute" if execute else "dry-run",
        "run_id": protocol["run_id"],
        "episode_seeds": protocol["episode_seeds"],
        "repeat_search_seed_base": protocol["repeat_search_seed_base"],
        "strata": protocol["strata"],
        "states_per_stratum": protocol["states_per_stratum"],
        "reference_simulations": protocol["reference_simulations"],
        "repeat_count": protocol["repeat_count"],
        "budget_probes": protocol["budget_probes"],
        "automatic_training": protocol["automatic_training"],
        "output_path": protocol["output_path"],
        "preregistration_sha256": sha256(path),
    }, indent=2))
    if not execute:
        return 0

    candidates, episodes = collect_candidates(protocol, base)
    selected, available = select_states(candidates, protocol)
    rows = [diagnose_state(row, ordinal, protocol, base)
            for ordinal, row in enumerate(selected)]
    summary, gate = summarize(rows, protocol, available)
    payload = {
        "kind": REPORT_KIND,
        "run_id": protocol["run_id"],
        "preregistration": str(path.relative_to(ROOT)),
        "preregistration_sha256": sha256(path),
        "base_teacher_config": protocol["base_teacher_config"],
        "base_teacher_config_sha256": protocol["base_teacher_config_sha256"],
        "episode_seeds": protocol["episode_seeds"],
        "repeat_search_seed_base": protocol["repeat_search_seed_base"],
        "episodes": episodes,
        "summary": summary,
        "decision_gate": gate,
        "states": rows,
        "automatic_training_started": False,
    }
    atomic_json(ROOT / protocol["output_path"], payload)
    print(json.dumps({"summary": summary, "decision_gate": gate}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--execute", action="store_true",
                        help="run the preregistered diagnostic; default is dry-run")
    args = parser.parse_args(argv)
    return run(args.config.resolve(), args.execute)


if __name__ == "__main__":
    raise SystemExit(main())
