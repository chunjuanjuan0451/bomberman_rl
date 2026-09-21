"""Freeze the s130000 lingering-flame failure and its corrected horizon."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import settings as s  # noqa: E402


FAILED_MANIFEST = ROOT / "experiments/logs/runs/model-a-v4-task4-bomb-credit-s130000/control/r1.json"
FAILED_MANIFEST_SHA256 = "519e008069c71ccce51b3d6cc9a9a11d657ae71ebb07b07e602e43ae562552ee"
AGENT_LOG = ROOT / "agent_code/model_a_v4_bomb_credit/logs/model_a_v4_bomb_credit.log"
AGENT_LOG_SHA256 = "c25ef6f0f86696962b531935dc00186999e1ae0dd7b01388c175cbec1d9cb0b4"
GAME_LOG = ROOT / "logs/game.log"
GAME_LOG_SHA256 = "11cb5eb76fa0c6d22fb76bfaab27e09cb009e8f7e1dedc96162cef207f0e6a64"
ENVIRONMENT_SOURCE = ROOT / "environment.py"
ENVIRONMENT_SOURCE_SHA256 = "75b40b8bc5dd1fec7d584ba1dc3a15de060e0bfe1afcfed769d553f5d5d0fd36"
AGENT_RUNTIME_SOURCE = ROOT / "agents.py"
AGENT_RUNTIME_SOURCE_SHA256 = "582a96cc27ba7dd246f248b316e7808d82a07b29fd0139dc44a6ec9a696a49a8"
OUTPUT = ROOT / "experiments/logs/diagnostics/model-a-v4-bomb-credit-lifecycle-s131000.json"
PREVIOUS_OUTPUT_SHA256 = "390b2a5c3119922ec8324743f6dd2d4647efd4828629909ab3b3e860650743b2"


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


def build_report() -> dict:
    sources = {
        FAILED_MANIFEST: FAILED_MANIFEST_SHA256,
        AGENT_LOG: AGENT_LOG_SHA256,
        GAME_LOG: GAME_LOG_SHA256,
        ENVIRONMENT_SOURCE: ENVIRONMENT_SOURCE_SHA256,
        AGENT_RUNTIME_SOURCE: AGENT_RUNTIME_SOURCE_SHA256,
    }
    for path, expected in sources.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"s130000 failure evidence changed: {path}")
    manifest = json.loads(FAILED_MANIFEST.read_text(encoding="utf-8"))
    if not (
        manifest.get("status") == "failed"
        and manifest.get("arm") == "control"
        and manifest.get("replica") == "r1"
        and manifest.get("exit_code") == 1
    ):
        raise RuntimeError("s130000 failure manifest semantics changed")
    agent_log = AGENT_LOG.read_text(encoding="utf-8")
    error = "official KILLED_SELF at 100/63 did not uniquely match step-4 BOMB"
    if error not in agent_log:
        raise RuntimeError("s130000 exact attribution failure is missing")
    game_log = GAME_LOG.read_text(encoding="utf-8")
    round_start = game_log.rfind("STARTING ROUND #100")
    round_end = game_log.rfind("WRAPPING UP ROUND #100")
    if round_start < 0 or round_end <= round_start:
        raise RuntimeError("s130000 round-100 log segment is incomplete")
    segment = game_log[round_start:round_end]
    evidence = (
        "STARTING STEP 58",
        "Agent <model_a_v4_bomb_credit> drops bomb at (np.int64(5), np.int64(6))",
        "STARTING STEP 62",
        "Agent <model_a_v4_bomb_credit>'s bomb at (np.int64(5), np.int64(6)) explodes",
        "Agent <seeded_rule_based_agent_0> blown up by agent <model_a_v4_bomb_credit>'s bomb",
        "STARTING STEP 63",
        "Agent <model_a_v4_bomb_credit> blown up by own bomb",
    )
    positions = [segment.find(item) for item in evidence]
    if any(value < 0 for value in positions) or positions != sorted(positions):
        raise RuntimeError("s130000 lingering-flame causal sequence changed")
    if s.BOMB_TIMER != 4 or s.EXPLOSION_TIMER != 2:
        raise RuntimeError("official bomb or explosion lifecycle changed")
    return {
        "schema_version": 1,
        "kind": "model-a-v4-bomb-credit-lifecycle-audit",
        "status": "completed",
        "failed_protocol_id": "model-a-v4-task4-bomb-credit-s130000",
        "failure": {
            "arm": "control",
            "replica": "r1",
            "round": 100,
            "bomb_step": 58,
            "explosion_and_kill_step": 62,
            "lingering_self_kill_step": 63,
            "exception": error,
        },
        "source_evidence": {
            "failed_manifest": {"path": str(FAILED_MANIFEST.relative_to(ROOT)), "sha256": FAILED_MANIFEST_SHA256},
            "agent_log": {"path": str(AGENT_LOG.relative_to(ROOT)), "sha256": AGENT_LOG_SHA256},
            "game_log": {"path": str(GAME_LOG.relative_to(ROOT)), "sha256": GAME_LOG_SHA256},
            "environment_source": {"path": str(ENVIRONMENT_SOURCE.relative_to(ROOT)), "sha256": ENVIRONMENT_SOURCE_SHA256},
            "agent_runtime_source": {"path": str(AGENT_RUNTIME_SOURCE.relative_to(ROOT)), "sha256": AGENT_RUNTIME_SOURCE_SHA256},
        },
        "environment_semantics": {
            "bomb_timer": s.BOMB_TIMER,
            "explosion_timer": s.EXPLOSION_TIMER,
            "explosion_step_lag": 4,
            "last_dangerous_flame_step_lag": 5,
            "allowed_official_outcome_lags": [4, 5],
            "minimum_complete_return_horizon": 6,
            "terminal_event_delivery": "dead agents skip per-step callbacks; later owned-bomb events are delivered at end_of_round on the last terminal transition",
        },
        "checks": {
            "s130000_preserved_as_failed": True,
            "round_100_sequence_is_ordered": True,
            "lag_four_explosion_observed": True,
            "lag_five_lingering_self_kill_observed": True,
            "six_transition_horizon_required": True,
            "posthumous_owned_bomb_kill_requires_terminal_deferred_attribution": True,
        },
        "decision": "bomb_outcome_lags_four_and_five_confirmed",
        "passed": True,
        "new_game_rounds": 0,
        "policy_updates": 0,
    }


def dry_run() -> dict:
    return {
        "mode": "dry-run",
        "failed_manifest_sha256": sha256_file(FAILED_MANIFEST),
        "agent_log_sha256": sha256_file(AGENT_LOG),
        "game_log_sha256": sha256_file(GAME_LOG),
        "allowed_official_outcome_lags": [4, 5],
        "minimum_complete_return_horizon": 6,
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
            if sha256_file(OUTPUT) != PREVIOUS_OUTPUT_SHA256:
                raise RuntimeError("refusing to overwrite changed lifecycle audit")
            atomic_json(OUTPUT, report)
    else:
        atomic_json(OUTPUT, report)
    print(json.dumps({
        "status": report["status"],
        "decision": report["decision"],
        "output": str(OUTPUT.relative_to(ROOT)),
        "sha256": sha256_file(OUTPUT),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
