"""Run the fixed-seed, no-training v10 Phase-1 tactical search smoke test."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from agent_code.model_a_v10.tactics import evaluate_tactics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result = {"phase": "v10-phase1", "created_at": datetime.now(timezone.utc).isoformat(),
              "configuration": {"simulations": args.simulations, "seeds": [case.seed for case in __import__(
                  "agent_code.model_a_v10.tactics", fromlist=["tactical_cases"]).tactical_cases()]},
              "machine": {"python": sys.version, "platform": platform.platform(), "git_commit": commit},
              "tactics": evaluate_tactics(args.simulations)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["tactics"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
