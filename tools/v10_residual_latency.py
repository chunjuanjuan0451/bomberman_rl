"""Official-container checkpoint compatibility and root-forward benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from agent_code.model_a_v10.residual_policy import load_residual_checkpoint, residual_forward


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=500)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite latency report: {args.output}")
    if args.samples < 100:
        raise ValueError("official latency report requires at least 100 samples")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if hasattr(torch.backends, "nnpack"):
        torch.backends.nnpack.set_flags(False)
    model, metadata = load_residual_checkpoint(args.checkpoint)
    generator = torch.Generator().manual_seed(103132)
    input_channels = int(getattr(model, "input_channels", 7))
    values = torch.randn((args.samples, input_channels, 33, 33), generator=generator)
    planner_prior = torch.full((args.samples, 6), 1.0 / 6.0)
    with torch.inference_mode():
        for index in range(20):
            residual_forward(model, values[index:index + 1],
                             planner_prior[index:index + 1])
        timings = []
        for index in range(args.samples):
            started = time.perf_counter()
            output = residual_forward(model, values[index:index + 1],
                                      planner_prior[index:index + 1])
            timings.append((time.perf_counter() - started) * 1000.0)
            if not torch.isfinite(output).all():
                raise FloatingPointError("non-finite official residual output")
    report = {
        "kind": "v10-residual-root-forward-latency-v1",
        "run_id": metadata["run_id"],
        "checkpoint_sha256": digest(args.checkpoint),
        "architecture": metadata["architecture"],
        "config_sha256": metadata["config_sha256"],
        "dataset_sha256": metadata["dataset_sha256"],
        "feature_schema": metadata["feature_schema"],
        "machine": platform.machine(),
        "torch": torch.__version__.split("+")[0],
        "torch_num_threads": torch.get_num_threads(),
        "samples": args.samples,
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "p99_ms": float(np.percentile(timings, 99)),
        "max_ms": float(max(timings)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
