"""Report and validate the CPU-only tournament runtime.

This deliberately uses only the standard library plus packages already required
by the project, so it can run inside the untouched official Docker image.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path


GIB = 1024 ** 3


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _cgroup_limits() -> dict[str, object]:
    cpu_max = _read("/sys/fs/cgroup/cpu.max")
    memory_max = _read("/sys/fs/cgroup/memory.max")
    cpu_quota = None
    if cpu_max:
        quota, period = cpu_max.split()
        if quota != "max":
            cpu_quota = int(quota) / int(period)
    memory_bytes = None if memory_max in (None, "max") else int(memory_max)
    return {
        "cpu_max_raw": cpu_max,
        "cpu_quota": cpu_quota,
        "memory_max_raw": memory_max,
        "memory_max_bytes": memory_bytes,
    }


def _load_default_v4_checkpoint(torch_module) -> dict[str, object]:
    checkpoint = Path("agent_code/model_a_dqn/model_a.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    try:
        payload = torch_module.load(checkpoint, map_location="cpu", weights_only=True)
        load_mode = "weights_only=True"
    except TypeError:
        payload = torch_module.load(checkpoint, map_location="cpu")
        load_mode = "legacy"
    if not isinstance(payload, dict):
        raise ValueError("default v4 checkpoint has an unexpected schema")
    state_key = "model_state" if "model_state" in payload else "online_net"
    if state_key not in payload:
        raise ValueError("default v4 checkpoint has no recognized model state")
    tensors = payload[state_key]
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("default v4 checkpoint has no model_state tensors")
    return {
        "path": str(checkpoint),
        "load_mode": load_mode,
        "state_key": state_key,
        "tensor_count": len(tensors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-container-limits", action="store_true")
    parser.add_argument("--load-v4-checkpoint", action="store_true")
    args = parser.parse_args()

    import numpy
    import scipy
    import torch

    # The tournament grants one CPU thread.  Applying the setting before model
    # construction also prevents PyTorch from creating an oversized pool.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    limits = _cgroup_limits()
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "cgroup": limits,
    }
    if args.load_v4_checkpoint:
        report["v4_checkpoint"] = _load_default_v4_checkpoint(torch)

    errors = []
    if torch.get_num_threads() != 1:
        errors.append("PyTorch intra-op thread count is not 1")
    if torch.cuda.is_available():
        errors.append("CUDA unexpectedly available in CPU-only validation")
    if args.require_container_limits:
        quota = limits["cpu_quota"]
        memory = limits["memory_max_bytes"]
        if quota is None or float(quota) > 1.01:
            errors.append(f"container CPU quota is not limited to one CPU: {quota!r}")
        if memory is None or int(memory) > 8 * GIB:
            errors.append(f"container memory is not limited to 8 GiB: {memory!r}")

    report["status"] = "failed" if errors else "passed"
    report["errors"] = errors
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
