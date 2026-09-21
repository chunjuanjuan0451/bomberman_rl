"""Recover the boundary-audit report after NumPy scalar JSON serialization failed.

This wrapper does not alter the hash-bound diagnostic runner or its scientific
logic.  It only teaches the standard JSON encoder how to convert NumPy scalar
values before deterministically replaying the same already revealed cases.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


_standard_dumps = json.dumps


def _dumps_with_numpy_scalars(payload, *args, **kwargs):
    if "default" not in kwargs:
        kwargs["default"] = lambda value: value.item() if hasattr(value, "item") else str(value)
    return _standard_dumps(payload, *args, **kwargs)


json.dumps = _dumps_with_numpy_scalars

from tools.v4_task3_invalid_boundary_audit import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["--execute"]))
