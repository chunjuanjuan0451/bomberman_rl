"""Run the project's function-style tests without requiring pytest."""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    test_dir = Path(__file__).with_name("unit_tests")
    failures = []
    count = 0
    for path in sorted(test_dir.glob("test_*.py")):
        module_name = ".".join(path.relative_to(repository_root).with_suffix("").parts)
        module = importlib.import_module(module_name)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            count += 1
            try:
                function()
            except Exception as exc:  # Keep running so one invocation reports every failure.
                failures.append(f"{module_name}.{name}: {type(exc).__name__}: {exc}")
    if failures:
        raise SystemExit("\n".join([f"{len(failures)}/{count} tests failed:", *failures]))
    print(f"{count} tests passed.")


if __name__ == "__main__":
    main()
