"""Contract tests for the CNN Task2 continuation protocol."""

import hashlib
import json
from pathlib import Path

from tools.cnn_n8_task2 import load_protocol, validate_task1_lineage


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments/configs/model-a-cnn-n8-task2-s148000.json"


def test_cnn_task2_uses_the_preregistered_task1_common_milestone():
    protocol, _ = load_protocol(PROTOCOL)
    validate_task1_lineage(protocol)
    report_path = ROOT / protocol["task1_parent_report"]["path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["selection"]["selected_common_milestone"] == 400
    for replica, parent in protocol["parents"].items():
        assert parent == report["selected_checkpoints"][replica]


def test_cnn_task2_seeds_are_unique_and_new_relative_to_task1():
    protocol, _ = load_protocol(PROTOCOL)
    values = []
    for replica in protocol["task2"]["replicas"].values():
        values.extend((replica["world_seed"], replica["agent_seed"]))
    for suite in protocol["validation"].values():
        if isinstance(suite, dict):
            for case in suite.get("cases", []):
                values.extend((case["world_seed"], case["agent_seed"]))
    task1_text = (ROOT / "experiments/configs/model-a-cnn-n8-task1-s147000.json").read_text(encoding="utf-8")
    assert len(values) == len(set(values))
    assert not any(str(seed) in task1_text for seed in values)


def test_cnn_task2_parent_report_hash_is_frozen():
    protocol, _ = load_protocol(PROTOCOL)
    path = ROOT / protocol["task1_parent_report"]["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == protocol["task1_parent_report"]["sha256"]
