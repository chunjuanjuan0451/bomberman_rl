"""Contract tests for the CNN Task3 peaceful-opponent course."""

import hashlib
import json
from pathlib import Path

from tools.cnn_n8_task3_peaceful import load_protocol, validate_task2_lineage


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments/configs/model-a-cnn-n8-task3-peaceful-s149000.json"


def test_cnn_task3p_uses_the_preregistered_task2_common_milestone():
    protocol, _ = load_protocol(PROTOCOL); validate_task2_lineage(protocol)
    report_path = ROOT / protocol["task2_parent_report"]["path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["selection"]["selected_common_milestone"] == 800
    for replica, parent in protocol["parents"].items():
        assert parent == report["selected_checkpoints"][replica]


def test_cnn_task3p_seeds_are_unique_and_new():
    protocol, _ = load_protocol(PROTOCOL); values = []
    for replica in protocol["task3_peaceful"]["replicas"].values():
        values.extend((replica["world_seed"], replica["agent_seed"], replica["opponent_seed"]))
    for suite in protocol["validation"].values():
        if isinstance(suite, dict):
            for case in suite.get("cases", []):
                values.extend(value for key, value in case.items() if key.endswith("_seed"))
    old = "".join(path.read_text(encoding="utf-8") for path in (
        ROOT / "experiments/configs/model-a-cnn-n8-task1-s147000.json",
        ROOT / "experiments/configs/model-a-cnn-n8-task2-s148000.json",
    ))
    assert len(values) == len(set(values))
    assert not any(str(seed) in old for seed in values)


def test_cnn_task3p_parent_report_hash_and_schedule_are_frozen():
    protocol, _ = load_protocol(PROTOCOL)
    path = ROOT / protocol["task2_parent_report"]["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == protocol["task2_parent_report"]["sha256"]
    assert protocol["learning"]["checkpoint_every_rounds"] == 150
    assert protocol["task3_peaceful"]["milestones"] == [150, 300, 450, 600]
