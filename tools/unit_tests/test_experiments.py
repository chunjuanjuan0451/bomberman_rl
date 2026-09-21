import json
import tempfile
from pathlib import Path

from experiments.evaluate import (evaluation_environment, metrics_by_agent,
                                  referenced_checkpoints, repository_state, sha256_file)
from experiments.plots import load_rows, summarize_rows
from experiments.train_model_a import load_config, resolve_checkpoint, training_command


def test_metrics_and_summary_round_trip():
    stats = {
        "by_agent": {
            "model_a_dqn": {
                "rounds": 100, "score": 250, "coins": 150, "kills": 20,
                "suicides": 25, "invalid": 3, "steps": 10_000, "time": 2.0,
            }
        }
    }
    metrics = metrics_by_agent(stats)
    assert metrics["model_a_dqn"]["score_per_round"] == 2.5
    assert metrics["model_a_dqn"]["mean_decision_time_ms"] == 0.2

    manifest = {
        "kind": "evaluation", "status": "completed", "run_id": "example",
        "variant": "safe-mask", "checkpoint": {"sha256": "abc"},
        "evaluation": {
            "target_agent": "model_a_dqn", "agents": ["model_a_dqn", "rule_based_agent"],
            "scenario": "classic", "seed": 4001,
        },
        "target_metrics": metrics["model_a_dqn"],
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "example.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        rows = load_rows([path])
        summary = summarize_rows(rows)
        assert len(summary) == 1
        assert summary[0]["score_per_round_mean"] == 2.5


def test_sha256_file():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "payload"
        path.write_bytes(b"abc")
        assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_repository_state_has_explicit_dirty_paths():
    # This assertion is independent of whether the developer's tree is clean.
    state = repository_state(Path("experiments/logs/evaluations"))
    assert state["dirty"] == bool(state["dirty_paths"])
    assert all("experiments/logs/evaluations/" not in path for path in state["dirty_paths"])
    status_paths = [line[3:].split(" -> ")[-1] for line in __import__("subprocess").run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        text=True, stdout=__import__("subprocess").PIPE, check=True,
    ).stdout.splitlines()]
    assert set(state["dirty_paths"]).issubset(set(status_paths))


def test_evaluation_environment_injects_exact_checkpoint_and_agent_seed():
    checkpoint = Path("experiments/checkpoints/candidate.pt")
    child, overrides = evaluation_environment(
        "model_a_dqn", 1234, checkpoint, {"UNCHANGED": "yes"},
    )
    assert child["UNCHANGED"] == "yes"
    assert child["MODEL_A_SEED"] == "1234"
    assert child["MODEL_A_CHECKPOINT_PATH"] == str(checkpoint.resolve())
    assert overrides["MODEL_A_CHECKPOINT_PATH"] == str(checkpoint.resolve())


def test_v6_evaluation_environment_requires_and_injects_config():
    checkpoint = Path("experiments/checkpoints/v6.pt")
    config = Path("experiments/configs/v6a-control.json")
    child, overrides = evaluation_environment(
        "model_a_v6", 4321, checkpoint, {}, agent_config=config,
    )
    assert child["MODEL_A_V6_SEED"] == "4321"
    assert child["MODEL_A_V6_CHECKPOINT_PATH"] == str(checkpoint.resolve())
    assert child["MODEL_A_V6_CONFIG_PATH"] == str(config.resolve())
    assert overrides == {
        "MODEL_A_V6_SEED": "4321",
        "MODEL_A_V6_CHECKPOINT_PATH": str(checkpoint.resolve()),
        "MODEL_A_V6_CONFIG_PATH": str(config.resolve()),
    }


def test_v10_evaluation_environment_requires_and_injects_config():
    checkpoint = Path("experiments/checkpoints/v10.npz")
    config = Path("experiments/configs/v10-official-value-only-s103004.json")
    child, overrides = evaluation_environment(
        "model_a_v10", 9876, checkpoint, {}, agent_config=config,
    )
    assert child["MODEL_A_V10_SEED"] == "9876"
    assert child["MODEL_A_V10_CHECKPOINT_PATH"] == str(checkpoint.resolve())
    assert child["MODEL_A_V10_CONFIG_PATH"] == str(config.resolve())
    assert set(overrides) == {
        "MODEL_A_V10_SEED", "MODEL_A_V10_CHECKPOINT_PATH", "MODEL_A_V10_CONFIG_PATH",
    }


def test_evaluation_provenance_tracks_residual_checkpoint():
    with tempfile.TemporaryDirectory(dir=".") as directory:
        root = Path(directory)
        checkpoint = root / "residual.pt"
        checkpoint.write_bytes(b"bounded residual")
        config = root / "agent.json"
        config.write_text(json.dumps({
            "residual_policy": {
                "enabled": True,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        }), encoding="utf-8")
        references = referenced_checkpoints(config)
        assert references == [{
            "role": "bounded_residual_root_policy",
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        }]


def test_training_runner_isolated_checkpoint_and_dry_configuration():
    with tempfile.TemporaryDirectory(dir=".") as directory:
        config_path = Path(directory) / "config.json"
        config = {
            "run_id": "model-a-infrastructure-test",
            "agent": "model_a_dqn",
            "agents": ["model_a_dqn", "rule_based_agent"],
            "scenario": "classic",
            "rounds": 10,
            "seed": 1001,
            "agent_seed": 2001,
            "checkpoint_path": f"{directory}/candidate.pt",
            "resume": False,
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")
        loaded = load_config(config_path)
        checkpoint = resolve_checkpoint(loaded)
        command = training_command(loaded, Path(directory) / "stats.json")
        assert checkpoint != Path("agent_code/model_a_dqn/model_a.pt").resolve()
        assert "--train" in command and command[command.index("--train") + 1] == "1"


def test_training_runner_rejects_default_checkpoint():
    config = {
        "run_id": "bad-default",
        "agents": ["model_a_dqn"],
        "scenario": "classic",
        "rounds": 1,
        "seed": 1,
        "agent_seed": 1,
        "checkpoint_path": "agent_code/model_a_dqn/model_a.pt",
    }
    try:
        resolve_checkpoint(config)
    except ValueError as exc:
        assert "tournament-default" in str(exc)
    else:
        raise AssertionError("Training runner must reject the default checkpoint")


def test_training_runner_accepts_enabled_v6_config_without_execution():
    path = Path("experiments/configs/v6a-control.json").resolve()
    config = load_config(path)
    checkpoint = resolve_checkpoint(config)
    command = training_command(config, Path("unused.stats.json"))
    assert config["agent"] == "model_a_v6"
    assert checkpoint.name == "model-a-v6a-control-s7101.pt"
    assert command[command.index("--agents") + 1] == "model_a_v6"


def test_training_runner_rejects_disabled_v6d_config():
    path = Path("experiments/configs/v6d-combined.json").resolve()
    try:
        load_config(path)
    except ValueError as exc:
        assert "disabled" in str(exc)
    else:
        raise AssertionError("v6d must stay disabled until a component passes")
