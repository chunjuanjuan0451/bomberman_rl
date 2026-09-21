from pathlib import Path

from agent_code.model_a_v6.features import GLOBAL_SIZE, LOCAL_SHAPE
from agent_code.model_a_v6_ensemble.config import architecture_name, load_config
from agent_code.model_a_v6_ensemble.network import EnsembleDQN, torch
from tools.build_v6_ensemble import build

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/configs/v6s3-mean-residual-ensemble.json"


def _load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def test_ensemble_config_pins_three_unique_members():
    config, _, _ = load_config(CONFIG)
    assert config["aggregation"] == "mean"
    assert [member["training_seed"] for member in config["members"]] == [7501, 7502, 7503]
    assert len({member["sha256"] for member in config["members"]}) == 3


def test_bundle_contains_only_validated_residual_states():
    config, _, config_sha256 = load_config(CONFIG)
    bundle = build(CONFIG, ROOT / "unused.pt")
    assert bundle["architecture"] == architecture_name()
    assert bundle["config_sha256"] == config_sha256
    assert bundle["member_sha256"] == [member["sha256"] for member in config["members"]]
    assert len(bundle["residual_states"]) == 3
    assert all(set(state) == {"0.weight", "0.bias", "2.weight", "2.bias"} for state in bundle["residual_states"])


def test_ensemble_output_is_mean_of_member_residuals_and_bounded():
    config, _, _ = load_config(CONFIG)
    bundle = build(CONFIG, ROOT / "unused.pt")
    model = EnsembleDQN(member_count=3, delta_cap=config["delta_cap"])
    parent = _load(ROOT / config["parent_checkpoint"])
    model.base.load_state_dict(parent["online_net"], strict=True)
    for head, state in zip(model.residual_heads, bundle["residual_states"]):
        head.load_state_dict(state, strict=True)
    model.freeze_all()
    local = torch.randn(7, *LOCAL_SHAPE)
    global_features = torch.randn(7, GLOBAL_SIZE)
    with torch.no_grad():
        final, mean_delta, member_deltas = model(local, global_features, return_delta=True)
        base_q = model.base(local, global_features)
    assert torch.allclose(mean_delta, member_deltas.mean(dim=0), atol=1e-7, rtol=0)
    assert torch.allclose(final, base_q + member_deltas.mean(dim=0), atol=1e-6, rtol=0)
    assert torch.all(mean_delta <= 0.25) and torch.all(mean_delta >= -0.25)
    assert all(not parameter.requires_grad for parameter in model.parameters())
