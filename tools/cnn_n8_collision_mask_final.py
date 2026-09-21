"""Selection-free final confirmation of the frozen collision-mask candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.model_a_cnn_n8.network import ARCHITECTURE  # noqa: E402
from agent_code.model_a_dqn.callbacks import MODEL_ARCHITECTURE  # noqa: E402
from tools.cnn_n8_task1 import _metrics, _torch_load, atomic_json, relative, sha256_file, utc_now  # noqa: E402
from tools.cnn_n8_task4 import combine_metrics  # noqa: E402


DEFAULT_PROTOCOL = ROOT / "experiments/configs/model-a-cnn-n8-collision-mask-final-s160000.json"
IDENTITIES = ("collision-candidate", "original-cnn", "frozen-v4")
KIND_MANIFEST = "model-a-cnn-n8-collision-mask-final-case"
KIND_REPORT = "model-a-cnn-n8-collision-mask-final-report"


def load_protocol(path: Path) -> tuple[dict, Path, str]:
    path = path.resolve(); raw = path.read_bytes(); protocol = json.loads(raw)
    if protocol.get("kind") != "model-a-cnn-n8-collision-mask-ab" or protocol.get("protocol_id") != "model-a-cnn-n8-collision-mask-final-s160000":
        raise RuntimeError("wrong collision-mask final protocol")
    if not protocol.get("selection_free") or protocol.get("training_allowed") or protocol.get("automatic_followup"):
        raise RuntimeError("collision-mask final must be selection-free and evaluation-only")
    if tuple(protocol.get("identities", ())) != IDENTITIES:
        raise RuntimeError("collision-mask final identities changed")
    source = protocol["selection_source"]; source_path = ROOT / source["path"]
    if sha256_file(source_path) != source["sha256"]:
        raise RuntimeError("collision-mask selection source drift")
    source_report = json.loads(source_path.read_text(encoding="utf-8"))
    if not source_report["gate"]["passed"] or source_report["result"]["decision"] != source["required_decision"]:
        raise RuntimeError("collision-mask source did not select candidate")
    cnn_path = ROOT / protocol["checkpoint"]["path"]
    if sha256_file(cnn_path) != protocol["checkpoint"]["sha256"]:
        raise RuntimeError("CNN checkpoint drift")
    cnn = _torch_load(cnn_path)
    if any(cnn.get(k) != v for k, v in {"architecture": ARCHITECTURE, "stage": "task4", "replica": "r3", "stage_rounds": 800}.items()):
        raise RuntimeError("CNN checkpoint identity mismatch")
    v4_path = ROOT / protocol["frozen_v4"]["path"]
    if sha256_file(v4_path) != protocol["frozen_v4"]["sha256"]:
        raise RuntimeError("frozen-v4 checkpoint drift")
    v4 = _torch_load(v4_path)
    if v4.get("architecture") not in (None, MODEL_ARCHITECTURE):
        raise RuntimeError("frozen-v4 architecture mismatch")
    collection = protocol["collection"]
    if collection["rounds_per_block"] != 50 or collection["policy_updates"] != 0 or len(collection["blocks"]) != 8:
        raise RuntimeError("collision-mask final collection changed")
    values=[]
    for block in collection["blocks"].values():
        values.extend(v for k,v in block.items() if k.endswith("_seed"))
    if len(values) != 40 or len(set(values)) != 40 or min(values) < 160000:
        raise RuntimeError("collision-mask final requires 40 unique fresh seeds")
    return protocol, path, hashlib.sha256(raw).hexdigest()


def _opponent_env(block: dict) -> dict[str, str]:
    result = {"TASK4_RULE_SEED": str(block["rule_seed"])}
    if "coin_seed" in block: result["TASK3_OPPONENT_SEED"] = str(block["coin_seed"])
    if "random_seed" in block: result["SEEDED_RANDOM_AGENT_SEED"] = str(block["random_seed"])
    return result


def _cnn_env(protocol: dict, path: Path, label: str, identity: str, block: dict) -> dict[str, str]:
    arm = "treatment" if identity == "collision-candidate" else "control"
    result = {
        "CNN_COLLISION_AB_PROTOCOL": str(path), "CNN_COLLISION_AB_CASE": label,
        "CNN_COLLISION_AB_ARM": arm, "CNN_COLLISION_AB_AGENT_SEED": str(block["cnn_agent_seed"]),
        "CNN_COLLISION_AB_CHECKPOINT": str((ROOT / protocol["checkpoint"]["path"]).resolve()),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }
    result.update(_opponent_env(block)); return result


def _trace_path(protocol: dict, label: str, identity: str) -> Path:
    arm = "treatment" if identity == "collision-candidate" else "control"
    return ROOT / protocol["trace_directory"] / f"{label}-{arm}.json"


def _validate_trace(protocol: dict, protocol_hash: str, label: str, identity: str, path: Path) -> dict:
    payload=json.loads(path.read_text(encoding="utf-8")); arm="treatment" if identity=="collision-candidate" else "control"
    expected={"kind":"model-a-cnn-n8-collision-mask-ab-trace","protocol_sha256":protocol_hash,
              "case":label,"arm":arm,"checkpoint_sha256":protocol["checkpoint"]["sha256"],
              "rounds":50,"policy_updates":0}
    for key,value in expected.items():
        if payload.get(key)!=value: raise RuntimeError(f"invalid final trace {label}/{identity}: {key}")
    if identity=="original-cnn" and payload["counts"]["actions_changed"]:
        raise RuntimeError("original CNN trace contains overrides")
    return payload


def collect(protocol: dict, path: Path, protocol_hash: str, label: str, identity: str, block: dict) -> dict:
    manifest=ROOT/protocol["manifest_directory"]/f"{label}-{identity}.json"; stats=manifest.with_suffix(".stats.json")
    trace=None if identity=="frozen-v4" else _trace_path(protocol,label,identity)
    if manifest.exists():
        existing=json.loads(manifest.read_text(encoding="utf-8"))
        if existing.get("status")=="completed" and existing.get("protocol_sha256")==protocol_hash:
            if trace: _validate_trace(protocol,protocol_hash,label,identity,trace)
            return existing
        raise RuntimeError(f"orphaned final manifest: {relative(manifest)}")
    if stats.exists() or (trace and trace.exists()): raise RuntimeError(f"orphaned final artifact: {label}/{identity}")
    if identity=="frozen-v4":
        target="model_a_dqn"; overrides={"MODEL_A_CHECKPOINT_PATH":str((ROOT/protocol["frozen_v4"]["path"]).resolve()),
            "MODEL_A_SEED":str(block["v4_agent_seed"]),"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
        overrides.update(_opponent_env(block)); train_args=[]
    else:
        target="model_a_cnn_n8_collision_mask_ab"; overrides=_cnn_env(protocol,path,label,identity,block)
        train_args=["--train","1","--continue-without-training"]
    command=[sys.executable,"main.py","play","--agents",target,*block["opponents"],*train_args,"--no-gui",
             "--scenario","classic","--n-rounds","50","--seed",str(block["world_seed"]),"--save-stats",str(stats)]
    hashes={"cnn":sha256_file(ROOT/protocol["checkpoint"]["path"]),"v4":sha256_file(ROOT/protocol["frozen_v4"]["path"])}
    record={"schema_version":1,"kind":KIND_MANIFEST,"protocol_id":protocol["protocol_id"],"protocol_sha256":protocol_hash,
            "case_label":label,"identity":identity,"block":block,"status":"running","started_at_utc":utc_now(),
            "command":command,"environment_overrides":overrides,"checkpoint_sha256_before":hashes,"raw_stats":relative(stats)}
    atomic_json(manifest,record); child=os.environ.copy(); child.update(overrides)
    completed=subprocess.run(command,cwd=ROOT,env=child,check=False); record.update({"ended_at_utc":utc_now(),"exit_code":completed.returncode})
    try:
        if completed.returncode or not stats.is_file(): raise RuntimeError("final evaluation process failed")
        after={"cnn":sha256_file(ROOT/protocol["checkpoint"]["path"]),"v4":sha256_file(ROOT/protocol["frozen_v4"]["path"])}
        if after!=hashes: raise RuntimeError("final evaluation mutated checkpoint")
        raw=json.loads(stats.read_text(encoding="utf-8"))["by_agent"][target]
        record.update({"status":"completed","checkpoint_sha256_after":after,"target_metrics":_metrics(raw)})
        if trace:
            payload=_validate_trace(protocol,protocol_hash,label,identity,trace)
            record.update({"trace_path":relative(trace),"trace_sha256":sha256_file(trace),"trace_counts":payload["counts"]})
    except Exception as exc: record.update({"status":"failed","error":f"{type(exc).__name__}: {exc}"})
    atomic_json(manifest,record)
    if record["status"]!="completed": raise RuntimeError(f"final evaluation failed: {label}/{identity}: {record.get('error')}")
    return record


def _delta(a: dict,b: dict,key: str)->float: return a[key]-b[key]


def summarize(protocol: dict, protocol_hash: str, manifests: dict[str,dict[str,dict]]) -> dict:
    pooled={identity:combine_metrics([arms[identity]["target_metrics"] for arms in manifests.values()]) for identity in IDENTITIES}
    strata={}
    for stratum in ("task4_rule","task4_mixed"):
        strata[stratum]={identity:combine_metrics([arms[identity]["target_metrics"] for label,arms in manifests.items()
            if protocol["collection"]["blocks"][label]["stratum"]==stratum]) for identity in IDENTITIES}
    paired={}; wins={"original-cnn":0,"frozen-v4":0}
    for label,arms in manifests.items():
        row={identity:arms[identity]["target_metrics"] for identity in IDENTITIES}
        row["stratum"]=protocol["collection"]["blocks"][label]["stratum"]
        row["candidate_minus_original_score"]=_delta(row["collision-candidate"],row["original-cnn"],"score_per_round")
        row["candidate_minus_v4_score"]=_delta(row["collision-candidate"],row["frozen-v4"],"score_per_round")
        wins["original-cnn"]+=row["candidate_minus_original_score"]>0; wins["frozen-v4"]+=row["candidate_minus_v4_score"]>0
        paired[label]=row
    candidate=pooled["collision-candidate"]; original=pooled["original-cnn"]; v4=pooled["frozen-v4"]
    deltas={
        "candidate_minus_original":{k:_delta(candidate,original,k) for k in ("score_per_round","coins_per_round","kills_per_round","suicides_per_round","invalid_actions_per_round","wait_fraction")},
        "candidate_minus_v4":{k:_delta(candidate,v4,k) for k in ("score_per_round","coins_per_round","kills_per_round","suicides_per_round","invalid_actions_per_round","wait_fraction")},
    }
    gate=protocol["decision_rule"]
    checks={
        "score_over_original":deltas["candidate_minus_original"]["score_per_round"]>=gate["minimum_score_gain_over_original_per_round"],
        "score_over_v4":deltas["candidate_minus_v4"]["score_per_round"]>=gate["minimum_score_gain_over_v4_per_round"],
        "blocks_over_original":wins["original-cnn"]>=gate["minimum_block_wins_over_each_out_of_8"],
        "blocks_over_v4":wins["frozen-v4"]>=gate["minimum_block_wins_over_each_out_of_8"],
        "rule_over_v4":_delta(strata["task4_rule"]["collision-candidate"],strata["task4_rule"]["frozen-v4"],"score_per_round")>=gate["minimum_rule_score_delta_over_v4"],
        "mixed_over_v4":_delta(strata["task4_mixed"]["collision-candidate"],strata["task4_mixed"]["frozen-v4"],"score_per_round")>=gate["minimum_mixed_score_delta_over_v4"],
        "suicide_vs_original":deltas["candidate_minus_original"]["suicides_per_round"]<=gate["maximum_suicide_increase_over_original_per_round"],
        "suicide_vs_v4":deltas["candidate_minus_v4"]["suicides_per_round"]<=gate["maximum_suicide_increase_over_v4_per_round"],
        "kills_vs_original":deltas["candidate_minus_original"]["kills_per_round"]>=-gate["maximum_kill_loss_per_round"],
        "kills_vs_v4":deltas["candidate_minus_v4"]["kills_per_round"]>=-gate["maximum_kill_loss_per_round"],
        "latency":candidate["mean_decision_time_ms"]<=gate["maximum_mean_decision_time_ms"],
    }
    passed=all(checks.values()); decision="collision_mask_finally_confirmed" if passed else "collision_mask_not_confirmed_retain_best_existing"
    treatment_counts={key:sum(arms["collision-candidate"]["trace_counts"][key] for arms in manifests.values()) for key in next(iter(manifests.values()))["collision-candidate"]["trace_counts"]}
    return {"schema_version":1,"kind":KIND_REPORT,"protocol_id":protocol["protocol_id"],"protocol_sha256":protocol_hash,
        "status":"completed","completed_at_utc":utc_now(),"rounds_per_identity":400,"pooled_metrics":pooled,
        "stratum_metrics":strata,"paired_blocks":paired,"candidate_block_wins":wins,"candidate_deltas":deltas,
        "candidate_trace_counts":treatment_counts,"gate":{"checks":checks,"passed":passed,"thresholds":gate},
        "result":{"decision":decision,"training_started":False,"checkpoint_modified":False,"candidate_deployed":False,"automatic_followup_started":False},
        "awaiting_user_instruction":True}


def execute(protocol: dict,path: Path,protocol_hash: str)->dict:
    manifests={label:{identity:collect(protocol,path,protocol_hash,label,identity,block) for identity in IDENTITIES}
               for label,block in protocol["collection"]["blocks"].items()}
    report=summarize(protocol,protocol_hash,manifests); output=ROOT/protocol["report_path"]
    if output.exists(): raise RuntimeError(f"refusing to overwrite final report: {relative(output)}")
    atomic_json(output,report); return report


def main(argv=None)->int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol",type=Path,default=DEFAULT_PROTOCOL); parser.add_argument("--execute",action="store_true")
    args=parser.parse_args(argv); protocol,path,protocol_hash=load_protocol(args.protocol)
    result=execute(protocol,path,protocol_hash) if args.execute else {"mode":"dry-run","protocol_id":protocol["protocol_id"],"protocol_sha256":protocol_hash,"identities":IDENTITIES,"rounds_per_identity":400,"training_started":False}
    print(json.dumps(result,indent=2,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
