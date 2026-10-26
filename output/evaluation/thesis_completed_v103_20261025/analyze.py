"""Reproduce thesis results from fully completed configurations, without generation."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import collections
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "notebook")]
OUT = Path(__file__).resolve().parent
SUITE = ROOT / "output/bottom_up_experiments/ofat_fixed_thresholds_20261024_211627"

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

manifest = json.loads((SUITE / "manifest.json").read_text(encoding="utf-8"))
documents = [json.loads((SUITE / b["dataset_file"]).read_text(encoding="utf-8"))
             for b in manifest["baselines"]]
configs = [c for c in manifest["configurations"]
           if all(doc["runs"].get(f"{c['config_id']}/repeat_{r:03d}", {}).get("status") == "completed"
                  for doc in documents for r in range(1, manifest["repetitions"] + 1))]
# Freeze the planned comparison established at analysis start, even if other runs finish later.
assert {c["config_id"] for c in configs} == {"reference", "E1_0_examples", "E1_1_examples"}
selected = {c["config_id"] for c in configs}
snapshot = {"suite_id": manifest["suite_id"], "policy": manifest["fixed_runtime"]["policy"],
            "configurations": configs, "repetitions": manifest["repetitions"],
            "documents": [{"name": doc["name"], "baseline": doc["baseline"],
                           "baseline_sha256": doc["baseline_sha256"],
                           "runs": {k: v for k, v in doc["runs"].items() if k.split('/')[0] in selected}}
                          for doc in documents]}
if (OUT / "source_snapshot.json").exists():
    snapshot = json.loads((OUT / "source_snapshot.json").read_text(encoding="utf-8"))
    configs = snapshot["configurations"]
    selected = {c["config_id"] for c in configs}
else:
    dump(OUT / "source_snapshot.json", snapshot)
print("Frozen configurations:", sorted(selected), flush=True)

import pandas as pd
import torch
torch.set_num_threads(2)
from groundtruth import GROUNDTRUTH_BY_NAME
from src.bottom_up.result_storage import result_payload
# Plain BERT does not use quantization. Disable discovery of the incompatible
# optional torchao/Triton extension only inside this analysis process.
import transformers.utils.import_utils as transformer_imports
transformer_imports._torchao_available = False
from src.evaluation.goal_evaluator import GoalEvaluator

notebook = json.loads((ROOT / "notebook/04_evaluation_all_configurations.ipynb").read_text(encoding="utf-8"))
ns = {"pd": pd}
for i in (3, 5):
    exec(compile("".join(notebook["cells"][i]["source"]), f"notebook04_cell{i}", "exec"), ns)
collection = {"entries": [], "availability": [], "configurations": [
    dict(c, suite_id=manifest["suite_id"], planned_datasets=12, planned_repetitions=manifest["repetitions"])
    for c in configs]}
for doc in snapshot["documents"]:
    baseline = doc["baseline"]
    key = ns["baseline_key"](baseline)
    base = dict(dataset=doc["name"], baseline_key=key, suite_id="", method="top_down",
                config_id="top_down", repetition=0, source_file="source_snapshot.json")
    collection["entries"].append(dict(base, payload=baseline,
        hlg=baseline["structuredHighLevelGoals"], llg=baseline["structuredLowLevelGoals"]))
    collection["availability"].append(dict(base, status="completed"))
    for run_id, record in doc["runs"].items():
        config_id = run_id.split('/')[0]
        identity = dict(base, suite_id=manifest["suite_id"], config_id=config_id,
                        method=manifest["suite_id"] + '/' + config_id, repetition=record["repetition"])
        # Read the current schema using a view with the stored schema marker.
        payload = result_payload(dict(doc, schema_version=3), record)
        collection["entries"].append(dict(identity, payload=payload,
            hlg=payload["finalHighLevelGoals"], llg=payload["finalLowLevelGoals"]))
        collection["availability"].append(dict(identity, status="completed"))

cache = Path.home() / ".cache/huggingface/hub/models--bert-base-uncased"
revision = (cache / "refs/main").read_text().strip()
evaluator = GoalEvaluator(model_name=str(cache / "snapshots" / revision), preprocess=True)
counter = 0
def score(preds, refs):
    global counter
    counter += 1
    value = evaluator.score_goals(preds, refs)
    if counter % 8 == 0:
        print(f"Scored {counter}/96 goal collections", flush=True)
    return value

metrics, matches = ns["evaluate_final_outputs"](collection, GROUNDTRUTH_BY_NAME, score)
metrics.to_csv(OUT / "final_metrics.csv", index=False)
matches.to_csv(OUT / "matches.csv", index=False)
comparison, cohort = ns["final_comparison_table"](metrics, collection["configurations"], collection["availability"])
tables = {"final_comparison": comparison,
          "dataset_comparison": ns["dataset_comparison_table"](metrics, collection["configurations"], collection["availability"]),
          "convergence_goal_counts": ns["convergence_goal_counts_table"](collection),
          "evaluator_decision_counts": ns["evaluator_decision_counts_table"](collection),
          "final_decision_counts": ns["final_decision_counts_table"](collection)}
for name, frame in tables.items():
    frame.to_csv(OUT / (name + ".csv"), index=False)
    frame.to_html(OUT / (name + ".html"), index=False, float_format=lambda value: f"{value:.4f}")
ns["export_decisions_by_configuration"](OUT, {k: v for k, v in tables.items() if k in {
    "evaluator_decision_counts", "final_decision_counts"}})
dump(OUT / "evidence_manifest.json", {
    "suite_id": manifest["suite_id"], "source_policy": snapshot["policy"],
    "configurations": configs, "dataset_count": 12, "run_count": 36,
    "source_snapshot_sha256": hashlib.sha256((OUT / "source_snapshot.json").read_bytes()).hexdigest(),
    "ground_truth_sha256": digest(GROUNDTRUTH_BY_NAME),
    "metric": "bert-base-uncased, original NLTK preprocessing, maximum-weight one-to-one matching",
    "aggregation": "arithmetic macro means of P, R, F1 independently; paired 12-dataset cohort",
    "bert_revision": revision,
    "execution_note": "CPU; offline cached BERT; unused incompatible optional torchao discovery disabled in this process only",
    "evaluation_source_sha256": hashlib.sha256((ROOT / "src/evaluation/goal_evaluator.py").read_bytes()).hexdigest(),
    "human_verification": "pending author review", "common_cohort": sorted(cohort),
    "note": "No generator/evaluator LLM calls. Only final hierarchies are scored; nonconvergent completed runs included."})
print(comparison.to_string(index=False), flush=True)
