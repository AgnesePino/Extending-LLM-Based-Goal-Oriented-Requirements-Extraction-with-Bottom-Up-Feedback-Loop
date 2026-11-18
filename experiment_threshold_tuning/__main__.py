"""HLG prime, original parent and cosine similarity.

Run: python -m experiment_threhold_tuning
Use --fresh to regenerate all reconstructions instead of resuming the CSV.
Primes use only their LLGs; cosine compares prime with parent description.
Parents without children have empty prime and score.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).with_name("hlg_cosine.csv")
FIELDS = ("dataset", "branch_id", "hlg_prime", "hlg_original", "cosine_similarity")


def save(rows):
    temporary = OUTPUT.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(OUTPUT)


def run(*, fresh=False):
    # Support direct execution from any working directory as well.
    sys.path.insert(0, str(ROOT))
    from src.data_model import (
        HighLevelGoals, LowLevelGoals, indexed_high_level_goals, low_level_goals_for_hlg,
    )

    previous = {}
    if OUTPUT.exists() and not fresh:
        with OUTPUT.open(encoding="utf-8-sig", newline="") as stream:
            previous = {(r["dataset"], r["branch_id"]): r for r in csv.DictReader(stream)}
    paths = sorted((ROOT / "output/top_down").glob(
        "*_ZERO_SHOT_feedback_original_top_down_top_down.json"))
    if not paths:
        raise ValueError("No top-down baselines found in output/top_down")
    rows, pending, datasets = [], [], set()
    for path in paths:
        baseline = json.loads(path.read_bytes())
        name = baseline["name"]
        if name in datasets:
            raise ValueError(f"Duplicate baseline: {name}")
        datasets.add(name)
        parents = HighLevelGoals.model_validate(baseline["structuredHighLevelGoals"])
        children = LowLevelGoals.model_validate(baseline["structuredLowLevelGoals"])
        for branch_id, parent in indexed_high_level_goals(parents).items():
            llgs = low_level_goals_for_hlg(parent, children)
            row = dict(dataset=name, branch_id=str(branch_id), hlg_prime="",
                       hlg_original=parent.description, cosine_similarity="")
            saved = previous.get((name, str(branch_id)))
            if saved and saved["hlg_original"] != parent.description:
                raise ValueError(f"Baseline changed for {name}/{branch_id}; use --fresh")
            if llgs:
                if saved and saved["hlg_prime"] and saved["cosine_similarity"]:
                    row.update(saved)
                else:
                    pending.append((row, llgs))
            rows.append(row)
    if pending:
        from src.bottom_up.goal_reconstructor import reconstruct_high_level_goal
        from src.bottom_up.goal_similarity import _cosine, _encode_text
        from src.llm_clients import loaded_local_model, QWEN_GENERATOR_MODEL, reset_model_clients

        try:
            with loaded_local_model(QWEN_GENERATOR_MODEL):
                for row, llgs in pending:
                    print(f"[HLG prime] {row['dataset']} / {row['branch_id']}", flush=True)
                    prime = reconstruct_high_level_goal(llgs).reconstructed_high_level_goal
                    score = _cosine(_encode_text(prime), _encode_text(row["hlg_original"]))
                    row.update(hlg_prime=prime, cosine_similarity=score)
                    save(rows)
                    print(f"[cosine] {score:.6f}", flush=True)
        finally:
            reset_model_clients()
    save(rows)
    print(f"Saved {len(rows)} HLGs across {len(datasets)} datasets: {OUTPUT}", flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", action="store_true", help="Regenerate all HLG primes")
    run(fresh=parser.parse_args().fresh)
