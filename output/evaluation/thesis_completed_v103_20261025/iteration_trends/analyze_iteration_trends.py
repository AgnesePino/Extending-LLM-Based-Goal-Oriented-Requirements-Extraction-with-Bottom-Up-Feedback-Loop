"""Reproduce per-iteration trajectories from the frozen thesis snapshot only."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["MPLBACKEND"] = "Agg"

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "notebook")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def tex(value):
    for old, new in [("&", r"\&"), ("_", r"\_"), ("%", r"\%")]:
        value = value.replace(old, new)
    return value


def main():
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    torch.set_num_threads(2)
    import transformers.utils.import_utils as imports
    imports._torchao_available = False  # Plain cached BERT needs no optional quantization backend.
    from groundtruth import GROUNDTRUTH_BY_NAME
    from src.evaluation.goal_evaluator import GoalEvaluator

    evidence = json.loads((BASE / "evidence_manifest.json").read_text(encoding="utf-8"))
    source = BASE / "source_snapshot.json"
    assert sha(source) == evidence["source_snapshot_sha256"], "Frozen snapshot changed"
    assert digest(GROUNDTRUTH_BY_NAME) == evidence["ground_truth_sha256"], "Ground truth changed"
    snapshot = json.loads(source.read_text(encoding="utf-8"))
    configs = {"E1_0_examples": "P0", "E1_1_examples": "P1", "reference": "P4"}
    names = {
        "Assegno Unico Universale - SIA Project 24 25": ("AUU", "auu"),
        "Fitness Tracker": ("Fitness Tracker", "fitness"),
        "La Reine Marlene - SIA Project 23 24": ("La Reine Marlene", "marlene"),
        "London Ambulance Service": ("London Ambulance Service", "london"),
        "Planning Poker": ("Planning Poker", "poker"),
        "SIA Project 25 26": ("SIA Project 25 26", "sia"),
    }
    selected = {doc["name"] for doc in snapshot["documents"] if any(
        run["globalGoalCycle"]["completed_iterations"] > 10 for run in doc["runs"].values())}
    assert selected <= set(names), "Add an explicit display name for another selected project"
    cache_path = Path.home() / ".cache/huggingface/hub/models--bert-base-uncased"
    revision = evidence["bert_revision"]
    evaluator = GoalEvaluator(model_name=str(cache_path / "snapshots" / revision), preprocess=True)
    cache_file = HERE / "score_cache.json"
    fingerprint = dict(snapshot=sha(source), ground_truth=digest(GROUNDTRUTH_BY_NAME),
                       bert_revision=revision, scorer=sha(ROOT / "src/evaluation/goal_evaluator.py"))
    stored = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    cache = stored.get("scores", {}) if stored.get("fingerprint") == fingerprint else {}
    rows, summaries, initial_reconciliations = [], [], []

    def score(predictions, references):
        key = digest({"predictions": predictions, "references": references})
        if key not in cache:
            result = evaluator.score_goals(predictions, references)
            cache[key] = {k: float(result[k]) for k in ("precision", "recall", "f1")}
            cache_file.write_text(json.dumps(dict(fingerprint=fingerprint, scores=cache)), encoding="utf-8")
            if len(cache) % 20 == 0:
                print(f"Scored {len(cache)} distinct goal collections", flush=True)
        return cache[key]

    final = pd.read_csv(BASE / "final_metrics.csv")
    verification = []
    for doc in snapshot["documents"]:
        name = doc["name"]
        if name not in selected:
            continue
        for run_id, run in doc["runs"].items():
            config = run_id.split("/")[0]
            assert config in configs and run["status"] == "completed"
            cycle = run["globalGoalCycle"]
            traces = cycle["iterations"]
            assert len(traces) == cycle["completed_iterations"]
            assert [t["iteration"] for t in traces] == list(range(1, len(traces) + 1))
            assert all(t.get("checkpoint_complete", True) for t in traces)
            baseline = doc["baseline"]
            states = [(0, baseline["structuredHighLevelGoals"], baseline["structuredLowLevelGoals"], None)]
            normalized_children = deepcopy(states[0][2])
            adjustments = cycle["configuration"].get("initial_parent_reference_reconciliations", [])
            for adjustment in adjustments:
                child = normalized_children["low_level_goals"][adjustment["llg_index"]]
                assert child["high_level_associated"] == adjustment["generated_parent_reference"]
                assert adjustment["approved_parent_reference"] in states[0][1]["goals"]
                assert adjustment["generated_parent_reference"]["description"] == adjustment["approved_parent_reference"]["description"]
                child["high_level_associated"] = adjustment["approved_parent_reference"]
            initial_reconciliations.append(dict(dataset=name, configuration=configs[config],
                                                parent_reference_adjustments=len(adjustments)))
            previous = (states[0][1], normalized_children)
            for trace in traces:
                incoming = trace["input_hierarchy"]
                assert (incoming["high_level_goals"], incoming["low_level_goals"]) == previous, (name, run_id, trace["iteration"])
                previous = trace["high_level_goals"], trace["low_level_goals"]
                states.append((trace["iteration"], *previous, trace))
            assert previous == (run["finalHighLevelGoals"], run["finalLowLevelGoals"])
            for iteration, hlgs, llgs, trace in states:
                actions = Counter((v.get("decision") or "INCONCLUSIVE")
                                  for v in (trace or {}).get("evaluations", {}).values())
                for level, container, field, ref_field in (
                    ("HLG", hlgs, "goals", "highLevelGoals"),
                    ("LLG", llgs, "low_level_goals", "lowLevelGoals")):
                    predictions = [g["description"] for g in container[field]]
                    metrics = score(predictions, GROUNDTRUTH_BY_NAME[name][ref_field])
                    row = dict(dataset=name, config_id=config, configuration=configs[config],
                               repetition=run["repetition"], iteration=iteration, level=level,
                               **metrics, generated_goal_count=len(predictions),
                               reference_goal_count=len(GROUNDTRUTH_BY_NAME[name][ref_field]),
                               converged=cycle["converged"], total_iterations=len(traces),
                               stop_reason=cycle["stop_reason"],
                               hierarchy_sha256=digest({"hlg": hlgs, "llg": llgs}),
                               keep=actions["KEEP_ORIGINAL_HIGH_LEVEL_GOAL"],
                               rewrite=actions["REWRITE_ORIGINAL_HIGH_LEVEL_GOAL"],
                               remove=actions["REMOVE_ORIGINAL_HIGH_LEVEL_GOAL"],
                               duplicate=actions["DUPLICATE_REMOVAL"],
                               inconclusive=actions["INCONCLUSIVE"])
                    rows.append(row)
                    if iteration in (0, len(traces)):
                        expected_config = "top_down" if iteration == 0 else config
                        expected = final[(final.dataset == name) & (final.config_id == expected_config) & (final.level == level)]
                        assert len(expected) == 1
                        differences = {k: abs(float(expected.iloc[0][k]) - metrics[k]) for k in metrics}
                        assert max(differences.values()) < 1e-6, (name, config, iteration, level, differences)
                        verification.append(dict(dataset=name, configuration=configs[config], iteration=iteration,
                                                 level=level, maximum_absolute_difference=max(differences.values())))
            summaries.append(dict(dataset=name, configuration=configs[config], iterations=len(traces),
                                  converged=cycle["converged"], stop_reason=cycle["stop_reason"],
                                  selected_long_run=len(traces) > 10))
        print("Finished:", names[name][0], flush=True)

    metrics = pd.DataFrame(rows)
    summary = pd.DataFrame(summaries)
    metrics.to_csv(HERE / "iteration_metrics.csv", index=False)
    summary.to_csv(HERE / "run_summary.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    colours = {"P0": "#1764a0", "P1": "#cf6b16", "P4": "#7953a2"}
    section = [r"\section{Andamento dei progetti con piu di dieci iterazioni}",
        r"\label{sec:results-iteration-trends}",
        "% Evidence: iteration_trends/{iteration_metrics.csv,run_summary.csv,evidence_manifest.json}.",
        "L'analisi seleziona i progetti per cui almeno una configurazione completa",
        "supera dieci iterazioni nel medesimo snapshot dei risultati precedenti.",
        "Sono nove run su sei progetti: sei raggiungono la convergenza e tre",
        "terminano al limite di 25 iterazioni. Per ciascun progetto si riportano",
        "anche le altre configurazioni P0, P1 e P4, comprese quelle piu brevi,",
        "per confrontare le traiettorie dalla stessa gerarchia iniziale.",
        "",
        "L'iterazione 0 indica la baseline top-down; ogni valore successivo valuta",
        "lo stato dopo una completa iterazione, incluse rimozioni e riparazioni",
        "effettivamente accettate. Si usa lo stesso scorer BERT, preprocessing e",
        "matching dei risultati finali. Una riparazione rifiutata puo lasciare",
        "invariata la gerarchia e quindi le metriche. Le curve terminano allo stop",
        "effettivo: i trattini nelle tabelle indicano iterazioni non eseguite,",
        "non valori nulli o mancata convergenza.",
        "L'eventuale riconciliazione iniziale dei riferimenti padre--figlio,",
        "registrata dal ciclo, non modifica le descrizioni usate dallo scorer.",
        "",
        "Queste traiettorie sono un'analisi descrittiva del processo: non vengono",
        "mediate per ricalcolare le prestazioni finali e non rappresentano repliche",
        "indipendenti. Il sottoinsieme selezionato per durata non descrive da solo",
        "le prestazioni su tutti i dodici dataset. I valori di Precision e Recall,",
        "i conteggi dei goal e delle decisioni per iterazione restano nel CSV",
        r"\path{output/evaluation/thesis_completed_v103_20261025/iteration_trends/iteration_metrics.csv}.",
        ""]
    # One project per page keeps its plot and complete numeric trajectory together.
    for name in sorted(selected):
        title, slug = names[name]
        data = metrics[metrics.dataset == name]
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55), sharex=True, sharey=True, layout="constrained")
        for ax, level in zip(axes, ("HLG", "LLG")):
            for config in ("P0", "P1", "P4"):
                series = data[(data.configuration == config) & (data.level == level)].sort_values("iteration")
                ax.plot(series.iteration, series.f1, label=config, color=colours[config],
                        marker="o", markersize=2.6, linewidth=1.25)
                last = series.iloc[-1]
                ax.plot(last.iteration, last.f1, marker="o" if last.converged else "X",
                        color=colours[config], markersize=6, markeredgecolor="white", markeredgewidth=.6)
            ax.set_title(f"F1 {level}")
            ax.set_xlabel("Iterazione")
            ax.set_ylim(0, 1)
            ax.set_xlim(-.3, int(data.iteration.max())+.5)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
            ax.grid(alpha=.22)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("F1 semantico")
        axes[1].legend(loc="lower right", ncol=3, frameon=True, fontsize=8)
        fig.savefig(HERE / f"trend_{slug}.pdf", bbox_inches="tight")
        fig.savefig(HERE / f"trend_{slug}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        statuses = summary[summary.dataset == name].set_index("configuration")
        status_text = "; ".join(f"{c}: {int(statuses.loc[c, 'iterations'])} iterazioni, " +
            ("convergenza" if statuses.loc[c, "converged"] else "limite raggiunto") for c in ("P0", "P1", "P4"))
        section += [r"\clearpage", r"\subsection{"+tex(title)+"}", status_text+".",
            r"\begin{figure}[htbp]", r"\centering",
            r"\includegraphics[width=\textwidth]{output/evaluation/thesis_completed_v103_20261025/iteration_trends/trend_"+slug+".pdf}",
            r"\caption{Traiettorie F1 per "+tex(title)+r". L'iterazione 0 e la baseline; una croce indica arresto senza convergenza.}",
            r"\label{fig:iteration-trend-"+slug+"}", r"\end{figure}",
            r"\begin{table}[htbp]", r"\centering", r"\small",
            r"\caption{F1 dopo ogni iterazione per "+tex(title)+". H: HLG; L: LLG.}",
            r"\label{tab:iteration-trend-"+slug+"}", r"\begin{tabular}{rrrrrrr}", r"\toprule",
            r"\textbf{Iter.} & \textbf{P0 H} & \textbf{P0 L} & \textbf{P1 H} & \textbf{P1 L} & \textbf{P4 H} & \textbf{P4 L} \\", r"\midrule"]
        indexed = data.set_index(["configuration", "iteration", "level"])
        for iteration in range(int(data.iteration.max())+1):
            cells = [str(iteration)]
            for config in ("P0", "P1", "P4"):
                for level in ("HLG", "LLG"):
                    key = (config, iteration, level)
                    cells.append(f"{indexed.loc[key, 'f1']:.4f}".replace(".", ",") if key in indexed.index else "--")
            section.append(" & ".join(cells)+r" \\")
        section += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    section += [r"\clearpage", ""]
    (HERE / "iteration_trends.tex").write_text("\n".join(section), encoding="utf-8")
    (HERE / "evidence_manifest.json").write_text(json.dumps(dict(
        source_snapshot_sha256=sha(source), ground_truth_sha256=digest(GROUNDTRUTH_BY_NAME),
        selected_projects=sorted(selected), selected_long_runs=int(summary.selected_long_run.sum()),
        compared_runs=len(summary), scored_states=len(metrics)//2, unique_goal_collections=len(cache),
        selection="Project has at least one completed run with strictly more than 10 iterations; compare all P0/P1/P4 runs of that project",
        metric=evidence["metric"], bert_revision=revision, final_and_baseline_verification=verification,
        initial_parent_reference_reconciliations=initial_reconciliations,
        script_sha256=sha(Path(__file__)), human_verification="pending author review",
        outputs={p.name: sha(p) for p in HERE.iterdir() if p.suffix in (".csv", ".tex", ".pdf", ".png")}
    ), ensure_ascii=False, indent=2), encoding="utf-8")
    assert sha(source) == evidence["source_snapshot_sha256"]
    print(f"Done: {len(selected)} projects; {len(summary)} compared runs; {len(metrics)//2} states", flush=True)


if __name__ == "__main__":
    main()
