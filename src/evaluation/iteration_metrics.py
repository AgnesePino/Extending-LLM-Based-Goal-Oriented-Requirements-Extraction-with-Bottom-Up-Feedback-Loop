"""Read-only evaluation of saved post-iteration hierarchies for notebook 04."""

import math
import textwrap

import pandas as pd


METRICS = ("precision", "recall", "f1")
IDENTITY = ("suite_id", "config_id", "method", "dataset", "baseline_key")
LEVELS = (("HLG", "high_level_goals", "goals", "highLevelGoals"),
          ("LLG", "low_level_goals", "low_level_goals", "lowLevelGoals"))


def evaluate_iterations(collection, references, score_goals, final_metrics, goal_texts):
    """Keep iterations/repetitions separate; reuse already evaluated endpoints.

    Only completed runs enter this analysis, as in the final-output tables.
    Iteration zero is the exact paired frozen baseline, once per configuration.
    Missing snapshots are reported, never reconstructed or carried forward.
    """
    entries = collection["entries"]
    baselines = {(e["dataset"], e["baseline_key"]): e for e in entries
                 if e["method"] == "top_down"}
    runs = {(e["method"], e["dataset"], e["baseline_key"], e["repetition"]): e
            for e in entries if e["method"] != "top_down"}
    finals = {(r.method, r.dataset, r.baseline_key, r.repetition, r.level): r
              for r in final_metrics.itertuples()}
    cache, rows, issues = {}, [], []

    def cache_key(dataset, level, container, field, ref_field):
        return (tuple(goal_texts(container, field)), tuple(references[dataset][ref_field]))

    # Seed the cache with final and baseline scores from this notebook execution.
    for entry in entries:
        for level, _, field, ref_field in LEVELS:
            key = (entry["method"], entry["dataset"], entry["baseline_key"],
                   entry["repetition"], level)
            if key in finals:
                values = {m: float(getattr(finals[key], m)) for m in METRICS}
                cache.setdefault(cache_key(entry["dataset"], level,
                    entry[level.lower()], field, ref_field), values)

    def add(identity, repetition, iteration, level, container, field, ref_field, source):
        key = cache_key(identity["dataset"], level, container, field, ref_field)
        if key not in cache:
            result = score_goals(list(key[0]), list(key[1]))
            cache[key] = {m: float(result[m] if m != "f1" else
                                  result.get("f1", result.get("f1_score"))) for m in METRICS}
        rows.append({**identity, "repetition": repetition, "iteration": iteration,
                     "level": level, **cache[key], "reference_goal_count": len(key[1]),
                     "source_file": source})

    planned = pd.DataFrame(collection["availability"])
    planned = planned.loc[planned["method"] != "top_down"]
    for identity_values, group in planned.groupby(list(IDENTITY), sort=False, dropna=False):
        identity = dict(zip(IDENTITY, identity_values))
        baseline = baselines[(identity["dataset"], identity["baseline_key"])]
        for level, _, field, ref_field in LEVELS:
            add(identity, 0, 0, level, baseline[level.lower()], field, ref_field,
                baseline["source_file"])
        for planned_run in group.to_dict("records"):
            repetition = planned_run["repetition"]
            entry = runs.get((identity["method"], identity["dataset"],
                              identity["baseline_key"], repetition))
            def issue(message):
                issues.append({**identity, "repetition": repetition, "issue": message})
            if entry is None:
                issue(f"Run non valutata: {planned_run['status']}")
                continue
            cycle = entry["payload"].get("globalGoalCycle", {})
            traces = cycle.get("iterations", [])
            completed = cycle.get("completed_iterations", len(traces))
            seen = set()
            for trace in traces:
                iteration = trace["iteration"]
                if type(iteration) is not int or iteration < 1 or iteration in seen:
                    raise ValueError(f"Indice iterazione non valido o duplicato: {identity}, {iteration}")
                seen.add(iteration)
                if not trace.get("checkpoint_complete", True):
                    issue(f"Iterazione {iteration}: checkpoint incompleto, escluso")
                    continue
                for level, snapshot_field, field, ref_field in LEVELS:
                    container = trace.get(snapshot_field)
                    if container is None:
                        issue(f"Iterazione {iteration}: snapshot {level} mancante")
                        continue
                    if iteration == completed:
                        if goal_texts(container, field) != goal_texts(entry[level.lower()], field):
                            raise ValueError(f"Ultima iterazione diversa dall'output finale: {identity}, {level}")
                    add(identity, repetition, iteration, level, container, field, ref_field,
                        entry["source_file"])
            missing = sorted(set(range(1, completed + 1)) - seen)
            if missing:
                issue(f"Snapshot mancanti per le iterazioni: {missing}")

    raw = pd.DataFrame(rows, columns=[*IDENTITY, "repetition", "iteration", "level",
                                     *METRICS, "reference_goal_count", "source_file"])
    keys = [*IDENTITY, "level", "iteration"]
    summary = raw.groupby(keys, as_index=False, sort=False).agg(
        **{m: (m, "mean") for m in METRICS},
        reference_goal_count=("reference_goal_count", "first"),
        repetitions=("repetition", "nunique"))
    # The baseline is a single observation, not an experimental repetition.
    summary.loc[summary["iteration"] == 0, "repetitions"] = 0
    diagnostics = pd.DataFrame(issues, columns=[*IDENTITY, "repetition", "issue"])
    return raw, summary, diagnostics


def varying_project_metrics(summary):
    """Keep the same cohort across configurations and levels within each suite.

    A project qualifies if any observed metric differs from its paired baseline
    in at least one configuration or level. Preserve every observed iteration.
    """
    comparison_keys = [*IDENTITY, "level"]
    baseline = summary.loc[summary["iteration"] == 0,
                           [*comparison_keys, *METRICS]]
    compared = summary.merge(baseline, on=comparison_keys, suffixes=("", "_baseline"),
                             validate="many_to_one")
    changed = pd.concat([(compared[m] - compared[f"{m}_baseline"]).abs() > 1e-8
                         for m in METRICS], axis=1).any(axis=1)
    cohort_keys = ["suite_id", "dataset", "baseline_key"]
    cohort = compared.loc[changed, cohort_keys].drop_duplicates()
    return summary.merge(cohort, on=cohort_keys, how="inner", validate="many_to_one")


def aggregate_varying_projects(summary):
    """Reference-weighted P/R and harmonic F1; export observed coverage.

    The cohort is fixed across configurations/levels within a suite. At later
    iterations only actual snapshots contribute; missing projects are not filled.
    """
    from src.evaluation.goal_evaluator import weighted_metric_table

    selected = varying_project_metrics(summary)
    keys = ["suite_id", "config_id", "method", "level", "iteration"]
    if selected.duplicated([*IDENTITY, "level", "iteration"]).any():
        raise ValueError("Expected one repetition mean per project/level/iteration")
    cohort = selected[["suite_id", "dataset", "baseline_key"]].drop_duplicates()
    sizes = cohort.groupby("suite_id").size().rename("cohort_datasets")
    coverage = selected.groupby(keys, as_index=False, sort=False).agg(
        repetitions_min=("repetitions", "min"),
        repetitions_max=("repetitions", "max"))
    result = weighted_metric_table(selected, keys).rename(
        columns={"dataset_count": "datasets_observed"})
    result = result.merge(coverage, on=keys, validate="one_to_one")
    result = result.merge(sizes, on="suite_id", validate="many_to_one")
    result["complete_cohort"] = result.datasets_observed.eq(result.cohort_datasets)
    return selected, result


def plot_aggregate_metrics(aggregate, *, level, labels=None):
    """Compare configurations, one panel each for precision, recall and F1."""
    import matplotlib.pyplot as plt

    selected = aggregate.loc[aggregate.level.eq(level)]
    if selected.empty:
        raise ValueError("No varying projects available for the requested level")
    if selected.suite_id.nunique() != 1:
        raise ValueError("Plot one suite at a time")
    labels = labels or {}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharex=True, sharey=True)
    for config, frame in selected.groupby("config_id", sort=False):
        frame = frame.sort_values("iteration")
        grid = frame.set_index("iteration").reindex(range(int(frame.iteration.max()) + 1))
        for ax, metric in zip(axes, METRICS):
            line, = ax.plot(grid.index, grid[metric], marker="o", markersize=3,
                            label=labels.get(config, config))
            partial = frame.loc[~frame.complete_cohort]
            ax.scatter(partial.iteration, partial[metric], facecolors="white",
                       edgecolors=line.get_color(), s=32, zorder=3)
    for ax, metric in zip(axes, METRICS):
        ax.set(title=metric.capitalize() if metric != "f1" else "F1",
               xlabel="Iterazione (0 = Top-Down)", ylim=(-.02, 1.02))
        ax.grid(alpha=.25)
    axes[0].set_ylabel(f"{level}: aggregazione pesata per riferimenti")
    axes[-1].legend(fontsize=8)
    fig.suptitle(f"{level} - dataset variabili aggregati")
    fig.text(.5, .015, "P/R pesate per goal di riferimento; F1 armonica delle P/R pesate. "
             "Cerchi vuoti: coorte incompleta, solo snapshot osservati; nessun riempimento.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .94))
    return fig


def plot_iteration_metrics(summary, raw, *, level, title):
    """One panel per dataset; no extrapolation beyond observed iterations."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    selected = summary.loc[summary["level"] == level]
    pairs = selected[["dataset", "baseline_key"]].drop_duplicates().sort_values("dataset")
    if pairs.empty:
        raise ValueError("No projects available for the requested plot")
    columns = min(2, len(pairs))
    fig, axes = plt.subplots(math.ceil(len(pairs) / columns), columns, squeeze=False,
                             figsize=(8 * columns, 3.7 * math.ceil(len(pairs) / columns)))
    styles = (("precision", "Precision", "#0072B2", "o", "-"),
              ("recall", "Recall", "#D55E00", "s", "--"),
              ("f1", "F1", "#009E73", "^", "-."))
    ymin = min(0, float(selected[list(METRICS)].min().min()))
    for ax, pair in zip(axes.flat, pairs.itertuples(index=False)):
        data = selected.loc[(selected["dataset"] == pair.dataset) &
                            (selected["baseline_key"] == pair.baseline_key)].sort_values("iteration")
        xmax = max(1, int(data["iteration"].max()))
        individuals = raw.loc[(raw["dataset"] == pair.dataset) &
                              (raw["baseline_key"] == pair.baseline_key) &
                              (raw["level"] == level) & (raw["iteration"] > 0)]
        # Reindex internal gaps to NaN so missing snapshots break the line.
        grid = data.set_index("iteration").reindex(range(int(data["iteration"].max()) + 1))
        for metric, label, color, marker, linestyle in styles:
            if individuals["repetition"].nunique() > 1:
                for _, repetition in individuals.groupby("repetition"):
                    points = pd.concat([data.loc[data["iteration"] == 0], repetition])
                    points = points.set_index("iteration").reindex(range(int(points["iteration"].max()) + 1))
                    ax.plot(points.index, points[metric], color=color, alpha=.18, linewidth=.8)
            ax.plot(grid.index, grid[metric], label=label, color=color, marker=marker,
                    linestyle=linestyle, markersize=3.5, linewidth=1.5)
        ax.set_title(textwrap.fill(pair.dataset, 43), fontsize=10)
        ax.set(xlim=(-.25, xmax + .25), ylim=(ymin - .02, 1.02),
               xlabel="Iterazione (0 = Top-Down)", ylabel=level)
        ax.set_xticks(range(xmax + 1))
        ax.tick_params(axis="x", labelsize=8, labelrotation=45 if xmax > 12 else 0)
        ax.grid(alpha=.25)
        if individuals.empty:
            ax.text(.5, .45, "Solo baseline: nessuna iterazione disponibile",
                    transform=ax.transAxes, ha="center", fontsize=9, wrap=True)
        if data["repetitions"].max() > 1:
            counts = data.loc[data["iteration"] > 0, "repetitions"]
            ax.text(.02, .02, f"Ripetizioni per punto: {counts.min()}–{counts.max()}",
                    transform=ax.transAxes, fontsize=8)
    for ax in list(axes.flat)[len(pairs):]:
        ax.set_visible(False)
    handles = [Line2D([0], [0], color=c, marker=m, linestyle=s, label=l)
               for _, l, c, m, s in styles]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .965), ncol=3)
    fig.suptitle(f"{title} — {level}", fontsize=14)
    fig.tight_layout(rect=(0, .025, 1, .93))
    fig.text(.5, .005, "0 = baseline Top-Down; t > 0 = gerarchia dopo l'iterazione t. "
             "Medie tra ripetizioni disponibili; linee tenui = singole run. Nessun prolungamento dopo l'arresto.",
             ha="center", fontsize=9)
    return fig
