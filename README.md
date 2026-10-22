# Goal Extraction with Bottom-up Feedback

This project extends a top-down goal extraction pipeline inspired by Arnaudo et al. The top-down stage identifies actors, generates High-Level Goals (HLGs), and decomposes them into Low-Level Goals (LLGs), using generator–critic feedback.

The bottom-up stage reconstructs an HLG from each group of LLGs and compares it with the current goals. Three evaluators review the results against the project description and vote on keeping, rewriting, or removing goals, including redundant goals. Approved rewrite requests return to the top-down pipeline, which generates revised HLGs and their LLGs before replacing the previous hierarchy.

Threshold calibration uses a single Excel workbook for manual review. Experiments compare zero-shot, one-shot, and few-shot evaluators and four evaluator temperatures across 12 datasets, with one fixed similarity threshold shared by equivalence and duplicate detection. External evaluation measures HLG and LLG quality separately against ground truth.

- [Detailed approach — Italian](docs/README_approach.md)
- [Experimental design and thesis guide](docs/PIANO_SPERIMENTALE_E_GUIDA_TESI.md)
- [Manual threshold tuning](experiment_threhold_tuning/README.md)
- [External evaluation methodology](docs/README_evaluation.md)

Execution notebooks are available in [notebook/](notebook/). The local environment requires the dependencies in [requirements.txt](requirements.txt) and the configured LM Studio models. Preparing this repository does not execute the experimental campaign.

The workflow has four notebooks:

| Notebook | Purpose |
|---|---|
| [01 - Top-down](notebook/01_pipeline_execution_top_down_only.ipynb) | Generate the initial hierarchies |
| [02 - Bottom-up](notebook/02_pipeline_execution_bottom_up_only.ipynb) | Review the saved top-down hierarchies |
| [03 - Experiments](notebook/03_bottom_up_experiments.ipynb) | Run every experimental configuration |
| [04 - Evaluation](notebook/04_evaluation_all_configurations.ipynb) | Evaluate top-down, bottom-up, and all experimental runs |

Execution and evaluation are independent: notebook 04 reads saved outputs without rerunning goal generation.

The modules in `src/bottom_up/` are all used by these notebooks:

| Module | Purpose |
|---|---|
| `goal_cycle_orchestrator.py` | Run the review cycle and apply approved corrections through the top-down pipeline |
| `goal_reconstructor.py` | Reconstruct an HLG from each group of LLGs |
| `goal_similarity.py` | Compute cosine similarities for equivalence and duplicate detection |
| `global_goal_evaluator.py` | Collect evaluator decisions and combine their votes |
| `result_storage.py` | Save dataset results and batch summaries for execution and evaluation |

Offline threshold selection and Excel handling live in `experiment_threhold_tuning/workbook.py`. Run `python -m experiment_threhold_tuning prepare` to prepare comparisons and `python -m experiment_threhold_tuning tune` to update the workbook after annotation.
