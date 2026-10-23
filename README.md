# Goal Extraction with Bottom-up Feedback

This project extends a top-down goal extraction pipeline inspired by Arnaudo et al. The top-down stage identifies actors, generates High-Level Goals (HLGs), and decomposes them into Low-Level Goals (LLGs), using generator–critic feedback.

The bottom-up stage reconstructs an HLG from each group of LLGs and compares it with the current goals. Three evaluators review the results against the project description and vote on keeping, rewriting, or removing goals, including redundant goals. Approved rewrite requests return to the top-down pipeline, which generates revised HLGs and their LLGs before replacing the previous hierarchy.

Reconstruction is zero-shot and uses only child names and descriptions. Related operations are abstracted into a concise verb-led functional goal, without introductory wording. Distinct capabilities, responsibilities, and scope constraints remain explicit when necessary. The reconstructor produces one goal, including a composite goal when needed, without judging or correcting the decomposition or receiving the original parent or ground truth.

Each evaluator reviews the complete hierarchy using a compact numerical payload. An incomplete or invalid response is discarded in full and retried only for that model, up to three attempts. Valid responses from the other evaluators are preserved. Defaults are a verified 16384-token loaded context, 4096 output tokens per evaluator, and 8192 output tokens for Qwen. New experiments require a new suite because the prompt and retry policy changed.

Each decision starts from the scope expressed by the reconstructed HLG. Action-specific instructions guide comparisons with its parent, all current HLGs, same-actor reconstructions, project description and actors before choosing KEEP, REWRITE, REMOVE or DUPLICATE_REMOVAL. The description establishes required functionality; generated goals and similarity results establish relationships and scope, without becoming additional requirements. Partial overlap calls for preserving independent capabilities, while duplicate removal requires full supported coverage by a direct candidate's retained parent.

Experiments compare zero-shot, one-shot, and few-shot evaluators and three evaluator temperatures (0, 0.3, 0.6) across 12 datasets. The fixed 0.68 cosine threshold applies to reconstructed goals versus their parent and all current HLGs. Duplicate review compares same-actor goal texts without prime/prime cosine or a numerical duplicate threshold. The separate threshold-tuning workbook remains an offline analysis tool. External evaluation measures HLG and LLG quality separately against ground truth.

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
| `goal_similarity.py` | Compare reconstructed goals with original HLGs; list same-actor branches for textual duplicate review |
| `global_goal_evaluator.py` | Collect evaluator decisions and combine their votes |
| `result_storage.py` | Save dataset results and batch summaries for execution and evaluation |

Offline threshold selection and Excel handling live in `experiment_threhold_tuning/workbook.py`. Run `python -m experiment_threhold_tuning prepare` to prepare comparisons and `python -m experiment_threhold_tuning tune` to update the workbook after annotation.
