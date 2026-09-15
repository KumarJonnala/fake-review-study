# Fake Review Classification: XGBoost, SVM and BERT

Can a classifier trained to spot human-written fake hotel reviews also spot LLM-generated
ones? This repo generates the synthetic reviews, builds the datasets that isolate that
question, and trains three model families across all of them.

## The eight datasets

Built by `src/make_datasets_and_splits.py` into `data/datasets/`. They differ **only in
what the fake half is made of**, so any gap in classifier performance between them is
attributable to composition rather than to sample size or to which rows were drawn.

The `.5` variants use the **large**-model generation pool (deepseek, glm-5.3, gpt5.6
luna/terra); `d2`/`d3` use the **small** local pool (gemma4, ministral, llama3.2, qwen3.5).

| dataset | rows | real | human fake | synthetic | pool | accuracy floor |
|---|---|---|---|---|---|---|
| `d1_human_real_vs_human_fake` | 1592 | 796 | 796 | — | — | 0.500 |
| `d2_sampled_human_real_vs_synthetic` | 1592 | 796 | — | 796 | small | 0.500 |
| `d2.5_sampled_human_real_vs_synthetic` | 1592 | 796 | — | 796 | large | 0.500 |
| `d3_sampled_human_real_vs_mixed` | 1592 | 796 | 398 | 398 | small | 0.500 |
| `d3.5_sampled_human_real_vs_mixed` | 1592 | 796 | 398 | 398 | large | 0.500 |
| `d3_unsampled_human_real_vs_mixed` | 2392 | 796 | 796 | 800 | small | **0.667** |
| `d2.5_unsampled_human_real_vs_synthetic` | 2716 | 796 | — | 1920 | large | **0.707** |
| `d3.5_unsampled_human_real_vs_mixed` | 3512 | 796 | 796 | 1920 | large | **0.773** |

See `data/datasets/dataset.md` for the full column reference and composition notes.

**The three unsampled datasets are class-imbalanced.** Their majority-class accuracy floors
are 0.667, 0.707 and 0.773 rather than 0.5, so accuracy is not comparable across them or
against the five balanced sets — on `d3.5_unsampled` a model can score 77% by always
guessing "fake". Quote F1 or PR-AUC. Every results file records its own floor alongside the
metrics, so this is visible at the point of reading.

## Splits

Each dataset carries its own **75/25 train/test boundary in a `split` column**, drawn once
by `make_datasets_and_splits.py` and stratified on `origin` so every generating model is
proportionally present in the test half.

The training scripts never recompute that boundary. `split_from_column` in `src/data.py`
reads it back and carves **15% of the train half** off as validation, leaving the test half
untouched until a single final evaluation:

| dataset | train | validation | test |
|---|---|---|---|
| the five sampled datasets | 1014 | 180 | 398 |
| d3_unsampled | 1524 | 270 | 598 |
| d2.5_unsampled | 1731 | 306 | 679 |
| d3.5_unsampled | 2238 | 396 | 878 |

## Feature policy

**`text` is the only feature.** The datasets also carry `origin`, `source_dataset` and
`cell_id_variation` for stratification and error analysis. Each one determines or narrows
the label — a generated filename identifies a model, hence `fake` — so none may ever reach
a vectorizer. The corpus's own `source` column is dropped outright at dataset-build time,
because it is perfectly confounded with the label (`MTurk` = fake, `TripAdvisor`/`Web` =
real).

## Running

```bash
# Build the datasets (read-only inputs; safe to re-run, byte-identical at seed 42)
python3 src/make_datasets_and_splits.py

# All three model families across all four datasets
python3 src/run_all.py

# Or one family, or one dataset
python3 -m src.svm_classifier.train
python3 -m src.xgboost_classifier.train --dataset d1_human_real_vs_human_fake
```

Results are written to `results/<model>/<dataset>/` as each dataset completes, so an
interrupted run keeps whatever already finished. BERT additionally accepts
`--skip-existing` to resume a run that was cut short.

### On the HPC cluster

One job script per model, each covering all four datasets. They write to disjoint output
directories and share only a read-only venv, so all three can queue at once:

```bash
sbatch src/slurm_svm.sh                    # ~45 min   (wall clock 2h)
sbatch src/slurm_xgboost.sh                # ~1-3h     (wall clock 6h)
sbatch src/slurm_bert.sh                   # ~1h30 on the H100 (wall clock 4h)

SETUP_ONLY=1 sbatch src/slurm_svm.sh       # build the shared venv and stop
SKIP_EXISTING=1 sbatch src/slurm_bert.sh   # resume an interrupted BERT run
```

`config.yaml` lists the datasets smallest-first, and each trainer writes a dataset's
results as it finishes, so a job that hits its wall clock keeps everything up to the cut
rather than losing the lot.

The three scripts are thin SBATCH headers over `src/slurm_common.sh`, which holds the venv
build, the thread pinning and the results summary so they cannot drift apart. All target
`--partition=gpu --nodelist=ant2`, the only partition available; the two CPU-only jobs
deliberately request no `--gres` so they do not hold an H100 idle.

Those runtimes are measured, not guessed. Job 1675581 ran a full 24-combination BERT grid
on one 1014-row dataset in 8 minutes on the H100. The worst-case XGBoost fit (800 trees,
depth 7) and an rbf `probability=True` SVM fit were benchmarked per dataset size:

| train rows | XGBoost worst-case fit | SVM fit |
|---|---|---|
| 1014 | 33s | 8s |
| 1524 | 52s | 16s |
| 2238 | 75s | 31s |

Wall clocks are set roughly 2x above the pessimistic total, which assumes every grid
combination is the slowest one.

### Tuning budget

| model | combinations | note |
|---|---|---|
| XGBoost | 36 | trimmed from 288; see the comment in `src/config/xgboost_grid.yaml` |
| SVM | 24 | 36 minus the linear-kernel duplicates where `gamma` is ignored |
| BERT | 24 | 24 x 4 datasets = 96 fine-tunings, longer than `slurm_bert.sh`'s wall clock |

## Layout

- `src/generate_synthetic_reviews.py` — the 2x2x2x2 factorial LLM review generator
- `src/validate_generated.py` — instruction-compliance and stylometry audit of generated output
- `src/config.py` — every generator constant, with the measurement behind it
- `src/make_datasets_and_splits.py` — builds the four datasets and their splits
- `src/{xgboost,svm,bert}_classifier/train.py` — the three model families
- `src/{data,vectorizer,evaluation}.py` — shared loading, features and metrics
- `src/slurm_{xgboost,svm,bert}.sh` — one cluster job per model
- `src/slurm_common.sh` — the shared job body those three source
- `src/config/` — experiment grids and reproducibility settings

## Research protocol

1. The test half stays untouched until model selection is complete.
2. All three model families see identical train/validation/test rows for a given dataset.
3. Hyperparameters are tuned on the training set and selected on validation F1.
4. The test set is evaluated once per selected configuration.
5. Report accuracy, precision, recall, F1, ROC-AUC, PR-AUC and the confusion matrix — and
   read accuracy against the majority-class floor recorded in each results file.
6. For stronger claims, repeat over multiple seeds and report mean +/- standard deviation.
7. Record exact package versions and random seeds.

**Known asymmetry:** XGBoost and SVM refit the selected configuration on train+validation
before evaluating test; BERT reloads the best run's checkpoint, which saw the train half
only. Refitting BERT would double the GPU time. This belongs in any write-up of the results.

The default seed is 42 throughout.
