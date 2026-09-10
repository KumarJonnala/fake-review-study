# Study datasets

Four datasets for the fake-review detection study, each with a seeded 75-25 stratified
train/test split. All four are **generated artifacts** — do not edit them by hand. They are
rebuilt from scratch by:

```
python3 src/make_datasets_and_splits.py [--config src/config/datasets.yaml]
```

Composition, inputs and the seed live in `src/config/datasets.yaml`. Re-running is
byte-identical (seed 42, verified).

## The four datasets

Every dataset answers the same question — can a classifier tell real hotel reviews from
fake ones — and they differ **only in what the fake half is made of**. The three sampled
datasets hold the real half and the totals fixed, so any gap in performance between them
is attributable to composition rather than to sample size or to which rows were drawn.

| file | rows | real | human fake | synthetic | train/test |
|---|---|---|---|---|---|
| `d1_human_real_vs_human_fake.csv` | 1592 | 796 | 796 | — | 1194 / 398 |
| `d2_sampled_human_real_vs_synthetic.csv` | 1592 | 796 | — | 796 | 1194 / 398 |
| `d3_sampled_human_real_vs_mixed.csv` | 1592 | 796 | 398 | 398 | 1194 / 398 |
| `d3_unsampled_human_real_vs_mixed.csv` | 2392 | 796 | 796 | 800 | 1794 / 598 |

**d1** — the human corpus in full, no sampling. Real vs. MTurk-written fakes.

**d2_sampled** — real humans vs. LLM-written fakes only. 796 of the 800 generated reviews,
balanced across the 4 models (199 each) and the 16 prompt variations (12-13 per
model-cell, 48-52 pooled).

**d3_sampled** — the mixed condition: real humans vs. a 50/50 blend of human and LLM
fakes. 398 human fakes plus 398 generated (100/100/99/99 per model, 6-7 per model-cell).

**d3_unsampled** — every human review and every generated review, with no subsampling at
all. 200 per model, the entire pool.

### ⚠️ d3_unsampled is class-imbalanced

796 real (33.3%) against 1596 fake (66.7%). The stratified split preserves that 1:2 ratio
in both halves. Two consequences:

- The majority-class accuracy baseline is **66.7%, not 50%**. Raw accuracy is not readable
  on its own for this dataset.
- Its accuracy is **not comparable** to d1 / d2_sampled / d3_sampled, which are all 50/50.

Quote F1 or PR-AUC, and consider `class_weight="balanced"` when training on it.

## Columns

| column | description |
|---|---|
| `text` | the review. The only feature. |
| `Binary_label` | `real` or `fake` |
| `label` | the same thing as 0/1 — **`real` = 0, `fake` = 1** |
| `hotel` | one of the 20 Chicago hotels (`affinia` … `talbott`). From the corpus's `Category` column. |
| `origin` | `human_real`, `human_fake`, or the generating model's name |
| `source_dataset` | filename the row was read from |
| `cell_id_variation` | the prompt variation, `{length}_{sentiment}_{structure}_{example_mode}`. **Empty for all human rows.** |
| `split` | `train` or `test` |
