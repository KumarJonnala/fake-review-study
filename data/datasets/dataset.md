# Study datasets

## The eight datasets


| file | rows | real | human fake | synthetic | pool | train/test |
|---|---|---|---|---|---|---|
| `d1_human_real_vs_human_fake.csv` | 1592 | 796 | 796 | — | — | 1194 / 398 |
| `d2_sampled_human_real_vs_synthetic.csv` | 1592 | 796 | — | 796 | small | 1194 / 398 |
| `d3_sampled_human_real_vs_mixed.csv` | 1592 | 796 | 398 | 398 | small | 1194 / 398 |
| `d3_unsampled_human_real_vs_mixed.csv` | 2392 | 796 | 796 | 800 | small | 1794 / 598 |
| `d2.5_sampled_human_real_vs_synthetic.csv` | 1592 | 796 | — | 796 | large | 1194 / 398 |
| `d2.5_unsampled_human_real_vs_synthetic.csv` | 2716 | 796 | — | 1920 | large | 2037 / 679 |
| `d3.5_sampled_human_real_vs_mixed.csv` | 1592 | 796 | 398 | 398 | large | 1194 / 398 |
| `d3.5_unsampled_human_real_vs_mixed.csv` | 3512 | 796 | 796 | 1920 | large | 2634 / 878 |

**d1** — the human corpus in full, no sampling. Real vs. MTurk-written fakes.

**d2_sampled** — real humans vs. LLM-written fakes only. 796 of the 800 generated reviews,
balanced across the 4 models (199 each) and the 16 prompt variations (12-13 per
model-cell, 48-52 pooled).

**d3_sampled** — the mixed condition: real humans vs. a 50/50 blend of human and LLM
fakes. 398 human fakes plus 398 generated (100/100/99/99 per model, 6-7 per model-cell).

**d3_unsampled** — every human review and every generated review, with no subsampling at
all. 200 per model, the entire pool.

**d2.5_sampled / d3.5_sampled** — the large-model equivalents of d2 and d3: 199 per model
(12–13 per model-cell), and 100/100/99/99 (6–7 per model-cell) respectively.

**d2.5_unsampled / d3.5_unsampled** — every human review and all 1920 large-model reviews,
no subsampling.

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
