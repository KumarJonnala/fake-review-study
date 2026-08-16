# Fake Review Classification: XGBoost, SVM and BERT

This project is structured for reproducible research experiments on the supplied hotel-review dataset.

## Dataset audit

The supplied CSV contains **1,592 reviews**, with **796 real** and **796 fake** reviews.
There are no missing values and no duplicate review texts in the supplied file.

Important: `source` is perfectly confounded with the target in this dataset:
- `fake` = MTurk
- `real` = TripAdvisor/Web

Therefore, the primary experiments intentionally use **review text only**. Do **not** include `source`, `is_synthetic`, `domain`, or `Category` as model features in the main paper experiments, because they would create target leakage or an artificial shortcut.

## Models

- `xgboost_classifier/`: TF-IDF + XGBoost
- `svm_classifier/`: TF-IDF + SVM
- `bert_classifier/`: Transformer fine-tuning (default: `bert-base-uncased`)
- `common/`: shared data splitting and evaluation utilities
- `config/`: experiment grids and reproducibility settings

## Recommended research protocol

1. Keep the test set untouched until model selection is complete.
2. Use the same fixed stratified train/validation/test split for XGBoost, SVM and BERT.
3. Tune hyperparameters on the training set using the validation set.
4. Report the final test set once per selected configuration.
5. For stronger claims, repeat the complete experiment over multiple random seeds and report mean ± standard deviation.
6. Report Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC and the confusion matrix.
7. Record the exact package versions and random seeds.

The default split is 70% train / 15% validation / 15% test with seed 42.
