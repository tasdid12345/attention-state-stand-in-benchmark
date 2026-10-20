# Stand-in attention-state benchmark (EEG Eye State)

This repository runs a small proxy experiment: can a class-balanced, CV-tuned,
isotonic-calibrated LightGBM classifier beat a standard random forest at detecting
eye open/closed state from EEG features?

## Setup

```
pip install -r requirements.txt
python analysis.py
```

The script writes `results.json` and `figures/macro_f1_comparison.png`.

## Data

- Primary source: OpenML dataset 1471 (EEG Eye State).
- Fallback: if OpenML is unavailable after retries (it returned HTTP 412 in CI),
  the same dataset is loaded from the UCI ML Repository (id 264) via `ucimlrepo`.
  The `dataset.data_source` field in `results.json` records which source was used.

## Method

- One stratified 80/20 split (seed 42).
- Baseline: RandomForestClassifier (300 trees), seeds 42–44, raw features.
- Improved: LGBMClassifier (class_weight="balanced"), grid searched with 5-fold
  stratified CV on the training set only, then wrapped in isotonic CalibratedClassifierCV.
- Main metric: macro-F1 on the held-out test split.

## Caveats

This is a proxy task, not an attention or mindful-eating dataset. Results come from a
single split.

## Results (latest run)

On the EEG Eye State proxy dataset (not an attention-labelled dataset), a standard random forest classifier reached 93.1% macro-F1 on the held-out test split, while a class-balanced, calibrated LightGBM reached 94.8%, a small gain of 1.7 points from a single split.

| Metric | Value |
|---|---|
| baseline_score | 0.9309 |
| improved_score | 0.9484 |
| improvement | 0.01748 |
| baseline_score_std | 0.001415 |
| improved_score_std | 0 |
| improved_cv_macro_f1 | 0.9408 |
| baseline_accuracy | 0.9319 |
| improved_accuracy | 0.9489 |
| baseline_brier | 0.07164 |
| improved_brier | 0.03967 |
| baseline_f1 | 0.9226 |
| improved_f1 | 0.9431 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Proxy dataset: EEG eye state is not a mindful-eating or attention-labelled dataset. Results are for a single split. LightGBM with default row/column sampling is nearly deterministic, so its seed std is near zero. Raw features are used for both models; EEG readings contain known amplitude outliers that tree models tolerate. Secondary metrics are computed on the same test split and were not used for selection. Data was loaded from 'uci'; if OpenML was unavailable (HTTP 412 in CI), the same EEG Eye State data was fetched from the UCI ML Repository (id 264).

## Reference

This project is motivated by the following paper:

> C Sas, L Guluzade, FA Bertran, K Höök (2025). *Opportunities and Challenges of Designing for Mindful Eating: Integrating HCI Research on Mindfulness, Human-Food Interaction and Somaesthetics*. 38th International. https://www.scienceopen.com/hosted-document?doi=10.14236/ewic/BCSHCI2025.3

Paper author: Prof. Corina Sas, Lancaster University (c.sas@lancaster.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
