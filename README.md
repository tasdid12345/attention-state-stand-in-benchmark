# Balanced, Calibrated Gradient Boosting vs. Random Forest for Attention-State Detection (Proxy Experiment)

This experiment is motivated by "Opportunities and Challenges of Designing for Mindful Eating: Integrating HCI Research on Mindfulness, Human-Food Interaction and Somaesthetics" (2025). That paper discusses how wearable and physiological sensing could support attentional awareness during eating. It does not report a classifier, so this repository tests a general question instead: whether a class-balanced, calibrated gradient-boosting model detects a binary attention-related state more reliably than a standard random forest.

**Question:** Can a class-balanced, calibrated gradient-boosting model detect a binary attention-related state from physiological sensor features more reliably than a standard random forest?

**Dataset:** OpenML "EEG Eye State" (id 1471), with 14 EEG electrode features and a binary eye-open/eye-closed label. It is used as a stand-in for a binary attentional state. It is not the paper's own dataset, and it is not a mindful-eating dataset.

**Baseline:** a standard random forest with 300 trees on the raw features.

**Improved model:** a LightGBM classifier with balanced class weights. Its hyperparameters are chosen by 5-fold stratified cross-validation on the training split only, and its probabilities are calibrated with isotonic regression.

**Evaluation:** one stratified train/test split with a fixed seed. The main metric is macro-F1 on the test split. Accuracy, Brier score, and positive-class F1 are reported as secondary metrics. The randomized models are run with three seeds.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

Results: written to `results.json` (charts are saved to `figures/`).

## Reference

This project is motivated by the following paper:

> C Sas, L Guluzade, FA Bertran, K Höök (2025). *Opportunities and Challenges of Designing for Mindful Eating: Integrating HCI Research on Mindfulness, Human-Food Interaction and Somaesthetics*. 38th International. https://www.scienceopen.com/hosted-document?doi=10.14236/ewic/BCSHCI2025.3

Paper author: Prof. Corina Sas, Lancaster University (c.sas@lancaster.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
