"""Stand-in experiment: can a class-balanced, calibrated LightGBM detect a binary
attention-related state (EEG eye open/closed, OpenML 1471) better than a standard
random forest? Seeds 42, 43, 44 for the randomized models; test set used once."""
import json
import sys
import time
from io import BytesIO

import matplotlib
matplotlib.use("Agg")  # headless plotting on CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMClassifier
from scipy.io import arff
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

SEED = 42
SEEDS = [42, 43, 44]
OPENML_ID = 1471
META_URL = f"https://www.openml.org/api/v1/json/data/{OPENML_ID}"
MAX_ROWS = 200_000  # subsample guard for the 20-minute CI budget (dataset is ~15k rows)


def get(url, as_json=False, retries=5):
    """GET with timeout=60 and exponential backoff on HTTP 429/5xx and network errors."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
        except requests.RequestException as e:
            err = str(e)
        else:
            if r.status_code < 400:
                return r.json() if as_json else r.content
            if r.status_code != 429 and r.status_code < 500:
                r.raise_for_status()  # non-retryable client error
            err = f"HTTP {r.status_code}"
        if attempt == retries - 1:
            raise RuntimeError(f"download failed after {retries} tries: {err}")
        print(f"  download failed ({err}); retrying in {2 ** attempt}s")
        time.sleep(2 ** attempt)


def load_data():
    """Download the ARFF file from OpenML and return features X and binary target y."""
    meta = get(META_URL, as_json=True)["data_set_description"]
    print(f"Downloading '{meta['name']}' from OpenML (id {OPENML_ID}) ...")
    data, _ = arff.loadarff(BytesIO(get(meta["url"])))
    df = pd.DataFrame(data)
    for c in df.columns:  # nominal attributes come back as bytes
        if df[c].dtype == object:
            df[c] = df[c].str.decode("utf-8")
    y = df.iloc[:, -1].astype(int).to_numpy()  # last column = eye state label
    X = df.iloc[:, :-1].astype(float).to_numpy()
    if len(y) > MAX_ROWS:  # stratified subsample for runtime safety
        X, _, y, _ = train_test_split(X, y, train_size=MAX_ROWS, stratify=y, random_state=SEED)
    return X, y, meta["url"]


def scores(model, X_te, y_te):
    """Main metric (macro-F1) plus secondary metrics: accuracy, Brier, positive-class F1."""
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    return {
        "macro_f1": f1_score(y_te, pred, average="macro"),
        "accuracy": accuracy_score(y_te, pred),
        "brier": brier_score_loss(y_te, proba),
        "f1": f1_score(y_te, pred),  # positive-class F1
    }


def main():
    X, y, source_url = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)  # split once
    print(f"Rows: {len(y)} | train: {len(y_tr)} | test: {len(y_te)}")

    # ---- Baseline: standard random forest on raw features, 3 seeds ----
    base = []
    for s in SEEDS:
        rf = RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=-1).fit(X_tr, y_tr)
        base.append(scores(rf, X_te, y_te))
        print(f"  baseline RF seed {s}: macro-F1 = {base[-1]['macro_f1']:.4f}")

    # ---- Improved: class-balanced LightGBM, small CV grid on train only ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid = GridSearchCV(
        LGBMClassifier(class_weight="balanced", random_state=SEED, verbose=-1),
        {"num_leaves": [15, 31], "learning_rate": [0.05, 0.1], "n_estimators": [200, 400]},
        scoring="f1_macro", cv=cv, n_jobs=-1).fit(X_tr, y_tr)
    print(f"  tuned LightGBM params: {grid.best_params_} (CV macro-F1 {grid.best_score_:.4f})")

    # ---- Improved: isotonic calibration of the tuned model, 3 seeds ----
    imp = []
    for s in SEEDS:
        lgb = clone(grid.best_estimator_).set_params(random_state=s)
        cal = CalibratedClassifierCV(lgb, method="isotonic", cv=5).fit(X_tr, y_tr)
        imp.append(scores(cal, X_te, y_te))
        print(f"  improved LGBM seed {s}: macro-F1 = {imp[-1]['macro_f1']:.4f}")

    def agg(runs, k):
        v = np.array([r[k] for r in runs])
        return float(v.mean()), float(v.std())

    b_f1, b_f1_sd = agg(base, "macro_f1")
    i_f1, i_f1_sd = agg(imp, "macro_f1")
    if not (np.isfinite(b_f1) and np.isfinite(i_f1)):
        sys.exit("Non-finite metrics; cannot produce meaningful results.")

    metrics = {
        "baseline_score": b_f1, "improved_score": i_f1,
        "improvement": i_f1 - b_f1,  # higher-is-better: improved - baseline
        "baseline_score_std": b_f1_sd, "improved_score_std": i_f1_sd,
        "improved_cv_macro_f1": float(grid.best_score_),
        "baseline_accuracy": agg(base, "accuracy")[0], "improved_accuracy": agg(imp, "accuracy")[0],
        "baseline_brier": agg(base, "brier")[0], "improved_brier": agg(imp, "brier")[0],
        "baseline_f1": agg(base, "f1")[0], "improved_f1": agg(imp, "f1")[0],
    }

    results = {
        "question": "Can a class-balanced, calibrated gradient-boosting model detect a binary "
                    "attention-related state from physiological sensor features more reliably "
                    "than a standard random forest?",
        "method": "Stand-in experiment on OpenML EEG Eye State: one stratified 80/20 split; "
                  "random forest baseline vs. class-balanced, CV-tuned, isotonic-calibrated LightGBM; "
                  "test set used only for final scoring.",
        "dataset": {
            "name": "EEG Eye State (OpenML 1471)", "source_url": source_url,
            "n_rows": int(len(y)), "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
            "same_as_paper": False, "baseline_is_papers_method": False,
        },
        "metric_name": "macro-F1 on held-out test split",
        "higher_is_better": True,
        "baseline": "RandomForestClassifier(n_estimators=300) on raw features, seeds 42-44",
        "improved": "LGBMClassifier with class_weight='balanced', hyperparameters chosen by 5-fold "
                    "stratified CV on train only, wrapped in isotonic CalibratedClassifierCV; seeds 42-44",
        "metrics": {k: float(v) for k, v in metrics.items()},
        "notes": "Proxy dataset: EEG eye state is not a mindful-eating or attention-labelled dataset. "
                 "Results are for a single split. LightGBM with default row/column sampling is nearly "
                 "deterministic, so its seed std is near zero. Raw features are used for both models; "
                 "EEG readings contain known amplitude outliers that tree models tolerate. "
                 "Secondary metrics are computed on the same test split and were not used for selection.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Wrote results.json")

    # ---- Optional chart: baseline vs improved macro-F1 ----
    import os
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.bar(["Random forest", "Balanced+calibrated LGBM"], [b_f1, i_f1],
           yerr=[b_f1_sd, i_f1_sd], capsize=5, color=["grey", "steelblue"])
    ax.set_ylabel("Test macro-F1")
    ax.set_title("EEG eye state (proxy task)")
    fig.tight_layout()
    fig.savefig("figures/macro_f1_comparison.png", dpi=120)
    print("Saved figures/macro_f1_comparison.png")


if __name__ == "__main__":
    main()
