# rf_runner.py

import numpy as np
import pandas as pd
import os
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, confusion_matrix, matthews_corrcoef,
    roc_curve, auc, precision_score, f1_score, balanced_accuracy_score,
    precision_recall_curve
)

# ================== CẤU HÌNH CHUNG ================== #

BASE_PREFIX = "CYP3A4"


# ================== TRAIN + EVALUATE RANDOM FOREST ================== #

def evaluate_rf(
    x_train, y_train, x_test, y_test,
    n_estimators=500,
    max_depth=None,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
):
    x_train = np.asarray(x_train)
    x_test  = np.asarray(x_test)
    y_train = np.asarray(y_train).ravel()
    y_test  = np.asarray(y_test).ravel()

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=n_jobs
    )

    rf.fit(x_train, y_train)

    # ===== Predict probabilities =====
    y_train_prob = rf.predict_proba(x_train)[:, 1]
    y_test_prob  = rf.predict_proba(x_test)[:, 1]

    y_train_pred = (y_train_prob >= 0.5).astype(int)
    y_test_pred  = (y_test_prob >= 0.5).astype(int)

    # ===== Metrics =====
    acc = accuracy_score(y_test, y_test_pred)
    bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    mcc = matthews_corrcoef(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)

    labels = np.unique(y_test)
    if set(labels) == {0, 1}:
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    else:
        spec = np.nan

    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    roc_auc = auc(fpr, tpr)

    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_test_prob)
    pr_auc = auc(rec_arr, prec_arr)

    metrics = {
        "Accuracy Test": acc,
        "Balanced Accuracy Test": bal_acc,
        "AUROC Test": roc_auc,
        "AUPRC Test": pr_auc,
        "MCC Test": mcc,
        "Precision Test": prec,
        "Sensitivity Test": rec,
        "Specificity Test": spec,
        "F1 Test": f1
    }

    return metrics, y_train_prob, y_test_prob, y_train, y_test


# ================== CHẠY QUA CÁC FEATURE SETS ================== #

def run_all_feature_sets(feature_sets, num_runs=3):
    results_all = {}
    all_metrics_raw = []

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prob_folder = f"Prob_CYP3A4_RF/Prob_{timestamp}"
    os.makedirs(prob_folder, exist_ok=True)

    print(f"\n📁 Saving probabilities to: {prob_folder}")

    for fs in feature_sets:
        print(f"\n=== 🔬 Evaluating feature set: {fs.upper()} ===")
        fs_file = fs.lower()

        try:
            x_train = pd.read_csv(f"{BASE_PREFIX}_x_train_{fs_file}.csv", index_col=0).values
            x_test  = pd.read_csv(f"{BASE_PREFIX}_x_test_{fs_file}.csv", index_col=0).values

            y_train = pd.read_csv(f"{BASE_PREFIX}_y_train.csv", index_col=0).values.ravel()
            y_test  = pd.read_csv(f"{BASE_PREFIX}_y_test.csv", index_col=0).values.ravel()
        except FileNotFoundError as e:
            print(f"[SKIP] Missing file for {fs.upper()}: {e}")
            continue

        metrics_keys = [
            "Accuracy Test", "Balanced Accuracy Test", "AUROC Test", "AUPRC Test",
            "MCC Test", "Precision Test", "Sensitivity Test", "Specificity Test", "F1 Test"
        ]
        metrics_summary = {k: [] for k in metrics_keys}

        for run in range(num_runs):
            seed = 42 + run
            print(f"\n🚀 Run {run+1}/{num_runs} (seed={seed})")

            metrics, y_train_prob, y_test_prob, y_train_true, y_test_true = evaluate_rf(
                x_train, y_train, x_test, y_test,
                random_state=seed
            )

            for k in metrics_keys:
                metrics_summary[k].append(metrics[k])

            metrics.update({
                "Feature_Set": fs.upper(),
                "Run": run + 1,
                "Seed": seed
            })
            all_metrics_raw.append(metrics)

            pd.DataFrame({
                "y_true": y_train_true,
                "y_prob": y_train_prob
            }).to_csv(
                f"{prob_folder}/{BASE_PREFIX}_train_prob_{fs_file}_run{run+1}.csv",
                index=False
            )

            pd.DataFrame({
                "y_true": y_test_true,
                "y_prob": y_test_prob
            }).to_csv(
                f"{prob_folder}/{BASE_PREFIX}_test_prob_{fs_file}_run{run+1}.csv",
                index=False
            )

        summary = {k: (np.nanmean(v), np.nanstd(v)) for k, v in metrics_summary.items()}
        results_all[fs] = summary

        print(f"\n📊 {fs.upper()} Mean ± SD")
        for k, (m, s) in summary.items():
            print(f"{k}: {m:.3f} ± {s:.3f}")

    pd.DataFrame(all_metrics_raw).to_csv(
        f"{BASE_PREFIX}_RF_feature_sets_metrics_raw.csv", index=False
    )

    return results_all


# ================== MAIN ================== #

def main():
    feature_sets = [
        "ecfp", "rdkit", "maccs", "phychem",
        "estate", "substruct", "all_features", "selected_features","selfies" 
    ]

    results_by_fs = run_all_feature_sets(feature_sets, num_runs=3)

    df_export = pd.DataFrame({
        fs.upper(): {
            metric: f"{mean:.3f} ± {std:.3f}"
            for metric, (mean, std) in metrics.items()
        }
        for fs, metrics in results_by_fs.items()
    }).T

    df_export.to_csv(f"{BASE_PREFIX}_RF_feature_sets_metrics.csv")
    print("\n✅ RF summary saved")


if __name__ == "__main__":
    main()
