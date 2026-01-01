import numpy as np
import pandas as pd
from datetime import datetime
import os

# ==== Chỉ cần chỉnh 1 dòng này nếu đổi prefix ====
BASE_PREFIX = "Hepatotoxicity"

# XGBoost
try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise SystemExit("XGBoost chưa được cài. Cài bằng: pip install xgboost") from e

from sklearn.metrics import (
    accuracy_score, recall_score, confusion_matrix, matthews_corrcoef,
    roc_curve, auc, precision_score, f1_score, precision_recall_curve,
    balanced_accuracy_score,
)

# === Huấn luyện 1 model XGBoost trên 1 feature set ===
def train_xgboost(
    x_train, x_test, y_train, y_test,
    n_estimators=500, max_depth=6, random_state=42,
    n_jobs=-1, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, gamma=0.1, min_child_weight=1
):
    x_train = np.asarray(x_train)
    x_test  = np.asarray(x_test)
    y_train = np.asarray(y_train).ravel()
    y_test  = np.asarray(y_test).ravel()

    # Tính scale_pos_weight nếu dữ liệu lệch lớp
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0

    params = dict(
        objective="binary:logistic",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        gamma=gamma,
        min_child_weight=min_child_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method="hist",  # đổi 'gpu_hist' nếu dùng GPU
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False
    )

    clf = XGBClassifier(**params)
    clf.fit(
        x_train, y_train,
        eval_set=[(x_test, y_test)],
        verbose=True
    )

    # Dự đoán
    y_pred      = clf.predict(x_test)
    y_prob_test = clf.predict_proba(x_test)[:, 1]
    y_prob_train = clf.predict_proba(x_train)[:, 1]

    # Metrics trên TEST
    accuracy     = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    mcc          = matthews_corrcoef(y_test, y_pred)
    precision    = precision_score(y_test, y_pred, zero_division=0)
    recall       = recall_score(y_test, y_pred, zero_division=0)
    f1           = f1_score(y_test, y_pred, zero_division=0)

    labels = np.unique(y_test)
    if set(labels) == {0, 1}:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    else:
        specificity = np.nan

    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    roc_auc = auc(fpr, tpr)

    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob_test)
    pr_auc = auc(rec_arr, prec_arr)

    return {
        "metrics": {
            "Accuracy": accuracy,
            "Balanced Accuracy": balanced_acc,
            "AUROC": roc_auc,
            "AUPRC": pr_auc,
            "MCC": mcc,
            "Precision": precision,
            "Sensitivity": recall,
            "Specificity": specificity,
            "F1": f1
        },
        "y_prob_train": y_prob_train,
        "y_prob_test": y_prob_test,
        "y_train_true": y_train,
        "y_test_true": y_test
    }

# === Chạy XGB qua tất cả feature sets ===
def run_all_feature_sets(feature_sets, num_runs=3):
    results_all = {}
    all_metrics_raw = []

    # === Tạo thư mục chứa y_prob theo timestamp ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prob_folder = f"Prob_Hepa_XGB/Prob_{timestamp}"
    os.makedirs(prob_folder, exist_ok=True)
    print(f"\n📁 Sẽ lưu y_prob vào: {prob_folder}")

    for fs in feature_sets:
        print(f"\n=== Evaluating feature set: {fs.upper()} ===")
        fs_file = fs.lower()

        try:
            # Lưu ý: index_col=0 vì cột đầu là Index
            x_train = pd.read_csv(f"{BASE_PREFIX}_x_train_{fs_file}.csv", index_col=0).values
            x_test  = pd.read_csv(f"{BASE_PREFIX}_x_test_{fs_file}.csv", index_col=0).values

            y_train = pd.read_csv(f"{BASE_PREFIX}_y_train.csv", index_col=0).values.ravel()
            y_test  = pd.read_csv(f"{BASE_PREFIX}_y_test.csv", index_col=0).values.ravel()
        except FileNotFoundError as e:
            print(f"[SKIP] Thiếu file cho {fs.upper()}: {e}")
            continue

        metrics_keys = [
            "Accuracy", "Balanced Accuracy", "AUROC", "AUPRC",
            "MCC", "Precision", "Sensitivity", "Specificity", "F1"
        ]
        metrics_summary = {k: [] for k in metrics_keys}

        for run in range(num_runs):
            seed = 42 + run
            print(f"\n🚀 Run {run+1}/{num_runs} for {fs.upper()} (seed={seed})...")
            result = train_xgboost(
                x_train, x_test, y_train, y_test,
                n_estimators=500, max_depth=6, random_state=seed,
                n_jobs=-1
            )

            metrics = result["metrics"]
            for k in metrics_keys:
                metrics_summary[k].append(metrics[k])

            metrics["Feature_Set"] = fs.upper()
            metrics["Run"] = run + 1
            metrics["Seed"] = seed
            all_metrics_raw.append(metrics)

            # === Lưu y_prob train/test ===
            train_df = pd.DataFrame({
                "y_true": result["y_train_true"],
                "y_prob": result["y_prob_train"]
            })
            test_df = pd.DataFrame({
                "y_true": result["y_test_true"],
                "y_prob": result["y_prob_test"]
            })

            train_path = f"{prob_folder}/{BASE_PREFIX}_train_prob_{fs_file}_run{run+1}.csv"
            test_path  = f"{prob_folder}/{BASE_PREFIX}_test_prob_{fs_file}_run{run+1}.csv"

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            print(f"💾 Đã lưu: {train_path}, {test_path}")

        # Mean ± SD cho từng feature set
        summary = {k: (np.nanmean(v), np.nanstd(v)) for k, v in metrics_summary.items()}
        results_all[fs] = summary

        print(f"\n📊 --- {fs.upper()} Results (Mean ± SD over {num_runs} runs) ---")
        for k, (mean_val, std_val) in summary.items():
            print(f"{k}: {mean_val:.3f} ± {std_val:.3f}")

    # Xuất file raw từng run
    df_raw = pd.DataFrame(all_metrics_raw)
    df_raw.to_csv(f"{BASE_PREFIX}_XGB_feature_sets_metrics_raw.csv", index=False)
    print(f"\n✅ Saved raw results: {BASE_PREFIX}_XGB_feature_sets_metrics_raw.csv")

    return results_all

# === Hàm chính ===
def main():
    # 8 bộ đặc trưng tương ứng với 8 file X bạn đang có
    feature_sets = [
        "ecfp",
        "rdkit",
        "maccs",
        "phychem",
        "estate",
        "substruct",         # Hepatotoxicity_x_train_substruct.csv
        "all_features",      # Hepatotoxicity_x_train_all_features.csv
        "selected_features"  # Hepatotoxicity_x_train_selected_features.csv
    ]

    results_by_fs = run_all_feature_sets(feature_sets, num_runs=3)

    # Xuất bảng Mean ± SD
    df_export = pd.DataFrame({
        fs.upper(): {metric: f"{mean:.3f} ± {std:.3f}" for metric, (mean, std) in metrics.items()}
        for fs, metrics in results_by_fs.items()
    }).T

    df_export.to_csv(f"{BASE_PREFIX}_XGB_feature_sets_metrics.csv")
    print(f"\n✅ Saved summary: {BASE_PREFIX}_XGB_feature_sets_metrics.csv")


if __name__ == "__main__":
    main()
