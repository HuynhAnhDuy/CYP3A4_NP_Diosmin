import numpy as np
import pandas as pd
from datetime import datetime
import os
import gc

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, recall_score, confusion_matrix, matthews_corrcoef,
    roc_auc_score, average_precision_score, precision_score, f1_score,
    balanced_accuracy_score
)

# ================== CẤU HÌNH CHUNG ================== #

BASE_PREFIX = "CYP3A4"

X_TRAIN_PATH = f"{BASE_PREFIX}_x_train_all_features.csv"
Y_TRAIN_PATH = f"{BASE_PREFIX}_y_train.csv"
X_TEST_PATH  = f"{BASE_PREFIX}_x_test_all_features.csv"
Y_TEST_PATH  = f"{BASE_PREFIX}_y_test.csv"

N_SPLITS   = 5        # CV trong mỗi Y-random run
N_Y_RANDOM = 30       # số lần Y-random

MODEL_NAME = f"{BASE_PREFIX}_XGB_all_features_YRandom"


# ================== HÀM TÍNH METRICS ================== #

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    labels = np.unique(y_true)
    if set(labels) == {0, 1}:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    else:
        spec = np.nan

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = np.nan

    try:
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auprc = np.nan

    metrics = {
        "Accuracy": acc,
        "Balanced_Accuracy": bal_acc,
        "AUROC": auroc,
        "AUPRC": auprc,
        "MCC": mcc,
        "Precision": prec,
        "Sensitivity": rec,
        "Specificity": spec,
        "F1": f1
    }
    return metrics


# ================== XÂY DỰNG MODEL XGB ================== #

def build_xgb_model(y_train, random_state=42):
    y_train = np.asarray(y_train).ravel()
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.1,
        min_child_weight=1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=4,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False
    )
    return model


# ================== Y-RANDOMIZATION + 5-FOLD CV ================== #

def run_y_randomization_cv(X, y, out_dir, n_runs=30, n_splits=5, base_seed=2025):
    """
    Y-randomization chuẩn:
      - Mỗi run: shuffle y -> y_perm
      - 5-fold Stratified CV trên (X, y_perm)
      - Lấy mean metrics qua 5 folds cho run đó
    Cuối cùng:
      - Lưu metrics từng fold cho tất cả runs
      - Lưu metrics mean cho mỗi run
      - Lưu summary Mean ± SD qua tất cả runs
    """
    print(f"\n===== Y-RANDOMIZATION (with {n_splits}-fold CV, {n_runs} runs) =====")

    runs_mean_metrics = []   # metrics trung bình mỗi run
    folds_all_metrics = []   # metrics từng fold cho tất cả runs

    for run in range(1, n_runs + 1):
        print(f"\n>>> Y-Random Run {run}/{n_runs}")
        rng = np.random.default_rng(base_seed + run)
        y_perm = rng.permutation(y)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=base_seed + run)
        fold_metrics_list = []

        for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y_perm), start=1):
            print(f"  - Fold {fold}/{n_splits}")

            X_tr, X_val = X[train_idx], X[valid_idx]
            y_tr, y_val = y_perm[train_idx], y_perm[valid_idx]

            model = build_xgb_model(y_tr, random_state=base_seed + run * 10 + fold)
            model.fit(X_tr, y_tr)

            y_val_prob = model.predict_proba(X_val)[:, 1]
            metrics_fold = compute_metrics(y_val, y_val_prob)
            metrics_fold["YRandom_Run"] = run
            metrics_fold["Fold"] = fold
            fold_metrics_list.append(metrics_fold)
            folds_all_metrics.append(metrics_fold)

            for k, v in metrics_fold.items():
                if k not in ["YRandom_Run", "Fold"]:
                    print(f"    {k}: {v:.3f}")

            # ===== GIẢI PHÓNG BỘ NHỚ SAU MỖI FOLD =====
            del model, X_tr, X_val, y_tr, y_val, y_val_prob
            gc.collect()
            # ===========================================

        # Tính mean metrics qua 5 folds cho run này
        mean_metrics_run = {}
        metric_keys = [k for k in fold_metrics_list[0].keys() if k not in ["YRandom_Run", "Fold"]]
        for m in metric_keys:
            vals = [fm[m] for fm in fold_metrics_list]
            mean_val = np.nanmean(vals)
            mean_metrics_run[m] = mean_val
        mean_metrics_run["YRandom_Run"] = run
        runs_mean_metrics.append(mean_metrics_run)

        print(f"--- Mean metrics for Y-Random Run {run} ---")
        for m in metric_keys:
            print(f"  {m}: {mean_metrics_run[m]:.3f}")

        # ===== GIẢI PHÓNG BỘ NHỚ SAU MỖI RUN =====
        del y_perm, fold_metrics_list, mean_metrics_run, metric_keys
        gc.collect()
        # =========================================

    # Lưu metrics từng fold cho tất cả runs
    df_folds = pd.DataFrame(folds_all_metrics)
    folds_path = os.path.join(out_dir, f"{MODEL_NAME}_YRandom_{n_runs}runs_5CV_folds.csv")
    df_folds.to_csv(folds_path, index=False)
    print(f"\n✅ Đã lưu metrics từng fold của tất cả Y-random runs: {folds_path}")

    # Lưu metrics trung bình mỗi run (1 hàng / run)
    df_runs = pd.DataFrame(runs_mean_metrics)
    runs_path = os.path.join(out_dir, f"{MODEL_NAME}_YRandom_{n_runs}runs_mean_per_run.csv")
    df_runs.to_csv(runs_path, index=False)
    print(f"✅ Đã lưu metrics mean (qua {n_splits} folds) cho từng Y-random run: {runs_path}")

    # Summary Mean ± SD qua tất cả runs (1 hàng, mỗi ô "mean ± sd")
    metric_cols = [c for c in df_runs.columns if c != "YRandom_Run"]
    summary_formatted = {}
    for m in metric_cols:
        vals = df_runs[m].values
        mean_val = np.nanmean(vals)
        std_val = np.nanstd(vals)
        summary_formatted[m] = f"{mean_val:.3f} ± {std_val:.3f}"

    df_summary = pd.DataFrame([summary_formatted])
    summary_path = os.path.join(out_dir, f"{MODEL_NAME}_YRandom_summary_mean_sd.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"✅ Đã lưu Y-randomization summary (Mean ± SD dạng 1 hàng): {summary_path}")

    print("\n===== Y-RANDOMIZATION summary (Mean ± SD across runs) =====")
    for m, v in summary_formatted.items():
        print(f"{m}: {v}")

    # ===== GIẢI PHÓNG BỘ NHỚ SAU KHI HOÀN THÀNH =====
    del runs_mean_metrics, folds_all_metrics, df_folds, df_runs, df_summary, summary_formatted, metric_cols
    gc.collect()
    # ================================================


# ================== MAIN ================== #

def main():
    # Tạo thư mục output theo timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = f"{MODEL_NAME}_Validation_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 Thư mục lưu kết quả: {out_dir}")

    # ----- Load TRAIN & TEST rồi gộp thành ALL ----- #
    print("\n📥 Đang load dữ liệu TRAIN...")
    X_train_df = pd.read_csv(X_TRAIN_PATH, index_col=0)
    y_train_df = pd.read_csv(Y_TRAIN_PATH, index_col=0)

    print("📥 Đang load dữ liệu TEST...")
    X_test_df = pd.read_csv(X_TEST_PATH, index_col=0)
    y_test_df = pd.read_csv(Y_TEST_PATH, index_col=0)

    X_all = np.vstack([
        X_train_df.values,
        X_test_df.values
    ]).astype(np.float32)

    y_all = np.concatenate([
        y_train_df.values.ravel(),
        y_test_df.values.ravel()
    ])

    print(f"X_all shape: {X_all.shape}")
    print(f"y_all shape: {y_all.shape}")

    # Giải phóng DataFrame gốc nếu không dùng nữa
    del X_train_df, X_test_df, y_train_df, y_test_df
    gc.collect()

    # ----- Y-randomization + 5-fold CV ----- #
    run_y_randomization_cv(X_all, y_all, out_dir, n_runs=N_Y_RANDOM, n_splits=N_SPLITS, base_seed=2025)

    # Sau khi xong có thể giải phóng luôn X_all, y_all
    del X_all, y_all
    gc.collect()

    print(f"\n🎯 Hoàn tất Y-randomization (5-CV trong mỗi run) cho {MODEL_NAME}")


if __name__ == "__main__":
    main()
