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

BASE_PREFIX = "CYP3A4"  # chỉ cần chỉnh dòng này nếu đổi prefix

# File all-features hiện có: train + test sẽ được gộp lại
X_TRAIN_PATH = f"{BASE_PREFIX}_x_train_all_features.csv"
Y_TRAIN_PATH = f"{BASE_PREFIX}_y_train.csv"
X_TEST_PATH  = f"{BASE_PREFIX}_x_test_all_features.csv"
Y_TEST_PATH  = f"{BASE_PREFIX}_y_test.csv"

N_SPLITS = 5        # dùng cho 5-CV

MODEL_NAME = f"{BASE_PREFIX}_XGB_all_features_5CV"


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
    """
    Xây XGBClassifier với hyperparameters best của anh.
    Có tính scale_pos_weight từ y_train để xử lý lệch lớp.
    """
    y_train = np.asarray(y_train).ravel()
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0

    model = XGBClassifier(
        # ======= THAY ĐỔI CÁC THAM SỐ NÀY THEO BEST MODEL CỦA ANH =======
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.1,
        min_child_weight=1,
        # ================================================================
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=4,
        tree_method="hist",  # đổi sang "gpu_hist" nếu dùng GPU
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False
    )
    return model


# ================== 5-FOLD CV TRÊN X_all, Y_all ================== #

def run_kfold_cv(X, y, out_dir, n_splits=5, base_seed=42):
    print(f"\n===== 5-fold CV trên toàn bộ dữ liệu ({n_splits} folds) =====")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=base_seed)

    fold_metrics_list = []
    oof_prob = np.zeros(len(y), dtype=float)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n--- Fold {fold}/{n_splits} ---")

        X_tr, X_val = X[train_idx], X[valid_idx]
        y_tr, y_val = y[train_idx], y[valid_idx]

        model = build_xgb_model(y_tr, random_state=base_seed + fold)
        model.fit(X_tr, y_tr)

        y_val_prob = model.predict_proba(X_val)[:, 1]
        oof_prob[valid_idx] = y_val_prob

        fold_metrics = compute_metrics(y_val, y_val_prob)
        fold_metrics["Fold"] = fold
        fold_metrics_list.append(fold_metrics)

        for k, v in fold_metrics.items():
            if k != "Fold":
                print(f"{k}: {v:.3f}")

        # ===== GIẢI PHÓNG BỘ NHỚ SAU MỖI FOLD =====
        del model, X_tr, X_val, y_tr, y_val, y_val_prob
        gc.collect()
        # ===========================================

    # Lưu metrics từng fold (mỗi fold = 1 hàng, metrics = cột)
    df_cv = pd.DataFrame(fold_metrics_list)
    cv_path = os.path.join(out_dir, f"{MODEL_NAME}_5CV_metrics.csv")
    df_cv.to_csv(cv_path, index=False)
    print(f"\n✅ Đã lưu 5-fold CV metrics: {cv_path}")

    # Lưu OOF predictions
    df_oof = pd.DataFrame({
        "Index": np.arange(len(y)),
        "y_true": y,
        "y_prob_oof": oof_prob,
    })
    oof_path = os.path.join(out_dir, f"{MODEL_NAME}_5CV_oof_predictions.csv")
    df_oof.to_csv(oof_path, index=False)
    print(f"✅ Đã lưu OOF predictions: {oof_path}")

    # Metrics tổng thể trên OOF (1 hàng, metrics = cột)
    overall_metrics = compute_metrics(y, oof_prob)
    print("\n===== OOF metrics (5-fold CV trên toàn bộ dữ liệu) =====")
    for k, v in overall_metrics.items():
        print(f"{k}: {v:.3f}")

    overall_metrics_path = os.path.join(out_dir, f"{MODEL_NAME}_5CV_oof_summary.csv")
    pd.DataFrame([overall_metrics]).to_csv(overall_metrics_path, index=False)
    print(f"✅ Đã lưu OOF summary metrics: {overall_metrics_path}")

    # Tính Mean ± SD theo từng metric trên 5 folds (1 hàng, dạng "mean ± sd")
    metric_cols = [c for c in df_cv.columns if c != "Fold"]
    summary_formatted = {}
    for m in metric_cols:
        vals = df_cv[m].values
        mean_val = np.nanmean(vals)
        std_val = np.nanstd(vals)
        summary_formatted[m] = f"{mean_val:.3f} ± {std_val:.3f}"

    df_cv_mean_sd = pd.DataFrame([summary_formatted])
    cv_mean_sd_path = os.path.join(out_dir, f"{MODEL_NAME}_5CV_metrics_mean_sd.csv")
    df_cv_mean_sd.to_csv(cv_mean_sd_path, index=False)
    print(f"✅ Đã lưu 5-fold CV metrics (Mean ± SD dạng 1 hàng): {cv_mean_sd_path}")

    print("\n===== 5-fold CV Mean ± SD =====")
    for m, v in summary_formatted.items():
        print(f"{m}: {v}")


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

    # Gộp lại
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

    # ----- 5-fold CV trên toàn bộ dữ liệu ----- #
    run_kfold_cv(X_all, y_all, out_dir, n_splits=N_SPLITS, base_seed=42)

    print(f"\n🎯 Hoàn tất 5-CV cho {MODEL_NAME}")


if __name__ == "__main__":
    main()
