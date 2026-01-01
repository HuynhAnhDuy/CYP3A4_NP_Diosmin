# mlp_runner.py

import numpy as np
import pandas as pd
import os
import random
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    accuracy_score, recall_score, confusion_matrix, matthews_corrcoef,
    roc_curve, auc, precision_score, f1_score, balanced_accuracy_score,
    precision_recall_curve
)

# ================== CẤU HÌNH CHUNG ================== #

BASE_PREFIX = "Hepatotoxicity"


# ================== XÂY DỰNG MLP ================== #

def build_mlp(input_dim):
    """
    MLP cho dữ liệu tabular (fingerprints + descriptors).
    input_dim: số lượng feature (số cột của X sau khi bỏ Index).
    """
    model = Sequential()
    # Layer 1
    model.add(Dense(512, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.4))
    # Layer 2
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    # Layer 3
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    # Output
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ================== TRAIN + EVALUATE ================== #

def evaluate_mlp(
    x_train, y_train, x_test, y_test,
    epochs=50, batch_size=64, run_id=1, seed=42
):
    """
    Huấn luyện MLP trên 1 feature set và tính metrics trên test.
    Trả về:
        - dict metrics
        - y_prob_train, y_prob_test
        - y_train_true, y_test_true
    """
    # Set seed cho reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    x_train = np.asarray(x_train)
    x_test  = np.asarray(x_test)
    y_train = np.asarray(y_train).ravel()
    y_test  = np.asarray(y_test).ravel()

    model = build_mlp(x_train.shape[1])

    # Early stopping để tránh overfit quá mức
    es = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[es],
        verbose=1
    )

    print(f"\n📉 Training loss/val_loss for Run {run_id}:")
    for epoch in range(len(history.history['loss'])):
        train_loss = history.history['loss'][epoch]
        val_loss = history.history['val_loss'][epoch]
        print(f"  Epoch {epoch+1:02d}: loss = {train_loss:.4f}, val_loss = {val_loss:.4f}")

    # Dự đoán xác suất
    y_test_prob = model.predict(x_test).ravel()
    y_test_pred = (y_test_prob > 0.5).astype(int)

    y_train_prob = model.predict(x_train).ravel()
    y_train_pred = (y_train_prob > 0.5).astype(int)

    # Metrics trên TEST
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
    """
    Chạy MLP trên từng feature set trong danh sách feature_sets.
    Mỗi feature set lặp lại num_runs để lấy Mean ± SD.
    """
    results_all = {}
    all_metrics_raw = []

    # Thư mục lưu y_prob theo timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prob_folder = f"Prob_Hepa_MLP/Prob_{timestamp}"
    os.makedirs(prob_folder, exist_ok=True)
    print(f"\n📁 Sẽ lưu y_prob vào thư mục: {prob_folder}")

    for fs in feature_sets:
        print(f"\n=== 🔬 Evaluating feature set: {fs.upper()} ===")
        fs_file = fs.lower()

        try:
            # X: bỏ cột Index bằng index_col=0
            x_train = pd.read_csv(f"{BASE_PREFIX}_x_train_{fs_file}.csv", index_col=0).values
            x_test  = pd.read_csv(f"{BASE_PREFIX}_x_test_{fs_file}.csv", index_col=0).values

            y_train = pd.read_csv(f"{BASE_PREFIX}_y_train.csv", index_col=0).values.ravel()
            y_test  = pd.read_csv(f"{BASE_PREFIX}_y_test.csv", index_col=0).values.ravel()
        except FileNotFoundError as e:
            print(f"[SKIP] Thiếu file cho {fs.upper()}: {e}")
            continue

        metrics_keys = [
            "Accuracy Test", "Balanced Accuracy Test", "AUROC Test", "AUPRC Test",
            "MCC Test", "Precision Test", "Sensitivity Test", "Specificity Test", "F1 Test"
        ]
        metrics_summary = {k: [] for k in metrics_keys}

        for run in range(num_runs):
            seed = 42 + run
            print(f"\n🚀 Run {run+1}/{num_runs} for {fs.upper()} (seed={seed})...")

            metrics, y_train_prob, y_test_prob, y_train_true, y_test_true = evaluate_mlp(
                x_train, y_train, x_test, y_test,
                epochs=50, batch_size=64, run_id=run+1, seed=seed
            )

            # Lưu metrics vào list để tính Mean ± SD
            for k in metrics_keys:
                metrics_summary[k].append(metrics[k])

            metrics["Feature_Set"] = fs.upper()
            metrics["Run"] = run + 1
            metrics["Seed"] = seed
            all_metrics_raw.append(metrics)

            # Lưu y_prob train/test
            train_df = pd.DataFrame({
                "y_true": y_train_true,
                "y_prob": y_train_prob
            })
            test_df = pd.DataFrame({
                "y_true": y_test_true,
                "y_prob": y_test_prob
            })

            train_path = f"{prob_folder}/{BASE_PREFIX}_train_prob_{fs_file}_run{run+1}.csv"
            test_path  = f"{prob_folder}/{BASE_PREFIX}_test_prob_{fs_file}_run{run+1}.csv"

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            print(f"💾 Đã lưu: {train_path}, {test_path}")

        # Tính Mean ± SD theo từng metric
        summary = {k: (np.nanmean(v), np.nanstd(v)) for k, v in metrics_summary.items()}
        results_all[fs] = summary

        print(f"\n📊 --- {fs.upper()} Results (Mean ± SD over {num_runs} runs) ---")
        for k, (mean_val, std_val) in summary.items():
            print(f"{k}: {mean_val:.3f} ± {std_val:.3f}")

    # Lưu raw metrics từng run
    df_raw = pd.DataFrame(all_metrics_raw)
    df_raw.to_csv(f"{BASE_PREFIX}_MLP_feature_sets_metrics_raw.csv", index=False)
    print(f"\n✅ Saved raw results: {BASE_PREFIX}_MLP_feature_sets_metrics_raw.csv")

    return results_all


# ================== MAIN ================== #

def main():
    # 6 feature sets đơn bạn đang dùng cho XGB/LGBM
    feature_sets = ["ecfp", "rdkit", "maccs", "phychem", "estate", "substruct","all_features","selected_features"]

    results_by_fs = run_all_feature_sets(feature_sets, num_runs=3)

    # Xuất bảng Mean ± SD dạng summary
    df_export = pd.DataFrame({
        fs.upper(): {
            metric: f"{mean:.3f} ± {std:.3f}" for metric, (mean, std) in metrics.items()
        }
        for fs, metrics in results_by_fs.items()
    }).T

    df_export.to_csv(f"{BASE_PREFIX}_MLP_feature_sets_metrics.csv")
    print(f"\n✅ Saved summary: {BASE_PREFIX}_MLP_feature_sets_metrics.csv")


if __name__ == "__main__":
    main()
