import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

BASE_PREFIX   = "CYP3A4"
CV_MODEL_NAME = f"{BASE_PREFIX}_XGB_all_features_CV"
YR_MODEL_NAME = f"{BASE_PREFIX}_XGB_all_features_YRandom"

# Thư mục 5-CV (chứa OOF summary)
CV_OUT_DIR = "CYP3A4_XGB_all_features_CV_Validation_2026-01-02_11-39-31"
# Thư mục Y-random (chứa YRandom_30runs_mean_per_run.csv)
YR_OUT_DIR = "CYP3A4_XGB_all_features_YRandom_Validation_2026-01-02_11-58-00"
# File metrics gốc (run trên train/test split, nhiều seed, all_features)
ORIG_METRICS_PATH = f"{BASE_PREFIX}_XGB_feature_sets_metrics.csv"
ORIG_FEATURE_SET_ROW = "ALL_FEATURES"   # dòng tương ứng all_features trong file summary

# 1. Đọc metrics Y-random (mean qua 5 folds / run)
yrand_runs_path = os.path.join(
    YR_OUT_DIR, f"{YR_MODEL_NAME}_YRandom_30runs_mean_per_run.csv"
)
df_yrand = pd.read_csv(yrand_runs_path)

# 2. Đọc metrics thật (OOF) từ 5-CV (dùng để vẽ đường ngang)
oof_path = os.path.join(
    CV_OUT_DIR, f"{CV_MODEL_NAME}_5CV_oof_summary.csv"
)
df_oof = pd.read_csv(oof_path)

# 3. Đọc metrics gốc (True Mean ± SD) từ file feature sets summary
try:
    df_orig = pd.read_csv(ORIG_METRICS_PATH, index_col=0)
    if ORIG_FEATURE_SET_ROW not in df_orig.index:
        print(f"[CẢNH BÁO] Không tìm thấy dòng {ORIG_FEATURE_SET_ROW} trong {ORIG_METRICS_PATH}")
        df_orig = None
    else:
        print(f"Đã đọc metrics gốc từ: {ORIG_METRICS_PATH}, dòng {ORIG_FEATURE_SET_ROW}")
except FileNotFoundError:
    df_orig = None
    print(f"[CẢNH BÁO] Không tìm thấy file metrics gốc: {ORIG_METRICS_PATH}")

runs = df_yrand["YRandom_Run"].values  # 1..30

metrics_to_plot = [
    ("Balanced_Accuracy", "BACC"),
    ("AUROC",             "AUROC"),
    ("AUPRC",             "AUPRC"),
    ("MCC",             "MCC"),
]

for metric_col, metric_label in metrics_to_plot:
    if metric_col not in df_yrand.columns:
        print(f"[SKIP] {metric_col} không có trong file Y-random.")
        continue
    if metric_col not in df_oof.columns:
        print(f"[SKIP] {metric_col} không có trong OOF summary.")
        continue

    metric_yrand = df_yrand[metric_col].values
    metric_true = float(df_oof[metric_col].iloc[0])

    mean_rand = metric_yrand.mean()
    std_rand  = metric_yrand.std()

    # Map tên metric để đọc đúng cột trong file metrics gốc
    # (Balanced_Accuracy trong CSV Y-random ↔ "Balanced Accuracy" trong file gốc)
    if metric_col == "Balanced_Accuracy":
        orig_metric_col = "Balanced Accuracy"
    else:
        orig_metric_col = metric_col

    if df_orig is not None and orig_metric_col in df_orig.columns:
        true_mean_sd_str = str(df_orig.loc[ORIG_FEATURE_SET_ROW, orig_metric_col])
        true_text = f"Original {metric_label}  = {true_mean_sd_str}"
    else:
        # fallback: chỉ dùng single value từ OOF nếu không có summary gốc
        true_text = f"True {metric_label} (OOF) = {metric_true:.3f}"

    print(f"\n{metric_label} thật (OOF): {metric_true:.3f}")
    print(f"Y-random {metric_label} mean ± SD: "
          f"{mean_rand:.3f} ± {std_rand:.3f}")

    # 4. Scatter plot metric theo run
    fig, ax = plt.subplots(figsize=(4, 4))

    # điểm Y-random
    ax.scatter(
        runs,
        metric_yrand,
        color="red",
        edgecolors="red",
        alpha=0.8
    )

    # đường ngang metric thật (dùng OOF)
    ax.axhline(metric_true, linestyle="--", linewidth=2, color="blue")

    ax.set_xlabel(
        "Y-randomization run (n = 30)",
        fontsize=12,
        fontweight="bold",
        fontstyle="italic"
    )
    ax.set_ylabel(
        metric_label,
        fontsize=12,
        fontweight="bold",
        fontstyle="italic"
    )

    ax.set_xlim(0.5, runs.max() + 1.0)

    # Text ở giữa figure: True (original) Mean ± SD + Y-random mean ± SD
    center_text = (
        f"{true_text}\n"
        f"Y-random mean = {mean_rand:.3f} ± {std_rand:.3f}"
    )

    ax.text(
        0.5,
        0.5,
        center_text,
        transform=ax.transAxes,
        fontweight="bold",
        horizontalalignment="center",
        verticalalignment="center"
    )

    plt.tight_layout()
    out_svg = os.path.join(
        YR_OUT_DIR,
        f"{YR_MODEL_NAME}_YRandom_{metric_col}_scatter.svg"
    )
    plt.savefig(out_svg, format="svg")
    plt.close(fig)

    print("Đã lưu hình SVG:", out_svg)
