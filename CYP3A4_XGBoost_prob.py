import numpy as np
import pandas as pd
from datetime import datetime
import os

BASE_PREFIX = "CYP3A4_external"
FEATURE_SET = "all_features"   # hoặc "selected_features"
THRESHOLD = 0.5                # ngưỡng gắn nhãn từ xác suất

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise SystemExit("Chưa cài XGBoost. Cài: pip install xgboost") from e


def load_train_data(feature_set: str):
    """Đọc X_train và y_train"""
    fs = feature_set.lower()

    x_train_path = f"{BASE_PREFIX}_x_train_{fs}.csv"
    y_train_path = f"{BASE_PREFIX}_y_train.csv"

    if not os.path.exists(x_train_path):
        raise FileNotFoundError(f"Không tìm thấy X_train: {x_train_path}")
    if not os.path.exists(y_train_path):
        raise FileNotFoundError(f"Không tìm thấy y_train: {y_train_path}")

    x_train_df = pd.read_csv(x_train_path, index_col=0)
    y_train = pd.read_csv(y_train_path, index_col=0).values.ravel()

    X_train = x_train_df.values

    if len(y_train) != X_train.shape[0]:
        raise ValueError(f"Mismatch số dòng: X_train={X_train.shape[0]} nhưng y_train={len(y_train)}")

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    return X_train, y_train


def load_external(feature_set: str):
    """Đọc X_external; cột đầu là Index/ID hợp chất."""
    fs = feature_set.lower()
    path = f"{BASE_PREFIX}_x_test_{fs}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file external: {path}")

    x_ext_df = pd.read_csv(path, index_col=0)
    X_ext = x_ext_df.values
    ext_index = x_ext_df.index.copy()

    print("X_external shape:", X_ext.shape)
    return X_ext, ext_index


def build_xgb(
    random_state=42,
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    gamma=0.1,
    min_child_weight=1,
):
    return XGBClassifier(
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
        n_jobs=-1,
        tree_method="hist",
        eval_metric="logloss",
        use_label_encoder=False,
    )


def main():
    # 1) Load train
    X_train, y_train = load_train_data(FEATURE_SET)

    # 2) scale_pos_weight tính từ TRAIN
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0
    print(f"n_pos(train)={n_pos}, n_neg(train)={n_neg}, scale_pos_weight={scale_pos_weight:.3f}")

    # 3) Train model trên TRAIN
    clf = build_xgb(random_state=42)
    clf.set_params(scale_pos_weight=scale_pos_weight)

    print("\n🚀 Training XGB model on TRAIN only...")
    clf.fit(X_train, y_train, verbose=True)

    # 4) Load external
    X_ext, ext_index = load_external(FEATURE_SET)

    # 5) Predict external
    y_prob_ext = clf.predict_proba(X_ext)[:, 1]
    y_pred_ext = (y_prob_ext >= THRESHOLD).astype(int)

    # 6) Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_folder = f"External_CYP3A4_XGB/Pred_{timestamp}"
    os.makedirs(out_folder, exist_ok=True)

    out_df = pd.DataFrame(
        {"Index": ext_index, "y_prob": y_prob_ext, "y_pred": y_pred_ext}
    )

    out_path = os.path.join(out_folder, f"{BASE_PREFIX}_external_pred_{FEATURE_SET}.csv")
    out_df.to_csv(out_path, index=False)

    print(f"\n✅ Đã lưu kết quả external tại: {out_path}")
    print(f"   (Threshold = {THRESHOLD})")


if __name__ == "__main__":
    main()
