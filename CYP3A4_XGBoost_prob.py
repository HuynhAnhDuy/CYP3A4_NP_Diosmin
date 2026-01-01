import numpy as np
import pandas as pd
from datetime import datetime
import os

BASE_PREFIX = "Hepatotoxicity"
FEATURE_SET = "all_features"   # đổi thành "selected_features" nếu muốn
THRESHOLD = 0.5                # ngưỡng gắn nhãn từ xác suất

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise SystemExit("Chưa cài XGBoost. Cài: pip install xgboost") from e


# --------- 1. Đọc train + test và gộp thành FULL ---------- #
def load_full_data(feature_set):
    """
    Đọc:
      - Hepatotoxicity_x_train_all_features.csv
      - Hepatotoxicity_x_test_all_features.csv
      - Hepatotoxicity_y_train.csv
      - Hepatotoxicity_y_test.csv
    và gộp chúng thành X_full, y_full để train model cuối.
    """
    fs = feature_set.lower()

    x_train_df = pd.read_csv(f"{BASE_PREFIX}_x_train_{fs}.csv", index_col=0)
    x_test_df  = pd.read_csv(f"{BASE_PREFIX}_x_test_{fs}.csv", index_col=0)

    x_train = x_train_df.values
    x_test  = x_test_df.values

    # label
    y_train = pd.read_csv(f"{BASE_PREFIX}_y_train.csv", index_col=0).values.ravel()
    y_test  = pd.read_csv(f"{BASE_PREFIX}_y_test.csv", index_col=0).values.ravel()

    X_full = np.vstack([x_train, x_test])
    y_full = np.concatenate([y_train, y_test])

    print("X_full shape:", X_full.shape)
    print("y_full shape:", y_full.shape)

    return X_full, y_full


# --------- 2. Đọc features cho external ---------- #
def load_external(feature_set):
    """
    Giả định tồn tại file:
      Hepatotoxicity_x_external_all_features.csv
    với cột đầu là Index (ID hợp chất).
    """
    fs = feature_set.lower()
    path = f"{BASE_PREFIX}_x_external_{fs}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file external: {path}")

    x_ext_df = pd.read_csv(path, index_col=0)
    X_ext = x_ext_df.values
    ext_index = x_ext_df.index.copy()

    print("X_external shape:", X_ext.shape)

    return X_ext, ext_index


# --------- 3. Xây dựng model XGB ---------- #
def build_xgb(random_state=42,
              n_estimators=500, max_depth=6,
              learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              reg_alpha=0.1, reg_lambda=1.0, gamma=0.1, min_child_weight=1):

    clf = XGBClassifier(
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
    return clf


# --------- 4. Train trên FULL và predict EXTERNAL ---------- #
def main():
    # 1) Gộp train + test
    X_full, y_full = load_full_data(FEATURE_SET)

    # Tính scale_pos_weight cho dữ liệu lệch lớp
    n_pos = np.sum(y_full == 1)
    n_neg = np.sum(y_full == 0)
    scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0
    print(f"n_pos = {n_pos}, n_neg = {n_neg}, scale_pos_weight = {scale_pos_weight:.3f}")

    # 2) Train model cuối trên FULL
    clf = build_xgb(random_state=42)
    clf.set_params(scale_pos_weight=scale_pos_weight)

    print("\n🚀 Training final XGB model on FULL data...")
    clf.fit(X_full, y_full, verbose=True)

    # 3) Đọc external
    X_ext, ext_index = load_external(FEATURE_SET)

    # 4) Dự đoán xác suất & gắn nhãn
    y_prob_ext = clf.predict_proba(X_ext)[:, 1]
    y_pred_ext = (y_prob_ext >= THRESHOLD).astype(int)

    # 5) Lưu kết quả
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_folder = f"External_Hepa_XGB/Pred_{timestamp}"
    os.makedirs(out_folder, exist_ok=True)

    out_df = pd.DataFrame({
        "Index": ext_index,
        "y_prob": y_prob_ext,
        "y_pred": y_pred_ext
    })

    out_path = os.path.join(out_folder, f"{BASE_PREFIX}_external_pred_{FEATURE_SET}.csv")
    out_df.to_csv(out_path, index=False)

    print(f"\n✅ Đã lưu kết quả external tại: {out_path}")
    print(f"   (Threshold = {THRESHOLD})")


if __name__ == "__main__":
    main()
