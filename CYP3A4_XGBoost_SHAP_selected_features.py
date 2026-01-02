import os
import numpy as np
import pandas as pd
from datetime import datetime

BASE_PREFIX = "CYP3A4"
FEATURE_SET = "selected_features"

# ===== XGBoost =====
from xgboost import XGBClassifier

# ===== SHAP =====
import shap


def load_train_selected_features(base_prefix: str, feature_set: str):
    fs = feature_set.lower()
    x_path = f"{base_prefix}_x_train_{fs}.csv"
    y_path = f"{base_prefix}_y_train.csv"

    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Không tìm thấy X_train: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Không tìm thấy y_train: {y_path}")

    X_df = pd.read_csv(x_path, index_col=0)
    y = pd.read_csv(y_path, index_col=0).values.ravel()

    if len(y) != X_df.shape[0]:
        raise ValueError("Mismatch số dòng giữa X_train và y_train")

    return X_df, y


def compute_scale_pos_weight(y: np.ndarray) -> float:
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    return (float(n_neg) / float(n_pos)) if n_pos > 0 else 1.0


def build_xgb(random_state=42):
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.1,
        min_child_weight=1,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="logloss",
        use_label_encoder=False,
    )


def get_shap_pos_class(shap_values):
    if isinstance(shap_values, list) and len(shap_values) == 2:
        return np.asarray(shap_values[1])
    return np.asarray(shap_values)


def main():
    # 1) Load TRAIN
    X_df, y_train = load_train_selected_features(BASE_PREFIX, FEATURE_SET)
    print("X_train:", X_df.shape, "| y_train:", y_train.shape)

    # 2) Train XGB
    spw = compute_scale_pos_weight(y_train)
    clf = build_xgb(random_state=42)
    clf.set_params(scale_pos_weight=spw)

    print("\n🚀 Training XGB...")
    clf.fit(X_df.values, y_train, verbose=True)

    # 3) Prepare output folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = f"SHAP_XGB_SelectedFeatures/{BASE_PREFIX}_SHAP_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    # 4) SHAP on FULL TRAIN (không sampling 2000)
    explainer = shap.TreeExplainer(clf)

    X_shap = X_df  # dùng toàn bộ train
    shap_values = explainer.shap_values(X_shap)
    shap_pos = get_shap_pos_class(shap_values)

    # 5) Feature sign list
    mean_shap = shap_pos.mean(axis=0)
    mean_abs = np.abs(shap_pos).mean(axis=0)

    sign_df = pd.DataFrame({
        "feature": X_shap.columns,
        "mean_shap": mean_shap,
        "mean_abs_shap": mean_abs,
    })
    sign_df["sign"] = np.where(sign_df["mean_shap"] >= 0, "pos", "neg")
    sign_df = sign_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    sign_df["rank_abs"] = np.arange(1, len(sign_df) + 1)

    # 6) Save MINIMAL outputs
    sign_df.to_csv(os.path.join(out_dir, "shap_feature_sign_list.csv"), index=False)
    np.save(os.path.join(out_dir, "shap_values.npy"), shap_pos)
    X_shap.to_csv(os.path.join(out_dir, "X_shap_used_for_summary.csv"))

    print(f"\n✅ Saved SHAP outputs to: {out_dir}")
    print(" - shap_feature_sign_list.csv")
    print(" - shap_values.npy")
    print(" - X_shap_used_for_summary.csv")

    print("\nGợi ý vẽ summary plot:")
    print("  X = pd.read_csv('X_shap_used_for_summary.csv', index_col=0)")
    print("  shap_values = np.load('shap_values.npy')")
    print("  shap.summary_plot(shap_values, X)")


if __name__ == "__main__":
    main()
