import os
import numpy as np
import pandas as pd
from datetime import datetime

BASE_PREFIX = "CYP3A4"
FEATURE_SET = "maccs"   # MACCS

# ===== XGBoost =====
from xgboost import XGBClassifier

# ===== SHAP =====
import shap


def load_X_train(base_prefix: str, feature_set: str) -> pd.DataFrame:
    fs = feature_set.lower()
    path = f"{base_prefix}_full_{fs}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy X_train: {path}")
    return pd.read_csv(path, index_col=0)


def load_y_train(base_prefix: str) -> np.ndarray:
    path = f"{base_prefix}_y_full.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy y_train: {path}")
    return pd.read_csv(path, index_col=0).values.ravel()


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
    # Binary classifier: có thể là list [class0, class1]
    if isinstance(shap_values, list) and len(shap_values) == 2:
        return np.asarray(shap_values[1])
    return np.asarray(shap_values)


def main():
    # 1) Load TRAIN
    X_train = load_X_train(BASE_PREFIX, FEATURE_SET)
    y_train = load_y_train(BASE_PREFIX)

    if len(y_train) != X_train.shape[0]:
        raise ValueError("Mismatch số dòng giữa X_train và y_train")

    print("X_train:", X_train.shape, "| y_train:", y_train.shape)

    # 2) Train XGB on TRAIN
    spw = compute_scale_pos_weight(y_train)
    clf = build_xgb(random_state=42)
    clf.set_params(scale_pos_weight=spw)

    print("\n🚀 Training XGB on TRAIN...")
    clf.fit(X_train.values, y_train, verbose=True)

    # 3) Prepare output folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = f"SHAP_XGB_MACCS_TRAIN/{BASE_PREFIX}_SHAP_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    # 4) Compute SHAP on TRAIN
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_train)
    shap_pos = get_shap_pos_class(shap_values)   # (n_train, n_features)

    # 5) Feature sign list (based on TRAIN SHAP)
    mean_shap = shap_pos.mean(axis=0)
    mean_abs = np.abs(shap_pos).mean(axis=0)

    sign_df = pd.DataFrame({
        "feature": X_train.columns,
        "mean_shap": mean_shap,
        "mean_abs_shap": mean_abs,
    })
    sign_df["sign"] = np.where(sign_df["mean_shap"] >= 0, "pos", "neg")
    sign_df = sign_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    sign_df["rank_abs"] = np.arange(1, len(sign_df) + 1)

    # 6) Save minimal outputs
    sign_df.to_csv(os.path.join(out_dir, "shap_feature_sign_list.csv"), index=False)
    np.save(os.path.join(out_dir, "shap_values.npy"), shap_pos)
    X_train.to_csv(os.path.join(out_dir, "X_shap_used_for_summary.csv"))

    # 7) Quick preview
    top_pos = sign_df[sign_df["sign"] == "pos"].head(15)
    top_neg = sign_df[sign_df["sign"] == "neg"].head(15)

    print(f"\n✅ Saved SHAP (computed on TRAIN) to: {out_dir}")
    print(" - shap_feature_sign_list.csv")
    print(" - shap_values.npy")
    print(" - X_train_used_for_summary.csv")

    print("\nTop POS features (TRAIN mean_shap >= 0):")
    print(top_pos[["feature", "mean_shap", "mean_abs_shap", "rank_abs"]].to_string(index=False))
    print("\nTop NEG features (TRAIN mean_shap < 0):")
    print(top_neg[["feature", "mean_shap", "mean_abs_shap", "rank_abs"]].to_string(index=False))

    print("\nGợi ý vẽ summary plot (TRAIN):")
    print("  X = pd.read_csv('X_train_used_for_summary.csv', index_col=0)")
    print("  shap_values = np.load('shap_values.npy')")
    print("  shap.summary_plot(shap_values, X)")


if __name__ == "__main__":
    main()
