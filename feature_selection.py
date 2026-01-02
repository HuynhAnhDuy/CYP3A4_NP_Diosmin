import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def load_data():
    X_train_all = pd.read_csv("CYP3A4_x_train_all_features.csv")
    X_test_all  = pd.read_csv("CYP3A4_x_test_all_features.csv")

    train_index = X_train_all["Index"].copy()
    test_index  = X_test_all["Index"].copy()

    # Bỏ cột Index khỏi ma trận feature
    X_train_all = X_train_all.drop(columns=["Index"])
    X_test_all  = X_test_all.drop(columns=["Index"])

    # Chỉnh lại đường dẫn / tên cột label cho đúng file của bạn
    y_train = pd.read_csv("CYP3A4_y_train.csv")["Label"]

    return X_train_all, X_test_all, y_train, train_index, test_index


def feature_selection_ecfp_rdkit_only():
    X_train_all, X_test_all, y_train, train_index, test_index = load_data()

    print("Original X_train shape:", X_train_all.shape)

    # 1) Xử lý missing (fill NaN = 0 cho fingerprint / count)
    X_train_all = X_train_all.fillna(0)
    X_test_all  = X_test_all.fillna(0)

    # 2) TÁCH FEATURE:
    #    - ECFP* và RDKit*  -> sẽ làm feature selection
    #    - Các cột còn lại  -> giữ nguyên (MACCS, EState, physchem, substruct, ...)
    fp_cols = [c for c in X_train_all.columns if c.startswith("ECFP") or c.startswith("RDKit")]
    other_cols = [c for c in X_train_all.columns if c not in fp_cols]

    X_train_fp = X_train_all[fp_cols]
    X_test_fp  = X_test_all[fp_cols]

    X_train_other = X_train_all[other_cols]
    X_test_other  = X_test_all[other_cols]

    print(f"Fingerprint block shape (train): {X_train_fp.shape}")
    print(f"Other features block shape (train): {X_train_other.shape}")

    # 3) VarianceThreshold CHỈ trên ECFP/RDKit
    var_selector = VarianceThreshold(threshold=0.0)
    X_train_fp_var = var_selector.fit_transform(X_train_fp)
    X_test_fp_var  = var_selector.transform(X_test_fp)

    fp_cols_after_var = np.array(fp_cols)[var_selector.get_support()]

    print("After variance filter on ECFP+RDKit, X_train_fp_var shape:", X_train_fp_var.shape)

    # 4) RandomForest để chọn feature quan trọng TRÊN KHỐI ECFP+RDKit
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    rf.fit(X_train_fp_var, y_train)

    selector = SelectFromModel(
        rf,
        prefit=True,
        threshold="median"   # có thể điều chỉnh: "mean" hoặc số cụ thể
    )

    X_train_fp_sel = selector.transform(X_train_fp_var)
    X_test_fp_sel  = selector.transform(X_test_fp_var)

    selected_fp_cols = fp_cols_after_var[selector.get_support()]

    print("Selected fingerprint feature count:", len(selected_fp_cols))
    print("X_train_fp_sel shape:", X_train_fp_sel.shape)

    # 5) GHÉP LẠI:
    #    - other_cols giữ nguyên
    #    - ECFP/RDKit đã được chọn (selected_fp_cols)

    # Chuyển fingerprint đã chọn về DataFrame để ghép với other features
    X_train_fp_sel_df = pd.DataFrame(X_train_fp_sel, columns=selected_fp_cols)
    X_test_fp_sel_df  = pd.DataFrame(X_test_fp_sel, columns=selected_fp_cols)

    # Ghép: [other_features | selected_fingerprints]
    X_train_sel_df = pd.concat([X_train_other.reset_index(drop=True),
                                X_train_fp_sel_df.reset_index(drop=True)], axis=1)
    X_test_sel_df  = pd.concat([X_test_other.reset_index(drop=True),
                                X_test_fp_sel_df.reset_index(drop=True)], axis=1)

    # Thêm lại cột Index ở vị trí đầu
    X_train_sel_df.insert(0, "Index", train_index.values)
    X_test_sel_df.insert(0, "Index", test_index.values)

    print("Final selected X_train shape (with other features + selected fp):",
          X_train_sel_df.shape)

    # 6) Lưu lại
    X_train_sel_df.to_csv("CYP3A4_x_train_selected_features.csv", index=False)
    X_test_sel_df.to_csv("CYP3A4_x_test_selected_features.csv", index=False)

    # Ngoài ra lưu danh sách fingerprint đã chọn (chỉ ECFP/RDKit)
    pd.Series(selected_fp_cols).to_csv(
        "CYP3A4_selected_fp_feature_names.csv",
        index=False,
        header=["feature_name"]
    )

    print("✅ Saved selected features:")
    print("   CYP3A4_x_train_selected_features.csv")
    print("   CYP3A4_x_test_selected_features.csv")
    print("   CYP3A4_selected_fp_feature_names.csv")


if __name__ == "__main__":
    feature_selection_ecfp_rdkit_only()
