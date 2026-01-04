import os
import pandas as pd

# Thư mục chứa các file CSV
base_path = "/home/andy/andy/CYP3A4_NP_Diosmin/Prob_CYP3A4_RF/Prob_2026-01-02_10-36-03"

# Định nghĩa mapping giữa "tên trong file" và "tên Model mong muốn"
feature_configs = {
    "all_features": "RF_Concatenated features",
    "ecfp": "RF_ECFP",
    "maccs": "RF_MACCS keys",
    "estate": "RF_EState",
    "phychem": "RF_PCP",
     "rdkit": "RF_RDKit",
    "substruct": "RF_Substructure",
}

all_models_dfs = []

for feature_key, model_name in feature_configs.items():
    # Tạo tên file cho 3 run
    run_files = [
        os.path.join(base_path, f"CYP3A4_test_prob_{feature_key}_run1.csv"),
        os.path.join(base_path, f"CYP3A4_test_prob_{feature_key}_run2.csv"),
        os.path.join(base_path, f"CYP3A4_test_prob_{feature_key}_run3.csv"),
    ]
    
    # Đọc 3 file
    dfs = []
    for f in run_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Không tìm thấy file: {f}")
        df_run = pd.read_csv(f)
        dfs.append(df_run)
    
    # Giả sử các file đều có cột 'y_true' và 'y_pred'
    # Kiểm tra y_true giữa các run có giống nhau không
    y_true_base = dfs[0]['y_true']
    for i, df_run in enumerate(dfs[1:], start=2):
        if not (df_run['y_true'].values == y_true_base.values).all():
            raise ValueError(f"y_true không khớp giữa các run cho feature {feature_key} (run1 vs run{i})")
    
    # Tính trung bình y_pred của 3 run (theo từng dòng)
    y_pred_stack = pd.concat([df_run['y_prob'] for df_run in dfs], axis=1)
    y_pred_mean = y_pred_stack.mean(axis=1)
    
    # Tạo dataframe cho model này
    df_model = pd.DataFrame({
        "y_true": y_true_base,
        "y_prob": y_pred_mean,
        "Model": model_name
    })
    
    all_models_dfs.append(df_model)

# Gộp tất cả các model lại
df_out = pd.concat(all_models_dfs, ignore_index=True)

# Lưu ra file CSV chung
out_file = os.path.join(base_path, "CYP3A4_test_prob_merged_models_RF.csv")
df_out.to_csv(out_file, index=False)

print(f"Đã lưu file gộp: {out_file}")
