import os
import random
import numpy as np
import pandas as pd
import shap
from datetime import datetime
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.MolStandardize import rdMolStandardize
import xgboost as xgb
from typing import Optional

# ==== 1. Set seed for reproducibility ====
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

# ==== 2. Molecule standardization + scaffold extraction ====
def _standardize_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Chuẩn hoá phân tử trước khi lấy scaffold"""
    if mol is None:
        return None
    try:
        params = rdMolStandardize.CleanupParameters()
        mol = rdMolStandardize.Cleanup(mol, params)
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)     # giữ mảnh lớn nhất
        mol = rdMolStandardize.Uncharger().uncharge(mol)                # trung hoá điện tích
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)   # canonical tautomer
        return mol
    except Exception:
        return None

def get_scaffold(smiles: str) -> Optional[str]:
    """Trích xuất Murcko scaffold đã chuẩn hoá từ SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    mol = _standardize_mol(mol)
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if core is None or core.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(
            core,
            isomericSmiles=False,
            kekuleSmiles=False,
            canonical=True
        )
    except Exception:
        return None

def smiles_to_ecfp(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    mol = _standardize_mol(mol)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    return np.zeros(n_bits)

# === 3. Main SHAP analysis pipeline (full dataset) ===
def main(random_state=42):
    set_seed(random_state)

    # === Load train & test, gộp thành full dataset ===
    path_train = "/home/andy/andy/CYP3A4_NP_Diosmin/CYP3A4_x_train.csv"
    path_test  = "/home/andy/andy/CYP3A4_NP_Diosmin/CYP3A4_x_test.csv"

    df_train = pd.read_csv(path_train)
    df_test  = pd.read_csv(path_test)

    # Gộp full dataset, chỉ giữ mẫu có canonical_smiles & Label
    df_full = pd.concat([df_train, df_test], ignore_index=True)
    df_full = df_full.dropna(subset=['canonical_smiles', 'Label'])

    # Chuẩn hoá & trích xuất scaffold, fingerprint cho full dataset
    df_full['scaffold'] = df_full['canonical_smiles'].apply(get_scaffold)
    df_full.dropna(subset=['scaffold'], inplace=True)

    df_full['fingerprint'] = df_full['canonical_smiles'].apply(smiles_to_ecfp)

    X_full = np.array(df_full['fingerprint'].tolist())
    y_full = df_full['Label'].astype(int).values

    # === Train XGBoost trên full dataset ===
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss"
    )
    model.fit(X_full, y_full)

    explainer = shap.TreeExplainer(model)

    # === Output folder ===
    output_dir = f"shap_XGB_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # === SHAP trên full dataset ===
    shap_values_full = explainer.shap_values(X_full)
    # mean SHAP theo sample (trung bình toàn bộ feature)
    per_sample_mean_full = shap_values_full.mean(axis=1)

    df_shap_full = pd.DataFrame({
        "scaffold": df_full['scaffold'].values,
        "mean_shap": per_sample_mean_full
    })

    # Gộp theo scaffold để xem xu hướng dương / âm
    df_summary_full = (
        df_shap_full
        .groupby("scaffold")["mean_shap"]
        .mean()
        .reset_index()
        .sort_values(by="mean_shap", ascending=False)
    )

    # Xu hướng: positive nếu mean_shap > 0, ngược lại negative
    df_summary_full["effect"] = df_summary_full["mean_shap"].apply(
        lambda x: "positive" if x > 0 else "negative"
    )

    # === Thống kê số lượng compound cho từng scaffold trên dataset gốc ===
    df_counts = (
        df_full
        .groupby("scaffold")
        .agg(
            n_total=("canonical_smiles", "size"),
            n_active=("Label", lambda x: (x == 1).sum()),
            n_inactive=("Label", lambda x: (x == 0).sum())
        )
        .reset_index()
    )

    # Thêm tỉ lệ active
    df_counts["active_ratio"] = df_counts["n_active"] / df_counts["n_total"]

    # Gộp với bảng SHAP summary để có cả effect + thống kê số lượng
    df_scaffold_final = (
        df_summary_full
        .merge(df_counts, on="scaffold", how="left")
        .sort_values(by="mean_shap", ascending=False)
    )

    # === Save ===
    df_shap_full.to_csv(f"{output_dir}/scaffold_shap_per_sample_full.csv", index=False)
    df_summary_full.to_csv(f"{output_dir}/scaffold_shap_summary_full.csv", index=False)
    df_scaffold_final.to_csv(f"{output_dir}/scaffold_shap_with_counts_full.csv", index=False)

    # === Log info ===
    n_scaffold_full = df_full['scaffold'].nunique()
    print(f"\n✅ SHAP (full dataset) done. Results saved in: {output_dir}")
    print(f"📊 Full set: {len(df_full)} molecules, {n_scaffold_full} scaffolds")
    print(f"💾 Per-sample SHAP → scaffold_shap_per_sample_full.csv")
    print(f"💾 Scaffold summary → scaffold_shap_summary_full.csv")
    print(f"💾 Scaffold + counts → scaffold_shap_with_counts_full.csv")

if __name__ == "__main__":
    main(random_state=42)
