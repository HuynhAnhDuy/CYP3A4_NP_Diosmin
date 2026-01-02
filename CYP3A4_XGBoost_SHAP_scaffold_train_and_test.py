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

# ==== 0. Config flag ====
RUN_SHAP_TRAIN = False   # Đổi True nếu muốn chạy SHAP trên tập train

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
        return Chem.MolToSmiles(core, isomericSmiles=False,
                                kekuleSmiles=False, canonical=True)
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

# === 3. Main SHAP analysis pipeline ===
def main(random_state=42):
    set_seed(random_state)

    # === Load and process train/test data ===
    df_train = pd.read_csv("/home/andy/andy/CYP3A4_NP_Diosmin/CYP3A4_x_train.csv").dropna(subset=['canonical_smiles', 'Label'])
    df_test = pd.read_csv("/home/andy/andy/CYP3A4_NP_Diosmin/CYP3A4_x_test.csv").dropna(subset=['canonical_smiles', 'Label'])

    for df in [df_train, df_test]:
        df['scaffold'] = df['canonical_smiles'].apply(get_scaffold)
        df.dropna(subset=['scaffold'], inplace=True)
        df['fingerprint'] = df['canonical_smiles'].apply(smiles_to_ecfp)

    X_train = np.array(df_train['fingerprint'].tolist())
    y_train = df_train['Label'].astype(int).values

    X_test = np.array(df_test['fingerprint'].tolist())
    y_test = df_test['Label'].astype(int).values

    # === Train XGBoost ===
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
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)

    # === Output folder ===
    output_dir = f"shap_XGB_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # === 1. SHAP on train set (optional) ===
    if RUN_SHAP_TRAIN:
        shap_values_train = explainer.shap_values(X_train)
        per_sample_mean_train = shap_values_train.mean(axis=1)
        df_shap_train = pd.DataFrame({
            "scaffold": df_train['scaffold'].values,
            "mean_shap": per_sample_mean_train
        })
        df_summary_train = (df_shap_train.groupby("scaffold")["mean_shap"]
                            .mean().reset_index()
                            .sort_values(by="mean_shap", ascending=False))
        df_summary_train["effect"] = df_summary_train["mean_shap"].apply(lambda x: "positive" if x > 0 else "negative")

        df_shap_train.to_csv(f"{output_dir}/scaffold_shap_per_sample_train.csv", index=False)
        df_summary_train.to_csv(f"{output_dir}/scaffold_shap_summary_train.csv", index=False)

    # === 2. SHAP on test set ===
    shap_values_test = explainer.shap_values(X_test)
    per_sample_mean_test = shap_values_test.mean(axis=1)
    df_shap_test = pd.DataFrame({
        "scaffold": df_test['scaffold'].values,
        "mean_shap": per_sample_mean_test
    })
    df_summary_test = (df_shap_test.groupby("scaffold")["mean_shap"]
                       .mean().reset_index()
                       .sort_values(by="mean_shap", ascending=False))
    df_summary_test["effect"] = df_summary_test["mean_shap"].apply(lambda x: "positive" if x > 0 else "negative")

    df_shap_test.to_csv(f"{output_dir}/scaffold_shap_per_sample_test.csv", index=False)
    df_summary_test.to_csv(f"{output_dir}/scaffold_shap_summary_test.csv", index=False)

    # === Log info ===
    print(f"\n✅ SHAP done. Results saved in: {output_dir}")
    print(f"📊 Train set: {len(df_train)} molecules, {df_train['scaffold'].nunique()} scaffolds")
    print(f"📊 Test set: {len(df_test)} molecules, {df_test['scaffold'].nunique()} scaffolds")
    if RUN_SHAP_TRAIN:
        print(f"💾 Train summary → scaffold_shap_summary_train.csv")
    print(f"💾 Test summary  → scaffold_shap_summary_test.csv")

if __name__ == "__main__":
    main(random_state=42)
