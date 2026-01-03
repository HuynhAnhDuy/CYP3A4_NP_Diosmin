import os
import numpy as np
import pandas as pd
from typing import Optional, List
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize

# =========================
# 1) Same scaffold function
# =========================
def _standardize_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    if mol is None:
        return None
    try:
        params = rdMolStandardize.CleanupParameters()
        mol = rdMolStandardize.Cleanup(mol, params)
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        return mol
    except Exception:
        return None

def get_scaffold(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(str(smiles))
    mol = _standardize_mol(mol)
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if core is None or core.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(core, isomericSmiles=False, kekuleSmiles=False, canonical=True)
    except Exception:
        return None

# =========================
# 2) Helpers
# =========================
def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def ensure_name_column(df: pd.DataFrame, name_col: Optional[str]) -> pd.DataFrame:
    df = df.copy()
    if name_col is None:
        df["compound_name"] = [f"mol_{i}" for i in range(len(df))]
    else:
        df["compound_name"] = df[name_col].astype(str)
    return df

def detect_smiles_col(df: pd.DataFrame) -> str:
    if "canonical_smiles" in df.columns:
        return "canonical_smiles"
    c = pick_first_existing(df, ["smiles", "SMILES"])
    if c is None:
        raise ValueError("Không tìm thấy cột SMILES. Cần canonical_smiles hoặc smiles/SMILES.")
    return c

def detect_label_col(df: pd.DataFrame) -> str:
    if "Label" in df.columns:
        return "Label"
    c = pick_first_existing(df, ["label", "y", "Y", "Activity", "activity"])
    if c is None:
        raise ValueError("Không tìm thấy cột nhãn (Label) trong train/test.")
    return c

# =========================
# 3) Main
# =========================
def export_pure_scaffold_illustrations_and_external(
    path_train: str,
    path_test: str,
    path_scaffold_summary_csv: str,
    path_external: Optional[str] = None,
    out_dir: str = "illustration_compounds_by_scaffold_pure",
    top_n_scaffolds: int = 6,      # ✅ top 6 scaffold mỗi nhóm
    per_scaffold: int = 3,         # ✅ mỗi scaffold tối đa 3 compounds
    min_n_total_scaffold: int = 3, # ✅ n_total >= 3
    random_state: int = 42
):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(random_state)

    # ---- Load scaffold SHAP summary ----
    df_sum = pd.read_csv(path_scaffold_summary_csv)
    if not {"scaffold", "mean_shap"}.issubset(df_sum.columns):
        raise ValueError("scaffold_shap_summary_full.csv phải có cột scaffold, mean_shap")

    df_sum = df_sum.copy()
    df_sum["effect"] = np.where(df_sum["mean_shap"] > 0, "positive", "negative")
    sum_map = df_sum.set_index("scaffold")[["mean_shap", "effect"]]

    # ---- Load train/test ----
    df_train = pd.read_csv(path_train)
    df_test  = pd.read_csv(path_test)

    smiles_col_train = detect_smiles_col(df_train)
    smiles_col_test  = detect_smiles_col(df_test)
    label_col_train  = detect_label_col(df_train)
    label_col_test   = detect_label_col(df_test)

    name_col_train = pick_first_existing(df_train, ["Name", "name", "compound_name", "ID", "id"])
    name_col_test  = pick_first_existing(df_test,  ["Name", "name", "compound_name", "ID", "id"])

    df_train = ensure_name_column(df_train, name_col_train)
    df_test  = ensure_name_column(df_test,  name_col_test)

    df_train["split"] = "train"
    df_test["split"]  = "test"

    df_train["smiles"] = df_train[smiles_col_train].astype(str)
    df_test["smiles"]  = df_test[smiles_col_test].astype(str)

    df_train["Label"] = df_train[label_col_train].astype(int)
    df_test["Label"]  = df_test[label_col_test].astype(int)

    df_full = pd.concat(
        [df_train[["compound_name", "smiles", "split", "Label"]],
         df_test[["compound_name", "smiles", "split", "Label"]]],
        ignore_index=True
    )

    # ---- Compute scaffold ----
    df_full["scaffold"] = df_full["smiles"].apply(get_scaffold)
    df_full = df_full.dropna(subset=["scaffold"]).reset_index(drop=True)

    # ---- Attach mean_shap/effect from summary ----
    df_full = df_full.join(sum_map, on="scaffold")
    df_full["effect"] = df_full["effect"].fillna("unknown")

    # ---- Count n_total / n_active / n_inactive per scaffold on full ----
    counts = (df_full
              .groupby("scaffold")["Label"]
              .agg(
                  n_total="size",
                  n_active=lambda x: int((x == 1).sum()),
                  n_inactive=lambda x: int((x == 0).sum())
              )
              .reset_index())

    # Merge counts back (for filtering)
    df_scaf = counts.join(sum_map, on="scaffold")
    df_scaf = df_scaf.dropna(subset=["mean_shap"])  # only scaffolds present in SHAP summary

    # ---- Apply eligibility: n_total>=3 and "pure" label presence ----
    df_scaf = df_scaf[df_scaf["n_total"] >= min_n_total_scaffold].copy()

    # Positive-pure: mean_shap>0 and n_inactive==0 (only active)
    pos_pure = df_scaf[(df_scaf["mean_shap"] > 0) & (df_scaf["n_inactive"] == 0)].copy()
    # Negative-pure: mean_shap<0 and n_active==0 (only inactive)
    neg_pure = df_scaf[(df_scaf["mean_shap"] < 0) & (df_scaf["n_active"] == 0)].copy()

    pos_pure = pos_pure.sort_values("mean_shap", ascending=False).head(top_n_scaffolds)
    neg_pure = neg_pure.sort_values("mean_shap", ascending=True).head(top_n_scaffolds)

    top_pos = pos_pure["scaffold"].tolist()
    top_neg = neg_pure["scaffold"].tolist()

    print(f"\nPure scaffold selection (n_total >= {min_n_total_scaffold}):")
    print(f"- positive-pure (only active): {len(top_pos)} / requested {top_n_scaffolds}")
    print(f"- negative-pure (only inactive): {len(top_neg)} / requested {top_n_scaffolds}")

    # ---- Pick compounds: prioritize test then train; cap per_scaffold ----
    def pick_compounds(scaffold_list: List[str], required_label: int) -> pd.DataFrame:
        rows = []
        for scaf in scaffold_list:
            g = df_full[(df_full["scaffold"] == scaf) & (df_full["Label"] == required_label)].copy()
            if g.empty:
                continue

            g_test = g[g["split"] == "test"]
            g_train = g[g["split"] == "train"]

            picked = []
            if len(g_test) > 0:
                k = min(per_scaffold, len(g_test))
                idx = rng.choice(g_test.index.values, size=k, replace=False)
                picked.append(g_test.loc[idx])

            need = per_scaffold - sum(len(x) for x in picked)
            if need > 0 and len(g_train) > 0:
                k2 = min(need, len(g_train))
                idx2 = rng.choice(g_train.index.values, size=k2, replace=False)
                picked.append(g_train.loc[idx2])

            out = pd.concat(picked, ignore_index=True) if picked else g.head(per_scaffold)

            # add scaffold-level meta
            meta = df_scaf[df_scaf["scaffold"] == scaf].iloc[0]
            out["n_total_scaffold_full"] = int(meta["n_total"])
            out["n_active_scaffold_full"] = int(meta["n_active"])
            out["n_inactive_scaffold_full"] = int(meta["n_inactive"])

            rows.append(out)

        if not rows:
            return pd.DataFrame(columns=[
                "compound_name","smiles","split","Label","scaffold","mean_shap","effect",
                "n_total_scaffold_full","n_active_scaffold_full","n_inactive_scaffold_full"
            ])
        return pd.concat(rows, ignore_index=True)

    # positive-pure -> active only
    out_pos = pick_compounds(top_pos, required_label=1)
    # negative-pure -> inactive only
    out_neg = pick_compounds(top_neg, required_label=0)

    out_pos_path = os.path.join(out_dir, f"purePOS_top{len(top_pos)}_scaffolds_ACTIVE_nge{min_n_total_scaffold}_k{per_scaffold}.csv")
    out_neg_path = os.path.join(out_dir, f"pureNEG_top{len(top_neg)}_scaffolds_INACTIVE_nge{min_n_total_scaffold}_k{per_scaffold}.csv")
    out_pos.to_csv(out_pos_path, index=False)
    out_neg.to_csv(out_neg_path, index=False)

    print("\nSaved illustration files:")
    print(f"- {out_pos_path} (rows={len(out_pos)})")
    print(f"- {out_neg_path} (rows={len(out_neg)})")

    # =========================
    # 4) External annotation (UNCHANGED)
    # =========================
    if path_external is not None:
        df_ext = pd.read_csv(path_external)
        smiles_col_ext = detect_smiles_col(df_ext)
        name_col_ext = pick_first_existing(df_ext, ["Name", "name", "compound_name", "ID", "id"])

        df_ext = ensure_name_column(df_ext, name_col_ext)
        df_ext["smiles"] = df_ext[smiles_col_ext].astype(str)

        df_ext["scaffold"] = df_ext["smiles"].apply(get_scaffold)
        df_ext = df_ext.dropna(subset=["scaffold"]).reset_index(drop=True)

        df_ext = df_ext.join(sum_map, on="scaffold")
        df_ext["in_scaffold_summary"] = df_ext["mean_shap"].notna()
        df_ext["effect"] = df_ext["effect"].fillna("unknown")

        df_ext["is_top_positive_scaffold"] = df_ext["scaffold"].isin(top_pos)
        df_ext["is_top_negative_scaffold"] = df_ext["scaffold"].isin(top_neg)

        df_ext["abs_mean_shap"] = df_ext["mean_shap"].abs()
        df_ext = df_ext.sort_values(
            by=["in_scaffold_summary", "abs_mean_shap"],
            ascending=[False, False]
        ).drop(columns=["abs_mean_shap"])

        out_ext_path = os.path.join(out_dir, "external_scaffold_annotation.csv")
        df_ext[[
            "compound_name", "smiles", "scaffold",
            "effect", "mean_shap",
            "in_scaffold_summary",
            "is_top_positive_scaffold",
            "is_top_negative_scaffold"
        ]].to_csv(out_ext_path, index=False)

        n_pos = int((df_ext["effect"] == "positive").sum())
        n_neg = int((df_ext["effect"] == "negative").sum())
        n_unk = int((df_ext["effect"] == "unknown").sum())
        print(f"\nExternal annotation saved: {out_ext_path}")
        print(f"External counts: positive={n_pos}, negative={n_neg}, unknown={n_unk}")

    print(f"\nAll outputs saved in: {out_dir}")


if __name__ == "__main__":
    export_pure_scaffold_illustrations_and_external(
        path_train="/home/andy/andy/CYP3A4_NP_Diosmin/CYP3A4_x_train.csv",
        path_test="/home/andy/andy/CYP3A4_NP_Diosmin/CYP3A4_x_test.csv",
        path_scaffold_summary_csv="shap_XGB_full_20260102_181750/scaffold_shap_summary_full.csv",
        path_external="External_CYP3A4_XGB/CYP3A4_external_x_test_preprocess.csv",
        out_dir="illustration_compounds_by_scaffold_pure",
        top_n_scaffolds=6,
        per_scaffold=3,
        min_n_total_scaffold=3,
        random_state=42
    )
