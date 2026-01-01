"""
This software is used for Advanced Pharmaceutical Analysis
Prof(Assist).Dr.Tarapong Srisongkram – Khon Kaen University
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, MACCSkeys
from rdkit.Chem.EState import Fingerprinter  # E-State

import os

# ------------------------- 1.  ECFP (Morgan) ------------------------- #
def calculate_ecfp(df, smiles_col, radius=10, nBits=4096):
    """
    Calculate ECFP (Morgan fingerprint) as bit vectors.
    radius=10 là giá trị bạn đang dùng; thông thường radius=2 hoặc 3,
    nhưng ở đây giữ nguyên theo script gốc.
    """
    def get_ecfp(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return [None] * nBits
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
            return [int(x) for x in fp.ToBitString()]
        except Exception:
            return [None] * nBits

    ecfp_df = df[smiles_col].apply(get_ecfp).apply(pd.Series)
    ecfp_df.columns = [f"ECFP{i}" for i in range(nBits)]
    return ecfp_df


# ------------------------- 2.  RDKit path-based ---------------------- #
def calculate_rdkit(df, smiles_col, nBits=2048):
    """
    RDKit topological path-based fingerprint.
    """
    def get_rdkit(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return [None] * nBits
            fp = Chem.RDKFingerprint(mol, fpSize=nBits)
            return [int(x) for x in fp.ToBitString()]
        except Exception:
            return [None] * nBits

    rdk_df = df[smiles_col].apply(get_rdkit).apply(pd.Series)
    rdk_df.columns = [f"RDKit{i}" for i in range(nBits)]
    return rdk_df


# ------------------------- 3.  MACCS keys (167) ---------------------- #
def calculate_maccs(df, smiles_col):
    """
    MACCS keys (167 bits). Bit0 thường không dùng về mặt hoá học,
    nhưng ở đây giữ nguyên 167 bits cho nhất quán.
    """
    def get_maccs(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return [None] * 167
            fp = MACCSkeys.GenMACCSKeys(mol)
            return [int(x) for x in fp.ToBitString()]
        except Exception:
            return [None] * 167

    maccs_df = df[smiles_col].apply(get_maccs).apply(pd.Series)
    maccs_df.columns = [f"MACCS{i}" for i in range(167)]
    return maccs_df


# ------------------------- 4.  E-State (79 continuous) --------------- #
def calculate_estate(df, smiles_col):
    """
    79-dimensional E-State indices (continuous values, rounded to 3 dp).
    """
    def get_estate(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return [None] * 79
            values = Fingerprinter.FingerprintMol(mol)[0]
            return [round(v, 3) for v in values]
        except Exception:
            return [None] * 79

    est_df = df[smiles_col].apply(get_estate).apply(pd.Series)
    est_df.columns = [f"EState_{i+1}" for i in range(79)]
    return est_df


# ------------------------- 5.  Physicochemical descriptors ----------- #
def calculate_phychem(df, smiles_col):
    """
    Basic physicochemical descriptors (scalar).
    """
    descriptor_funcs = {
        "MolWt": Descriptors.MolWt,
        "LogP": Descriptors.MolLogP,
        "NumHDonors": Descriptors.NumHDonors,
        "NumHAcceptors": Descriptors.NumHAcceptors,
        "TPSA": rdMolDescriptors.CalcTPSA,
        "NumRotatableBonds": Descriptors.NumRotatableBonds,
        "NumAromaticRings": Descriptors.NumAromaticRings,
        "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings,
        "NumHeteroatoms": rdMolDescriptors.CalcNumHeteroatoms,
        "RingCount": rdMolDescriptors.CalcNumRings,
        "HeavyAtomCount": rdMolDescriptors.CalcNumHeavyAtoms,
        "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings,
    }

    def get_desc(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return [None] * len(descriptor_funcs)
            return [func(mol) for func in descriptor_funcs.values()]
        except Exception:
            return [None] * len(descriptor_funcs)

    desc_df = df[smiles_col].apply(get_desc).apply(pd.Series)
    desc_df.columns = list(descriptor_funcs.keys())
    return desc_df


# ------------------------- 6.  Substructure Counts ------------------- #
def calculate_substructure_counts(df, smiles_col):
    """
    Đếm số lần xuất hiện của một số nhóm thế/substructure quan trọng.
    Dùng để phục vụ interpretability: mối liên hệ nhóm thế ↔ độc tính.
    Bạn có thể chỉnh sửa/ mở rộng dict SMARTS bên dưới theo nhu cầu.
    """

    substructure_smarts = {
        "Nitro": "[O-][N+](=O)[!#8]",
        "Aromatic_Ring": "a1aaaaa1",
        "Hydroxyl_Aliphatic": "[CX4][OH]",
        "Phenol": "c[OH]",
        "Halogen": "[F,Cl,Br,I]",
        "Ketone": "C(=O)C",
        "Aldehyde": "[CX3H1](=O)[#6,#1]",
        "Ester": "C(=O)OC",
        "Carboxylic_Acid": "C(=O)[OH]",
        "Amine_Primary": "[NX3;H2]",
        "Amine_Secondary": "[NX3;H1][#6]",
        "Amide": "C(=O)N",
        "Sulfonamide": "S(=O)(=O)N",
        "Thiol": "[SH]",
        "Thioether": "C-S-C",
        "Imidazole": "c1cnc[nH]1",
        "Thiazole": "c1ncsc1",
        "Heteroaromatic_N": "n1cccc1",
    }

    patterns = {name: Chem.MolFromSmarts(s) for name, s in substructure_smarts.items()}

    def count_substructures(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return [None] * len(patterns)
        counts = []
        for pat in patterns.values():
            if pat is None:
                counts.append(None)
            else:
                matches = mol.GetSubstructMatches(pat)
                counts.append(len(matches))
        return counts

    sub_df = df[smiles_col].apply(count_substructures).apply(pd.Series)
    sub_df.columns = list(substructure_smarts.keys())
    return sub_df


# --------------------------------------------------------------------- #
# 7. Helper: add index & save
# --------------------------------------------------------------------- #
def add_index_and_save(df_fps, df_src_index, filename):
    df_fps.insert(0, "Index", df_src_index)
    df_fps.to_csv(filename, index=False)


# --------------------------------------------------------------------- #
# 8. Main processing function (NO Atom Pair)
# --------------------------------------------------------------------- #
def process_and_save_features(df, smiles_col, prefix):
    # ECFP
    ecfp_df = calculate_ecfp(df, smiles_col)
    add_index_and_save(ecfp_df, df.index, f"{prefix}_ecfp.csv")

    # RDKit path-based
    rdk_df = calculate_rdkit(df, smiles_col)
    add_index_and_save(rdk_df, df.index, f"{prefix}_rdkit.csv")

    # MACCS
    maccs_df = calculate_maccs(df, smiles_col)
    add_index_and_save(maccs_df, df.index, f"{prefix}_maccs.csv")

    # E-State
    est_df = calculate_estate(df, smiles_col)
    add_index_and_save(est_df, df.index, f"{prefix}_estate.csv")

    # Physicochemical
    phychem_df = calculate_phychem(df, smiles_col)
    add_index_and_save(phychem_df, df.index, f"{prefix}_phychem.csv")

    # Substructure counts (for interpretability)
    substruct_df = calculate_substructure_counts(df, smiles_col)
    add_index_and_save(substruct_df, df.index, f"{prefix}_substruct.csv")

    print(f"✅  Finished feature extraction for {prefix}")


# --------------------------------------------------------------------- #
def main():
    #x_train = pd.read_csv("/home/andy/andy/hepatoxicity_VoiVoi/Hepatotoxicity_x_train.csv", index_col=0)
    x_test  = pd.read_csv("/home/andy/andy/hepatoxicity_VoiVoi/PAs_VoiVoi_preprocess.csv", index_col=0)

    #process_and_save_features(x_train, "canonical_smiles", "Hepatotoxicity_x_train")
    process_and_save_features(x_test,  "canonical_smiles", "Hepatotoxicity_x_external")


if __name__ == "__main__":
    main()
