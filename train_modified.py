import pandas as pd
from rdkit import Chem

# ================== CẤU HÌNH ================== #
FULL_FILE     = "/home/andy/andy/hepatoxicity_VoiVoi/Hepatotoxicity.csv"
EXTERNAL_FILE = "/home/andy/andy/hepatoxicity_VoiVoi/PAs_VoiVoi.csv"

SMILES_COL = "SMILES"   # đổi thành "canonical_smiles" nếu file bạn dùng tên cột đó

# Tên file output sau khi loại trùng khỏi full
FULL_FILTERED_FILE = "Hepatotoxicity_full_no_overlap_with_external.csv"
# ============================================= #


def canonicalize_smiles(smi):
    """
    Chuẩn hóa về canonical SMILES bằng RDKit.
    Nếu SMILES không hợp lệ -> trả về None.
    """
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def main():
    # 1) Đọc full + external
    full_df = pd.read_csv(FULL_FILE)
    ext_df  = pd.read_csv(EXTERNAL_FILE)

    if SMILES_COL not in full_df.columns or SMILES_COL not in ext_df.columns:
        raise ValueError(f"Không tìm thấy cột '{SMILES_COL}' trong một trong hai file.")

    print("🔄 Chuẩn hóa canonical SMILES cho FULL...")
    full_df["can_smiles_std"] = full_df[SMILES_COL].apply(canonicalize_smiles)

    print("🔄 Chuẩn hóa canonical SMILES cho EXTERNAL...")
    ext_df["can_smiles_std"] = ext_df[SMILES_COL].apply(canonicalize_smiles)

    # 2) Tạo set canonical SMILES của external
    ext_smiles_set = set(ext_df["can_smiles_std"].dropna().unique())
    print(f"\nExternal có {len(ext_smiles_set)} canonical SMILES (sau chuẩn hóa, bỏ NA).")

    # 3) Đánh dấu trong FULL xem SMILES có nằm trong external không
    full_df["is_overlap_with_external"] = full_df["can_smiles_std"].isin(ext_smiles_set)

    n_full = len(full_df)
    n_overlap = full_df["is_overlap_with_external"].sum()
    n_remain = n_full - n_overlap

    print(f"\nFULL tổng: {n_full}")
    print(f" - Số mẫu FULL trùng SMILES với EXTERNAL: {n_overlap}")
    print(f" - Số mẫu FULL còn lại (không trùng): {n_remain}")

    # 4) Loại các hàng trùng khỏi FULL
    full_filtered_df = full_df[~full_df["is_overlap_with_external"]].copy()

    # 5) Lưu full đã lọc (bỏ các mẫu trùng SMILES với external)
    full_filtered_df.to_csv(FULL_FILTERED_FILE, index=False)
    print(f"\n✅ Đã lưu FULL đã loại trùng vào: {FULL_FILTERED_FILE}")

    # (tuỳ chọn) In vài SMILES trùng để bạn kiểm tra
    if n_overlap > 0:
        print("\nMột vài SMILES trùng (ví dụ):")
        dup_smiles = full_df.loc[full_df["is_overlap_with_external"], "can_smiles_std"].dropna().unique()
        for s in list(dup_smiles)[:10]:
            print("  ", s)


if __name__ == "__main__":
    main()
