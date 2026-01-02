import pandas as pd
from rdkit import Chem

# ================== CẤU HÌNH ================== #
FULL_FILE     = "/home/andy/andy/CYP3A4_NP_Diosmin/CYP3A4.csv"
EXTERNAL_FILE = "/home/andy/andy/CYP3A4_NP_Diosmin/CYP3A4_x_external.csv"

SMILES_COL = "SMILES"        # đổi thành "canonical_smiles" nếu file bạn dùng tên cột đó
ID_COL     = "Name"          # <-- CỘT "tên mẫu" / mã mẫu. Ví dụ: "Name", "ID", "MolID", ...

FULL_FILTERED_FILE = "CYP3A4_modified.csv"

# In tối đa bao nhiêu dòng trùng ra terminal
PRINT_MAX_ROWS = 50
# ============================================= #


def canonicalize_smiles(smi):
    """Chuẩn hóa về canonical SMILES bằng RDKit. Nếu SMILES không hợp lệ -> None."""
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def ensure_id_col(df: pd.DataFrame, id_col: str) -> str:
    """
    Đảm bảo có cột định danh để in tên mẫu.
    Nếu không có id_col trong df -> tạo cột '__row_index__' từ index.
    Trả về tên cột định danh sẽ dùng.
    """
    if id_col in df.columns:
        return id_col
    df["__row_index__"] = df.index.astype(str)
    return "__row_index__"


def main():
    # 1) Đọc full + external
    full_df = pd.read_csv(FULL_FILE)
    ext_df  = pd.read_csv(EXTERNAL_FILE)

    if SMILES_COL not in full_df.columns or SMILES_COL not in ext_df.columns:
        raise ValueError(f"Không tìm thấy cột '{SMILES_COL}' trong một trong hai file.")

    full_id_col = ensure_id_col(full_df, ID_COL)
    ext_id_col  = ensure_id_col(ext_df, ID_COL)

    print("🔄 Chuẩn hóa canonical SMILES cho FULL...")
    full_df["can_smiles_std"] = full_df[SMILES_COL].apply(canonicalize_smiles)

    print("🔄 Chuẩn hóa canonical SMILES cho EXTERNAL...")
    ext_df["can_smiles_std"] = ext_df[SMILES_COL].apply(canonicalize_smiles)

    # 2) Tạo set canonical SMILES của external
    ext_smiles_set = set(ext_df["can_smiles_std"].dropna().unique())
    print(f"\nExternal có {len(ext_smiles_set)} canonical SMILES (sau chuẩn hóa, bỏ NA).")

    # 3) Đánh dấu overlap
    ext_df["is_overlap_with_full"] = ext_df["can_smiles_std"].isin(
        set(full_df["can_smiles_std"].dropna().unique())
    )
    full_df["is_overlap_with_external"] = full_df["can_smiles_std"].isin(ext_smiles_set)

    n_full = len(full_df)
    n_ext  = len(ext_df)
    n_overlap_full_rows = int(full_df["is_overlap_with_external"].sum())
    n_overlap_ext_rows  = int(ext_df["is_overlap_with_full"].sum())

    # unique smiles overlap (giữa 2 tập)
    overlap_smiles = sorted(
        set(full_df.loc[full_df["is_overlap_with_external"], "can_smiles_std"].dropna().unique())
        .intersection(set(ext_df["can_smiles_std"].dropna().unique()))
    )
    n_overlap_unique_smiles = len(overlap_smiles)

    print("\n========== THỐNG KÊ ==========")
    print(f"FULL tổng: {n_full}")
    print(f"EXTERNAL tổng: {n_ext}")
    print(f"- FULL: số dòng trùng (SMILES nằm trong EXTERNAL): {n_overlap_full_rows}")
    print(f"- EXTERNAL: số dòng trùng (SMILES nằm trong FULL): {n_overlap_ext_rows}")
    print(f"- Số unique canonical SMILES bị overlap giữa 2 tập: {n_overlap_unique_smiles}")

    # 4) In danh sách tên mẫu trùng trong EXTERNAL
    if n_overlap_ext_rows > 0:
        print("\n========== TÊN MẪU TRÙNG TRONG EXTERNAL ==========")

        overlap_ext = ext_df.loc[ext_df["is_overlap_with_full"], [ext_id_col, SMILES_COL, "can_smiles_std"]].copy()
        overlap_ext = overlap_ext.rename(columns={ext_id_col: "external_sample_name"})

        # In ra danh sách tên mẫu (unique)
        unique_names = overlap_ext["external_sample_name"].dropna().astype(str).unique().tolist()
        print(f"Số tên mẫu trùng (unique) trong EXTERNAL: {len(unique_names)}")
        print("Danh sách (tối đa 200 tên đầu tiên):")
        for name in unique_names[:200]:
            print("  -", name)

        # In ra bảng chi tiết (top PRINT_MAX_ROWS)
        print(f"\nChi tiết {min(PRINT_MAX_ROWS, len(overlap_ext))} dòng trùng đầu tiên (EXTERNAL):")
        print(overlap_ext.head(PRINT_MAX_ROWS).to_string(index=False))
    else:
        print("\nKhông có mẫu nào trong EXTERNAL trùng với FULL (theo canonical SMILES).")

    # 5) (giữ logic cũ) Loại các hàng trùng khỏi FULL và lưu
    full_filtered_df = full_df[~full_df["is_overlap_with_external"]].copy()
    full_filtered_df.to_csv(FULL_FILTERED_FILE, index=False)
    print(f"\n✅ Đã lưu FULL đã loại trùng vào: {FULL_FILTERED_FILE}")

    # 6) (tuỳ chọn) nếu bạn cũng muốn in tên mẫu trùng trong FULL
    if n_overlap_full_rows > 0:
        print("\n========== (TUỲ CHỌN) TÊN MẪU TRÙNG TRONG FULL ==========")
        overlap_full = full_df.loc[full_df["is_overlap_with_external"], [full_id_col, SMILES_COL, "can_smiles_std"]].copy()
        overlap_full = overlap_full.rename(columns={full_id_col: "full_sample_name"})
        print(f"Chi tiết {min(PRINT_MAX_ROWS, len(overlap_full))} dòng trùng đầu tiên (FULL):")
        print(overlap_full.head(PRINT_MAX_ROWS).to_string(index=False))


if __name__ == "__main__":
    main()
