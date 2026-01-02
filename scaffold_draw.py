import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


def draw_scaffold_grid_by_effect(csv_path: str,
                                 effect_filter: str,
                                 out_path: str,
                                 sort_col: str = "mean_shap",
                                 top_n: int = 20,
                                 mols_per_row: int = 5):
    """
    Đọc file scaffold_shap_with_counts_full.csv,
    lọc theo effect (positive / negative / neutral),
    chọn top_n scaffold theo sort_col và vẽ thành 1 hình grid.
    """
    df = pd.read_csv(csv_path)

    # Lọc bỏ hàng không có scaffold (an toàn)
    df = df.dropna(subset=["scaffold"])

    # Lọc theo effect mong muốn
    df = df[df["effect"] == effect_filter]

    if df.empty:
        raise ValueError(f"No scaffolds found with effect = {effect_filter}")

    # Sắp xếp:
    #   - positive: mean_shap càng lớn càng ưu tiên
    #   - negative: mean_shap càng âm (nhỏ) càng ưu tiên
    ascending = True if effect_filter == "negative" else False

    # Sắp xếp và lấy top_n
    df = (
        df.sort_values(by=sort_col, ascending=ascending)
          .head(top_n)
          .reset_index(drop=True)
    )

    # Tạo RDKit mols
    mols = []
    legends = []
    for idx, row in df.iterrows():
        smi = row["scaffold"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue  # bỏ qua nếu scaffold lỗi
        mols.append(mol)

        # Chuẩn bị legend (hiển thị dưới mỗi khung)
        n_total = row.get("n_total", None)
        active_ratio = row.get("active_ratio", None)
        effect = row.get("effect", "")

        legend_parts = [f"{idx+1}. {effect}"]
        if n_total is not None:
            legend_parts.append(f"N={int(n_total)}")
        if active_ratio is not None:
            legend_parts.append(f"AR={active_ratio:.2f}")  # AR = active_ratio

        legend = " | ".join(legend_parts)
        legends.append(legend)

    if not mols:
        raise ValueError(f"No valid scaffolds to draw for effect = {effect_filter}")

    # Vẽ grid: RDKit tự sắp xếp layout
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,   # số scaffold mỗi hàng (anh có thể chỉnh 4, 5, 6...)
        subImgSize=(300, 300),
        legends=legends
    )
    img.save(out_path)
    print(f"✅ Saved scaffold grid for effect={effect_filter} to: {out_path}")


# =========================
# Ví dụ dùng tách 2 hình
# =========================
if __name__ == "__main__":
    csv_file = "shap_XGB_full_20260102_181750/scaffold_shap_with_counts_full.csv"

    # Top 20 positive
    draw_scaffold_grid_by_effect(
        csv_path=csv_file,
        effect_filter="positive",
        out_path="top20_scaffolds_positive.png",
        sort_col="mean_shap",
        top_n=20,
        mols_per_row=4
    )

    # Top 20 negative
    draw_scaffold_grid_by_effect(
        csv_path=csv_file,
        effect_filter="negative",
        out_path="top20_scaffolds_negative.png",
        sort_col="mean_shap",
        top_n=20,
        mols_per_row=4
    )
