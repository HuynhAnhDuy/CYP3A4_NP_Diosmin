import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ CSV
file_path = "/home/andy/andy/CYP3A4_NP_Diosmin/Visualization/CYP3A4_XGB_feature_sets_metrics_Specificity.csv"
df = pd.read_csv(file_path)

# Chuẩn hoá tên cột, tránh dính khoảng trắng
df.columns = df.columns.str.strip()

print(df.head())
print("Columns:", df.columns.tolist())

# Kiểm tra cột dữ liệu có tồn tại không
if {"XGB model", "Specificity", "Error"}.issubset(df.columns):

    # Sort theo BACC TĂNG dần -> BACC nhỏ ở dưới, lớn ở TRÊN
    df_sorted = df.sort_values(by="Specificity", ascending=True).reset_index(drop=True)

    # Dùng index số làm vị trí y
    y_pos = range(len(df_sorted))

    # Tạo biểu đồ ngang
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(
        y_pos,
        df_sorted["Specificity"],
        xerr=df_sorted["Error"],
        capsize=4,
        color="#2A45B1",
        edgecolor="#372699",
        height=0.5,
        alpha=0.9
    )

    # Xác định vị trí cột số liệu
    max_value = df_sorted["Specificity"].max()
    offset = max_value + 0.05
    ax.set_xlim(0, offset + 0.05)  # mở rộng trục x để text không bị cắt

    for i, value in enumerate(df_sorted["Specificity"]):
        text_color = "red" if value >= 0.9 else "black"
        ax.text(
            offset, i, f"{value:.3f}",
            ha="left", va="center",
            fontsize=12,
            fontname="sans-serif",
            color=text_color
        )

    # Gán nhãn trục y theo XGB model đã sort
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["XGB model"], fontsize=12, fontname="sans-serif")

    # Tiêu đề trục
    ax.set_ylabel("XGB model", fontsize=12, fontweight='bold',
                  fontname='sans-serif', fontstyle='italic')
    ax.set_xlabel("Specificity*", fontsize=12, fontweight='bold',
                  fontname='sans-serif', fontstyle='italic')

    # Chỉnh font trục x
    ax.tick_params(axis="x", labelsize=12)

    # Bỏ viền trên & phải
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Ghi chú
    plt.figtext(
        0.65, 0.01,
        "*Specificity values ≥ 0.9 are highlighted in red",
        ha="right", fontsize=10, color="red",
        fontname="sans-serif", fontstyle='italic', fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig("CYP3A4_metrics_XGB_Specificity.svg", format="svg", dpi=300)
    print("Biểu đồ đã được lưu!!!")

else:
    print("Error: CSV file phải có các cột: XGB model, BACC, Error")
