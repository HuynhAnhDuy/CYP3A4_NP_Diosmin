import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Đọc dữ liệu ===
df = pd.read_csv("CYP3A4_scaffold_stat_analysis.csv", index_col=0)
df = df[(df['Inhibitor'] + df['Noninhibitor']) >= 3].copy()

# Gán ID tạm thời (Scaffold_001, ...)
df = df.reset_index().rename(columns={'index': 'SMILES'})
df['ID'] = [f"Scaffold_{i+1:03d}" for i in range(len(df))]
df = df.set_index('ID')

# === Tính toán tần suất log10 ===
df['log10_inhibitor'] = np.log10(df['Inhibitor'] + 1)
df['log10_noninhibitor'] = np.log10(df['Noninhibitor'] + 1)

# === Tính OddsRatio, p-value (đã có), rồi phân nhóm theo volcano logic ===
df['log2_odds_ratio'] = np.log2(df['OddsRatio'].replace(0, np.nan))
df['neg_log10_p'] = -np.log10(df['p_value'].replace(0, 1e-300))

def assign_group(row):
    if row['p_value'] < 0.05 and row['log2_odds_ratio'] > 1:
        return 'Inhibitor'
    elif row['p_value'] < 0.05 and row['log2_odds_ratio'] < -1:
        return 'Noninhibitor'
    else:
        return 'Not significant'

df['Group'] = df.apply(assign_group, axis=1)

# Màu cho từng nhóm
colors = {
    'Inhibitor': 'red',
    'Noninhibitor': 'blue',
    'Not significant': 'gray'
}

# === Vẽ scatter plot theo log10(count), tô màu theo nhóm ===
plt.figure(figsize=(7, 5))

for group, color in colors.items():
    sub = df[df['Group'] == group]
    plt.scatter(
        sub['log10_noninhibitor'],
        sub['log10_inhibitor'],
        c=color,
        label=group,
        alpha=0.7,
        s=40
    )

# Đường tham chiếu chéo (x = y)
xy_max = max(df['log10_noninhibitor'].max(), df['log10_inhibitor'].max()) + 0.1
plt.plot([0, xy_max], [0, xy_max], 'k--', linewidth=1)

# Trang trí
plt.xlabel("log₁₀(Noninhibitor count + 1)", fontsize=12, fontweight='bold',
                  fontname='sans-serif', fontstyle='italic')
plt.ylabel("log₁₀(Inhibitor count + 1)", fontsize=12, fontweight='bold',
                  fontname='sans-serif', fontstyle='italic')
plt.title("Scaffold distribution: CYP3A4 Inhibitors and Noninhibitors", fontsize=12, fontweight='bold')
plt.legend(title="Enriched in", loc='upper right')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

# Lưu file
plt.savefig("scaffold_distribution_log10_grouped.svg", format='svg', dpi=300)
