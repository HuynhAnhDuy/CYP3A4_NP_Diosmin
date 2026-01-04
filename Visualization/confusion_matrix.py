import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# ====== Cấu hình ======
csv_path = "/home/andy/andy/CYP3A4_NP_Diosmin/Visualization/CYP3A4_XGB_all_features_CV_5CV_oof_predictions.csv"
output_folder = "/home/andy/andy/CYP3A4_NP_Diosmin/Visualization"
threshold = 0.5
labels = ['Negative', 'Positive']
# =======================

os.makedirs(output_folder, exist_ok=True)

# Đọc dữ liệu
df = pd.read_csv(csv_path)
models = df['model'].unique()

# Tính accuracy cho mỗi mô hình
model_acc = {}
for model in models:
    data = df[df['model'] == model]
    y_true = data['y_true'].astype(int)
    y_pred = (data['y_prob'] >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    model_acc[model] = acc

# Lấy 4 model có accuracy cao nhất
top4 = sorted(model_acc.items(), key=lambda x: x[1], reverse=True)[:6]

# Hàm làm sạch tên file an toàn
def safe_name(s: str) -> str:
    s = re.sub(r"[^\w\-_. ]", "_", s)
    return s.strip().replace(" ", "_")

# Vẽ và lưu từng hình độc lập
for model, acc in top4:
    data = df[df['model'] == model]
    y_true = data['y_true'].astype(int)
    y_pred = (data['y_prob'] >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # ==== % THEO TỪNG HÀNG (row-wise) ====
    # row_sums: tổng mỗi hàng (True 0 và True 1)
    row_sums = cm.sum(axis=1, keepdims=True)
    # Chia an toàn: nếu hàng = 0 thì cho 0%
    cm_percent = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0
    # =====================================

    # Tạo annotation: số đếm + %
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"

    plt.figure(figsize=(5, 5))
    ax = sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="black",
        annot_kws={"fontsize": 11, "weight": "bold"}
    )

    # Đảo màu chữ tùy theo nền (nền tối thì chữ trắng, nền sáng thì chữ đen)
    # Lấy giá trị max để tính ngưỡng
    vmax = cm.max()
    threshold_val = vmax / 2
    for text, val in zip(ax.texts, cm.flatten()):
        text.set_color("white" if val > threshold_val else "black")

    plt.xlabel('Predicted', fontsize=12, fontweight='bold', fontstyle='italic', family='sans-serif')
    plt.ylabel('Actual', fontsize=12, fontweight='bold', fontstyle='italic', family='sans-serif')
    plt.title(f'{model}', fontsize=12, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(output_folder, f"confusion_matrix_{safe_name(model)}.svg")
    plt.savefig(out_path, format="svg")
    plt.close()
    print(f"✅ Đã lưu: {out_path}")
