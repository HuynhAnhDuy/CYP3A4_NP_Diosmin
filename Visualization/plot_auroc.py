import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ====== Cấu hình ======
csv_path = "/home/andy/andy/CYP3A4_NP_Diosmin/Visualization/CYP3A4_test_prob_merged_models.csv"
output_path = "CYP3A4_test_AUROC_plot.svg"
colormap_name = 'tab10'  # 🔁 Bạn có thể thử: 'Set3', 'Dark2', 'Paired', ...
# =======================

# Đọc dữ liệu
df = pd.read_csv(csv_path)
models = df['model'].unique()

# Tính ROC và AUROC cho từng model
roc_data = []
for model in models:
    data = df[df['model'] == model]
    y_true = data['y_true']
    y_score = data['y_prob']
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc = auc(fpr, tpr)
    
    roc_data.append((model, fpr, tpr, auroc))

# Sắp xếp theo AUROC giảm dần
roc_data_sorted = sorted(roc_data, key=lambda x: x[3], reverse=True)

# Lấy colormap
cmap = plt.get_cmap(colormap_name)
colors = [cmap(i % cmap.N) for i in range(len(roc_data_sorted))]

# Khởi tạo figure
plt.figure(figsize=(8, 6))

# Vẽ từng đường ROC theo thứ tự AUROC
for idx, (model, fpr, tpr, auroc) in enumerate(roc_data_sorted):
    plt.plot(fpr, tpr,
             label=f'{model} (AUROC={auroc:.3f})',
             color=colors[idx],
             linewidth=1.8,
             linestyle='-')

# Vẽ thêm đường chéo tham chiếu (random)
plt.plot([0, 1], [0, 1], 'k--', linewidth=1.0, label='Random')

# Tùy chỉnh biểu đồ
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold', fontstyle='italic', family='sans-serif') 
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold', fontstyle='italic', family='sans-serif') 
plt.title('AUROC for CYP3A4 predictive models', fontsize=12, fontweight='bold', family='sans-serif') 
plt.legend(loc='lower right', fontsize='9', ncol=2)
plt.grid(True)

# Lưu biểu đồ
plt.tight_layout()
plt.savefig(output_path, format='svg')
print(f"✅ Đã lưu file: {output_path}")
