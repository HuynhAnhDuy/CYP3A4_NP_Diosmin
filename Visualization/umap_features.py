import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler

# Đường dẫn tới file CSV
file_path = "/home/andy/andy/CYP3A4_NP_Diosmin/CYP3A4_external_x_train_rdkit.csv"  # Thay bằng đường dẫn thực tế của bạn

# Đọc dữ liệu
df = pd.read_csv(file_path)

# Giả sử cột nhãn là 'Label' (1 = ACP, 0 = non-ACP)
if 'Label' not in df.columns:
    raise ValueError("Không tìm thấy cột 'Label' trong file CSV.")

X = df.drop(columns=['Label'])
y = df['Label']

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Giảm chiều bằng UMAP
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# Tạo biểu đồ
plt.figure(figsize=(4, 3))
colors = ['#1B4F72', '#C0392B'] 
labels = ['Inhibitor', 'Non-inhibitor']

for class_value in [1, 0]:
    plt.scatter(X_umap[y == class_value, 1], X_umap[y == class_value, 0],
                label=labels[class_value], alpha=0.7, s=40, edgecolors="white",c=colors[class_value])

# Tùy chỉnh chú thích (legend)
legend = plt.legend(title='', loc='lower right', bbox_to_anchor=(1.3, 0), fontsize=11)
for text in legend.get_texts():
    text.set_fontsize(11)  # Set kích thước chữ trong legend
    text.set_fontweight('bold')
    text.set_fontname('sans-serif')  # Định dạng font chữ của legend

plt.title("UMAP of MACCS", fontsize=12, fontweight='bold', fontname='sans-serif')
plt.xlabel('UMAP1', fontsize=12, fontweight='bold', fontname='sans-serif', fontstyle='italic')
plt.ylabel('UMAP2', fontsize=12, fontweight='bold', fontname='sans-serif', fontstyle='italic')
plt.grid(alpha=0.3)
plt.tight_layout()

# Lưu biểu đồ dưới dạng SVG với DPI = 300
output_path = "CYP3A4_UMAP_MACCS.svg"
plt.savefig(output_path, format='svg', dpi=600)

# Không hiển thị
plt.close()
