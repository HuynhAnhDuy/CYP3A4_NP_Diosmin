import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ====== CONFIG: sửa đúng folder output của bạn ======
OUT_DIR = "/home/andy/andy/CYP3A4_NP_Diosmin/SHAP_XGB_SelectedFeatures/CYP3A4_SHAP_2026-01-02_16-59-45"   # ví dụ: SHAP_XGB_MACCS_TEST/CYP3A4_SHAP_2026-01-02_15-30-10
SHAP_NPY = os.path.join(OUT_DIR, "shap_values.npy")
X_TEST_CSV = os.path.join(OUT_DIR, "X_shap_used_for_summary.csv")
OUT_SVG = os.path.join(OUT_DIR, "shap_summary_beeswarm_test.svg")
# ====================================================

# Load data used for summary plot
X_test = pd.read_csv(X_TEST_CSV, index_col=0)
shap_values = np.load(SHAP_NPY)

# Sanity check shapes
if shap_values.shape != X_test.shape:
    raise ValueError(
        f"Shape mismatch: shap_values{shap_values.shape} vs X_test{X_test.shape}. "
        "Phải dùng đúng X_test_used_for_summary.csv tương ứng với shap_values.npy."
    )

# Beeswarm summary plot (TEST) -> SVG
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    X_test,
    plot_type="dot",     # beeswarm
    max_display=30,      # top N features
    show=False
)

plt.tight_layout()
plt.savefig(OUT_SVG, format="svg")   # SVG output
plt.close()

print("✅ Saved:", OUT_SVG)
