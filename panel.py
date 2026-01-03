import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_scaffold_enrichment_minimal(
    csv_path: str,
    out_dir: str = "enrichment_panel",
    save_name: str = "scaffold_enrichment_distribution",
    min_n_total: int = 3,
    size_range: tuple = (20, 400),
    alpha: float = 0.6,
    y_jitter: float = 0.015,
    y_margin: float = 0.03,
    random_state: int = 42,
    show_counts_box: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(random_state)

    df = pd.read_csv(csv_path)

    required = {"scaffold", "mean_shap", "n_total", "n_active"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")

    # ---- Clean + compute active_ratio ----
    df = df.dropna(subset=["scaffold", "mean_shap", "n_total", "n_active"]).copy()
    df["n_total"] = pd.to_numeric(df["n_total"], errors="coerce")
    df["n_active"] = pd.to_numeric(df["n_active"], errors="coerce")
    df["mean_shap"] = pd.to_numeric(df["mean_shap"], errors="coerce")
    df = df.dropna(subset=["n_total", "n_active", "mean_shap"]).copy()
    df = df[df["n_total"] > 0].copy()

    df["active_ratio"] = df["n_active"] / df["n_total"]
    df = df[df["n_total"] >= min_n_total].copy()

    if df.empty:
        raise ValueError("No scaffolds left after filtering min_n_total.")

    # ---- Baseline (global active ratio) ----
    baseline = df["n_active"].sum() / df["n_total"].sum()

    # ---- Descriptive counts relative to reference lines (NOT thresholds) ----
    shap_pos_ar_hi = df[(df["mean_shap"] > 0) & (df["active_ratio"] >= baseline)]
    shap_neg_ar_hi = df[(df["mean_shap"] < 0) & (df["active_ratio"] >= baseline)]
    shap_neg_ar_lo = df[(df["mean_shap"] < 0) & (df["active_ratio"] <  baseline)]
    shap_pos_ar_lo = df[(df["mean_shap"] > 0) & (df["active_ratio"] <  baseline)]

    # ---- Scale marker sizes ----
    n = df["n_total"].astype(float).values
    s_min, s_max = size_range
    if n.max() == n.min():
        sizes = np.full_like(n, (s_min + s_max) / 2)
    else:
        sizes = s_min + (n - n.min()) * (s_max - s_min) / (n.max() - n.min())

    # ---- Add slight jitter on y to reduce overlap ----
    y = df["active_ratio"].values.astype(float)
    if y_jitter and y_jitter > 0:
        y = y + rng.uniform(-y_jitter, y_jitter, size=len(y))
        y = np.clip(y, 0.0 - y_margin, 1.0 + y_margin)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    ax.scatter(df["mean_shap"].values, y, s=sizes, alpha=alpha)

    # Reference lines (visual aid): dashed + slightly transparent
    ax.axvline(0.0, linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axhline(baseline, linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_xlabel(
        f"Mean SHAP (scaffold-level, n_total ≥ {min_n_total})",
        fontsize=12, fontweight="bold", fontstyle="italic"
    )
    ax.set_ylabel(
        "Active ratio",
        fontsize=12, fontweight="bold", fontstyle="italic"
    )

    ax.set_ylim(0.0 - y_margin, 1.0 + y_margin)

    # ---- Blue box (bottom-right): descriptive distribution counts ----
    if show_counts_box:
        stats_text = (
            
            f"Baseline AR =  {baseline:.3f} (reference)\n"
            f"SHAP+ & enriched active: {len(shap_pos_ar_hi)}\n"
            f"SHAP− & enriched active: {len(shap_neg_ar_hi)}\n"
            f"SHAP− & enriched inactive: {len(shap_neg_ar_lo)}\n"
            f"SHAP+ & enriched inactive: {len(shap_pos_ar_lo)}"
        )
        ax.text(
            0.98, 0.02,
            stats_text,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=11,
            color="blue",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor="blue",
                linewidth=0.9,
                alpha=0.9
            )
        )

    
    fig.tight_layout()

    out_svg = os.path.join(out_dir, f"{save_name}.svg")
    fig.savefig(out_svg, format="svg")
    plt.close(fig)

    # ---- Print short ----
    print(f"Saved: {out_svg}")
    print(f"Shown: n_total >= {min_n_total} | Baseline AR = {baseline:.3f}")
    print(f"Scaffolds plotted = {len(df)}")
    print(
        f"SHAP+&AR≥={len(shap_pos_ar_hi)}, SHAP−&AR≥={len(shap_neg_ar_hi)}, "
        f"SHAP−&AR<={len(shap_neg_ar_lo)}, SHAP+&AR<={len(shap_pos_ar_lo)}"
    )


if __name__ == "__main__":
    plot_scaffold_enrichment_minimal(
        csv_path="shap_XGB_full_20260102_181750/scaffold_shap_with_counts_full.csv",
        out_dir="enrichment_panel",
        min_n_total=3,
        y_jitter=0.015,
        y_margin=0.03,
        show_counts_box=True
    )
