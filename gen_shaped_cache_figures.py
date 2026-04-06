"""
Generate publication-quality figures for the Asymmetric Shaped KV Cache experiment.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("/home/user/workspace/spectralquant/results/shaped_cache")
FIG_DIR    = Path("/home/user/workspace/spectralquant/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style constants ───────────────────────────────────────────────────────────
QUANT_COLORS = {
    "FP16":  "#2166ac",   # blue
    "4-bit": "#4dac26",   # green
    "3-bit": "#f07b00",   # orange
    "2-bit": "#d7191c",   # red
}
QUANT_ORDER  = ["FP16", "4-bit", "3-bit", "2-bit"]
MODEL_COLORS = {
    "Qwen-1.5B": "#9467bd",
    "Qwen-7B":   "#1f77b4",
    "Llama-8B":  "#d62728",
}
MODEL_MARKERS = {
    "Qwen-1.5B": "o",
    "Qwen-7B":   "s",
    "Llama-8B":  "^",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        12,
    "axes.titlesize":   14,
    "axes.labelsize":   12,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

# ── Load data ─────────────────────────────────────────────────────────────────
MODEL_FILES = {
    "Qwen-1.5B": "shaped_cache_Qwen-1.5B.json",
    "Qwen-7B":   "shaped_cache_Qwen-7B.json",
    "Llama-8B":  "shaped_cache_Llama-8B.json",
}

datasets = {}
for short_name, fname in MODEL_FILES.items():
    with open(DATA_DIR / fname) as f:
        datasets[short_name] = json.load(f)

# ── Helper ────────────────────────────────────────────────────────────────────
def get_results(data, quant_label=None, m=None, p=None):
    out = data["sweep_results"]
    if quant_label is not None:
        out = [r for r in out if r["quant_label"] == quant_label]
    if m is not None:
        out = [r for r in out if r["m"] == m]
    if p is not None:
        out = [r for r in out if r["p"] == p]
    return out

# =============================================================================
# FIGURE 1: Shaped Cache Quality vs Compression (Pareto plot)
# =============================================================================
fig1, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig1.suptitle("Shaped Cache: Quality vs Compression Across Models",
              fontsize=14, fontweight="bold", y=1.01)

# Point size mapping for m values
M_VALS = [2, 4, 8, 16, 32]
M_TO_SIZE = {m: 20 + 12 * i for i, m in enumerate(M_VALS)}   # 20..68

for ax, (model_name, data) in zip(axes, datasets.items()):
    baseline = data["baselines"]["SQ_noQJL_3bit"]

    # Horizontal baseline line
    ax.axhline(
        baseline["attn_cos_sim"],
        color="gray", linestyle="--", linewidth=1.2,
        label=f'SQ (no QJL) 3-bit\n(CR={baseline["compression_ratio"]:.1f}x)',
        zorder=2,
    )

    # Scatter per quant level
    for ql in QUANT_ORDER:
        rows = get_results(data, quant_label=ql)
        for r in rows:
            ax.scatter(
                r["compression_ratio"],
                r["attn_cos_sim"],
                color=QUANT_COLORS[ql],
                s=M_TO_SIZE[r["m"]],
                alpha=0.82,
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )

    # Annotate sweet-spot configs: m=32, p=128 for FP16, 4-bit, 3-bit
    sweet_spots = [
        (32, 128, "FP16"),
        (32, 128, "4-bit"),
        (32, 128, "3-bit"),
    ]
    for m_ss, p_ss, ql_ss in sweet_spots:
        rows = get_results(data, quant_label=ql_ss, m=m_ss, p=p_ss)
        if not rows:
            continue
        r = rows[0]
        ax.annotate(
            f'm={m_ss}\np={p_ss}\n{ql_ss}',
            xy=(r["compression_ratio"], r["attn_cos_sim"]),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=6.5,
            color=QUANT_COLORS[ql_ss],
            arrowprops=dict(arrowstyle="-", color=QUANT_COLORS[ql_ss],
                            lw=0.8, shrinkA=2, shrinkB=2),
            zorder=5,
        )

    ax.set_xscale("log")
    ax.set_xlim(1.5, 110)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Compression Ratio (log scale)", fontsize=12)
    ax.set_title(model_name, fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:.0f}x" if x >= 1 else f"{x:.1f}x"))
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

axes[0].set_ylabel("Attention Cosine Similarity", fontsize=12)

# Shared legend (quant colors)
quant_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=QUANT_COLORS[ql],
           markersize=8, label=ql)
    for ql in QUANT_ORDER
]
# Size legend for m
size_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
           markersize=np.sqrt(M_TO_SIZE[m]), label=f"m={m}")
    for m in M_VALS
]
baseline_handle = Line2D([0], [0], color="gray", linestyle="--",
                          linewidth=1.2, label="SQ 3-bit baseline")

fig1.legend(
    handles=quant_handles + [baseline_handle] + size_handles,
    loc="lower center",
    ncol=len(QUANT_ORDER) + 1 + len(M_VALS),
    fontsize=9,
    frameon=False,
    bbox_to_anchor=(0.5, -0.08),
)

fig1.tight_layout()
fig1.savefig(FIG_DIR / "shaped_cache_pareto.png")
plt.close(fig1)
print("Saved: shaped_cache_pareto.png")

# =============================================================================
# FIGURE 2: Key vs Value Truncation Sensitivity
# =============================================================================
fig2, (ax_key, ax_val) = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Keys Tolerate Truncation, Values Do Not",
              fontsize=14, fontweight="bold")

P_FIXED = 128   # fixed p for key panel
M_FIXED = 32    # fixed m for value panel
QL_FP16 = "FP16"

for model_name, data in datasets.items():
    color  = MODEL_COLORS[model_name]
    marker = MODEL_MARKERS[model_name]

    # Left: key_cos_sim vs m  (p=128, FP16)
    rows = sorted(get_results(data, quant_label=QL_FP16, p=P_FIXED),
                  key=lambda r: r["m"])
    ms_   = [r["m"]            for r in rows]
    k_sim = [r["key_cos_sim"]  for r in rows]
    ax_key.plot(ms_, k_sim, marker=marker, color=color,
                linewidth=1.8, markersize=7, label=model_name)

    # Right: val_cos_sim vs p  (m=32, FP16)
    rows = sorted(get_results(data, quant_label=QL_FP16, m=M_FIXED),
                  key=lambda r: r["p"])
    ps_   = [r["p"]           for r in rows]
    v_sim = [r["val_cos_sim"] for r in rows]
    ax_val.plot(ps_, v_sim, marker=marker, color=color,
                linewidth=1.8, markersize=7, label=model_name)

# Key panel
ax_key.set_xlabel("Key Dims Kept (m)", fontsize=12)
ax_key.set_ylabel("Key Cosine Similarity", fontsize=12)
ax_key.set_title(f"Key Sensitivity to Truncation\n(p={P_FIXED}, FP16)", fontsize=13)
ax_key.set_ylim(0.0, 1.05)
ax_key.set_xticks(M_VALS)
ax_key.legend(fontsize=10, frameon=False)
ax_key.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
ax_key.tick_params(labelsize=10)

# Value panel
ax_val.set_xlabel("Value Dims Kept (p)", fontsize=12)
ax_val.set_ylabel("Value Cosine Similarity", fontsize=12)
ax_val.set_title(f"Value Sensitivity to Truncation\n(m={M_FIXED}, FP16)", fontsize=13)
ax_val.set_ylim(0.0, 1.05)
ax_val.set_xticks([8, 16, 32, 48, 64, 96, 128])
ax_val.legend(fontsize=10, frameon=False)
ax_val.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
ax_val.tick_params(labelsize=10)

fig2.tight_layout()
fig2.savefig(FIG_DIR / "shaped_cache_kv_sensitivity.png")
plt.close(fig2)
print("Saved: shaped_cache_kv_sensitivity.png")

# =============================================================================
# FIGURE 3: Compression Heatmap (Llama-8B, 3-bit)
# =============================================================================
llama_data = datasets["Llama-8B"]
QL_3BIT = "3-bit"

rows_3bit = get_results(llama_data, quant_label=QL_3BIT)
ms_uniq   = sorted(set(r["m"] for r in rows_3bit))
ps_uniq   = sorted(set(r["p"] for r in rows_3bit))

# Build 2-D arrays indexed by (m_idx, p_idx)
n_m, n_p = len(ms_uniq), len(ps_uniq)
sim_grid  = np.full((n_m, n_p), np.nan)
cr_grid   = np.full((n_m, n_p), np.nan)

m_idx = {m: i for i, m in enumerate(ms_uniq)}
p_idx = {p: i for i, p in enumerate(ps_uniq)}

for r in rows_3bit:
    i, j = m_idx[r["m"]], p_idx[r["p"]]
    sim_grid[i, j] = r["attn_cos_sim"]
    cr_grid[i, j]  = r["compression_ratio"]

fig3, ax = plt.subplots(figsize=(10, 6))

# Mask NaN cells so they render as a distinct gray
masked_sim = np.ma.array(sim_grid, mask=np.isnan(sim_grid))
cmap_viridis = plt.cm.viridis.copy()
cmap_viridis.set_bad(color="#cccccc")

im = ax.imshow(
    masked_sim,
    cmap=cmap_viridis,
    vmin=0.0,
    vmax=1.0,
    aspect="auto",
    origin="lower",
)

cbar = fig3.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Attention Cosine Similarity", fontsize=12)
cbar.ax.tick_params(labelsize=10)

# Annotate each cell with compression ratio
for i in range(n_m):
    for j in range(n_p):
        cr  = cr_grid[i, j]
        sim = sim_grid[i, j]
        if np.isnan(sim):
            ax.text(j, i, "N/A",
                    ha="center", va="center",
                    fontsize=8, color="#888888", fontstyle="italic")
        else:
            txt_color = "white" if sim < 0.55 else "black"
            ax.text(j, i, f"{cr:.1f}x",
                    ha="center", va="center",
                    fontsize=8, color=txt_color, fontweight="bold")

ax.set_xticks(range(n_p))
ax.set_xticklabels([str(p) for p in ps_uniq], fontsize=10)
ax.set_yticks(range(n_m))
ax.set_yticklabels([str(m) for m in ms_uniq], fontsize=10)
ax.set_xlabel("Value Dims Kept (p)", fontsize=12)
ax.set_ylabel("Key Dims Kept (m)", fontsize=12)
ax.set_title("Attention Quality Heatmap (Llama 3.1-8B, 3-bit)\n"
             "Cell values show compression ratio",
             fontsize=13, fontweight="bold")

# Grid lines between cells
ax.set_xticks(np.arange(-0.5, n_p, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n_m, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.0)
ax.tick_params(which="minor", bottom=False, left=False)

fig3.tight_layout()
fig3.savefig(FIG_DIR / "shaped_cache_heatmap.png")
plt.close(fig3)
print("Saved: shaped_cache_heatmap.png")

print("\nAll figures saved to:", FIG_DIR)
