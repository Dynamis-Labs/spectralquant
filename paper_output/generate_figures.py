"""
generate_figures.py
-------------------
Generates all publication-quality figures for the SpectralQuant paper.
Output: /home/user/workspace/spectralquant/paper_output/figures/
"""

import json
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = "/home/user/workspace/spectralquant/paper_output/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR = "/home/user/workspace/spectralquant/results"
PATH_ALL  = os.path.join(DATA_DIR, "memory_efficiency/all_models.json")
PATH_SEQ  = os.path.join(DATA_DIR, "seqlen_sweep/seqlen_sweep.json")
PATH_DEFF = os.path.join(DATA_DIR, "deff_sweep/deff_sweep_results.json")
PATH_AGG  = os.path.join(DATA_DIR, "aggressive/aggressive.json")
PATH_EIGS = os.path.join(DATA_DIR, "eigenspectral/summary_statistics.json")

# ── Load data ─────────────────────────────────────────────────────────────────
with open(PATH_ALL)  as f: all_models = json.load(f)
with open(PATH_SEQ)  as f: seqlen     = json.load(f)
with open(PATH_DEFF) as f: deff       = json.load(f)
with open(PATH_AGG)  as f: aggressive = json.load(f)
with open(PATH_EIGS) as f: eigs       = json.load(f)

# ── Publication rcParams ──────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        300,
    "pdf.fonttype":      42,   # embed fonts as TrueType
    "ps.fonttype":       42,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "lines.linewidth":   1.4,
    "patch.linewidth":   0.6,
})

# ── Color palette ─────────────────────────────────────────────────────────────
# TurboQuant → blue family
# SpectralQuant variants → orange/red family
C_TQ        = "#2171B5"   # TQ blue
C_TQ_2bit   = "#9ECAE1"   # TQ 2bit, lighter blue
C_SQ_V3     = "#D94801"   # SQ noQJL_v3 — deep orange
C_SQ_V2T    = "#FD8D3C"   # SQ noQJL_v2tail — light orange
C_SQ_SELV3  = "#8B1A4A"   # SQ selQJL — dark rose
C_SQ_SELV2T = "#D4618E"   # SQ selQJL_v2tail — pink

# Model colors (for Pareto)
MODEL_COLORS = {
    "Qwen2.5-1.5B-Instruct": "#4292C6",
    "Qwen2.5-7B-Instruct":   "#2CA25F",
    "Qwen2.5-14B-Instruct":  "#D94801",
}
MODEL_LABELS = {
    "Qwen2.5-1.5B-Instruct": "1.5B",
    "Qwen2.5-7B-Instruct":   "7B",
    "Qwen2.5-14B-Instruct":  "14B",
}

# ── Helper ────────────────────────────────────────────────────────────────────
def save_fig(fig, name):
    pdf_path = os.path.join(OUT_DIR, f"{name}.pdf")
    png_path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"  Saved {name}.pdf / {name}.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Main comparison: grouped bar chart
# ═══════════════════════════════════════════════════════════════════════════════
def fig_main_comparison():
    configs = ["TQ_3bit", "SQ_noQJL_v3", "SQ_noQJL_v2tail"]
    config_labels = ["TQ-3bit", "SQ-noQJL-v3", "SQ-noQJL-v2tail"]
    bar_colors = [C_TQ, C_SQ_V3, C_SQ_V2T]

    models = list(all_models.keys())
    model_short = ["1.5B", "7B", "14B"]

    # Gather cosine sim + compression ratio
    cos_data   = {c: [] for c in configs}
    ratio_data = {c: [] for c in configs}
    for m in models:
        for c in configs:
            cos_data[c].append(all_models[m]["configs"][c]["cos_sim"])
            ratio_data[c].append(all_models[m]["configs"][c]["ratio"])

    n_models  = len(models)
    n_configs = len(configs)
    x = np.arange(n_models)
    total_width = 0.72
    w = total_width / n_configs

    fig, ax = plt.subplots(figsize=(6.75, 3.0))

    bars_list = []
    for i, (cfg, label, color) in enumerate(zip(configs, config_labels, bar_colors)):
        offsets = x + (i - (n_configs - 1) / 2) * w
        vals = cos_data[cfg]
        bars = ax.bar(offsets, vals, width=w, color=color, label=label,
                      zorder=3, linewidth=0.5, edgecolor="white")
        bars_list.append(bars)

        # Annotate compression ratio inside / above each bar
        for bar_rect, r_val, v_val in zip(bars, ratio_data[cfg], vals):
            ax.text(
                bar_rect.get_x() + bar_rect.get_width() / 2,
                v_val + 0.003,
                f"{r_val:.1f}×",
                ha="center", va="bottom",
                fontsize=6.5, color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(model_short)
    ax.set_xlabel("Model size")
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(0.80, 0.99)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.01))
    ax.grid(axis="y", linewidth=0.4, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Annotation: "compression ratio ×" label
    ax.text(1.01, 0.99, "Numbers above bars\nshow compression ratio",
            transform=ax.transAxes, fontsize=6.5, va="top",
            color="#555555", ha="left")

    legend = ax.legend(loc="lower right", frameon=True,
                       framealpha=0.9, edgecolor="#CCCCCC",
                       handlelength=1.2, handletextpad=0.5,
                       borderpad=0.5, labelspacing=0.3)

    fig.tight_layout()
    save_fig(fig, "fig_main_comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Pareto frontier
# ═══════════════════════════════════════════════════════════════════════════════
def fig_pareto():
    # Config → marker style
    config_markers = {
        "TQ_3bit":          ("o",  C_TQ,       "TQ-3bit"),
        "TQ_2bit":          ("o",  C_TQ_2bit,  "TQ-2bit"),
        "SQ_noQJL_v3":      ("s",  C_SQ_V3,    "SQ-noQJL-v3"),
        "SQ_noQJL_v2tail":  ("s",  C_SQ_V2T,   "SQ-noQJL-v2t"),
        "SQ_selQJL":        ("D",  C_SQ_SELV3, "SQ-selQJL"),
        "SQ_selQJL_v2tail": ("D",  C_SQ_SELV2T,"SQ-selQJL-v2t"),
        "SQ_noQJL_v1tail":  ("^",  "#FDAE6B",  "SQ-noQJL-v1t"),
        "SQ_selQJL_v1tail": ("^",  "#C7849A",  "SQ-selQJL-v1t"),
        "SQ_k1_noQJL_v2t":  ("v",  "#A65628",  "SQ-k1-v2t"),
    }

    fig, ax = plt.subplots(figsize=(6.75, 3.4))

    # Collect all points and build Pareto frontier
    all_pts = []  # (ratio, cos_sim, model_key, cfg_key)

    for m_key, m_data in all_models.items():
        for cfg_key, cfg_data in m_data["configs"].items():
            ratio   = cfg_data["ratio"]
            cos_sim = cfg_data["cos_sim"]
            marker_info = config_markers.get(cfg_key, ("x", "gray", cfg_key))
            marker, _, _ = marker_info
            color = MODEL_COLORS[m_key]
            ax.scatter(ratio, cos_sim, marker=marker, color=color,
                       s=30, zorder=5, linewidths=0.4,
                       edgecolors="white", alpha=0.88)
            all_pts.append((ratio, cos_sim))

    # Compute Pareto frontier (maximize cos_sim for each ratio threshold)
    # Sort by ratio ascending; keep points where cos_sim is non-dominated
    pts_arr = np.array(sorted(all_pts, key=lambda p: p[0]))
    pareto = []
    best_cos = -np.inf
    for ratio, cos in pts_arr:
        if cos > best_cos:
            best_cos = cos
            pareto.append((ratio, cos))
    pareto = np.array(pareto)

    # Draw step-wise Pareto frontier
    ax.step(pareto[:, 0], pareto[:, 1], where="post",
            color="#333333", linewidth=1.0, linestyle="--",
            zorder=4, label="Pareto frontier")
    ax.scatter(pareto[:, 0], pareto[:, 1],
               color="#333333", s=22, zorder=6, marker="*")

    # Label key operating points (one per model for TQ-3bit and SQ-noQJL-v3)
    key_configs = [
        ("Qwen2.5-14B-Instruct", "SQ_noQJL_v3",  " SQ-v3\n 14B"),
        ("Qwen2.5-14B-Instruct", "TQ_3bit",       " TQ\n 14B"),
        ("Qwen2.5-14B-Instruct", "SQ_noQJL_v2tail"," SQ-v2t\n 14B"),
        ("Qwen2.5-14B-Instruct", "SQ_k1_noQJL_v2t"," SQ-k1\n 14B"),
    ]
    for m_key, cfg_key, lbl in key_configs:
        d = all_models[m_key]["configs"][cfg_key]
        ax.annotate(lbl, xy=(d["ratio"], d["cos_sim"]),
                    fontsize=6.5, color="#222222",
                    xytext=(3, -1), textcoords="offset points")

    # Legend: model colors (circles)
    model_handles = [
        mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m])
        for m in list(all_models.keys())
    ]
    # Legend: config shapes
    shape_handles = [
        Line2D([0], [0], marker="o", color="gray", label="TQ",
               markersize=5, linestyle="None"),
        Line2D([0], [0], marker="s", color="gray", label="SQ-noQJL",
               markersize=5, linestyle="None"),
        Line2D([0], [0], marker="D", color="gray", label="SQ-selQJL",
               markersize=5, linestyle="None"),
        Line2D([0], [0], marker="^", color="gray", label="v1tail",
               markersize=5, linestyle="None"),
        Line2D([0], [0], marker="v", color="gray", label="k1-v2t",
               markersize=5, linestyle="None"),
        Line2D([0], [0], color="#333333", linestyle="--", label="Pareto frontier",
               linewidth=1.0),
    ]

    leg1 = ax.legend(handles=model_handles, title="Model", loc="lower left",
                     frameon=True, framealpha=0.9, edgecolor="#CCCCCC",
                     fontsize=7.5, title_fontsize=7.5,
                     handlelength=1.0, labelspacing=0.3)
    ax.add_artist(leg1)
    ax.legend(handles=shape_handles, title="Config type", loc="upper right",
              frameon=True, framealpha=0.9, edgecolor="#CCCCCC",
              fontsize=7.5, title_fontsize=7.5,
              handlelength=1.4, labelspacing=0.3)

    ax.set_xlabel("Compression ratio")
    ax.set_ylabel("Cosine similarity")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.grid(linewidth=0.3, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_fig(fig, "fig_pareto")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Sequence length sweep
# ═══════════════════════════════════════════════════════════════════════════════
def fig_seqlen():
    seq_lengths = sorted(int(k) for k in seqlen.keys())
    tq_vals = [seqlen[str(s)]["tq"] for s in seq_lengths]
    sq_vals = [seqlen[str(s)]["sq"] for s in seq_lengths]

    fig, ax = plt.subplots(figsize=(3.25, 2.5))

    ax.plot(seq_lengths, tq_vals, color=C_TQ, marker="o", markersize=4,
            label="TQ-3bit", zorder=4)
    ax.plot(seq_lengths, sq_vals, color=C_SQ_V3, marker="s", markersize=4,
            label="SQ-noQJL-v3", zorder=4)

    # Shade the region where SQ > TQ
    sq_arr = np.array(sq_vals)
    tq_arr = np.array(tq_vals)
    ax.fill_between(seq_lengths, tq_arr, sq_arr,
                    where=(sq_arr >= tq_arr),
                    alpha=0.18, color=C_SQ_V3, label="SQ advantage")

    # Annotate delta at each point
    for sl, tq, sq in zip(seq_lengths, tq_vals, sq_vals):
        delta = sq - tq
        if delta > 0:
            ax.annotate(f"+{delta:.4f}",
                        xy=(sl, (tq + sq) / 2),
                        fontsize=5.5, ha="center", color="#666666")

    ax.set_xscale("log", base=2)
    ax.set_xticks(seq_lengths)
    ax.set_xticklabels([str(s) for s in seq_lengths])
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("Cosine similarity")

    y_min = min(min(tq_vals), min(sq_vals)) - 0.002
    y_max = max(max(tq_vals), max(sq_vals)) + 0.004
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.002))
    ax.grid(linewidth=0.3, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(loc="lower right", frameon=True, framealpha=0.9,
              edgecolor="#CCCCCC", handlelength=1.4, labelspacing=0.3)

    fig.tight_layout()
    save_fig(fig, "fig_seqlen")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Memory savings at scale (14B model)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_memory_savings():
    m14 = all_models["Qwen2.5-14B-Instruct"]["configs"]

    # Select configs to display (ordered by MB descending = lowest compression first)
    display_cfgs = [
        ("TQ_3bit",          "TQ-3bit",         C_TQ),
        ("SQ_selQJL",        "SQ-selQJL",       C_SQ_SELV3),
        ("SQ_noQJL_v3",      "SQ-noQJL-v3",     C_SQ_V3),
        ("TQ_2bit",          "TQ-2bit",          C_TQ_2bit),
        ("SQ_selQJL_v2tail", "SQ-selQJL-v2t",   C_SQ_SELV2T),
        ("SQ_noQJL_v2tail",  "SQ-noQJL-v2t",    C_SQ_V2T),
        ("SQ_noQJL_v1tail",  "SQ-noQJL-v1t",    "#FDAE6B"),
        ("SQ_selQJL_v1tail", "SQ-selQJL-v1t",   "#C7849A"),
        ("SQ_k1_noQJL_v2t",  "SQ-k1-v2t",       "#A65628"),
    ]

    cfg_keys   = [d[0] for d in display_cfgs]
    cfg_labels = [d[1] for d in display_cfgs]
    cfg_colors = [d[2] for d in display_cfgs]
    mb_vals    = [m14[k]["mb_8k"] for k in cfg_keys]

    # Sort by MB descending (largest memory first)
    order = np.argsort(mb_vals)[::-1]
    cfg_labels = [cfg_labels[i] for i in order]
    cfg_colors = [cfg_colors[i] for i in order]
    mb_vals    = [mb_vals[i]    for i in order]
    cfg_keys   = [cfg_keys[i]   for i in order]

    tq3_mb = m14["TQ_3bit"]["mb_8k"]

    fig, ax = plt.subplots(figsize=(6.75, 3.2))

    y_pos = np.arange(len(cfg_labels))
    bars = ax.barh(y_pos, mb_vals, color=cfg_colors, height=0.65,
                   edgecolor="white", linewidth=0.5, zorder=3)

    # Annotate savings vs TQ-3bit
    for i, (bar_rect, mb, ck) in enumerate(zip(bars, mb_vals, cfg_keys)):
        savings = tq3_mb - mb
        # MB value label
        ax.text(mb + 1.5, bar_rect.get_y() + bar_rect.get_height() / 2,
                f"{mb:.0f} MB", va="center", fontsize=7, color="#333333")
        # Savings label (only for non-TQ3bit)
        if ck != "TQ_3bit" and savings > 0:
            ax.text(mb / 2, bar_rect.get_y() + bar_rect.get_height() / 2,
                    f"−{savings:.0f} MB vs TQ-3bit",
                    va="center", ha="center", fontsize=6.5, color="white",
                    fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cfg_labels)
    ax.set_xlabel("KV cache memory @ 8K context (MB)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(25))
    ax.set_xlim(0, tq3_mb * 1.18)
    ax.grid(axis="x", linewidth=0.3, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    # Reference line for TQ-3bit
    ax.axvline(tq3_mb, color=C_TQ, linewidth=1.0, linestyle="--",
               zorder=4, alpha=0.8)
    ax.text(tq3_mb + 1.5, len(cfg_labels) - 0.3,
            f"TQ-3bit\n{tq3_mb:.0f} MB", fontsize=6.5,
            color=C_TQ, va="top")

    fig.tight_layout()
    save_fig(fig, "fig_memory_savings")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Model scaling: SQ advantage (Δ cosine sim vs TQ) across model sizes
# ═══════════════════════════════════════════════════════════════════════════════
def fig_scaling():
    models_ordered = [
        ("Qwen2.5-1.5B-Instruct", "1.5B"),
        ("Qwen2.5-7B-Instruct",   "7B"),
        ("Qwen2.5-14B-Instruct",  "14B"),
    ]
    x_pos = [0, 1, 2]

    target_configs = [
        ("SQ_noQJL_v3",     "SQ-noQJL-v3",     C_SQ_V3,   "s"),
        ("SQ_noQJL_v2tail", "SQ-noQJL-v2tail",  C_SQ_V2T,  "D"),
        ("SQ_selQJL",       "SQ-selQJL",        C_SQ_SELV3, "^"),
    ]

    fig, ax = plt.subplots(figsize=(3.25, 2.8))

    for cfg_key, cfg_label, color, marker in target_configs:
        deltas = []
        for m_key, _ in models_ordered:
            tq_cos  = all_models[m_key]["configs"]["TQ_3bit"]["cos_sim"]
            sq_cos  = all_models[m_key]["configs"][cfg_key]["cos_sim"]
            deltas.append(sq_cos - tq_cos)
        ax.plot(x_pos, deltas, color=color, marker=marker, markersize=5,
                label=cfg_label, zorder=4, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=0.4)

    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--", zorder=2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(["1.5B", "7B", "14B"])
    ax.set_xlabel("Model size")
    ax.set_ylabel(r"$\Delta$ cosine sim (SQ $-$ TQ-3bit)")
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.005))
    ax.grid(linewidth=0.3, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(loc="upper left", frameon=True, framealpha=0.92,
              edgecolor="#CCCCCC", handlelength=1.6, labelspacing=0.35,
              bbox_to_anchor=(0.01, 0.99))

    fig.tight_layout()
    save_fig(fig, "fig_scaling")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating publication figures...")

    print("[1/5] Main comparison (grouped bar)...")
    fig_main_comparison()

    print("[2/5] Pareto frontier (scatter)...")
    fig_pareto()

    print("[3/5] Sequence length sweep (line)...")
    fig_seqlen()

    print("[4/5] Memory savings at scale (horizontal bar)...")
    fig_memory_savings()

    print("[5/5] Model scaling (line)...")
    fig_scaling()

    print(f"\nAll figures saved to: {OUT_DIR}")
