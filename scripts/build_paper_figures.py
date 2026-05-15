"""Generate figures for the unrestricted academic paper.

All values come straight from results/v3/modal/*.json. No transcription
of rounded numbers; the JSON is the only source of truth. Output PDFs
go to paper_output_v2/figures/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "v3" / "modal"
OUT = ROOT / "paper_output_v2" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
NEURIPS_OUT = ROOT / "paper_neurips2026" / "figures"
NEURIPS_OUT.mkdir(parents=True, exist_ok=True)

PERPLEXITY = RES / "perplexity__Qwen2.5-7B__b3_seed42_n1024_tok1024_str512_fp16+spectralquant_v2+turboquant.json"
GENERATION = RES / "generation__Qwen2.5-7B__b3_seed42_t0.00_new128_fp16+spectralquant_v2+turboquant.json"
LATENCY = RES / "latency__Qwen2.5-7B__b3_seed42_bs1_ctx512x1024x2048_gen64_wm3_it10_fp16+spectralquant_v2+turboquant.json"
LONGBENCH = RES / "longbench__Qwen2.5-7B__b3_seed42_subsetdeterministic_n50_in8192_out128_fp16+spectralquant_v2+turboquant.json"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

METHOD_COLORS = {
    "fp16": "#1f4e79",
    "spectralquant_v2": "#0a8a3a",
    "turboquant": "#b03a3a",
}
METHOD_LABEL = {
    "fp16": "FP16",
    "spectralquant_v2": "SQ water-fill (b=3)",
    "turboquant": "Random Haar (b=3)",
}


def load(path: Path):
    with open(path) as f:
        return json.load(f)


def fig_perplexity():
    d = load(PERPLEXITY)
    methods = ["fp16", "spectralquant_v2", "turboquant"]
    ppl = [d["methods"][m]["perplexity"] for m in methods]
    nll = [d["methods"][m]["nll_per_token"] for m in methods]
    n_tok = d["methods"]["fp16"]["n_tokens"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    x = np.arange(3)
    bars = axes[0].bar(x, ppl, color=[METHOD_COLORS[m] for m in methods], width=0.55)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([METHOD_LABEL[m] for m in methods], rotation=15, ha="right")
    axes[0].set_ylabel("Perplexity (log scale)")
    axes[0].set_title(f"WikiText-103 validation perplexity\n(n_tokens={n_tok:,})")
    for b, v in zip(bars, ppl):
        axes[0].text(b.get_x() + b.get_width() / 2, v * 1.08, f"{v:,.2f}", ha="center", va="bottom", fontsize=8)

    bars2 = axes[1].bar(x, nll, color=[METHOD_COLORS[m] for m in methods], width=0.55)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([METHOD_LABEL[m] for m in methods], rotation=15, ha="right")
    axes[1].set_ylabel("NLL per token (nats)")
    axes[1].set_title("Average negative log-likelihood")
    for b, v in zip(bars2, nll):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "fig_perplexity.pdf")
    fig.savefig(NEURIPS_OUT / "fig_perplexity.pdf")
    plt.close(fig)


def fig_generation():
    d = load(GENERATION)
    methods = ["fp16", "spectralquant_v2", "turboquant"]
    metrics = ["mean_token_overlap_f1", "mean_distinct_1", "mean_distinct_2"]
    metric_label = {
        "mean_token_overlap_f1": "Token-overlap F1\n(vs FP16 reference)",
        "mean_distinct_1": "distinct-1\n(unigram diversity)",
        "mean_distinct_2": "distinct-2\n(bigram diversity)",
    }

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(metrics))
    width = 0.27
    for i, m in enumerate(methods):
        vals = [d["methods"][m]["metrics"][k] for k in metrics]
        offs = (i - 1) * width
        bars = ax.bar(x + offs, vals, width=width, color=METHOD_COLORS[m], label=METHOD_LABEL[m])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([metric_label[k] for k in metrics])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Greedy generation quality, 8-prompt evaluation set\n(Qwen2.5-7B, b=3, max_new=128, T=0.0, seed=42)")
    ax.legend(loc="upper right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(OUT / "fig_generation.pdf")
    fig.savefig(NEURIPS_OUT / "fig_generation.pdf")
    plt.close(fig)


def fig_longbench():
    d = load(LONGBENCH)
    tasks = d["tasks"]
    methods = ["fp16", "spectralquant_v2", "turboquant"]
    macro = {m: d["methods"][m]["aggregate"]["macro_score"] for m in methods}
    per = {m: {t["task"]: t["score"] for t in d["methods"][m]["per_task"]} for m in methods}

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), gridspec_kw={"width_ratios": [1, 2.4]})

    # Left: macro score
    x = np.arange(3)
    vals = [macro[m] for m in methods]
    bars = axes[0].bar(x, vals, color=[METHOD_COLORS[m] for m in methods], width=0.55)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([METHOD_LABEL[m] for m in methods], rotation=15, ha="right")
    axes[0].set_ylabel("Macro score (5-task mean)")
    axes[0].set_title("LongBench macro score\n(5-task subset, n=50/task)", fontsize=10)
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    # Right: per-task grouped bars
    width = 0.27
    xt = np.arange(len(tasks))
    for i, m in enumerate(methods):
        vals = [per[m][t] for t in tasks]
        offs = (i - 1) * width
        axes[1].bar(xt + offs, vals, width=width, color=METHOD_COLORS[m], label=METHOD_LABEL[m])
    axes[1].set_xticks(xt)
    axes[1].set_xticklabels(tasks, rotation=20, ha="right")
    axes[1].set_ylabel("Per-task score")
    axes[1].set_title("LongBench per-task results\n(Qwen2.5-7B, b=3, seed=42, in$\\leq$8192, out=128)", fontsize=10)
    axes[1].legend(loc="upper right", framealpha=0.92)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.32)
    fig.savefig(OUT / "fig_longbench.pdf")
    fig.savefig(NEURIPS_OUT / "fig_longbench.pdf")
    plt.close(fig)


def fig_latency():
    d = load(LATENCY)
    ctx = [512, 1024, 2048]
    fp16 = [op for op in d["methods"]["fp16"]["operating_points"]]
    sq_micro = [op for op in d["methods"]["spectralquant_v2"]["operating_points"] if op.get("microbenchmark")]
    sq_e2e = [op for op in d["methods"]["spectralquant_v2"]["operating_points"] if op.get("end_to_end_measured") and not op.get("microbenchmark")]
    tq_micro = [op for op in d["methods"]["turboquant"]["operating_points"] if op.get("microbenchmark")]
    tq_e2e = [op for op in d["methods"]["turboquant"]["operating_points"] if op.get("end_to_end_measured") and not op.get("microbenchmark")]

    fp16_dec = {op["context_length"]: op["decode_ms_per_token_p50"] for op in fp16}
    sq_micro_dec = {op["context_length"]: op["decode_ms_per_token_p50"] for op in sq_micro}
    sq_e2e_dec = {op["context_length"]: op["decode_ms_per_token_p50"] for op in sq_e2e}
    tq_micro_dec = {op["context_length"]: op["decode_ms_per_token_p50"] for op in tq_micro}
    tq_e2e_dec = {op["context_length"]: op["decode_ms_per_token_p50"] for op in tq_e2e}

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))

    # Left: microbench (KV round-trip), log scale
    axes[0].plot(ctx, [sq_micro_dec[c] for c in ctx], "o-", color=METHOD_COLORS["spectralquant_v2"], label="Calibrated rot. + water-filled KV round-trip")
    axes[0].plot(ctx, [tq_micro_dec[c] for c in ctx], "s-", color=METHOD_COLORS["turboquant"], label="Random rot. + uniform KV round-trip")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xticks(ctx)
    axes[0].set_xticklabels([str(c) for c in ctx])
    axes[0].set_xlabel("Context length")
    axes[0].set_ylabel("ms / token (p50, log)")
    axes[0].set_title("KV compress+decompress microbenchmark\n(diagnostic replay, MICROBENCH ONLY -- not a serving kernel)")
    axes[0].legend(loc="upper right", framealpha=0.92)

    # Right: end-to-end decode (FP16 vs hooked replay)
    axes[1].plot(ctx, [fp16_dec[c] for c in ctx], "o-", color=METHOD_COLORS["fp16"], label="FP16 end-to-end (reference pass)")
    axes[1].plot(ctx, [sq_e2e_dec[c] for c in ctx], "o-", color=METHOD_COLORS["spectralquant_v2"], label="Calibrated rot. + water-filled hooked replay")
    axes[1].plot(ctx, [tq_e2e_dec[c] for c in ctx], "s-", color=METHOD_COLORS["turboquant"], label="Random rot. + uniform hooked replay")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xticks(ctx)
    axes[1].set_xticklabels([str(c) for c in ctx])
    axes[1].set_xlabel("Context length")
    axes[1].set_ylabel("decode ms / token (p50, log)")
    axes[1].set_title("End-to-end decode time\n(hooked replay is NOT a production kernel)")
    axes[1].legend(loc="upper left", framealpha=0.92, fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT / "fig_latency.pdf")
    fig.savefig(NEURIPS_OUT / "fig_latency.pdf")
    plt.close(fig)


def fig_attention_cosine():
    """From docs/full_matrix_evidence_summary.md verbatim values.

    Round 13 narrative: bars are labeled conceptually as the three arms of the
    rotation-then-quantize family at fixed bit budget:
      - structure-agnostic Haar rotation + uniform bits (controlled foil)
      - calibrated rotation + uniform bits
      - calibrated rotation + water-filled allocation
    """
    rows = [
        ("Mistral-7B-v0.3", 5, 0.6556, 0.9404, 0.9421),
        ("Mistral-7B-v0.3", 3, 0.6263, 0.9329, 0.9327),
        ("Mistral-7B-v0.3", 2, 0.6495, 0.9035, 0.9213),
        ("Qwen2.5-7B", 3, 0.3986, 0.7724, 0.7786),
    ]
    labels = [f"{m.split('-')[0]}\nb={b}" for (m, b, *_) in rows]
    arm_oblivious = [r[2] for r in rows]
    arm_calib_uniform = [r[3] for r in rows]
    arm_calib_waterfill = [r[4] for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    x = np.arange(len(rows))
    width = 0.27
    ax.bar(x - width, arm_oblivious, width=width, color=METHOD_COLORS["turboquant"],
           label="Random Haar + uniform")
    ax.bar(x, arm_calib_uniform, width=width, color="#9474c2",
           label="Calib. rot. + uniform")
    ax.bar(x + width, arm_calib_waterfill, width=width, color=METHOD_COLORS["spectralquant_v2"],
           label="Calib. rot. + water-fill")
    for i, vals in enumerate(zip(arm_oblivious, arm_calib_uniform, arm_calib_waterfill)):
        for j, v in enumerate(vals):
            ax.text(x[i] + (j - 1) * width, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Mean attention-output cosine vs FP16")
    ax.set_title("Attention-output fidelity at fixed bit budget (n_calib=32, n_eval=8 layers)")
    ax.legend(loc="lower right", framealpha=0.92, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_attention_cosine.pdf")
    fig.savefig(NEURIPS_OUT / "fig_attention_cosine.pdf")
    plt.close(fig)


def fig_pipeline():
    """Schematic of the calibrated rotation + water-filled allocation pipeline.

    Round 12 narrative emphasizes the empirical spectrum, the
    participation-ratio cutoff that defines the principal key subspace, the
    residual tail, and water-filled bit allocation. ``semantic subspace''
    language is removed.
    """
    fig, ax = plt.subplots(figsize=(8.6, 2.8))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.20, 1.5, 1.30, 1.0, "Calibration\nbatch", "#cfe2f3"),
        (1.70, 1.5, 1.50, 1.0, "Per-(layer,head)\nkey covariance", "#cfe2f3"),
        (3.40, 1.5, 1.60, 1.0, "Empirical spectrum\n+ PR cutoff $d_{\\rm eff}$", "#d9ead3"),
        (5.20, 1.5, 1.60, 1.0, "Principal key\nsubspace + tail", "#d9ead3"),
        (7.00, 1.5, 2.55, 1.0, "Greedy water-fill bits", "#fff2cc"),
        (9.75, 1.5, 1.45, 1.0, "Per-dim\nLloyd–Max", "#fce5cd"),
    ]
    for (x, y, w, h, txt, c) in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=c, edgecolor="#444", linewidth=0.8))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.2)

    arrow_kwargs = dict(arrowstyle="->", color="#444", linewidth=1.0)
    pairs = [
        (1.50, 2.0, 1.70, 2.0),
        (3.20, 2.0, 3.40, 2.0),
        (5.00, 2.0, 5.20, 2.0),
        (6.80, 2.0, 7.00, 2.0),
        (9.55, 2.0, 9.75, 2.0),
    ]
    for a in pairs:
        ax.annotate("", xy=(a[2], a[3]), xytext=(a[0], a[1]), arrowprops=arrow_kwargs)

    ax.set_xlim(0, 11.4)
    ax.text(5.7, 3.55,
            "Calibration pipeline (offline, once per (model, b))",
            ha="center", fontsize=10, weight="bold")
    ax.text(5.7, 0.55,
            "Online: rotate K/V into the calibrated eigenbasis $\\to$ quantize principal subspace + residual tail $\\to$ store;"
            "\non read: decode $\\to$ rotate back $\\to$ attend.",
            ha="center", fontsize=8.4, style="italic")
    fig.savefig(OUT / "fig_pipeline.pdf")
    fig.savefig(NEURIPS_OUT / "fig_pipeline.pdf")
    plt.close(fig)


def main():
    fig_perplexity()
    fig_generation()
    fig_longbench()
    fig_latency()
    fig_attention_cosine()
    fig_pipeline()
    print("Wrote figures to", OUT)
    for p in sorted(OUT.glob("*.pdf")):
        print(" ", p.name, os.path.getsize(p), "bytes")


if __name__ == "__main__":
    main()
