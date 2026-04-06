#!/usr/bin/env python3
"""
run_all_experiments.py — Master Runner for SpectralQuant Experiments

Runs all SpectralQuant experiment phases in order:
  Phase 0 → Phase 1 → Gate check → Phase 2 → Phase 3 (all experiments)

Features:
  --phase N          Run single phase (0, 1, 2, 3.1, 3.2, …, 3.7)
  --phases N N …     Run specified phases
  --quick            Use reduced sample sizes (for debugging)
  --skip-gate        Skip gate checks (for debugging)
  --no-phase0        Skip baseline reproduction (if already done)
  --no-phase1        Skip eigenspectral discovery (use existing calibration)

Logs everything to experiments/experiment_log.txt.
Generates summary report at experiments/summary_report.json.

Usage examples:
  python run_all_experiments.py                   # Run all phases
  python run_all_experiments.py --quick           # Quick debug run
  python run_all_experiments.py --phase 1         # Only phase 1
  python run_all_experiments.py --phases 3.1 3.2  # Only experiments 1 & 2
  python run_all_experiments.py --no-phase0 --no-phase1  # Skip setup/calib
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
EXPERIMENTS_DIR = Path(__file__).parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LOG_FILE = EXPERIMENTS_DIR / "experiment_log.txt"
SUMMARY_FILE = EXPERIMENTS_DIR / "summary_report.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
log = logging.getLogger("run_all")


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

PHASE_SCRIPTS = {
    "0":   ("phase0_setup.py",                    "Baseline Setup and Reproduction"),
    "1":   ("phase1_eigenspectral.py",             "Eigenspectral Discovery"),
    "2":   ("phase2_integration.py",               "SpectralQuant Integration"),
    "3.1": ("phase3_exp1_attention_quality.py",    "Experiment 1: Attention Quality"),
    "3.2": ("phase3_exp2_ablation.py",             "Experiment 2: Ablation Study"),
    "3.3": ("phase3_exp3_generation.py",           "Experiment 3: Text Generation"),
    "3.4": ("phase3_exp4_benchmarks.py",           "Experiment 4: Downstream Benchmarks"),
    "3.5": ("phase3_exp5_vector_search.py",        "Experiment 5: Vector Search"),
    "3.6": ("phase3_exp6_latency.py",              "Experiment 6: Latency Benchmarking"),
    "3.7": ("phase3_exp7_calibration_cost.py",     "Experiment 7: Calibration Cost"),
}

# Default full pipeline order
PIPELINE_ORDER = ["0", "1", "2", "3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7"]

# Gate phases — failing these halts the pipeline unless --skip-gate
GATE_PHASES = {"1", "3.1"}

# Phase 3.x experiments can be run in parallel (optional future extension)
PHASE3_EXPS = ["3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def banner(text: str, width: int = 68) -> str:
    border = "=" * width
    return f"\n{border}\n  {text}\n{border}"


def run_script(
    script_name: str,
    extra_args: list[str],
    cwd: Path = EXPERIMENTS_DIR,
    timeout: int | None = None,
) -> tuple[int, float]:
    """Run a Python script as a subprocess. Returns (returncode, elapsed_seconds)."""
    cmd = [sys.executable, str(cwd / script_name)] + extra_args
    log.info("Launching: %s", " ".join(cmd))

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        timeout=timeout,
    )
    elapsed = time.time() - t0
    return proc.returncode, elapsed


def load_gate1_result() -> str | None:
    """Read Gate 1 result from Phase 1 metadata JSON."""
    meta_path = PROJECT_ROOT / "results" / "eigenspectral" / "phase1_metadata.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("gate1_status")
    except Exception:
        return None


def load_gate2_result() -> bool | None:
    """Read Gate 2 result from Experiment 1 results JSON."""
    meta_path = PROJECT_ROOT / "results" / "attention_quality" / "attention_quality_results.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            data = json.load(f)
        return data.get("gate2_pass")
    except Exception:
        return None


def generate_summary_report(
    phase_results: dict,
    t_start: float,
    args: argparse.Namespace,
) -> None:
    """Write a summary report of all phases."""
    report = {
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "total_wall_time_s": round(time.time() - t_start, 2),
        "quick_mode": args.quick,
        "phases_run": list(phase_results.keys()),
        "phase_results": phase_results,
        "results_dirs": {
            "baseline": str(PROJECT_ROOT / "results" / "baseline_reproduction"),
            "eigenspectral": str(PROJECT_ROOT / "results" / "eigenspectral"),
            "integration": str(PROJECT_ROOT / "results" / "integration"),
            "attention_quality": str(PROJECT_ROOT / "results" / "attention_quality"),
            "ablation": str(PROJECT_ROOT / "results" / "ablation"),
            "generation": str(PROJECT_ROOT / "results" / "generation"),
            "benchmarks": str(PROJECT_ROOT / "results" / "benchmarks"),
            "vector_search": str(PROJECT_ROOT / "results" / "vector_search"),
            "latency": str(PROJECT_ROOT / "results" / "latency"),
            "calibration_cost": str(PROJECT_ROOT / "results" / "calibration_cost"),
        },
    }

    # Try to pull key metrics from result files
    try:
        attn_path = PROJECT_ROOT / "results" / "attention_quality" / "attention_quality_results.json"
        if attn_path.exists():
            with open(attn_path) as f:
                attn_data = json.load(f)
            report["headline_results"] = {
                "gate2_pass": attn_data.get("gate2_pass"),
                "results_table": attn_data.get("results", []),
            }
    except Exception:
        pass

    try:
        eigen_path = PROJECT_ROOT / "results" / "eigenspectral" / "summary_statistics.json"
        if eigen_path.exists():
            with open(eigen_path) as f:
                eigen_data = json.load(f)
            report["eigenspectral_summary"] = eigen_data
    except Exception:
        pass

    with open(SUMMARY_FILE, "w") as f:
        json.dump(report, f, indent=2)

    log.info("Summary report saved → %s", SUMMARY_FILE)

    # Print human-readable summary to console
    print(banner("SPECTRALQUANT EXPERIMENT SUMMARY"))
    print(f"  Total time: {report['total_wall_time_s']:.0f}s")
    print(f"  Phases run: {', '.join(report['phases_run'])}")
    print()
    print("  Phase results:")
    for phase, result in phase_results.items():
        status = result.get("status", "unknown")
        elapsed = result.get("elapsed_s", 0)
        script = PHASE_SCRIPTS.get(phase, ("?", "?"))[1]
        status_sym = "✓" if status == "OK" else ("✗" if status == "FAILED" else "!")
        print(f"    [{status_sym}] Phase {phase}: {script}  ({elapsed:.0f}s)  [{status}]")

    if "headline_results" in report and report["headline_results"]:
        print()
        print("  Attention quality results:")
        for r in report["headline_results"].get("results_table", []):
            cos = r.get("cosine_similarity_mean")
            print(f"    {r['config']:8s} ({r['method']:14s}, {r['avg_bits']:.1f}bit): "
                  f"cosine_sim={cos:.5f}" if cos else f"    {r['config']}: N/A")
        gate2 = report["headline_results"].get("gate2_pass")
        if gate2 is not None:
            status = "PASSED" if gate2 else "FAILED"
            print(f"\n  GATE 2: {status}")

    print(banner("END OF SUMMARY"))


# ---------------------------------------------------------------------------
# Extra args builder
# ---------------------------------------------------------------------------

def build_extra_args(phase: str, args: argparse.Namespace) -> list[str]:
    """Build extra arguments to pass to each script."""
    extra = []

    # --quick flag propagated to all scripts
    if args.quick:
        extra.append("--quick")

    # --seed propagated universally
    extra += ["--seed", str(args.seed)]

    # Phase-specific overrides
    if phase == "0":
        if args.skip_clone:
            extra.append("--skip-clone")
        if args.skip_model:
            extra.append("--skip-model")

    if phase == "1":
        if args.n_seqs:
            extra += ["--n-seqs", str(args.n_seqs)]

    if phase in PHASE3_EXPS:
        if args.avg_bits:
            extra += ["--avg-bits", str(args.avg_bits)]
        if args.calib_dir:
            extra += ["--calib-dir", str(args.calib_dir)]

    return extra


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(phases_to_run: list[str], args: argparse.Namespace) -> dict:
    """
    Run the specified phases in order.
    Returns dict of {phase_id: {status, elapsed_s, returncode}}.
    """
    phase_results = {}

    for phase_id in phases_to_run:
        if phase_id not in PHASE_SCRIPTS:
            log.warning("Unknown phase '%s' — skipping.", phase_id)
            continue

        script_name, description = PHASE_SCRIPTS[phase_id]
        script_path = EXPERIMENTS_DIR / script_name

        if not script_path.exists():
            log.error("Script not found: %s — skipping phase %s.", script_path, phase_id)
            phase_results[phase_id] = {"status": "SCRIPT_NOT_FOUND", "elapsed_s": 0}
            continue

        print(banner(f"Phase {phase_id}: {description}"))
        log.info("Starting Phase %s: %s", phase_id, description)

        extra_args = build_extra_args(phase_id, args)
        returncode, elapsed = run_script(script_name, extra_args)

        status = "OK" if returncode == 0 else "FAILED"
        phase_results[phase_id] = {
            "status": status,
            "elapsed_s": round(elapsed, 2),
            "returncode": returncode,
            "script": script_name,
        }

        log.info("Phase %s finished: status=%s  elapsed=%.1fs", phase_id, status, elapsed)

        # Gate checks
        if phase_id == "1" and not args.skip_gate:
            gate1 = load_gate1_result()
            log.info("Gate 1 result: %s", gate1)
            if gate1 == "FAILED":
                log.error(
                    "GATE 1 FAILED — spectral gap is too weak. "
                    "Pipeline halted. Use --skip-gate to override."
                )
                phase_results[phase_id]["gate1_status"] = gate1
                break
            phase_results[phase_id]["gate1_status"] = gate1

        if phase_id == "3.1" and not args.skip_gate:
            gate2 = load_gate2_result()
            log.info("Gate 2 result: %s", gate2)
            if gate2 is False:
                log.warning(
                    "GATE 2 FAILED — SpectralQuant does not beat TurboQuant. "
                    "Continuing remaining experiments for diagnostic value, "
                    "but debug Exp 1 before writing paper. "
                    "Use --skip-gate to suppress this warning."
                )
            phase_results[phase_id]["gate2_pass"] = gate2

        if status == "FAILED" and not args.continue_on_error:
            log.error(
                "Phase %s FAILED (rc=%d). Halting pipeline. "
                "Use --continue-on-error to continue despite failures.",
                phase_id, returncode,
            )
            break

    return phase_results


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SpectralQuant master experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_all_experiments.py --quick\n"
            "  python run_all_experiments.py --phase 1\n"
            "  python run_all_experiments.py --phases 3.1 3.2 3.3\n"
            "  python run_all_experiments.py --no-phase0 --no-phase1\n"
        ),
    )

    # Phase selection
    phase_group = p.add_mutually_exclusive_group()
    phase_group.add_argument(
        "--phase", type=str, metavar="PHASE",
        help="Run a single phase (e.g. 0, 1, 3.1, 3.5)"
    )
    phase_group.add_argument(
        "--phases", nargs="+", type=str, metavar="PHASE",
        help="Run specific phases in order (e.g. --phases 1 2 3.1 3.2)"
    )

    # Phase skip flags
    p.add_argument("--no-phase0", action="store_true",
                   help="Skip Phase 0 (baseline reproduction)")
    p.add_argument("--no-phase1", action="store_true",
                   help="Skip Phase 1 (eigenspectral discovery) — use existing calibration")
    p.add_argument("--no-phase2", action="store_true",
                   help="Skip Phase 2 (integration tests)")

    # Experiment selection
    p.add_argument("--exp", nargs="+", type=str, metavar="N",
                   help="Run specific Phase 3 experiments (e.g. --exp 1 3 5)")

    # General
    p.add_argument("--quick", action="store_true",
                   help="Use reduced sample sizes for all scripts (debugging)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-gate", action="store_true",
                   help="Skip gate checks (Gate 1 and Gate 2)")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Continue pipeline even if a phase fails")

    # Phase-specific overrides
    p.add_argument("--n-seqs", type=int, default=None,
                   help="Override number of calibration sequences for Phase 1")
    p.add_argument("--avg-bits", type=float, default=None,
                   help="Override average bits for Phase 3 experiments")
    p.add_argument("--calib-dir", type=Path, default=None,
                   help="Override calibration directory (Phase 1 output dir)")

    # Phase 0 options
    p.add_argument("--skip-clone", action="store_true",
                   help="Skip TurboQuant clone in Phase 0")
    p.add_argument("--skip-model", action="store_true",
                   help="Skip model download in Phase 0")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    t_start = time.time()

    log.info("=" * 68)
    log.info("SpectralQuant Master Runner  %s", datetime.utcnow().isoformat())
    log.info("  quick=%s  skip_gate=%s  seed=%d", args.quick, args.skip_gate, args.seed)
    log.info("=" * 68)

    # ------------------------------------------------------------------
    # Determine which phases to run
    # ------------------------------------------------------------------
    if args.phase:
        # Single phase
        phases_to_run = [args.phase]
    elif args.phases:
        # Explicit list
        phases_to_run = args.phases
    else:
        # Full pipeline with optional exclusions
        phases_to_run = list(PIPELINE_ORDER)

        if args.no_phase0:
            phases_to_run = [p for p in phases_to_run if p != "0"]
            log.info("Skipping Phase 0 (--no-phase0)")
        if args.no_phase1:
            phases_to_run = [p for p in phases_to_run if p != "1"]
            log.info("Skipping Phase 1 (--no-phase1)")
        if args.no_phase2:
            phases_to_run = [p for p in phases_to_run if p != "2"]
            log.info("Skipping Phase 2 (--no-phase2)")

        if args.exp:
            # Replace all 3.x with only specified experiments
            phase3_requested = [f"3.{e}" for e in args.exp]
            phases_to_run = [p for p in phases_to_run if not p.startswith("3.")]
            phases_to_run += [p for p in phase3_requested if p in PHASE_SCRIPTS]
            log.info("Running Phase 3 experiments: %s", phase3_requested)

    log.info("Phases to run: %s", phases_to_run)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    phase_results = run_pipeline(phases_to_run, args)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    generate_summary_report(phase_results, t_start, args)

    # Exit code: 0 if all completed OK, 1 otherwise
    any_failed = any(r.get("status") == "FAILED" for r in phase_results.values())
    if any_failed:
        log.warning("One or more phases FAILED. Check experiment_log.txt for details.")
        sys.exit(1)
    else:
        log.info("All phases completed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
