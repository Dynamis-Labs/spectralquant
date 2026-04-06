#!/bin/bash
# =============================================================================
# run_all.sh — Master script to run all SpectralQuant experiments in order.
#
# Prerequisites: Run scripts/setup_b200.sh first.
# Usage: bash scripts/run_all.sh [--config CONFIG_PATH] [--quick]
#
# Flags:
#   --config PATH   Path to YAML config file (default: configs/default.yaml)
#   --quick         Use quick.yaml config for a fast debug run
#   --skip-phase0   Skip baseline reproduction (useful if already done)
#   --skip-phase1   Skip eigenspectral analysis (load from results/)
#   --results-dir   Override results output directory
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
CONFIG="configs/default.yaml"
SKIP_PHASE0=false
SKIP_PHASE1=false
RESULTS_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)    CONFIG="$2";           shift 2 ;;
        --quick)     CONFIG="configs/quick.yaml"; shift ;;
        --skip-phase0) SKIP_PHASE0=true;    shift ;;
        --skip-phase1) SKIP_PHASE1=true;    shift ;;
        --results-dir) RESULTS_DIR="$2";    shift 2 ;;
        *)           echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

step() { echo -e "\n${BLUE}━━━ $* ${NC}\n"; }
ok()   { echo -e "${GREEN}✓ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠ $*${NC}"; }

# ---------------------------------------------------------------------------
# Activate environment
# ---------------------------------------------------------------------------
if [[ ! -d "venv" ]]; then
    echo -e "${RED}ERROR: venv/ not found. Run: bash scripts/setup_b200.sh${NC}"
    exit 1
fi
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

[[ -n "$RESULTS_DIR" ]] && export SPECTRALQUANT_RESULTS_DIR="$RESULTS_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         SpectralQuant Full Experiment Run                ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Config:    $CONFIG"
echo "  Start:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Python:    $(python --version)"
echo ""

# ---------------------------------------------------------------------------
# Phase 0: Setup and baseline reproduction
# ---------------------------------------------------------------------------
if [[ "$SKIP_PHASE0" == false ]]; then
    step "Phase 0: Setup and baseline reproduction"
    python experiments/phase0_setup.py --config "$CONFIG"
    ok "Phase 0 complete."
else
    warn "Skipping Phase 0 (--skip-phase0 flag)."
fi

# ---------------------------------------------------------------------------
# Phase 1: Eigenspectral discovery
# ---------------------------------------------------------------------------
if [[ "$SKIP_PHASE1" == false ]]; then
    step "Phase 1: Eigenspectral analysis"
    python experiments/phase1_eigenspectral.py --config "$CONFIG"
    ok "Phase 1 complete."

    echo ""
    echo "  ┌─ REFLECTION GATE 1 ─────────────────────────────────────┐"
    echo "  │ Check results/phase1/summary.json for spectral gap κ.   │"
    echo "  │ If κ < 5 for most heads, consider pivoting the paper     │"
    echo "  │ focus to vector search (see docs/SPEC.md: Failure Mode 1)│"
    echo "  └──────────────────────────────────────────────────────────┘"
    echo ""
else
    warn "Skipping Phase 1 (--skip-phase1 flag). Loading cached calibration."
fi

# ---------------------------------------------------------------------------
# Phase 2: SpectralQuant integration
# ---------------------------------------------------------------------------
step "Phase 2: SpectralQuant integration"
python experiments/phase2_integration.py --config "$CONFIG"
ok "Phase 2 complete."

# ---------------------------------------------------------------------------
# Phase 3: Experiments
# ---------------------------------------------------------------------------
step "Phase 3, Experiment 1: Attention quality (main result)"
python experiments/phase3_exp1_attention_quality.py --config "$CONFIG"
ok "Experiment 1 complete."

echo ""
echo "  ┌─ REFLECTION GATE 2 ─────────────────────────────────────┐"
echo "  │ Check results/phase3_exp1/summary.json                   │"
echo "  │ SpectralQuant should match or beat TurboQuant on at      │"
echo "  │ least one comparison. If not, see SPEC.md Failure Mode 2.│"
echo "  └──────────────────────────────────────────────────────────┘"
echo ""

step "Phase 3, Experiment 2: Ablation study"
python experiments/phase3_exp2_ablation.py --config "$CONFIG"
ok "Experiment 2 complete."

step "Phase 3, Experiment 3: Text generation quality"
python experiments/phase3_exp3_generation.py --config "$CONFIG"
ok "Experiment 3 complete."

step "Phase 3, Experiment 4: Downstream benchmarks (LongBench / NIAH)"
python experiments/phase3_exp4_benchmarks.py --config "$CONFIG"
ok "Experiment 4 complete."

step "Phase 3, Experiment 5: Vector search (secondary contribution)"
python experiments/phase3_exp5_vector_search.py --config "$CONFIG"
ok "Experiment 5 complete."

step "Phase 3, Experiment 6: Latency benchmarks"
python experiments/phase3_exp6_latency.py --config "$CONFIG"
ok "Experiment 6 complete."

step "Phase 3, Experiment 7: Calibration cost and stability"
python experiments/phase3_exp7_calibration_cost.py --config "$CONFIG"
ok "Experiment 7 complete."

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         All experiments complete!                        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "  Results in: results/"
echo ""
echo "  Next step: compile the paper"
echo "    cd paper_output && python generate_paper.py"
echo ""
