#!/bin/bash
# =============================================================================
# setup_b200.sh — Complete B200 setup script for SpectralQuant
#
# Run this on your RunPod B200 instance ONCE before running experiments.
# Usage: bash scripts/setup_b200.sh
#
# What this does:
#   1. Creates a Python virtual environment
#   2. Installs all SpectralQuant dependencies (dev + benchmarks)
#   3. Downloads the Qwen2.5-1.5B-Instruct model to HuggingFace cache
#   4. Downloads the WikiText-103 calibration dataset
#   5. Clones the TurboQuant baseline (turboquant_cutile)
#   6. Verifies GPU availability
#   7. Runs the test suite to confirm everything is working
# =============================================================================

set -euo pipefail

# Colours for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Colour

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo ""
echo "========================================================"
echo "       SpectralQuant B200 Setup"
echo "========================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 0: Verify we are in the repo root
# ---------------------------------------------------------------------------
if [[ ! -f "pyproject.toml" ]]; then
    error "Run this script from the repo root: bash scripts/setup_b200.sh"
fi

# ---------------------------------------------------------------------------
# Step 1: Create virtual environment
# ---------------------------------------------------------------------------
info "Step 1/7: Creating virtual environment..."
if [[ ! -d "venv" ]]; then
    python3 -m venv venv
    success "Created venv/"
else
    warn "venv/ already exists, skipping creation."
fi

source venv/bin/activate
info "Python: $(python --version)"
info "pip:    $(pip --version)"

# ---------------------------------------------------------------------------
# Step 2: Install dependencies
# ---------------------------------------------------------------------------
info "Step 2/7: Installing SpectralQuant with dev and benchmarks extras..."
pip install --upgrade pip setuptools wheel
pip install -e ".[dev,benchmarks]"
success "Dependencies installed."

# ---------------------------------------------------------------------------
# Step 3: Download model
# ---------------------------------------------------------------------------
info "Step 3/7: Downloading Qwen/Qwen2.5-1.5B-Instruct..."
python - <<'PYEOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"  Downloading tokenizer for {model_name}...", flush=True)
AutoTokenizer.from_pretrained(model_name)
print(f"  Downloading model weights for {model_name}...", flush=True)
AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
print("  Model downloaded successfully.", flush=True)
PYEOF
success "Model downloaded."

# ---------------------------------------------------------------------------
# Step 4: Download calibration data
# ---------------------------------------------------------------------------
info "Step 4/7: Downloading WikiText-103 calibration data..."
python - <<'PYEOF'
from datasets import load_dataset
print("  Downloading wikitext-103-raw-v1 (train split, 1000 samples)...")
load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:1000]")
print("  Dataset downloaded.")
PYEOF
success "Calibration data downloaded."

# ---------------------------------------------------------------------------
# Step 5: Clone TurboQuant baseline
# ---------------------------------------------------------------------------
info "Step 5/7: Cloning TurboQuant baseline..."
mkdir -p baseline
cd baseline
if [[ ! -d "turboquant_cutile/.git" ]]; then
    git clone https://github.com/DevTechJr/turboquant_cutile.git
    success "Cloned turboquant_cutile."
else
    warn "turboquant_cutile already cloned, pulling latest..."
    cd turboquant_cutile && git pull && cd ..
fi
cd ..

# ---------------------------------------------------------------------------
# Step 6: Verify GPU
# ---------------------------------------------------------------------------
info "Step 6/7: Verifying GPU..."
python - <<'PYEOF'
import torch
import sys

if not torch.cuda.is_available():
    print("ERROR: No CUDA GPU detected!")
    print("  torch.cuda.is_available() returned False")
    print("  Check that CUDA drivers are installed and the GPU is visible.")
    sys.exit(1)

n_gpus = torch.cuda.device_count()
print(f"  Found {n_gpus} GPU(s):")
for i in range(n_gpus):
    props = torch.cuda.get_device_properties(i)
    mem_gb = props.total_memory / (1024 ** 3)
    print(f"    GPU {i}: {props.name}  ({mem_gb:.1f} GB)")

print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA version:    {torch.version.cuda}")
PYEOF
success "GPU verified."

# ---------------------------------------------------------------------------
# Step 7: Run test suite
# ---------------------------------------------------------------------------
info "Step 7/7: Running test suite..."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
python -m pytest tests/ -v --tb=short

echo ""
echo "========================================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Activate venv:      source venv/bin/activate"
echo "  2. Set PYTHONPATH:     export PYTHONPATH=\"\$(pwd)/src\""
echo "  3. Run experiments:    bash scripts/run_all.sh"
echo "  4. Or step by step:    python experiments/phase1_eigenspectral.py"
echo "========================================================"
