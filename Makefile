# =============================================================================
# Makefile — SpectralQuant project automation
#
# Usage:
#   make setup        Complete B200 setup (run once)
#   make test         Run the full pytest test suite
#   make test-quick   Run tests with reduced verbosity
#   make lint         Check code style (ruff + black)
#   make format       Auto-format code (black + ruff --fix)
#   make typecheck    Run mypy type checks
#   make experiments  Run all experiments
#   make paper        Compile the paper
#   make clean        Remove generated files (keeps results structure)
#   make clean-all    Remove everything including results/
# =============================================================================

.PHONY: setup test test-quick lint format typecheck experiments paper \
        clean clean-all help

PYTHON     := python
PYTEST     := python -m pytest
RUFF       := ruff
BLACK      := black
MYPY       := mypy
PYTHONPATH := src

# Default target
all: help

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

setup:
	@echo "=== Running B200 setup script ==="
	bash scripts/setup_b200.sh

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test:
	@echo "=== Running test suite ==="
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/ -v

test-quick:
	@echo "=== Running test suite (brief output) ==="
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/ -q

test-cov:
	@echo "=== Running test suite with coverage ==="
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/ -v \
		--cov=spectralquant \
		--cov-report=term-missing \
		--cov-report=html:htmlcov/

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

lint:
	@echo "=== Running ruff ==="
	$(RUFF) check src/ experiments/ tests/ scripts/
	@echo "=== Checking black formatting ==="
	$(BLACK) --check src/ experiments/ tests/

format:
	@echo "=== Formatting with black ==="
	$(BLACK) src/ experiments/ tests/
	@echo "=== Fixing with ruff ==="
	$(RUFF) check --fix src/ experiments/ tests/

typecheck:
	@echo "=== Running mypy ==="
	$(MYPY) src/spectralquant/

# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

experiments:
	@echo "=== Running all experiments ==="
	bash scripts/run_all.sh

experiments-quick:
	@echo "=== Running quick experiment run (debug config) ==="
	bash scripts/run_all.sh --quick

phase1:
	@echo "=== Running Phase 1: Eigenspectral analysis ==="
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/phase1_eigenspectral.py

phase2:
	@echo "=== Running Phase 2: SpectralQuant integration ==="
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/phase2_integration.py

exp1:
	@echo "=== Running Experiment 1: Attention quality ==="
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/phase3_exp1_attention_quality.py

exp2:
	@echo "=== Running Experiment 2: Ablation ==="
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/phase3_exp2_ablation.py

# ---------------------------------------------------------------------------
# Paper
# ---------------------------------------------------------------------------

paper:
	@echo "=== Compiling paper ==="
	cd paper_output && $(PYTHON) generate_paper.py

# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

clean:
	@echo "=== Cleaning generated files (preserving results structure) ==="
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf dist/ build/
	@echo "Clean complete."

clean-results:
	@echo "=== Cleaning results (plots, CSVs, JSONs) ==="
	find results/ -name "*.png" -delete 2>/dev/null || true
	find results/ -name "*.csv" -delete 2>/dev/null || true
	find results/ -name "*.json" -delete 2>/dev/null || true
	find results/ -name "*.npz" -delete 2>/dev/null || true
	find results/ -name "*.npy" -delete 2>/dev/null || true
	@echo "Results cleaned."

clean-all: clean clean-results

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

help:
	@echo ""
	@echo "SpectralQuant Makefile"
	@echo "======================"
	@echo ""
	@echo "  make setup            Complete B200 setup (run once)"
	@echo "  make test             Run full pytest test suite"
	@echo "  make test-quick       Run tests with brief output"
	@echo "  make test-cov         Run tests with coverage report"
	@echo "  make lint             Check code style (ruff + black)"
	@echo "  make format           Auto-format code in place"
	@echo "  make typecheck        Run mypy type checks"
	@echo "  make experiments      Run all experiments (default config)"
	@echo "  make experiments-quick  Run with quick.yaml config"
	@echo "  make phase1           Run Phase 1 only"
	@echo "  make paper            Compile the paper"
	@echo "  make clean            Remove __pycache__, .pyc, etc."
	@echo "  make clean-results    Remove all result files"
	@echo "  make clean-all        clean + clean-results"
	@echo ""
